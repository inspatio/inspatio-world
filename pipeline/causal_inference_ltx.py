"""LTX-2 Causal Inference Pipeline for InSpatio-World.

Drop-in replacement for CausalInferencePipeline using LTX-2.3 as backbone.
Supports three inference modes:

1. Multi-step denoising (Phase 1) — standard LTX-2 sigma schedule, 8-30 steps
2. VFM 1-step inference (Phase 3) — NoiseAdapterV1b + SigmaHead, single forward pass
3. SCD streaming (Phase 4) — encoder/decoder split with KV-cache, frame-by-frame

Architecture differences from Wan-based pipeline:
- Latent: 128 channels (vs 16), smaller spatial resolution
- Tokens/frame: ~336 (vs 1560) — less KV memory per frame
- No temporal frame grouping — each latent frame is independent
- Depth conditioning via channel concat + projection (same concept as Wan)
"""

from __future__ import annotations

import time
from typing import List, Optional

import torch
import torch.nn as nn
from contextlib import nullcontext

from utils.ltx_wrapper import LTXDiffusionWrapper, LTXTextEncoder, LTXVAEWrapper
from utils.ltx_scheduler import LTXFlowMatchScheduler, LTX2Scheduler


def denoise_block_ltx(
    generator: LTXDiffusionWrapper,
    sigma_schedule: torch.FloatTensor,
    noisy_input: torch.Tensor,
    conditional_dict: dict,
    kv_cache: list[dict] | None = None,
    *,
    context_frames: torch.Tensor | None = None,
    context_no_grad: bool = True,
    render_block: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    """Block-based denoising using LTX-2 sigma schedule (Euler ODE solver).

    Unlike the Wan version that uses discrete timesteps [1000, 750, 500, 250],
    this uses continuous sigma values from LTX2Scheduler for smoother denoising.

    Args:
        generator: LTX-2 diffusion model wrapper
        sigma_schedule: [N+1] sigma values from high to low (e.g. [1.0, ..., 0.0])
        noisy_input: [B, F, C, H, W] noisy latent to denoise
        conditional_dict: text embeddings
        kv_cache: optional KV cache for causal attention
        context_frames: [B, F_ctx, C, H, W] clean frames for context encoding
        context_no_grad: whether to disable gradients for context pass
        render_block: [B, F, C_depth, H, W] depth guidance

    Returns:
        (denoised_pred, noise_before_last_step)
    """
    B, F = noisy_input.shape[:2]
    device = noisy_input.device
    noise_before_last_step = None

    # Context encoding pass (clean frames -> populate KV cache)
    if context_frames is not None and kv_cache is not None:
        sigma_zero = torch.zeros([B], device=device)
        timestep_zero = sigma_zero * 1000  # sigma=0 -> timestep=0
        timestep_zero = timestep_zero.unsqueeze(1).expand(-1, context_frames.shape[1])

        ctx = torch.no_grad() if context_no_grad else nullcontext()
        with ctx:
            generator(
                noisy_image_or_video=context_frames,
                conditional_dict=conditional_dict,
                timestep=timestep_zero,
                kv_cache=kv_cache,
                kv_size=(0, -1),  # append to cache
            )

    # Denoising loop using sigma schedule
    num_steps = len(sigma_schedule) - 1
    x_t = noisy_input

    for step_idx in range(num_steps):
        sigma_current = sigma_schedule[step_idx].item()
        sigma_next = sigma_schedule[step_idx + 1].item()
        is_last_step = (step_idx == num_steps - 1)

        # Build timestep from sigma
        timestep = torch.full(
            [B, F], sigma_current * 1000,
            device=device, dtype=torch.long,
        )

        ctx = torch.no_grad() if not is_last_step else nullcontext()
        with ctx:
            flow_pred, pred_x0 = generator(
                noisy_image_or_video=x_t,
                conditional_dict=conditional_dict,
                timestep=timestep,
                kv_cache=kv_cache,
                render_latent_input=render_block,
            )

        if is_last_step:
            noise_before_last_step = x_t.clone()
            denoised_pred = pred_x0
        else:
            # Euler step: x_{t+1} = x_t + flow_pred * (sigma_next - sigma_current)
            dt = sigma_next - sigma_current
            x_t = x_t + flow_pred * dt

    return denoised_pred, noise_before_last_step


def denoise_block_vfm(
    generator: LTXDiffusionWrapper,
    structured_noise: torch.Tensor,
    conditional_dict: dict,
    kv_cache: list[dict] | None = None,
    *,
    context_frames: torch.Tensor | None = None,
    render_block: torch.Tensor | None = None,
    per_token_sigma: torch.Tensor | None = None,
) -> torch.Tensor:
    """1-step VFM denoising — single forward pass through DiT.

    The noise adapter has already produced structured noise z that encodes
    text-conditioned layout/motion information. One DiT pass at sigma=1
    converts z directly to x0.

    Args:
        generator: LTX-2 diffusion model wrapper
        structured_noise: [B, F, C, H, W] from NoiseAdapterV1b
        conditional_dict: text embeddings
        kv_cache: optional KV cache
        context_frames: clean reference frames for context
        render_block: depth guidance
        per_token_sigma: [B, seq] optional per-token sigma from SigmaHead

    Returns:
        denoised_pred: [B, F, C, H, W]
    """
    B, F = structured_noise.shape[:2]
    device = structured_noise.device

    # Context encoding (same as multi-step)
    if context_frames is not None and kv_cache is not None:
        sigma_zero = torch.zeros([B], device=device)
        timestep_zero = (sigma_zero * 1000).unsqueeze(1).expand(-1, context_frames.shape[1])
        with torch.no_grad():
            generator(
                noisy_image_or_video=context_frames,
                conditional_dict=conditional_dict,
                timestep=timestep_zero,
                kv_cache=kv_cache,
                kv_size=(0, -1),
            )

    # Single forward pass at sigma=1 (pure noise -> clean video)
    timestep = torch.full([B, F], 1000, device=device, dtype=torch.long)

    flow_pred, pred_x0 = generator(
        noisy_image_or_video=structured_noise,
        conditional_dict=conditional_dict,
        timestep=timestep,
        kv_cache=kv_cache,
        render_latent_input=render_block,
    )

    # x0 = z - v (flow matching convention)
    return pred_x0


class LTXCausalInferencePipeline(nn.Module):
    """LTX-2.3 based causal inference pipeline for InSpatio-World.

    Replaces CausalInferencePipeline with LTX-2 backbone while maintaining
    the same block-by-block generation pattern with KV caching.
    """

    def __init__(self, args, device):
        super().__init__()

        # Config
        self.args = args
        self.device = device

        checkpoint_path = getattr(args, "ltx_checkpoint_path", None)
        text_encoder_path = getattr(args, "ltx_text_encoder_path", None)
        quantization = getattr(args, "ltx_quantization", "int8-quanto")
        lora_path = getattr(args, "ltx_lora_path", None)
        depth_channels = getattr(args, "ltx_depth_channels", 0)

        # Initialize models
        print("Initializing LTX-2.3 pipeline...")
        t0 = time.time()

        self.generator = LTXDiffusionWrapper(
            checkpoint_path=checkpoint_path,
            quantization=quantization,
            device=str(device),
            lora_path=lora_path,
            depth_in_channels=depth_channels,
        )
        print(f"  Generator loaded in {time.time() - t0:.1f}s")

        t0 = time.time()
        if text_encoder_path and not getattr(args, "use_cached_embeddings", False):
            self.text_encoder = LTXTextEncoder(
                gemma_path=text_encoder_path,
                device=str(device),
            )
            print(f"  Text encoder loaded in {time.time() - t0:.1f}s")
        else:
            self.text_encoder = None
            print("  Text encoder: using cached embeddings")

        t0 = time.time()
        self.vae = LTXVAEWrapper(
            checkpoint_path=checkpoint_path,
            device=str(device),
        )
        print(f"  VAE loaded in {time.time() - t0:.1f}s")

        # Scheduler
        self.num_inference_steps = getattr(args, "ltx_num_inference_steps", 30)
        self.scheduler = LTX2Scheduler()

        # Block parameters
        self.num_frame_per_block = getattr(args, "num_frame_per_block", 1)
        # LTX: smaller spatial, so fewer tokens per frame
        # For 768x448 -> latent 24x14 = 336 tokens/frame
        self.frame_seq_length = getattr(args, "ltx_frame_seq_length", 336)

        # KV cache
        self.kv_cache = None

        # VFM components (loaded separately if available)
        self.noise_adapter = None
        self.sigma_head = None

        print(f"  KV inference: {self.num_frame_per_block} frames/block, "
              f"{self.frame_seq_length} tokens/frame")

    def load_vfm_adapter(
        self,
        adapter_path: str,
        sigma_head_path: str | None = None,
        text_dim: int = 3840,
        latent_dim: int = 128,
    ):
        """Load VFM noise adapter and optional sigma head for 1-step inference."""
        from utils.noise_adapter import NoiseAdapterV1b, SigmaHead
        from safetensors.torch import load_file

        print(f"  Loading VFM noise adapter from {adapter_path}...")
        self.noise_adapter = NoiseAdapterV1b(
            text_dim=text_dim,
            latent_dim=latent_dim,
        )
        state_dict = load_file(adapter_path)
        self.noise_adapter.load_state_dict(state_dict)
        self.noise_adapter = self.noise_adapter.to(self.device, torch.bfloat16)
        self.noise_adapter.eval()

        if sigma_head_path:
            print(f"  Loading SigmaHead from {sigma_head_path}...")
            self.sigma_head = SigmaHead(latent_dim=latent_dim)
            sh_state = load_file(sigma_head_path)
            self.sigma_head.load_state_dict(sh_state)
            self.sigma_head = self.sigma_head.to(self.device, torch.bfloat16)
            self.sigma_head.eval()

    def _initialize_kv_cache(self, batch_size: int, dtype: torch.dtype, device: torch.device):
        """Initialize KV cache for LTX-2 transformer blocks.

        LTX-2 has 48 transformer blocks. Each stores K,V tensors for
        cached context frames.
        """
        max_context_frames = 6
        kv_cache_size = self.frame_seq_length * max_context_frames
        num_heads = 32  # LTX-2.3 default
        head_dim = 128  # 4096 / 32

        if self.kv_cache is not None:
            for block_cache in self.kv_cache:
                block_cache["k"].detach_().zero_()
                block_cache["v"].detach_().zero_()
            return

        print(f"  Initializing KV cache: {kv_cache_size} tokens, "
              f"{self.generator.num_layers} layers")
        self.kv_cache = []
        for _ in range(self.generator.num_layers):
            self.kv_cache.append({
                "k": torch.zeros(
                    [batch_size, kv_cache_size, num_heads, head_dim],
                    dtype=dtype, device=device,
                ),
                "v": torch.zeros(
                    [batch_size, kv_cache_size, num_heads, head_dim],
                    dtype=dtype, device=device,
                ),
            })

    def inference(
        self,
        noise: torch.Tensor,
        text_prompts: List[str] | None = None,
        conditional_dict: dict | None = None,
        ref_latent: Optional[torch.Tensor] = None,
        render_latent: Optional[torch.Tensor] = None,
        mask_latent: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Main inference entry point.

        Supports both text prompt encoding and pre-computed embeddings.

        Args:
            noise: [B, F, C, H, W] random noise
            text_prompts: list of text prompts (if text_encoder loaded)
            conditional_dict: pre-computed embeddings (alternative to text_prompts)
            ref_latent: [B, F, C, H, W] source video latents
            render_latent: [B, F, C_r, H, W] depth-rendered guidance
            mask_latent: [B, F, C_m, H, W] depth mask

        Returns:
            video: [B, T, 3, H, W] pixel video in [0, 1]
        """
        batch_size, num_frames, num_channels, height, width = noise.shape
        assert num_frames % self.num_frame_per_block == 0
        num_blocks = num_frames // self.num_frame_per_block
        device = noise.device
        dtype = noise.dtype

        # Encode text
        if conditional_dict is None:
            if self.text_encoder is not None and text_prompts is not None:
                conditional_dict = self.text_encoder(text_prompts)
            else:
                raise ValueError("Either text_prompts+text_encoder or conditional_dict required")

        # Initialize output
        output = torch.zeros_like(noise)

        # Initialize KV cache
        self._initialize_kv_cache(batch_size, dtype, device)

        # Compute sigma schedule
        # Create dummy latent for token-aware schedule
        dummy = torch.empty(1, 1, self.num_frame_per_block, height, width)
        sigma_schedule = self.scheduler.execute(
            steps=self.num_inference_steps,
            latent=dummy,
        ).to(device)

        # Combine depth guidance
        render_combined = None
        if render_latent is not None and mask_latent is not None:
            render_combined = torch.cat([mask_latent, render_latent], dim=2)

        # Block-by-block generation
        print(f"Generating {num_blocks} blocks ({self.num_inference_steps} steps each)...")
        t_start = time.time()

        start_index = 0
        last_pred = None

        for block_idx in range(num_blocks):
            block_start = start_index
            block_end = start_index + self.num_frame_per_block

            noisy_input = noise[:, block_start:block_end].to(device, dtype)
            ref_block = ref_latent[:, block_start:block_end].to(device, dtype) if ref_latent is not None else None

            # Render block
            render_block = None
            if render_combined is not None:
                render_block = render_combined[:, block_start:block_end].to(device, dtype)

            # Prepare context frames
            context_frames = None
            if ref_block is not None:
                if block_idx == 0:
                    context_frames = ref_block
                elif last_pred is not None:
                    context_frames = torch.cat([ref_block, last_pred], dim=1)

            # Choose inference mode
            if self.noise_adapter is not None:
                # VFM 1-step mode
                positions = self.generator._build_positions(
                    num_frames=self.num_frame_per_block,
                    latent_h=height, latent_w=width,
                    batch_size=batch_size, device=device, dtype=dtype,
                )
                prompt_embeds = conditional_dict["prompt_embeds"].to(device, dtype)
                prompt_mask = conditional_dict.get("prompt_attention_mask")
                if prompt_mask is not None:
                    prompt_mask = prompt_mask.to(device)

                from utils.noise_adapter import TASK_CLASSES
                task_class = torch.full(
                    [batch_size], TASK_CLASSES["v2v"],
                    device=device, dtype=torch.long,
                )

                with torch.no_grad():
                    mu, log_sigma = self.noise_adapter(
                        text_embeddings=prompt_embeds,
                        text_mask=prompt_mask.bool() if prompt_mask is not None else None,
                        positions=positions,
                        task_class=task_class,
                    )

                # Spherical Cauchy sampling
                from utils.spherical_utils import normalize, sample_spherical_cauchy
                mu_hat = normalize(mu)
                mu_norm = mu.norm(p=2, dim=-1)
                kappa = torch.exp(log_sigma.mean(dim=-1)).clamp(0.1, 50.0)

                B_flat, seq, D = mu.shape
                z_dir = sample_spherical_cauchy(
                    mu_hat.reshape(-1, D),
                    kappa.reshape(-1),
                ).reshape(B_flat, seq, D)

                structured_noise = mu_norm.unsqueeze(-1) * z_dir
                # Unpatchify back to spatial
                structured_noise_spatial = self.generator._unpatchify(
                    structured_noise, self.num_frame_per_block, height, width,
                ).permute(0, 2, 1, 3, 4)  # [B, F, C, H, W]

                denoised_pred = denoise_block_vfm(
                    self.generator, structured_noise_spatial,
                    conditional_dict, self.kv_cache,
                    context_frames=context_frames,
                    render_block=render_block,
                )
            else:
                # Multi-step denoising mode
                denoised_pred, _ = denoise_block_ltx(
                    self.generator, sigma_schedule,
                    noisy_input, conditional_dict, self.kv_cache,
                    context_frames=context_frames,
                    render_block=render_block,
                )

            output[:, block_start:block_end] = denoised_pred
            last_pred = denoised_pred.clone().detach()
            start_index += self.num_frame_per_block

            elapsed = time.time() - t_start
            print(f"  Block {block_idx+1}/{num_blocks} done ({elapsed:.1f}s)")

        # Decode latent -> pixel
        # LTX VAE expects [B, C, F, H, W]
        output_bcfhw = output.permute(0, 2, 1, 3, 4)  # [B, C, F, H, W]
        video = self.vae.decode_to_pixel(output_bcfhw)
        # Output: [B, 3, F, H, W] -> [B, F, 3, H, W]
        video = video.permute(0, 2, 1, 3, 4)
        video = (video * 0.5 + 0.5).clamp(0, 1)

        total_time = time.time() - t_start
        print(f"Generation complete: {total_time:.1f}s total, "
              f"{total_time/num_blocks:.1f}s/block")

        return video

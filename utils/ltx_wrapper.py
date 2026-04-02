"""LTX-2.3 (22B) model wrappers for InSpatio-World.

Drop-in replacements for WanDiffusionWrapper, WanTextEncoder, and WanVAEWrapper
using LTX-2.3 22B DiT as the diffusion backbone.

Model files (from https://huggingface.co/Lightricks/LTX-2.3):
- ltx-2.3-22b-dev.safetensors        (43GB bf16, for training)
- ltx-2.3-22b-distilled.safetensors  (43GB bf16, 8-step distilled)

Architecture:
- LTXDiffusionWrapper: 22B DiT (48-layer), supports LoRA, quantization, KV-cache
- LTXTextEncoder: Gemma-3 12B text encoder
- LTXVAEWrapper: Video VAE (128-channel latent space)

The flow matching convention matches InSpatio's existing interface:
    flow_pred = noise - x0
    x_t = (1 - sigma) * x0 + sigma * noise
    x0 = x_t - sigma * flow_pred
"""

from __future__ import annotations

import os
import re
import types
from pathlib import Path
from typing import List, Optional, Tuple

import torch
import torch.nn as nn
from safetensors.torch import load_file as safe_load_file

from utils.ltx_scheduler import LTXFlowMatchScheduler


# ─────────────────────────────────────────────────────────────────────
# Text Encoder (Gemma-3 12B)
# ─────────────────────────────────────────────────────────────────────

class LTXTextEncoder(nn.Module):
    """Gemma-3 based text encoder for LTX-2.3.

    Loads the AVGemma encoder and produces video-conditioning embeddings.
    Output shape: [B, 1024, 3840] with attention mask [B, 1024].
    """

    def __init__(
        self,
        gemma_path: str,
        device: str = "cuda",
        dtype: torch.dtype = torch.bfloat16,
        load_in_8bit: bool = False,
    ):
        super().__init__()
        self.device = device
        self.dtype = dtype

        from ltx_trainer.model_loader import load_text_encoder
        self.encoder = load_text_encoder(
            gemma_model_path=gemma_path,
            device=device, dtype=dtype,
            load_in_8bit=load_in_8bit,
        )
        self.encoder.eval()

    @torch.inference_mode()
    def forward(self, text_prompts: List[str]) -> dict:
        """Encode text prompts.

        Returns:
            dict with 'prompt_embeds' [B, 1024, 3840] and
            'prompt_attention_mask' [B, 1024]
        """
        video_embeds, _audio_embeds, attention_mask = self.encoder(text_prompts[0])
        return {
            "prompt_embeds": video_embeds.to(self.dtype),
            "prompt_attention_mask": attention_mask,
        }

    @staticmethod
    def load_cached_embedding(path: str, dtype: torch.dtype = torch.bfloat16) -> dict:
        """Load precomputed text embeddings from .pt file.

        This skips loading the 28GB Gemma encoder entirely.
        """
        emb = torch.load(path, map_location="cpu", weights_only=True)
        return {
            "prompt_embeds": emb["video_prompt_embeds"].unsqueeze(0).to(dtype),
            "prompt_attention_mask": emb["prompt_attention_mask"].unsqueeze(0),
        }


# ─────────────────────────────────────────────────────────────────────
# Video VAE
# ─────────────────────────────────────────────────────────────────────

class LTXVAEWrapper(nn.Module):
    """LTX-2.3 Video VAE for encoding/decoding video latents.

    Latent space: 128 channels, spatial compression 32x, temporal compression 8x.
    Frame rule: pixel frames must satisfy (F - 1) % 8 == 0
    """

    def __init__(
        self,
        checkpoint_path: str,
        device: str = "cuda",
        dtype: torch.dtype = torch.bfloat16,
    ):
        super().__init__()
        self.device = device
        self.dtype = dtype

        from ltx_trainer.model_loader import load_video_vae_encoder, load_video_vae_decoder

        self._encoder = load_video_vae_encoder(
            checkpoint_path, device=device, dtype=dtype,
        )
        self._decoder = load_video_vae_decoder(
            checkpoint_path, device=device, dtype=dtype,
        )

    def encode_to_latent(self, pixel: torch.Tensor) -> torch.Tensor:
        """Encode pixel video to latent space.

        Args:
            pixel: [B, C, F, H, W] in [-1, 1] range

        Returns:
            [B, 128, F', H', W'] where F'=(F-1)/8+1, H'=H/32, W'=W/32
        """
        with torch.inference_mode():
            latents = []
            for i in range(pixel.shape[0]):
                z = self._encoder(pixel[i:i+1].to(self.device, self.dtype))
                latents.append(z.float())
            return torch.cat(latents, dim=0)

    def decode_to_pixel(self, latent: torch.Tensor, use_cache: bool = False) -> torch.Tensor:
        """Decode latent to pixel video.

        Args:
            latent: [B, 128, F', H', W']

        Returns:
            [B, 3, F, H, W] in [-1, 1] range
        """
        with torch.inference_mode():
            pixels = []
            for i in range(latent.shape[0]):
                p = self._decoder(latent[i:i+1].to(self.device, self.dtype))
                pixels.append(p.float().clamp(-1, 1))
            return torch.cat(pixels, dim=0)


# ─────────────────────────────────────────────────────────────────────
# LoRA Utilities
# ─────────────────────────────────────────────────────────────────────

def extract_lora_target_modules(state_dict: dict[str, torch.Tensor]) -> list[str]:
    """Extract target module names from LoRA checkpoint keys."""
    target_modules = set()
    pattern = re.compile(r"(.+)\.lora_[AB]\.")
    for key in state_dict:
        match = pattern.match(key)
        if match:
            target_modules.add(match.group(1))
    return sorted(target_modules)


def load_lora_weights(
    model: nn.Module,
    lora_path: str | Path,
) -> nn.Module:
    """Load LoRA weights into a transformer model."""
    from peft import LoraConfig, get_peft_model, set_peft_model_state_dict

    print(f"  Loading LoRA from {lora_path}...")
    state_dict = safe_load_file(str(lora_path))

    # Remove common prefixes
    state_dict = {k.replace("diffusion_model.", "", 1): v for k, v in state_dict.items()}
    # Normalize SCD paths
    normalized = {}
    for key, value in state_dict.items():
        if key.startswith("encoder_blocks.") or key.startswith("decoder_blocks."):
            continue
        if key.startswith("base_model."):
            key = key[len("base_model."):]
        normalized[key] = value
    state_dict = normalized

    target_modules = extract_lora_target_modules(state_dict)
    if not target_modules:
        raise ValueError(f"No LoRA modules found in {lora_path}")

    lora_rank = None
    for key, value in state_dict.items():
        if "lora_A" in key and value.ndim == 2:
            lora_rank = value.shape[0]
            break
    if lora_rank is None:
        raise ValueError("Could not detect LoRA rank")

    print(f"  {len(target_modules)} target modules, rank {lora_rank}")

    config = LoraConfig(
        r=lora_rank,
        lora_alpha=lora_rank,
        target_modules=target_modules,
        lora_dropout=0.0,
        init_lora_weights=True,
    )
    model = get_peft_model(model, config)
    set_peft_model_state_dict(model.get_base_model(), state_dict)
    print("  LoRA weights loaded")
    return model


# ─────────────────────────────────────────────────────────────────────
# Diffusion Model Wrapper
# ─────────────────────────────────────────────────────────────────────

class LTXDiffusionWrapper(nn.Module):
    """LTX-2.3 DiT wrapper for InSpatio-World.

    Provides a forward() interface compatible with InSpatio's causal inference
    pipeline. Internally uses the LTX Modality dataclass for model input.

    Supports:
    - Standard 48-layer full model (multi-step denoising)
    - SCD encoder/decoder split (autoregressive streaming)
    - VFM 1-step inference (via noise adapter)
    - int8-quanto / fp8-quanto quantization
    - LoRA fine-tuning weights
    - Depth conditioning via channel concatenation
    """

    def __init__(
        self,
        checkpoint_path: str,
        quantization: str = "int8-quanto",
        device: str = "cuda:0",
        dtype: torch.dtype = torch.bfloat16,
        lora_path: str | None = None,
        depth_in_channels: int = 0,
    ):
        super().__init__()
        self.device_name = device
        self.dtype = dtype
        self.depth_in_channels = depth_in_channels

        # Load transformer
        print(f"  Loading LTX-2.3 transformer from {checkpoint_path}...")
        from ltx_trainer.model_loader import load_transformer
        self.model = load_transformer(
            checkpoint_path, device="cpu", dtype=dtype,
        )

        # Quantize if requested
        if quantization != "none":
            print(f"  Applying {quantization} quantization...")
            from ltx_trainer.quantization import quantize_model
            self.model = quantize_model(
                self.model, quantization, device=device,
            )
        else:
            self.model = self.model.to(device)

        # Apply LoRA if provided
        if lora_path:
            self.model = load_lora_weights(self.model, lora_path)

        # Depth input projection (for channel concatenation)
        # Concatenate depth latent channels with video latent channels
        if depth_in_channels > 0:
            self.depth_proj = nn.Linear(
                128 + depth_in_channels, 128
            ).to(device, dtype)
        else:
            self.depth_proj = None

        self.model.eval()

        # Scheduler
        self.scheduler = LTXFlowMatchScheduler(num_inference_steps=30)

        # Model config
        self._num_layers = 48  # LTX-2.3 default

    def get_scheduler(self):
        return self.scheduler

    @property
    def num_layers(self) -> int:
        return self._num_layers

    def _build_positions(
        self,
        num_frames: int,
        latent_h: int,
        latent_w: int,
        batch_size: int,
        fps: float = 24.0,
        device: torch.device | str = "cuda",
        dtype: torch.dtype = torch.bfloat16,
    ) -> torch.Tensor:
        """Build spatiotemporal positions [B, 3, seq, 2] for LTX-2.

        Each token has (time, height, width) bounds as (start, end) pairs.
        The LTX patchifier uses patch_size=1, so each token is one latent pixel.
        """
        tokens_per_frame = latent_h * latent_w
        total_tokens = num_frames * tokens_per_frame

        # Build coordinate grids
        positions = torch.zeros(batch_size, 3, total_tokens, 2, device=device, dtype=dtype)

        for f in range(num_frames):
            frame_start = f * tokens_per_frame
            t_val = f / max(fps, 1.0)

            for h in range(latent_h):
                for w in range(latent_w):
                    token_idx = frame_start + h * latent_w + w
                    # Time bounds
                    positions[:, 0, token_idx, 0] = t_val
                    positions[:, 0, token_idx, 1] = t_val + 1.0 / fps
                    # Height bounds (normalized)
                    positions[:, 1, token_idx, 0] = h / latent_h
                    positions[:, 1, token_idx, 1] = (h + 1) / latent_h
                    # Width bounds (normalized)
                    positions[:, 2, token_idx, 0] = w / latent_w
                    positions[:, 2, token_idx, 1] = (w + 1) / latent_w

        return positions

    def _patchify(self, latent: torch.Tensor) -> torch.Tensor:
        """Convert [B, C, F, H, W] latent to [B, F*H*W, C] token sequence.

        LTX-2 uses patch_size=1, so this is just a reshape.
        """
        B, C, F, H, W = latent.shape
        return latent.permute(0, 2, 3, 4, 1).reshape(B, F * H * W, C)

    def _unpatchify(
        self, tokens: torch.Tensor, F: int, H: int, W: int,
    ) -> torch.Tensor:
        """Convert [B, F*H*W, C] token sequence back to [B, C, F, H, W]."""
        B = tokens.shape[0]
        C = tokens.shape[-1]
        return tokens.reshape(B, F, H, W, C).permute(0, 4, 1, 2, 3)

    def forward(
        self,
        noisy_image_or_video: torch.Tensor,
        conditional_dict: dict,
        timestep: torch.Tensor,
        kv_cache: list[dict] | None = None,
        render_latent_input: torch.Tensor | None = None,
        kv_size: tuple[int, int] = (0, 0),
        freqs_offset: int = 0,
        **kwargs,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Forward pass through LTX-2.3 DiT.

        Args:
            noisy_image_or_video: [B, F, C, H, W] noisy latent
            conditional_dict: {'prompt_embeds': [B, T, D], 'prompt_attention_mask': [B, T]}
            timestep: [B, F] or [B] timestep values (0-1000 range)
            kv_cache: Optional KV cache for causal inference
            render_latent_input: [B, F, C_depth, H, W] depth guidance (optional)
            kv_size: KV cache size parameters
            freqs_offset: RoPE frequency offset

        Returns:
            (flow_pred, pred_x0) both [B, F, C, H, W]
        """
        B, F_frames, C, H, W = noisy_image_or_video.shape
        device = noisy_image_or_video.device

        # Patchify: [B, F, C, H, W] -> [B, seq, C]
        tokens = self._patchify(noisy_image_or_video)

        # Handle depth conditioning via channel concat + projection
        if render_latent_input is not None and self.depth_proj is not None:
            depth_tokens = self._patchify(render_latent_input)
            tokens = torch.cat([tokens, depth_tokens], dim=-1)  # [B, seq, C + C_depth]
            tokens = self.depth_proj(tokens)  # [B, seq, C]

        # Convert timestep to sigma
        if timestep.dim() == 2:
            sigma = timestep[:, 0].float() / 1000.0  # [B]
        else:
            sigma = timestep.float() / 1000.0  # [B]

        # Build per-token timesteps
        seq_len = tokens.shape[1]
        per_token_timesteps = sigma.unsqueeze(1).expand(-1, seq_len)  # [B, seq]

        # Build positions
        positions = self._build_positions(
            num_frames=F_frames, latent_h=H, latent_w=W,
            batch_size=B, device=device, dtype=self.dtype,
        )

        # Build Modality for LTX model
        from ltx_core.model.transformer.modality import Modality

        video_modality = Modality(
            enabled=True,
            latent=tokens,
            sigma=sigma,
            timesteps=per_token_timesteps,
            positions=positions,
            context=conditional_dict["prompt_embeds"].to(device, self.dtype),
            context_mask=conditional_dict.get("prompt_attention_mask"),
        )

        # Forward through LTX model
        from ltx_core.model.transformer.perturbation import BatchedPerturbationConfig
        perturbations = BatchedPerturbationConfig(batch_size=B)

        video_output, _audio_output = self.model(
            video=video_modality,
            audio=None,
            perturbations=perturbations,
        )

        # Unpatchify: [B, seq, C] -> [B, C, F, H, W] -> [B, F, C, H, W]
        flow_pred = self._unpatchify(video_output, F_frames, H, W)
        flow_pred = flow_pred.permute(0, 2, 1, 3, 4)  # [B, F, C, H, W]

        # Convert flow prediction to x0
        # flow_pred = noise - x0, and x_t = (1-sigma)*x0 + sigma*noise
        # => x0 = x_t - sigma * flow_pred
        sigma_expanded = sigma.view(B, 1, 1, 1, 1)
        pred_x0 = noisy_image_or_video - sigma_expanded * flow_pred

        return flow_pred, pred_x0

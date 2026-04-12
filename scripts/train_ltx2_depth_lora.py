#!/usr/bin/env python3
"""Train depth-conditioned LoRA for LTX-2.3 on InSpatio video pairs.

This is Stage 1 of the VFM v1f upgrade plan:
- Train LTX-2.3 22B (LoRA r=32) to perform depth-guided novel-view synthesis
- Uses standard multi-step flow matching (not VFM yet)
- Input: source video + depth render + mask -> target video
- Depth conditioning via channel concatenation

Once this produces good results, move to Stage 2 (VFM v1f adapter training).

Usage:
    # Basic training (single GPU, int8-quanto)
    python scripts/train_ltx2_depth_lora.py \
        --config configs/train_ltx2_depth.yaml

    # Resume from checkpoint
    python scripts/train_ltx2_depth_lora.py \
        --config configs/train_ltx2_depth.yaml \
        --resume checkpoints/lora_step_2000.safetensors

Architecture:
    source_video -> LTX VAE encode -> ref_latent [B, 128, F, H', W']
    render_video -> LTX VAE encode -> render_latent [B, 128, F, H', W']
    mask_video   -> spatial downsample -> mask_latent [B, 1, F, H', W']

    noisy_target = (1-sigma) * target_latent + sigma * noise

    input_to_dit = cat([noisy_target, render_latent, mask_latent], dim=channel)
                 = [B, 128+128+1, F, H', W']
    -> depth_proj(257, 128) -> [B, 128, F, H', W']
    -> LTX-2.3 DiT (48 layers, LoRA r=32) -> velocity
    -> loss = MSE(velocity, noise - target_latent)
"""

from __future__ import annotations

import argparse
import gc
import os
import time
from pathlib import Path

import torch
import torch.nn as nn
from peft import LoraConfig, get_peft_model
from safetensors.torch import save_file, load_file
from torch.utils.data import DataLoader

# These are available after running scripts/setup_ltx2.sh
from ltx_trainer.model_loader import (
    load_transformer,
    load_video_vae_encoder,
    load_video_vae_decoder,
)
from ltx_core.components.schedulers import LTX2Scheduler


def parse_args():
    parser = argparse.ArgumentParser(description="Train depth-conditioned LoRA for LTX-2.3")

    # Model
    parser.add_argument("--checkpoint", type=str, required=True,
                        help="Path to ltx-2.3-22b-dev.safetensors")
    parser.add_argument("--quantization", type=str, default="int8-quanto",
                        choices=["int8-quanto", "fp8-quanto", "none"])
    parser.add_argument("--resume", type=str, default=None,
                        help="Resume from LoRA checkpoint")

    # LoRA
    parser.add_argument("--lora-rank", type=int, default=32)
    parser.add_argument("--lora-alpha", type=int, default=32)
    parser.add_argument("--lora-targets", type=str, nargs="+",
                        default=["to_q", "to_k", "to_v", "to_out.0"],
                        help="LoRA target modules in the transformer")

    # Training
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--steps", type=int, default=5000)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--grad-accum", type=int, default=4)
    parser.add_argument("--log-every", type=int, default=50)
    parser.add_argument("--save-every", type=int, default=500)
    parser.add_argument("--val-every", type=int, default=500)
    parser.add_argument("--seed", type=int, default=42)

    # Data
    parser.add_argument("--data-json", type=str, required=True,
                        help="JSON file with video pairs (source, target, render, mask)")
    parser.add_argument("--resolution", type=int, nargs=2, default=[448, 768],
                        help="Video resolution [H, W]")
    parser.add_argument("--num-frames", type=int, default=25,
                        help="Pixel frames per clip (must satisfy (F-1)%%8==0)")

    # Depth conditioning
    parser.add_argument("--depth-channels", type=int, default=129,
                        help="Depth input channels (128 VAE-encoded render + 1 mask)")

    # Output
    parser.add_argument("--output-dir", type=str, default="./output/ltx2_depth_lora")
    parser.add_argument("--wandb-project", type=str, default="inspatio-ltx2")

    # Device
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--vae-device", type=str, default="cuda:0",
                        help="Device for VAE (can be different GPU)")

    return parser.parse_args()


class DepthProjection(nn.Module):
    """Projects concatenated [video + render + mask] latents back to 128ch.

    Input: [B, 128 + 128 + 1, F, H, W] = [B, 257, F, H, W]
    Output: [B, 128, F, H, W]
    """

    def __init__(self, in_channels: int = 257, out_channels: int = 128):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size=1, bias=True),
            nn.SiLU(),
            nn.Conv3d(out_channels, out_channels, kernel_size=1, bias=True),
        )
        # Initialize near-identity for the video channels
        nn.init.zeros_(self.proj[0].weight)
        nn.init.zeros_(self.proj[0].bias)
        # Copy video channels through
        with torch.no_grad():
            self.proj[0].weight[:, :out_channels, 0, 0, 0] = torch.eye(out_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.proj(x)


class DepthConditionedTrainer:
    """Trains LTX-2.3 with depth conditioning via channel concatenation."""

    def __init__(self, args):
        self.args = args
        self.device = torch.device(args.device)
        self.vae_device = torch.device(args.vae_device)
        self.global_step = 0

        # Setup output
        os.makedirs(args.output_dir, exist_ok=True)
        os.makedirs(os.path.join(args.output_dir, "checkpoints"), exist_ok=True)

        # Seed
        torch.manual_seed(args.seed)

        self._load_models()
        self._setup_training()

    def _load_models(self):
        args = self.args
        print("\n=== Loading Models ===")

        # Load VAE encoder (for preprocessing data)
        print("Loading VAE encoder...")
        self.vae_encoder = load_video_vae_encoder(
            args.checkpoint, device=str(self.vae_device), dtype=torch.bfloat16,
        )
        self.vae_encoder.eval()

        # Load VAE decoder (for validation visualization)
        print("Loading VAE decoder...")
        self.vae_decoder = load_video_vae_decoder(
            args.checkpoint, device=str(self.vae_device), dtype=torch.bfloat16,
        )
        self.vae_decoder.eval()

        # Load transformer
        print(f"Loading LTX-2.3 transformer...")
        self.transformer = load_transformer(
            args.checkpoint, device="cpu", dtype=torch.bfloat16,
        )

        # Quantize
        if args.quantization != "none":
            print(f"Applying {args.quantization}...")
            from ltx_trainer.quantization import quantize_model
            self.transformer = quantize_model(
                self.transformer, args.quantization, device=str(self.device),
            )
        else:
            self.transformer = self.transformer.to(self.device)

        # Apply LoRA
        print(f"Applying LoRA (rank={args.lora_rank})...")
        lora_config = LoraConfig(
            r=args.lora_rank,
            lora_alpha=args.lora_alpha,
            target_modules=args.lora_targets,
            lora_dropout=0.0,
        )
        self.transformer = get_peft_model(self.transformer, lora_config)
        trainable = sum(p.numel() for p in self.transformer.parameters() if p.requires_grad)
        total = sum(p.numel() for p in self.transformer.parameters())
        print(f"  Trainable: {trainable/1e6:.1f}M / {total/1e6:.1f}M ({100*trainable/total:.2f}%)")

        # Resume from checkpoint
        if args.resume:
            print(f"Resuming from {args.resume}...")
            state_dict = load_file(args.resume)
            from peft import set_peft_model_state_dict
            set_peft_model_state_dict(self.transformer.get_base_model(), state_dict)

        # Depth projection layer
        print(f"Creating depth projection ({args.depth_channels + 128} -> 128)...")
        self.depth_proj = DepthProjection(
            in_channels=128 + args.depth_channels,
            out_channels=128,
        ).to(self.device, torch.bfloat16)

        # Scheduler
        self.scheduler = LTX2Scheduler()

        self.transformer.train()

    def _setup_training(self):
        args = self.args

        # Collect trainable parameters
        train_params = []
        train_params.extend(p for p in self.transformer.parameters() if p.requires_grad)
        train_params.extend(self.depth_proj.parameters())

        self.optimizer = torch.optim.AdamW(train_params, lr=args.lr, weight_decay=0.01)

        self.scaler = torch.amp.GradScaler('cuda')

        # W&B
        try:
            import wandb
            wandb.init(project=args.wandb_project, config=vars(args))
            self.use_wandb = True
        except Exception:
            self.use_wandb = False
            print("W&B not available, logging to stdout only")

    @torch.inference_mode()
    def encode_video(self, video: torch.Tensor) -> torch.Tensor:
        """Encode pixel video to latent. video: [B, C, F, H, W] in [-1, 1]"""
        latents = []
        for i in range(video.shape[0]):
            z = self.vae_encoder(video[i:i+1].to(self.vae_device, torch.bfloat16))
            latents.append(z.to(self.device))
        return torch.cat(latents, dim=0)

    def train_step(self, batch: dict) -> dict:
        """Single training step.

        batch should contain:
            target_video: [B, C, F, H, W] target view video (pixel, [-1,1])
            render_latent: [B, 128, F', H', W'] VAE-encoded render
            mask_latent: [B, 1, F', H', W'] spatial mask at latent resolution
        """
        args = self.args

        target_latent = batch["target_latent"].to(self.device, torch.bfloat16)
        render_latent = batch["render_latent"].to(self.device, torch.bfloat16)
        mask_latent = batch["mask_latent"].to(self.device, torch.bfloat16)

        B, C, F, H, W = target_latent.shape

        # Sample sigma from LTX2Scheduler
        dummy = torch.empty(1, 1, F, H, W)
        sigmas_schedule = self.scheduler.execute(steps=30, latent=dummy)
        # Random sigma per sample
        step_idx = torch.randint(0, len(sigmas_schedule) - 1, (B,))
        sigmas = sigmas_schedule[step_idx].to(self.device).view(B, 1, 1, 1, 1)

        # Flow matching: x_t = (1 - sigma) * x_0 + sigma * noise
        noise = torch.randn_like(target_latent)
        noisy_target = (1 - sigmas) * target_latent + sigmas * noise

        # Channel concat: [noisy_target(128) + render(128) + mask(1)] = 257 channels
        conditioned_input = torch.cat([noisy_target, render_latent, mask_latent], dim=1)
        # Project back to 128 channels
        dit_input = self.depth_proj(conditioned_input)

        # Build modality for LTX model
        from ltx_core.model.transformer.modality import Modality
        from ltx_core.model.transformer.perturbation import BatchedPerturbationConfig

        # Patchify: [B, C, F, H, W] -> [B, F*H*W, C]
        tokens = dit_input.permute(0, 2, 3, 4, 1).reshape(B, F * H * W, C)

        # Build positions
        positions = torch.zeros(B, 3, F * H * W, 2, device=self.device, dtype=torch.bfloat16)
        # Simplified position encoding (proper one would use build_positions)
        for f in range(F):
            for h in range(H):
                for w_idx in range(W):
                    idx = f * H * W + h * W + w_idx
                    positions[:, 0, idx] = torch.tensor([f / 24.0, (f + 1) / 24.0])
                    positions[:, 1, idx] = torch.tensor([h / H, (h + 1) / H])
                    positions[:, 2, idx] = torch.tensor([w_idx / W, (w_idx + 1) / W])

        sigma_scalar = sigmas.squeeze()  # [B]
        per_token_ts = sigma_scalar.unsqueeze(1).expand(-1, F * H * W)

        video_mod = Modality(
            enabled=True,
            latent=tokens,
            sigma=sigma_scalar,
            timesteps=per_token_ts,
            positions=positions,
            context=batch["text_embeds"].to(self.device, torch.bfloat16),
            context_mask=batch.get("text_mask", None),
        )

        perturbations = BatchedPerturbationConfig(batch_size=B)

        # Forward
        with torch.amp.autocast('cuda', dtype=torch.bfloat16):
            video_output, _ = self.transformer(
                video=video_mod, audio=None, perturbations=perturbations,
            )

        # Unpatchify: [B, seq, C] -> [B, C, F, H, W]
        velocity_pred = video_output.reshape(B, F, H, W, C).permute(0, 4, 1, 2, 3)

        # Flow matching target: v = noise - x_0
        target_velocity = noise - target_latent

        # MSE loss
        loss = (velocity_pred - target_velocity).pow(2).mean()

        return {
            "loss": loss,
            "sigma_mean": sigmas.mean().item(),
        }

    def train(self):
        args = self.args
        print(f"\n=== Training for {args.steps} steps ===")

        # Simple data loading placeholder
        # In practice, you'd use VideoDataset with proper preprocessing
        print("NOTE: This training script expects preprocessed latent data.")
        print("Run the preprocessing script first to create latent tensors.")
        print("")
        print("Expected data format (preprocessed .pt files):")
        print("  target_latent: [B, 128, F', H', W']  (VAE-encoded target)")
        print("  render_latent: [B, 128, F', H', W']  (VAE-encoded render)")
        print("  mask_latent:   [B, 1, F', H', W']    (downsampled mask)")
        print("  text_embeds:   [B, 1024, 3840]       (Gemma embeddings)")
        print("")

        # Training loop
        running_loss = 0.0
        t_start = time.time()

        for step in range(1, args.steps + 1):
            self.global_step = step

            # TODO: Load real batch from dataloader
            # For now, create dummy data to verify the pipeline runs
            F_lat, H_lat, W_lat = 4, 14, 24  # 448x768 @ 32x compression
            dummy_batch = {
                "target_latent": torch.randn(1, 128, F_lat, H_lat, W_lat),
                "render_latent": torch.randn(1, 128, F_lat, H_lat, W_lat),
                "mask_latent": torch.randn(1, 1, F_lat, H_lat, W_lat),
                "text_embeds": torch.randn(1, 1024, 3840),
            }

            # Train step
            metrics = self.train_step(dummy_batch)
            loss = metrics["loss"]

            # Backward
            loss = loss / args.grad_accum
            loss.backward()

            if step % args.grad_accum == 0:
                torch.nn.utils.clip_grad_norm_(
                    [p for p in self.transformer.parameters() if p.requires_grad],
                    max_norm=1.0,
                )
                self.optimizer.step()
                self.optimizer.zero_grad()

            running_loss += metrics["loss"].item()

            # Log
            if step % args.log_every == 0:
                avg_loss = running_loss / args.log_every
                elapsed = time.time() - t_start
                steps_per_sec = step / elapsed

                print(f"Step {step}/{args.steps} | loss={avg_loss:.4f} | "
                      f"sigma={metrics['sigma_mean']:.3f} | "
                      f"{steps_per_sec:.1f} steps/s")

                if self.use_wandb:
                    import wandb
                    wandb.log({
                        "train/loss": avg_loss,
                        "train/sigma_mean": metrics["sigma_mean"],
                        "train/steps_per_sec": steps_per_sec,
                    }, step=step)

                running_loss = 0.0

            # Save
            if step % args.save_every == 0:
                self._save_checkpoint(step)

        # Final save
        self._save_checkpoint(args.steps)
        print(f"\nTraining complete! ({time.time() - t_start:.0f}s)")

    def _save_checkpoint(self, step: int):
        """Save LoRA weights + depth projection."""
        ckpt_dir = os.path.join(self.args.output_dir, "checkpoints")

        # Save LoRA
        lora_path = os.path.join(ckpt_dir, f"lora_weights_step_{step:05d}.safetensors")
        lora_state = {}
        for name, param in self.transformer.named_parameters():
            if param.requires_grad and ("lora_" in name):
                lora_state[name] = param.detach().cpu()
        save_file(lora_state, lora_path)

        # Save depth projection
        proj_path = os.path.join(ckpt_dir, f"depth_proj_step_{step:05d}.safetensors")
        proj_state = {k: v.detach().cpu() for k, v in self.depth_proj.state_dict().items()}
        save_file(proj_state, proj_path)

        print(f"  Saved checkpoint at step {step}: {lora_path}")


def main():
    args = parse_args()
    trainer = DepthConditionedTrainer(args)
    trainer.train()


if __name__ == "__main__":
    main()

"""LTX-2 compatible schedulers for InSpatio-World.

Provides both the LTX2Scheduler (sigma schedule generation) and a
FlowMatchScheduler adapter that bridges the LTX-2 sigma schedule with
InSpatio's denoising loop interface.
"""

from __future__ import annotations

import math
from typing import Literal

import torch
from torch import Tensor

BASE_SHIFT_ANCHOR = 1024
MAX_SHIFT_ANCHOR = 4096


class LTX2Scheduler:
    """Default scheduler for LTX-2 diffusion sampling.

    Generates a sigma schedule with token-count-dependent shifting and optional
    stretching to a terminal value.
    """

    def execute(
        self,
        steps: int,
        latent: torch.Tensor | None = None,
        max_shift: float = 2.05,
        base_shift: float = 0.95,
        stretch: bool = True,
        terminal: float = 0.1,
        default_number_of_tokens: int = MAX_SHIFT_ANCHOR,
    ) -> torch.FloatTensor:
        tokens = math.prod(latent.shape[2:]) if latent is not None else default_number_of_tokens
        sigmas = torch.linspace(1.0, 0.0, steps + 1)

        x1 = BASE_SHIFT_ANCHOR
        x2 = MAX_SHIFT_ANCHOR
        mm = (max_shift - base_shift) / (x2 - x1)
        b = base_shift - mm * x1
        sigma_shift = tokens * mm + b

        sigmas = torch.where(
            sigmas != 0,
            math.exp(sigma_shift) / (math.exp(sigma_shift) + (1 / sigmas - 1)),
            0,
        )

        if stretch:
            non_zero_mask = sigmas != 0
            non_zero_sigmas = sigmas[non_zero_mask]
            one_minus_z = 1.0 - non_zero_sigmas
            scale_factor = one_minus_z[-1] / (1.0 - terminal)
            stretched = 1.0 - (one_minus_z / scale_factor)
            sigmas[non_zero_mask] = stretched

        return sigmas.to(torch.float32)


class LTXFlowMatchScheduler:
    """Flow matching scheduler that bridges LTX-2 sigma schedules with InSpatio's
    denoising loop.

    This replaces InSpatio's FlowMatchScheduler when using LTX-2 backbone.
    Flow matching formulation: x_t = (1 - sigma) * x_0 + sigma * noise
    """

    def __init__(
        self,
        num_inference_steps: int = 30,
        latent: torch.Tensor | None = None,
    ):
        self.ltx_scheduler = LTX2Scheduler()
        self.sigmas = self.ltx_scheduler.execute(
            steps=num_inference_steps,
            latent=latent,
        )
        # Convert sigmas to timesteps (0-1000 range for compatibility)
        self.timesteps = (self.sigmas * 1000).long()

    def set_timesteps(self, num_inference_steps: int, latent: torch.Tensor | None = None):
        self.sigmas = self.ltx_scheduler.execute(
            steps=num_inference_steps,
            latent=latent,
        )
        self.timesteps = (self.sigmas * 1000).long()

    def get_sigma_schedule(self, num_steps: int, latent: torch.Tensor | None = None) -> torch.FloatTensor:
        """Get sigma values for denoising steps."""
        return self.ltx_scheduler.execute(steps=num_steps, latent=latent)

    def add_noise(
        self,
        original_samples: Tensor,
        noise: Tensor,
        sigma: float | Tensor,
    ) -> Tensor:
        """Flow matching forward process: x_t = (1 - sigma) * x_0 + sigma * noise."""
        if isinstance(sigma, (int, float)):
            sigma = torch.tensor(sigma, device=noise.device, dtype=noise.dtype)
        if sigma.dim() == 0:
            sigma = sigma.view(1)
        while sigma.dim() < noise.dim():
            sigma = sigma.unsqueeze(-1)

        return (1 - sigma) * original_samples + sigma * noise

    def euler_step(
        self,
        velocity: Tensor,
        x_t: Tensor,
        sigma_current: float,
        sigma_next: float,
    ) -> Tensor:
        """Single Euler ODE step.

        For flow matching: dx/dt = v(x_t, t)
        x_{t+1} = x_t + v * (sigma_next - sigma_current)
        """
        dt = sigma_next - sigma_current
        return x_t + velocity * dt

    def predict_x0(
        self,
        velocity: Tensor,
        x_t: Tensor,
        sigma: float | Tensor,
    ) -> Tensor:
        """Convert velocity prediction to x0.

        flow_pred = noise - x0 (LTX-2 convention)
        x_t = (1 - sigma) * x_0 + sigma * noise
        x_0 = x_t - sigma * flow_pred
        """
        if isinstance(sigma, (int, float)):
            sigma = torch.tensor(sigma, device=velocity.device, dtype=velocity.dtype)
        while sigma.dim() < velocity.dim():
            sigma = sigma.unsqueeze(-1)

        return x_t - sigma * velocity

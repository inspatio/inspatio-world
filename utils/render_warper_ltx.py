"""Depth mask conversion for LTX-2.3 latent space.

LTX-2 latent dimensions differ from Wan2.1:
- LTX-2: 128 channels, spatial compression 32x, temporal compression 8x
  - 480x832 pixel -> 15x26 latent spatial
  - 275 pixel frames -> 35 latent frames (via (F-1)/8 + 1)
  - No temporal frame grouping (each latent frame = 1 channel set)

- Wan2.1: 16 channels, spatial compression 8x, temporal 4-frame groups
  - 480x832 pixel -> 60x104 latent spatial
  - 275 pixel frames -> groups of 4 -> 4ch per group

This module adapts the mask/render processing for LTX-2's latent space.
"""

import torch
import torch.nn.functional as F
from einops import rearrange


def convert_mask_video_ltx(
    mask_video: torch.Tensor,
    target_latent_h: int | None = None,
    target_latent_w: int | None = None,
    spatial_compression: int = 32,
    temporal_compression: int = 8,
) -> torch.Tensor:
    """Convert pixel-space depth mask to LTX-2 latent-space mask.

    Unlike Wan's convert_mask_video() which groups 4 temporal frames into
    4-channel latent frames, LTX-2 treats each latent frame independently
    and uses 128 latent channels. We produce a single-channel spatial mask
    per latent frame, which can then be tiled across channels or used
    as-is for gating.

    Args:
        mask_video: [B, C, T_pixel, H_pixel, W_pixel] depth mask in pixel space
                    (typically C=3 or C=1, values in [-1,1] or [0,1])
        target_latent_h: Override latent height (default: H_pixel // spatial_compression)
        target_latent_w: Override latent width (default: W_pixel // spatial_compression)
        spatial_compression: LTX-2 spatial compression factor (default: 32)
        temporal_compression: LTX-2 temporal compression factor (default: 8)

    Returns:
        mask_latent: [B, T_latent, 1, H_latent, W_latent]
        Where T_latent = (T_pixel - 1) // temporal_compression + 1
    """
    b = mask_video.shape[0]
    h_pixel, w_pixel = mask_video.shape[-2:]

    # Target spatial dimensions in latent space
    lat_h = target_latent_h or (h_pixel // spatial_compression)
    lat_w = target_latent_w or (w_pixel // spatial_compression)

    # Take single channel (if RGB, use first channel as mask)
    mask = mask_video[:, :1]  # [B, 1, T, H, W]

    # Spatial downsampling to latent resolution
    T_pixel = mask.shape[2]
    mask = rearrange(mask, 'b c t h w -> (b t) c h w')
    mask = F.interpolate(
        mask, size=(lat_h, lat_w),
        mode='bilinear', align_corners=False,
    )
    mask = rearrange(mask, '(b t) c h w -> b t c h w', b=b)
    # mask: [B, T_pixel, 1, lat_h, lat_w]

    # Temporal downsampling to match LTX-2 VAE temporal compression
    # LTX-2 VAE: T_latent = (T_pixel - 1) // 8 + 1
    # We sample every 8th frame starting from frame 0
    T_latent = (T_pixel - 1) // temporal_compression + 1
    indices = torch.linspace(0, T_pixel - 1, T_latent).long()
    mask = mask[:, indices]  # [B, T_latent, 1, lat_h, lat_w]

    return mask


def convert_render_video_ltx(
    render_video: torch.Tensor,
    target_latent_h: int | None = None,
    target_latent_w: int | None = None,
    spatial_compression: int = 32,
    temporal_compression: int = 8,
) -> torch.Tensor:
    """Convert pixel-space rendered depth guide to LTX-2 latent-space guide.

    For LTX-2, instead of encoding the render through the VAE (which would
    produce 128-channel latents), we can either:
    1. Encode through VAE (recommended for quality, done externally)
    2. Spatial downsample to latent resolution (fast, less info)

    This function does option 2 (simple downsample). For option 1,
    use the LTXVAEWrapper.encode_to_latent() directly.

    Args:
        render_video: [B, C, T_pixel, H_pixel, W_pixel] rendered guide (C=3 typically)
        target_latent_h: Override latent height
        target_latent_w: Override latent width
        spatial_compression: LTX-2 spatial compression factor (default: 32)
        temporal_compression: LTX-2 temporal compression factor (default: 8)

    Returns:
        render_latent: [B, T_latent, C, H_latent, W_latent]
    """
    b, c = render_video.shape[:2]
    h_pixel, w_pixel = render_video.shape[-2:]

    lat_h = target_latent_h or (h_pixel // spatial_compression)
    lat_w = target_latent_w or (w_pixel // spatial_compression)

    T_pixel = render_video.shape[2]

    # Spatial downsampling
    render = rearrange(render_video, 'b c t h w -> (b t) c h w')
    render = F.interpolate(
        render, size=(lat_h, lat_w),
        mode='bilinear', align_corners=False,
    )
    render = rearrange(render, '(b t) c h w -> b t c h w', b=b)

    # Temporal downsampling
    T_latent = (T_pixel - 1) // temporal_compression + 1
    indices = torch.linspace(0, T_pixel - 1, T_latent).long()
    render = render[:, indices]

    return render


def prepare_depth_conditioning_ltx(
    render_video: torch.Tensor,
    mask_video: torch.Tensor,
    vae_encoder=None,
    spatial_compression: int = 32,
    temporal_compression: int = 8,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Prepare depth conditioning tensors for LTX-2 pipeline.

    Two modes:
    1. With VAE encoder: encode render through LTX VAE for 128-channel latents
       (higher quality, recommended for final pipeline)
    2. Without VAE: simple spatial/temporal downsampling
       (faster, good for prototyping)

    Args:
        render_video: [B, C, T_pixel, H_pixel, W_pixel] rendered depth guide
        mask_video: [B, C, T_pixel, H_pixel, W_pixel] depth mask
        vae_encoder: Optional LTXVAEWrapper for encoding render through VAE
        spatial_compression: LTX-2 spatial compression
        temporal_compression: LTX-2 temporal compression

    Returns:
        (render_latent, mask_latent) ready for the LTX pipeline
        - With VAE: render_latent [B, 128, T_lat, H_lat, W_lat], mask_latent [B, T_lat, 1, H_lat, W_lat]
        - Without VAE: render_latent [B, T_lat, C, H_lat, W_lat], mask_latent [B, T_lat, 1, H_lat, W_lat]
    """
    if vae_encoder is not None:
        # Encode render through VAE for proper 128-channel latent representation
        render_latent = vae_encoder.encode_to_latent(render_video)
        # render_latent: [B, 128, T_lat, H_lat, W_lat]

        # Get latent dimensions from VAE output
        _, _, T_lat, H_lat, W_lat = render_latent.shape

        # Create mask at matching dimensions
        mask_latent = convert_mask_video_ltx(
            mask_video,
            target_latent_h=H_lat,
            target_latent_w=W_lat,
            spatial_compression=spatial_compression,
            temporal_compression=temporal_compression,
        )

        # Ensure temporal dimension matches
        if mask_latent.shape[1] != T_lat:
            # Truncate or pad to match
            if mask_latent.shape[1] > T_lat:
                mask_latent = mask_latent[:, :T_lat]
            else:
                pad = T_lat - mask_latent.shape[1]
                mask_latent = torch.cat([
                    mask_latent,
                    mask_latent[:, -1:].expand(-1, pad, -1, -1, -1),
                ], dim=1)

        return render_latent, mask_latent
    else:
        # Simple spatial/temporal downsample (no VAE)
        h_pixel, w_pixel = render_video.shape[-2:]
        lat_h = h_pixel // spatial_compression
        lat_w = w_pixel // spatial_compression

        render_latent = convert_render_video_ltx(
            render_video,
            target_latent_h=lat_h, target_latent_w=lat_w,
            spatial_compression=spatial_compression,
            temporal_compression=temporal_compression,
        )

        mask_latent = convert_mask_video_ltx(
            mask_video,
            target_latent_h=lat_h, target_latent_w=lat_w,
            spatial_compression=spatial_compression,
            temporal_compression=temporal_compression,
        )

        return render_latent, mask_latent


def down_sample_video_ltx(
    video: torch.Tensor,
    scale_factor: float = 1.0 / 32.0,
) -> torch.Tensor:
    """Downsample video to LTX-2 latent spatial resolution.

    Args:
        video: [B, C, T, H, W]
        scale_factor: Spatial downsample factor (default: 1/32 for LTX-2)

    Returns:
        [B, C, T, H//32, W//32]
    """
    bs = video.shape[0]
    video = rearrange(video, 'b c t h w -> (b t) c h w')
    video = F.interpolate(
        video, scale_factor=(scale_factor, scale_factor),
        mode='bilinear', align_corners=False,
    )
    video = rearrange(video, '(b t) c h w -> b c t h w', b=bs)
    return video

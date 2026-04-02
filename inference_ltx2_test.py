"""LTX-2.3 Inference Script for InSpatio-World.

Phase 1: Multi-step denoising with LTX-2 backbone.
Phase 3: VFM 1-step inference (when --vfm-adapter is provided).

Usage:
    # Multi-step (Phase 1)
    python inference_ltx2_test.py \
        --config_path configs/inference_ltx2.yaml \
        --output_folder outputs/ltx2 \
        --json_path data/test.json

    # VFM 1-step (Phase 3)
    python inference_ltx2_test.py \
        --config_path configs/inference_ltx2.yaml \
        --vfm-adapter checkpoints/vfm/noise_adapter.safetensors \
        --output_folder outputs/ltx2_vfm

    # With cached text embeddings
    python inference_ltx2_test.py \
        --config_path configs/inference_ltx2.yaml \
        --cached-embedding data/conditions/000000.pt \
        --output_folder outputs/ltx2
"""

import argparse
import os
import time

import torch
from omegaconf import OmegaConf
from tqdm import tqdm
from torchvision.io import write_video
from einops import rearrange
import torch.distributed as dist
from torch.utils.data import DataLoader, SequentialSampler
from torch.utils.data.distributed import DistributedSampler

from pipeline.causal_inference_ltx import LTXCausalInferencePipeline
from utils.misc import set_seed

parser = argparse.ArgumentParser()
parser.add_argument("--config_path", type=str, required=True)
parser.add_argument("--output_folder", type=str, required=True)
parser.add_argument("--num_samples", type=int, default=1)
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--json_path", type=str, default=None)
parser.add_argument("--version", type=str, default="version_0")
parser.add_argument("--cached-embedding", type=str, default=None,
                    help="Path to cached conditions .pt file (skips Gemma)")
parser.add_argument("--vfm-adapter", type=str, default=None,
                    help="Path to VFM noise adapter checkpoint")
parser.add_argument("--vfm-sigma-head", type=str, default=None,
                    help="Path to VFM sigma head checkpoint")
args = parser.parse_args()

# Initialize distributed
if "LOCAL_RANK" in os.environ:
    dist.init_process_group(backend='nccl')
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    device = torch.device(f"cuda:{local_rank}")
    world_size = dist.get_world_size()
    rank = dist.get_rank()
    set_seed(args.seed + local_rank)
else:
    device = torch.device("cuda")
    local_rank = 0
    world_size = 1
    rank = 0
    set_seed(args.seed)

torch.set_grad_enabled(False)

# Load config
config = OmegaConf.load(args.config_path)
if os.path.exists("configs/default_config.yaml"):
    default_config = OmegaConf.load("configs/default_config.yaml")
    config = OmegaConf.merge(default_config, config)

# Initialize LTX pipeline
pipeline = LTXCausalInferencePipeline(config, device=device)

# Load VFM adapter if provided
if args.vfm_adapter:
    pipeline.load_vfm_adapter(
        adapter_path=args.vfm_adapter,
        sigma_head_path=args.vfm_sigma_head,
    )
    print(f"[Rank {rank}] VFM 1-step mode enabled")
else:
    print(f"[Rank {rank}] Multi-step mode ({config.get('ltx_num_inference_steps', 30)} steps)")

# Load cached embeddings if provided
cached_conditional = None
if args.cached_embedding:
    from utils.ltx_wrapper import LTXTextEncoder
    cached_conditional = LTXTextEncoder.load_cached_embedding(
        args.cached_embedding, dtype=torch.bfloat16,
    )
    print(f"[Rank {rank}] Using cached embedding from {args.cached_embedding}")

# Load dataset
from datasets.video_dataset import VideoDataset
dataset_config = OmegaConf.to_container(config.dataset, resolve=True)
if args.json_path:
    dataset_config['json_path'] = args.json_path
dataset = VideoDataset(**dataset_config)
print(f"[Rank {rank}] Number of videos: {len(dataset)}")

if dist.is_initialized():
    sampler = DistributedSampler(dataset, shuffle=False, drop_last=False)
else:
    sampler = SequentialSampler(dataset)
dataloader = DataLoader(dataset, batch_size=1, sampler=sampler, num_workers=0, drop_last=False)

output_dir = os.path.join(args.output_folder, args.version)
os.makedirs(output_dir, exist_ok=True)

if dist.is_initialized():
    dist.barrier()

from utils.render_warper import convert_mask_video

for i, batch_data in tqdm(enumerate(dataloader), total=len(dataloader),
                          disable=(rank != 0), desc=f"Rank {rank}"):
    global_idx = i * world_size + rank if dist.is_initialized() else i
    batch = batch_data if isinstance(batch_data, dict) else batch_data[0]

    # Encode source video
    source_video = batch["source_video"].to(device, dtype=torch.bfloat16)
    source_video = rearrange(source_video, 'b t c h w -> b c t h w')

    # For LTX-2, we need to handle the different latent space
    # LTX VAE: [B, C, F, H, W] -> [B, 128, F', H', W']
    ref_latent = pipeline.vae.encode_to_latent(source_video).to(device, dtype=torch.bfloat16)

    # Handle render/mask if present
    render_latent = None
    mask_latent = None
    if "render_video" in batch and "mask_video" in batch:
        render_videos = batch["render_video"].to(device, dtype=torch.bfloat16)
        render_videos = rearrange(render_videos, 'b t c h w -> b c t h w')
        mask_videos = batch["mask_video"].to(device, dtype=torch.bfloat16)
        mask_videos = rearrange(mask_videos, 'b t c h w -> b c t h w')

        render_latent = pipeline.vae.encode_to_latent(render_videos).to(device, dtype=torch.bfloat16)
        # For LTX, mask needs adaptation (different spatial dims than Wan)
        # Simple approach: encode mask through VAE like render
        mask_latent = pipeline.vae.encode_to_latent(mask_videos).to(device, dtype=torch.bfloat16)

    # Determine conditional
    if cached_conditional is not None:
        cond = {k: v.to(device) for k, v in cached_conditional.items()}
    else:
        text_prompts = batch["text"]
        cond = None  # Pipeline will encode text

    # Get latent dimensions for noise
    # ref_latent: [B, 128, F', H', W'] -> need [B, F', 128, H', W'] for noise
    B, C_lat, F_lat, H_lat, W_lat = ref_latent.shape

    # Ensure frame count is compatible with block size
    num_frame_per_block = config.get("num_frame_per_block", 1)
    if F_lat % num_frame_per_block != 0:
        F_lat = F_lat - F_lat % num_frame_per_block

    # Generate noise [B, F', C, H', W']
    sampled_noise = torch.randn(
        [args.num_samples, F_lat, C_lat, H_lat, W_lat],
        device=device, dtype=torch.bfloat16,
    )

    # Permute ref_latent to [B, F, C, H, W] for pipeline
    ref_latent_perm = ref_latent[:, :, :F_lat].permute(0, 2, 1, 3, 4)

    render_perm = None
    mask_perm = None
    if render_latent is not None:
        render_perm = render_latent[:, :, :F_lat].permute(0, 2, 1, 3, 4)
    if mask_latent is not None:
        mask_perm = mask_latent[:, :, :F_lat].permute(0, 2, 1, 3, 4)

    # Run inference
    video = pipeline.inference(
        noise=sampled_noise,
        text_prompts=batch.get("text"),
        conditional_dict=cond,
        ref_latent=ref_latent_perm,
        render_latent=render_perm,
        mask_latent=mask_perm,
    )

    # Save output
    pred_video = (255.0 * video).cpu()

    for seed_idx in range(args.num_samples):
        write_video(
            os.path.join(output_dir, f'{global_idx}-pred_ltx2_rank{rank}.mp4'),
            pred_video[seed_idx], fps=24,
        )

if dist.is_initialized():
    dist.barrier()
    dist.destroy_process_group()

print(f"[Rank {rank}] LTX-2 inference completed!")

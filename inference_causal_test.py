import argparse
import gc
import os
import time

import torch
import torch.distributed as dist
from einops import rearrange
from omegaconf import OmegaConf
from safetensors.torch import load_file
from torch.utils.data import DataLoader, SequentialSampler
from torch.utils.data.distributed import DistributedSampler
from torchvision.io import write_video
from tqdm import tqdm

from demo_utils.memory import DynamicSwapInstaller, get_cuda_free_memory_gb, gpu
from pipeline import CausalInferencePipeline
from pipeline.causal_inference import denoise_block
from utils.misc import set_seed
from utils.render_warper import convert_mask_video

# ============================================================================
# Argument parsing
# ============================================================================
parser = argparse.ArgumentParser()
parser.add_argument("--config_path", type=str, help="Path to the config file")
parser.add_argument("--checkpoint_path", type=str, help="Path to the checkpoint file")
parser.add_argument("--output_folder", type=str, help="Output folder")
parser.add_argument("--num_samples", type=int, default=1, help="Number of samples to generate per prompt")
parser.add_argument("--seed", type=int, default=0, help="Random seed")
parser.add_argument("--json_path", type=str, help="Path to the json file")
parser.add_argument("--version", type=str, default="version_0", help="Output version subfolder name")

# --- Acceleration options ---
parser.add_argument("--use_tae", action="store_true", help="Use Tiny Auto Encoder (TAE) instead of WanVAE")
parser.add_argument("--tae_checkpoint_path", type=str, default=None, help="Path to TAE checkpoint file")
parser.add_argument("--compile_dit", action="store_true", help="Apply torch.compile to the DiT model")

args = parser.parse_args()

# ============================================================================
# Distributed setup
# ============================================================================
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

print(f'[Rank {rank}] Free VRAM {get_cuda_free_memory_gb(gpu)} GB')
low_memory = get_cuda_free_memory_gb(gpu) < 40

torch.set_grad_enabled(False)

# ============================================================================
# Config
# ============================================================================
config = OmegaConf.load(args.config_path)
default_config = OmegaConf.load("configs/default_config.yaml")
config = OmegaConf.merge(default_config, config)

num_frame_per_block = getattr(config, "num_frame_per_block", 3)

# ============================================================================
# Initialize pipeline
# ============================================================================
pipeline = CausalInferencePipeline(config, device=device)

checkpoint_name = "None"
method_name = "default"
if args.checkpoint_path:
    print(f"[Rank {rank}] Loading checkpoint from {args.checkpoint_path}")
    state_dict = load_file(args.checkpoint_path)
    mismatch, missing = pipeline.generator.load_state_dict(state_dict, strict=False)
    print(f"[Rank {rank}] Mismatch: {mismatch}, Missing: {missing}")
    checkpoint_name = args.checkpoint_path.split("/")[-2]
    method_name = args.checkpoint_path.split("/")[-3]

pipeline = pipeline.to(dtype=torch.bfloat16)
if low_memory:
    DynamicSwapInstaller.install_model(pipeline.text_encoder, device=device)
else:
    pipeline.text_encoder.to(device=device)
pipeline.generator.to(device=device)

# ============================================================================
# Initialize VAE or TAE
# ============================================================================
tae_model = None

if args.use_tae:
    from utils.taehv import TAEHV

    assert args.tae_checkpoint_path is not None, "--tae_checkpoint_path is required when --use_tae is set"
    print(f"[Rank {rank}] Loading TAE from {args.tae_checkpoint_path}...")

    tae_model = TAEHV(checkpoint_path=args.tae_checkpoint_path).to(device, torch.float16)
    tae_model.eval()

    # TAE warmup
    print(f"[Rank {rank}] Warming up TAE...")
    with torch.no_grad():
        dummy_enc = torch.randn(1, 9, 3, 480, 832, device=device, dtype=torch.float16)
        _ = tae_model.encode_video(dummy_enc, show_progress_bar=False)
        dummy_lat = torch.randn(1, 3, tae_model.latent_channels, 60, 104, device=device, dtype=torch.float16)
        _ = tae_model.decode_video(dummy_lat, show_progress_bar=False)
        del dummy_enc, dummy_lat
    torch.cuda.synchronize(device)
    print(f"[Rank {rank}] TAE warmup complete.")
else:
    pipeline.vae.to(device=device)

# ============================================================================
# torch.compile for DiT
# ============================================================================
if args.compile_dit:
    print(f"[Rank {rank}] Compiling DiT model with torch.compile (mode=max-autotune)...")

    import torch._inductor.config as inductor_config
    inductor_config.fx_graph_cache = True
    torch._dynamo.config.cache_size_limit = 32

    # Use /dev/shm (tmpfs) for inductor cache to avoid fcntl.flock issues
    # on certain filesystems where unlink + flock causes FileNotFoundError
    cache_dir = f"/dev/shm/torchinductor_cache_rank{rank}"
    os.makedirs(cache_dir, exist_ok=True)
    os.environ["TORCHINDUCTOR_CACHE_DIR"] = cache_dir

    pipeline.generator.model = torch.compile(
        pipeline.generator.model,
        mode="max-autotune",
        fullgraph=False,
        dynamic=False,
        backend="inductor",
    )
    print(f"[Rank {rank}] DiT model compiled.")

# ============================================================================
# DiT warmup
# ============================================================================
pipeline._initialize_kv_cache(batch_size=1, dtype=torch.bfloat16, device=device)


def reset_kv_cache():
    for block_cache in pipeline.kv_cache1:
        block_cache['k'].detach_().zero_()
        block_cache['v'].detach_().zero_()


print(f"[Rank {rank}] Warming up DiT...")
t_warmup_start = time.time()

with torch.no_grad():
    F_warmup = num_frame_per_block
    dummy_noise = torch.randn(1, F_warmup, 16, 60, 104, device=device, dtype=torch.bfloat16)
    dummy_render = torch.randn(1, F_warmup, 20, 60, 104, device=device, dtype=torch.bfloat16)
    dummy_cond = {"prompt_embeds": torch.randn(1, 512, 4096, device=device, dtype=torch.bfloat16)}

    if args.compile_dit:
        # Warm up each distinct kv_size pattern to trigger compilation
        if num_frame_per_block == 1:
            warmup_ctx_sizes = [1, 3, 5, 6]
        else:
            warmup_ctx_sizes = [3, 6]

        for wi, n_ctx in enumerate(warmup_ctx_sizes):
            kv_size = n_ctx * 1560
            dummy_ctx = torch.randn(1, n_ctx, 36, 60, 104, device=device, dtype=torch.bfloat16)
            print(f"[Rank {rank}]   Compile warmup pattern {wi + 1}/{len(warmup_ctx_sizes)} (kv_size={kv_size})...")
            t_pat = time.time()

            for _ in range(3):
                reset_kv_cache()
                denoise_block(
                    pipeline.generator, pipeline.scheduler, dummy_noise, dummy_cond,
                    pipeline.kv_cache1,
                    context_frames=dummy_ctx, context_no_grad=True, context_freqs_offset=0,
                    render_block=dummy_render, denoising_kv_size=kv_size,
                    denoising_steps=pipeline.denoising_step_list,
                )

            torch.cuda.synchronize(device)
            print(f"[Rank {rank}]     Pattern {wi + 1} done ({time.time() - t_pat:.1f}s)")
            torch.cuda.empty_cache()
            gc.collect()
    else:
        # Simple warmup
        dummy_ctx = torch.randn(1, 3, 36, 60, 104, device=device, dtype=torch.bfloat16)
        reset_kv_cache()
        denoise_block(
            pipeline.generator, pipeline.scheduler, dummy_noise, dummy_cond,
            pipeline.kv_cache1,
            context_frames=dummy_ctx, context_no_grad=True, context_freqs_offset=0,
            render_block=dummy_render, denoising_kv_size=1560 * 3,
            denoising_steps=pipeline.denoising_step_list,
        )
        torch.cuda.synchronize(device)

    reset_kv_cache()

del dummy_noise, dummy_render, dummy_cond
torch.cuda.empty_cache()
gc.collect()
print(f"[Rank {rank}] DiT warmup complete ({time.time() - t_warmup_start:.1f}s).")

# ============================================================================
# VAE warmup (only when not using TAE)
# ============================================================================
if not args.use_tae:
    print(f"[Rank {rank}] Warming up VAE...")
    with torch.no_grad():
        vae_mean = pipeline.vae.mean.to(device=device, dtype=torch.bfloat16)
        vae_inv_std = (1.0 / pipeline.vae.std).to(device=device, dtype=torch.bfloat16)
        scale = [vae_mean, vae_inv_std]
        dummy_enc = torch.randn(1, 3, 9, 480, 832, device=device, dtype=torch.bfloat16)
        _ = pipeline.vae.model.encode(dummy_enc, scale)
        pipeline.vae.model.clear_cache()
        dummy_dec = torch.randn(1, 16, 3, 60, 104, device=device, dtype=torch.bfloat16)
        _ = pipeline.vae.model.decode(dummy_dec, scale)
        pipeline.vae.model.clear_cache()
        del dummy_enc, dummy_dec
    torch.cuda.synchronize(device)
    print(f"[Rank {rank}] VAE warmup complete.")

# ============================================================================
# TAE encode / decode helpers
# ============================================================================
def tae_encode(video_bcthw: torch.Tensor) -> torch.Tensor:
    """Encode [B,C,T,H,W] [-1,1] bf16 -> [B,T_lat,C_lat,H_lat,W_lat] bf16."""
    video = video_bcthw.permute(0, 2, 1, 3, 4)                       # -> [B,T,C,H,W]
    video = ((video * 0.5 + 0.5).clamp(0, 1)).to(torch.float16)      # -> [0,1] fp16
    latent = tae_model.encode_video(video, show_progress_bar=False)   # NTCHW
    return latent.to(torch.bfloat16)


def tae_decode(latent: torch.Tensor) -> torch.Tensor:
    """Decode [B,T_lat,C_lat,H_lat,W_lat] bf16 -> [B,T,C,H,W] [0,1] float32."""
    video = tae_model.decode_video(latent.to(torch.float16), show_progress_bar=False)
    return video.float()


def encode_video(video_bcthw: torch.Tensor) -> torch.Tensor:
    """Unified encode: TAE or VAE depending on args."""
    if args.use_tae:
        return tae_encode(video_bcthw)
    return pipeline.vae.encode_to_latent(video_bcthw).to(device, dtype=torch.bfloat16)


# ============================================================================
# Dataset
# ============================================================================
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

output_dir = os.path.join(args.output_folder, method_name, checkpoint_name)
os.makedirs(output_dir, exist_ok=True)

if dist.is_initialized():
    dist.barrier()

# ============================================================================
# Inference loop
# ============================================================================
for i, batch_data in tqdm(enumerate(dataloader), total=len(dataloader), disable=(rank != 0), desc=f"Rank {rank}"):
    if dist.is_initialized():
        global_idx = i * world_size + rank
    else:
        global_idx = i

    batch = batch_data if isinstance(batch_data, dict) else batch_data[0]

    # Load pre-rendered render/mask videos from batch (produced by offline point cloud rendering)
    render_videos_ori = batch["render_video"].to(device, dtype=torch.bfloat16)
    render_videos_ori = rearrange(render_videos_ori, 'b t c h w -> b c t h w')
    mask_videos_ori = batch["mask_video"].to(device, dtype=torch.bfloat16)
    mask_videos_ori = rearrange(mask_videos_ori, 'b t c h w -> b c t h w')

    # --- VAE Encode ---
    torch.cuda.synchronize(device)
    t_enc_start = time.time()

    render_latent = encode_video(render_videos_ori)
    mask_latent = convert_mask_video(mask_videos_ori)

    text_prompts = batch["text"]
    if "target_video" in batch:
        target_video = batch["target_video"].to(device=device, dtype=torch.bfloat16)
    else:
        target_video = batch["source_video"].to(device=device, dtype=torch.bfloat16)
    target_video = rearrange(target_video, 'b t c h w -> b c t h w')
    latent = encode_video(target_video)

    ref_video = batch["source_video"].to(device=device, dtype=torch.bfloat16)
    ref_video = rearrange(ref_video, 'b t c h w -> b c t h w')
    ref_latent = encode_video(ref_video)

    torch.cuda.synchronize(device)
    t_enc_end = time.time()

    latent_length = latent.shape[1]
    if latent_length % config.num_frame_per_block != 0:
        num_output_frames = latent_length - latent_length % config.num_frame_per_block
    else:
        num_output_frames = latent_length
    sampled_noise = torch.randn(
        [args.num_samples, num_output_frames, 16, 60, 104], device=device, dtype=torch.bfloat16
    )

    render_latent = render_latent[:, :num_output_frames, ...].to(device=device, dtype=torch.bfloat16)
    mask_latent = mask_latent[:, :num_output_frames, ...].to(device=device, dtype=torch.bfloat16)
    latent = latent[:, :num_output_frames, ...].to(device=device, dtype=torch.bfloat16)

    # --- DiT inference (decode=False when using TAE, True when using VAE) ---
    torch.cuda.synchronize(device)
    t_dit_start = time.time()

    result = pipeline.inference(
        noise=sampled_noise,
        text_prompts=text_prompts,
        ref_latent=ref_latent,
        render_latent=render_latent,
        mask_latent=mask_latent,
        decode=not args.use_tae,
    )

    torch.cuda.synchronize(device)
    t_dit_end = time.time()

    # --- VAE Decode ---
    torch.cuda.synchronize(device)
    t_dec_start = time.time()

    if args.use_tae:
        # result is denoised latents, decode with TAE
        video_out = tae_decode(result)
        current_video = rearrange(video_out, 'b t c h w -> b t h w c').cpu()
    else:
        # result is already decoded [0,1] video
        current_video = rearrange(result, 'b t c h w -> b t h w c').cpu()

    torch.cuda.synchronize(device)
    t_dec_end = time.time()

    # --- Timing summary ---
    print(f"[Rank {rank}] Video {global_idx} timing: "
          f"VAE Encode={t_enc_end - t_enc_start:.2f}s, "
          f"DiT={'(+VAE Dec) ' if not args.use_tae else ''}{t_dit_end - t_dit_start:.2f}s, "
          f"{'TAE' if args.use_tae else 'VAE'} Decode={t_dec_end - t_dec_start:.2f}s, "
          f"Total={t_dec_end - t_enc_start:.2f}s")

    source_video = rearrange(target_video, 'b c t h w -> b t h w c').cpu()
    source_video = (source_video * 0.5 + 0.5).clamp(0, 1)

    render_video = rearrange(render_videos_ori, 'b c t h w -> b t h w c').cpu()
    render_video = (render_video * 0.5 + 0.5).clamp(0, 1)

    pred_video = 255.0 * current_video
    source_video_out = 255.0 * source_video
    render_video_out = 255.0 * render_video

    if not args.use_tae:
        pipeline.vae.model.clear_cache()

    output_dir = os.path.join(args.output_folder, method_name, checkpoint_name, args.version)
    os.makedirs(output_dir, exist_ok=True)

    for seed_idx in range(args.num_samples):
        write_video(os.path.join(output_dir, f'{global_idx}-pred_video_rank{rank}.mp4'), pred_video[seed_idx], fps=24)
        write_video(os.path.join(output_dir, f'{global_idx}-source_video_rank{rank}.mp4'), source_video_out[seed_idx], fps=24)
        write_video(os.path.join(output_dir, f'{global_idx}-render_video_rank{rank}.mp4'), render_video_out[seed_idx], fps=24)

        if 'target_extrinsics' in batch:
            target_extrinsics = batch["target_extrinsics"].float().to(device=device)
            torch.save(target_extrinsics, os.path.join(output_dir, f'extrinsics_{global_idx}.pt'))

if dist.is_initialized():
    dist.barrier()
    dist.destroy_process_group()

print(f"[Rank {rank}] Inference completed!")

# InSpatio-World


[![HuggingFace](https://img.shields.io/badge/HuggingFace-Model-yellow?logo=huggingface)](https://huggingface.co/inspatio/world)
[![Discord](https://img.shields.io/badge/Discord-Join-7289da?logo=discord&logoColor=white)](https://discord.gg/SyyjR3Z57w)
[![Project Page](https://img.shields.io/badge/Project-Page-green)](https://inspatio.github.io/inspatio-world/)
[![License](https://img.shields.io/badge/License-Apache--2.0-orange)](https://github.com/inspatio/inspatio-world/blob/main/LICENSE)

## Requirements

- Python 3.10
- CUDA 12.1

**1. Create conda environment:**
```bash
conda env create -f environment.yml
conda activate inspatio_world
```

**2. Install flash-attn:**
```bash
pip install https://github.com/Dao-AILab/flash-attention/releases/download/v2.7.4.post1/flash_attn-2.7.4.post1+cu12torch2.5cxx11abiFALSE-cp310-cp310-linux_x86_64.whl
```

## Model Weights

Download the following model checkpoints into the `checkpoints/` directory:

| Model | Purpose | Source |
|---|---|---|
| **InSpatio-World** | v2v inference — 14B (Step 3) | [HuggingFace](https://huggingface.co/inspatio/world) |
| **InSpatio-World-1.3B** | v2v inference — 1.3B (Step 3) | [HuggingFace](https://huggingface.co/inspatio/world) |
| **Wan2.1-T2V-1.3B** | Text encoder + VAE + base model for 1.3B (Step 3) | [HuggingFace](https://huggingface.co/Wan-AI/Wan2.1-T2V-1.3B) |
| **Wan2.1-I2V-14B-480P** | Base diffusion model for 14B (Step 3) | [HuggingFace](https://huggingface.co/Wan-AI/Wan2.1-I2V-14B-480P) |
| **DA3 (Depth-Anything-3)** | Depth estimation (Step 2) | [HuggingFace](https://huggingface.co/depth-anything/DA3NESTED-GIANT-LARGE) |
| **Florence-2-large** | Video captioning (Step 1) | [HuggingFace](https://huggingface.co/microsoft/Florence-2-large) |
| **TAEHV** | Speed up (Optional) | [Github](https://github.com/madebyollin/taehv.git) |

```bash
bash scripts/download.sh
```

Expected directory structure after downloading:
```
checkpoints/
├── InSpatio-World/
│   └── InSpatio-World.safetensors
├── InSpatio-World-1.3B/
│   └── InSpatio-World-1.3B.safetensors
├── Wan2.1-T2V-1.3B/
├── Wan2.1-I2V-14B-480P/
├── DA3/
├── Florence-2-large/
└── taehv/
```

## Inference

The full pipeline runs in three steps:
1. **Step 1** — Generate video captions using Florence-2。
2. **Step 2** — Estimate depth with DA3, convert to inference format, render point clouds
3. **Step 3** — Run InSpatio-World v2v inference

All steps are wrapped in a single script:

```bash
# Using 14B model (default)
bash run_test_pipeline.sh \
  --input_dir ./test/example \
  --traj_txt_path ./traj/x_y_circle_cycle.txt

# Using 1.3B model (lighter, faster)
bash run_test_pipeline.sh \
  --input_dir ./test/example \
  --traj_txt_path ./traj/x_y_circle_cycle.txt \
  --config_path ./configs/inference_1.3b.yaml \
  --checkpoint_path ./checkpoints/InSpatio-World-1.3B/InSpatio-World-1.3B.safetensors
```

### Quick Start

```bash
# 1. Place your .mp4 video(s) in a folder
mkdir -p my_videos
cp your_video.mp4 my_videos/

# 2. Run the full pipeline
bash run_test_pipeline.sh \
  --input_dir ./my_videos \
  --traj_txt_path ./traj/x_y_circle_cycle.txt

# 3. Results will be saved to ./output/my_videos/x_y_circle_cycle/
```


### Trajectory Control

The `--traj_txt_path` argument controls the camera trajectory for novel-view synthesis. Predefined trajectories are provided in the `traj/` directory:

| File | Motion |
|---|---|
| `x_y_circle_cycle.txt` | Cyclic combined pitch + yaw orbit |
| `zoom_out_in.txt` | Dolly zoom out + Dolly zoom in|

#### Trajectory File Format

A trajectory file is a plain text file with **3 lines**, each containing space-separated keyframe values that are automatically interpolated to match the output frame count:

```
<line 1>  pitch (degrees): positive = orbit up, negative = orbit down
<line 2>  yaw (degrees):   positive = orbit left, negative = orbit right
<line 3>  displacement:    relative camera displacement scale
```

**Line 3 (displacement)** is a relative scale multiplied by the scene's estimated foreground depth:
- When pitch/yaw are non-zero, it controls the orbit radius (typically set to `1`)
- When both pitch and yaw are zero, it becomes a dolly zoom: positive = move forward (zoom in), negative = move backward (zoom out)

### All Arguments

| Argument | Required | Default | Description |
|---|---|---|---|
| `--input_dir` | Yes | — | Input folder containing `.mp4` files |
| `--traj_txt_path` | Yes | — | Trajectory file (e.g. `./traj/x_y_circle_cycle.txt`) |
| `--checkpoint_path` | No | `./checkpoints/InSpatio-World/InSpatio-World.safetensors` | InSpatio-World checkpoint |
| `--config_path` | No | `configs/inference.yaml` | Config file (`inference_1.3b.yaml` for 1.3B) |
| `--da3_model_path` | No | `./checkpoints/DA3` | DA3 depth model path |
| `--florence_model_path` | No | `./checkpoints/Florence-2-large` | Florence-2 model path |
| `--step1_gpus` | No | `0` | GPU ID(s) for Step 1 (comma-separated for parallel) |
| `--step2_gpus` | No | `0` | GPU ID(s) for Step 2 (comma-separated for parallel) |
| `--step3_gpus` | No | `0` | GPU ID(s) for Step 3 |
| `--step3_nproc` | No | `1` | Number of GPUs for Step 3 |
| `--output_folder` | No | `./output/<name>/<traj>` | Custom output directory |
| `--master_port` | No | `29513` | Master port for torchrun (Step 3) |
| `--skip_step1` | No | false | Skip caption generation |
| `--skip_step2` | No | false | Skip depth estimation |
| `--skip_step3` | No | false | Skip v2v inference |
| `--relative_to_source` | No | false | Compose trajectory poses relative to initial view |
| `--rotation_only` | No | false | Only apply rotation from trajectory, ignore translation (tripod pan/tilt) |
| `--disable_adaptive_frame` | No | false | Disable adaptive frame expansion/subsampling (use original frame count as-is) |
| `--freeze_repeat` | No | `0` | Repeat a specific frame N extra times to create a time-freeze (pause) effect |
| `--freeze_frame` | No | middle frame | Frame index to freeze; defaults to the middle frame if not specified |
| `--use_tae` | No |	false |	Use Tiny Auto Encoder (TAE) instead of WanVAE |
| `--tae_checkpoint_path` |	No | `./checkpoints/taehv/taew2_1.pth`	| Path to TAE checkpoint file (required when --use_tae is set) |
| `--compile_dit` |	No	| false |	Apply torch.compile to the DiT model |


### Skip Already-Completed Steps

If Step 1 or Step 2 outputs already exist, you can skip them:

```bash
bash run_test_pipeline.sh \
  --input_dir ./my_videos \
  --traj_txt_path ./traj/x_y_circle_cycle.txt \
  --skip_step1 --skip_step2
```

### Generate Temporal Control Videos

```bash
bash run_test_pipeline.sh \
  --input_dir ./test/example \
  --traj_txt_path ./traj/x_y_circle_cycle.txt \
  --freeze_repeat 150 \
  --output_folder ./output/example_freeze_repeat_150 \
  --disable_adaptive_frame
```

You can control the time-stop behavior using two specific parameters: use `--freeze_frame` to choose which frame to freeze (default middle frame), and `--freeze_repeat` to determine the duration (number of frames) of the pause.

### Autonomous Driving Applications

```bash
bash run_test_pipeline.sh \
  --input_dir ./test/example3 \
  --traj_txt_path ./traj/x_y_circle_cycle.txt \
  --relative_to_source \
  --rotation_only \
  --disable_adaptive_frame
```

### Speed Up

```bash
bash run_test_pipeline.sh \
  --input_dir ./test/example \
  --traj_txt_path ./traj/x_y_circle_cycle.txt \
  --use_tae \
  --config_path ./configs/inference_1.3b.yaml  \
  --checkpoint_path ./checkpoints/InSpatio-World-1.3B/InSpatio-World-1.3B.safetensors \
  --disable_adaptive_frame 
```

You can switch from VAE to TAE to accelerate the process. Furthermore, you can use `--compile_dit` to further boost the speed, reaching 24 fps on an H-series NVIDIA GPU（1.3B). However, please note that this operation requires a relatively long warm-up time when triggered for the first time. It is suitable for scenarios where you need to deploy as a service and pursue extreme speed.

## License

This project is licensed under the [Apache-2.0 License](https://github.com/inspatio/inspatio-world/blob/main/LICENSE). Note that this license only applies to code in our library, the dependencies and submodules of which ([Depth-Anything-3](https://github.com/ByteDance-Seed/depth-anything-3), [Florence-2](https://github.com/anyantudre/Florence-2-Vision-Language-Model), [TAEHV](https://github.com/madebyollin/taehv.git)) are separate and individually licensed.

---

## Citation

If you use InSpatio-World in your research, please use the following BibTeX entry.

```bibtex
@misc{inspatio-world,
    title={InSpatio-World},
    author={InSpatio Team},
    howpublished={\url{https://github.com/inspatio/inspatio-world}},
    year={2026}
}
```

## Acknowledgement
InSpatio-World utilizes a backbone based on [Wan2.1](https://github.com/Wan-Video/Wan2.1), with its training code referencing [Self-Forcing](https://github.com/guandeh17/Self-Forcing). Additionally, the TAE component for inference speed-up is built upon [TAEV](https://github.com/madebyollin/taehv.git). We sincerely thank the Self-Forcing, Wan and TAEV team for their foundational work and open-source contribution. We also gratefully acknowledge [Depth-Anything-3](https://github.com/ByteDance-Seed/depth-anything-3), [Florence-2](https://github.com/anyantudre/Florence-2-Vision-Language-Model) and [ReCamMaster](https://github.com/KlingAIResearch/ReCamMaster) for their excellent work that inspired and supported this project.

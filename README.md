# InSpatio-World


[![HuggingFace](https://img.shields.io/badge/HuggingFace-Model-yellow?logo=huggingface)](https://huggingface.co/inspatio/world)
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

```bash
# Download InSpatio-World checkpoint (14B)
mkdir -p checkpoints/InSpatio-World
wget -O checkpoints/InSpatio-World/InSpatio-World.safetensors \
  "https://huggingface.co/inspatio/world/resolve/main/InSpatio-World.safetensors"

# Download InSpatio-World checkpoint (1.3B)
mkdir -p checkpoints/InSpatio-World-1.3B
wget -O checkpoints/InSpatio-World-1.3B/InSpatio-World-1.3B.safetensors \
  "https://huggingface.co/inspatio/world/resolve/main/InSpatio-World-1.3B.safetensors"

# Download Wan models
huggingface-cli download Wan-AI/Wan2.1-T2V-1.3B --local-dir checkpoints/Wan2.1-T2V-1.3B
huggingface-cli download Wan-AI/Wan2.1-I2V-14B-480P --local-dir checkpoints/Wan2.1-I2V-14B-480P

# Download DA3
huggingface-cli download depth-anything/Depth-Anything-3-DA3 --local-dir checkpoints/DA3

# Download Florence-2
huggingface-cli download microsoft/Florence-2-large --local-dir checkpoints/Florence-2-large
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
└── Florence-2-large/
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

### Skip Already-Completed Steps

If Step 1 or Step 2 outputs already exist, you can skip them:

```bash
bash run_test_pipeline.sh \
  --input_dir ./my_videos \
  --traj_txt_path ./traj/x_y_circle_cycle.txt \
  --skip_step1 --skip_step2
```

## Repository Structure

```
├── configs/
│   ├── inference.yaml            # Inference config for 14B model
│   ├── inference_1.3b.yaml       # Inference config for 1.3B model
│   └── default_config.yaml       # Default config values
├── datasets/
│   ├── video_dataset.py          # Video dataset loader (reads JSON metadata)
│   ├── test_dataset.py            # Test dataset with adaptive frame count
│   └── utils.py                  # Crop/resize and trajectory generation utilities
├── depth/
│   ├── depth_predict_da3.py      # DA3 depth estimation core
│   ├── depth_predict_da3_cli.py  # DA3 CLI entry point
│   └── depth_utils.py            # PLY writing, ground plane alignment, pose smoothing
├── pipeline/
│   └── causal_inference.py       # STAR + Joint DMD inference pipeline
├── utils/
│   ├── wan_wrapper.py            # WanTextEncoder, WanVAEWrapper, WanDiffusionWrapper
│   ├── render_warper.py          # Depth-based mask conversion for latent space
│   └── misc.py                   # Seed setting and misc utilities
├── wan/                          # Wan model architecture (causal attention, KV cache)
├── traj/                         # Camera trajectory files (see Trajectory Control)
├── scripts/
│   ├── gen_json.py               # Step 1: Florence-2 video captioning
│   ├── convert_da3_to_pi3.py     # Step 2: DA3 → inference format conversion
│   └── render_point_cloud.py     # Step 2: Offline point cloud rendering
├── inference_causal_test.py      # Step 3: v2v inference entry point
├── run_test_pipeline.sh          # End-to-end inference pipeline script
└── requirements.txt
```

## License

This project is licensed under the [Apache-2.0 License](https://github.com/inspatio/inspatio-world/blob/main/LICENSE). Note that this license only applies to code in our library, the dependencies and submodules of which ([Depth-Anything-3](https://github.com/ByteDance-Seed/depth-anything-3), [Florence-2](https://github.com/anyantudre/Florence-2-Vision-Language-Model)) are separate and individually licensed.

---

**Note:**
- The current release of InSpatio‑World is a version that has not yet been specifically optimized for speed.  The fully optimized version, consistent with our online live demo and reaching ~24 FPS on a datacenter GPU and 10 FPS on an RTX 4090, will be released together with our technical report within 2 weeks.


## Citation

If you use InSpatio-World in your research, please use the following BibTeX entry.

```bibtex
@misc{inspatio-world,
    title={InSpatio-World},
    author={InSpatio-World Contributors},
    howpublished={\url{https://github.com/inspatio/inspatio-world}},
    year={2026}
}
```

## Acknowledgement

InSpatio-World is built upon [Self-Forcing](https://github.com/guandeh17/Self-Forcing) and [Wan2.1](https://github.com/Wan-Video/Wan2.1). We sincerely thank the Self-Forcing and Wan team for their foundational work and open-source contribution. We also gratefully acknowledge [Depth-Anything-3](https://github.com/ByteDance-Seed/depth-anything-3), [Florence-2](https://github.com/anyantudre/Florence-2-Vision-Language-Model) and [ReCamMaster](https://github.com/KlingAIResearch/ReCamMaster) for their excellent work that inspired and supported this project.

#!/usr/bin/env python3
"""Run DA3 depth estimation + format conversion in parallel across GPUs.

Usage:
    python run_da3_parallel.py \
        --json_path /path/to/new.json \
        --gpu_list 0,1,2,3 \
        --da3_cli /path/to/depth_predict_da3_cli.py \
        --da3_config '{"model_path":...}' \
        --convert_script /path/to/convert_da3_to_pi3.py
"""

import argparse
import json
import multiprocessing
import os
import subprocess
import sys


def process_video(args):
    idx, entry, gpu_id, total_videos, da3_cli, da3_config, convert_script = args
    video_path = entry["video_path"]
    final_output = entry["vggt_depth_path"]
    da3_output = final_output + "_da3_tmp"
    video_name = os.path.basename(video_path)

    print(f"[GPU {gpu_id}] [{idx+1}/{total_videos}] Processing: {video_name}")

    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)

    # --- DA3 depth estimation ---
    if os.path.isdir(da3_output) and os.path.isdir(os.path.join(da3_output, "frames_pcd")):
        print(f"[GPU {gpu_id}] [{idx+1}/{total_videos}] DA3 output exists, skipping")
    else:
        cmd_da3 = [
            sys.executable, da3_cli,
            "--input", video_path,
            "--output", da3_output,
            "--config-json", da3_config,
        ]
        result = subprocess.run(cmd_da3, env=env)
        if result.returncode != 0:
            print(f"[GPU {gpu_id}] DA3 failed for {video_name}", file=sys.stderr)
            return False

    # --- Convert DA3 -> Pi3 format ---
    cmd_convert = [
        sys.executable, convert_script,
        "--da3_dir", da3_output,
        "--output_dir", final_output,
        "--video_path", video_path,
    ]
    result = subprocess.run(cmd_convert, env=env)
    if result.returncode != 0:
        print(f"[GPU {gpu_id}] Convert failed for {video_name}", file=sys.stderr)
        return False

    print(f"[GPU {gpu_id}] [{idx+1}/{total_videos}] Done: {video_name}")
    return True


def main():
    parser = argparse.ArgumentParser(description="DA3 depth + convert in parallel across GPUs.")
    parser.add_argument("--json_path", required=True, help="Path to the JSON file with video entries")
    parser.add_argument("--gpu_list", required=True, help="Comma-separated GPU IDs (e.g. 0,1,2,3)")
    parser.add_argument("--da3_cli", required=True, help="Path to depth_predict_da3_cli.py")
    parser.add_argument("--da3_config", required=True, help="DA3 config as JSON string")
    parser.add_argument("--convert_script", required=True, help="Path to convert_da3_to_pi3.py")
    args = parser.parse_args()

    gpu_ids = args.gpu_list.split(",")
    num_gpus = len(gpu_ids)

    with open(args.json_path) as f:
        data = json.load(f)

    total_videos = len(data)
    print(f"Total videos: {total_videos}, GPUs: {num_gpus}")

    # Assign videos to GPUs round-robin
    tasks = []
    for i, entry in enumerate(data):
        gpu_id = gpu_ids[i % num_gpus]
        tasks.append((i, entry, gpu_id, total_videos, args.da3_cli, args.da3_config, args.convert_script))

    if num_gpus == 1:
        results = [process_video(t) for t in tasks]
    else:
        with multiprocessing.Pool(processes=num_gpus) as pool:
            results = pool.map(process_video, tasks)

    failed = sum(1 for r in results if not r)
    if failed > 0:
        print(f"{failed}/{total_videos} videos failed", file=sys.stderr)
        sys.exit(1)

    print(f"All {total_videos} videos processed successfully.")


if __name__ == "__main__":
    main()

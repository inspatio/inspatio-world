#!/usr/bin/env python3
"""Render point clouds in parallel across GPUs.

Usage:
    python run_render_parallel.py \
        --json_path /path/to/new.json \
        --gpu_list 0,1,2,3 \
        --render_script /path/to/render_point_cloud.py \
        --traj_txt_path /path/to/traj.txt \
        --width 832 --height 480
"""

import argparse
import json
import multiprocessing
import os
import subprocess
import sys


def render_video(args):
    idx, entry, gpu_id, total_videos, render_script, traj_txt_path, width, height = args
    video_path = entry["video_path"]
    final_output = entry["vggt_depth_path"]
    da3_output = final_output + "_da3_tmp"
    render_output = os.path.join(final_output, "render")
    video_name = os.path.basename(video_path)

    print(f"[GPU {gpu_id}] [{idx+1}/{total_videos}] Rendering: {video_name}")

    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)

    cmd_render = [
        sys.executable, render_script,
        "--da3_dir", da3_output,
        "--traj_txt_path", traj_txt_path,
        "--output_dir", render_output,
        "--width", str(width),
        "--height", str(height),
    ]
    result = subprocess.run(cmd_render, env=env)
    if result.returncode != 0:
        print(f"[GPU {gpu_id}] Render failed for {video_name}", file=sys.stderr)
        return False

    print(f"[GPU {gpu_id}] [{idx+1}/{total_videos}] Done: {video_name}")
    return True


def main():
    parser = argparse.ArgumentParser(description="Render point clouds in parallel across GPUs.")
    parser.add_argument("--json_path", required=True, help="Path to the JSON file with video entries")
    parser.add_argument("--gpu_list", required=True, help="Comma-separated GPU IDs (e.g. 0,1,2,3)")
    parser.add_argument("--render_script", required=True, help="Path to render_point_cloud.py")
    parser.add_argument("--traj_txt_path", required=True, help="Trajectory file path")
    parser.add_argument("--width", type=int, default=832, help="Render width")
    parser.add_argument("--height", type=int, default=480, help="Render height")
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
        tasks.append((i, entry, gpu_id, total_videos, args.render_script, args.traj_txt_path, args.width, args.height))

    if num_gpus == 1:
        results = [render_video(t) for t in tasks]
    else:
        with multiprocessing.Pool(processes=num_gpus) as pool:
            results = pool.map(render_video, tasks)

    failed = sum(1 for r in results if not r)
    if failed > 0:
        print(f"{failed}/{total_videos} videos failed", file=sys.stderr)
        sys.exit(1)

    print(f"All {total_videos} renders completed.")


if __name__ == "__main__":
    main()

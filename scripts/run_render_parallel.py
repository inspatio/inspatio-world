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
    idx, entry, gpu_id, total_videos, render_script, traj_txt_path, width, height, render_backend, relative_to_source, rotation_only, freeze_repeat, freeze_frame = args
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
        "--render_backend", render_backend,
    ]
    if relative_to_source:
        cmd_render.append("--relative_to_source")
    if rotation_only:
        cmd_render.append("--rotation_only")
    if freeze_repeat > 0:
        cmd_render.extend(["--freeze_repeat", str(freeze_repeat)])
    if freeze_frame is not None:
        cmd_render.extend(["--freeze_frame", str(freeze_frame)])
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
    parser.add_argument("--render_backend", choices=["warper", "ply"], default="warper",
                        help="Rendering backend passed through to render_point_cloud.py")
    parser.add_argument("--relative_to_source", action="store_true",
                        help="Compose trajectory poses relative to initial view")
    parser.add_argument("--rotation_only", action="store_true",
                        help="Only apply rotation from the trajectory, ignore translation")
    parser.add_argument("--freeze_repeat", type=int, default=0,
                        help="Number of times to repeat the freeze frame (0 = disabled)")
    parser.add_argument("--freeze_frame", type=int, default=None,
                        help="Frame index to freeze (default: middle frame)")
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
        tasks.append((i, entry, gpu_id, total_videos, args.render_script, args.traj_txt_path, args.width, args.height, args.render_backend, args.relative_to_source, args.rotation_only, args.freeze_repeat, args.freeze_frame))

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

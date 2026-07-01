#!/usr/bin/env python3
"""Offline rendering using DA3 output + traj_txt camera poses.

By default this script uses a fast depth-image forward-warp backend that reads
DA3 RGB frames + RGBA-float32 depth maps directly. The original PLY point cloud
renderer is still available with --render_backend ply.

Usage:
    python render_point_cloud.py \
        --da3_dir /path/to/da3_output \
        --traj_txt_path ./traj/x_y_circle_cycle.txt \
        --output_dir /path/to/output \
        --width 832 --height 480
"""

import argparse
import glob
import logging
import os
import subprocess
import sys
import time

import cv2
import numpy as np
import torch
from PIL import Image

# Add project root to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from datasets.utils import generate_traj_txt

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

# ── Constants ──
MIN_DEPTH_THRESHOLD = 0.1
DEPTH_EPSILON = 1e-4
_splat_offset_cache = {}


# ── Core rendering (from reference render_utils.py) ──

def load_ply_data(ply_path, device):
    """Load point cloud from PLY file. Returns (points, colors) tensors."""
    import open3d as o3d

    pcd = o3d.io.read_point_cloud(ply_path)
    if not pcd.has_points():
        logger.warning(f"Point cloud has no points: {ply_path}")
        return None, None
    pts = np.asarray(pcd.points, dtype=np.float32)
    colors = np.asarray(pcd.colors, dtype=np.float32)
    return torch.from_numpy(pts).to(device), torch.from_numpy(colors).to(device)


def render_batch(points, colors, c2w, K, width, height, point_size=2,
                 ss_ratio=2.0, bg_color=0):
    """Render point cloud from given camera pose with supersampling.

    Returns (rgb_bgr, mask) as numpy arrays.
    """
    if points is None or colors is None:
        return (np.zeros((height, width, 3), dtype=np.uint8),
                np.zeros((height, width), dtype=np.uint8))

    H_high = int(height * ss_ratio)
    W_high = int(width * ss_ratio)

    K_high = K.clone()
    K_high[0, :] *= ss_ratio
    K_high[1, :] *= ss_ratio

    p_size_high = int(point_size * ss_ratio)
    if p_size_high % 2 == 0:
        p_size_high += 1

    # Transform points to camera space
    w2c = torch.linalg.inv(c2w)
    N = points.shape[0]
    points_h = torch.cat([points, torch.ones((N, 1), device=points.device)], dim=1)
    cam_xyz = (points_h @ w2c.T)[:, :3]
    z = cam_xyz[:, 2]

    mask_z = z > MIN_DEPTH_THRESHOLD
    if mask_z.sum() == 0:
        return (np.zeros((height, width, 3), dtype=np.uint8),
                np.zeros((height, width), dtype=np.uint8))

    xyz = cam_xyz[mask_z]
    rgb = colors[mask_z]
    z = z[mask_z]

    # Project to image plane
    u_float = (K_high[0, 0] * xyz[:, 0] / z) + K_high[0, 2]
    v_float = (K_high[1, 1] * xyz[:, 1] / z) + K_high[1, 2]

    # Point splatting
    if p_size_high > 1:
        cache_key = (p_size_high, points.device)
        if cache_key not in _splat_offset_cache:
            radius = p_size_high // 2
            offset_range = torch.arange(-radius, radius + 1, device=points.device)
            dy, dx = torch.meshgrid(offset_range, offset_range, indexing='ij')
            _splat_offset_cache[cache_key] = (dx.flatten(), dy.flatten())
        dx, dy = _splat_offset_cache[cache_key]

        u_final = torch.round(u_float.unsqueeze(1) + dx.unsqueeze(0)).long().view(-1)
        v_final = torch.round(v_float.unsqueeze(1) + dy.unsqueeze(0)).long().view(-1)
        z_final = z.unsqueeze(1).expand(-1, dx.shape[0]).reshape(-1)
        rgb_final = rgb.unsqueeze(1).expand(-1, dx.shape[0], 3).reshape(-1, 3)
    else:
        u_final = torch.round(u_float).long()
        v_final = torch.round(v_float).long()
        z_final = z
        rgb_final = rgb

    # Filter out-of-bounds
    valid = (u_final >= 0) & (u_final < W_high) & (v_final >= 0) & (v_final < H_high)
    u = u_final[valid]
    v = v_final[valid]
    rgb = rgb_final[valid]
    z = z_final[valid]

    # Z-buffer depth test
    indices = v * W_high + u
    depth_buffer = torch.full((H_high * W_high,), float('inf'), device=points.device)
    depth_buffer.scatter_reduce_(0, indices, z, reduce='min', include_self=True)
    is_closest = z <= depth_buffer[indices] + DEPTH_EPSILON

    final_u = u[is_closest]
    final_v = v[is_closest]
    final_rgb = rgb[is_closest]

    # Canvas
    canvas = torch.full((3, H_high, W_high), bg_color / 255.0, device=points.device)
    canvas[:, final_v, final_u] = final_rgb.permute(1, 0)

    mask_canvas = torch.zeros((H_high, W_high), dtype=torch.uint8, device=points.device)
    mask_canvas[final_v, final_u] = 255

    # Downsample
    img_high = (canvas.permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
    img_final = cv2.resize(img_high, (width, height), interpolation=cv2.INTER_AREA)
    mask_final = cv2.resize(mask_canvas.cpu().numpy(), (width, height),
                            interpolation=cv2.INTER_NEAREST)

    img_bgr = cv2.cvtColor(img_final, cv2.COLOR_RGB2BGR)
    return img_bgr, mask_final


# ── Data loading ──

def load_intrinsic(da3_dir, device):
    """Load first-frame intrinsic (3x3) from DA3 intrinsic.txt."""
    path = os.path.join(da3_dir, "intrinsic.txt")
    data = np.loadtxt(path)
    K = data[0:3, :3].astype(np.float32)
    return torch.tensor(K, device=device, dtype=torch.float32)


def load_extrinsic_c2w(da3_dir, device):
    """Load extrinsics from DA3 extrinsic.txt.

    The file contains N frames, each stored as 3 rows of a 3x4 w2c matrix.

    Returns:
        initial_c2w: (4, 4) tensor, first-frame c2w
        source_c2ws: list of (4, 4) tensors, all-frame c2ws
    """
    path = os.path.join(da3_dir, "extrinsic.txt")
    data = np.loadtxt(path)
    num_frames = data.shape[0] // 3

    source_c2ws = []
    for i in range(num_frames):
        w2c_34 = data[i * 3:(i + 1) * 3, :4].astype(np.float32)
        w2c = np.vstack([w2c_34, np.array([[0, 0, 0, 1]], dtype=np.float32)])
        w2c_t = torch.tensor(w2c, dtype=torch.float32, device=device)
        c2w = torch.linalg.inv(w2c_t)
        source_c2ws.append(c2w)

    initial_c2w = source_c2ws[0]
    return initial_c2w, source_c2ws


def load_ply_sequence(da3_dir, device, max_frames=None):
    """Load PLY sequence from frames_pcd/. Returns lists of (points, colors)."""
    ply_folder = os.path.join(da3_dir, "frames_pcd")
    ply_files = sorted(glob.glob(os.path.join(ply_folder, "*.ply")))
    if not ply_files:
        raise FileNotFoundError(f"No PLY files in {ply_folder}")
    if max_frames is not None:
        ply_files = ply_files[:max_frames]

    logger.info(f"Loading {len(ply_files)} point clouds from {ply_folder}")
    points_list, colors_list = [], []
    for pf in ply_files:
        pts, cols = load_ply_data(pf, device)
        points_list.append(pts)
        colors_list.append(cols)
    return points_list, colors_list


def scale_intrinsic(K, target_width, target_height):
    """Scale intrinsic matrix to target resolution based on principal point."""
    orig_cx = K[0, 2].item()
    orig_cy = K[1, 2].item()
    scale_x = target_width / (orig_cx * 2)
    scale_y = target_height / (orig_cy * 2)
    K_scaled = K.clone()
    K_scaled[0, 0] *= scale_x
    K_scaled[1, 1] *= scale_y
    K_scaled[0, 2] = target_width / 2.0
    K_scaled[1, 2] = target_height / 2.0
    return K_scaled


# ── Camera pose generation ──

def generate_target_c2ws(traj_txt_path, initial_c2w, source_c2ws, num_frames, device,
                         relative_to_source=False, rotation_only=False):
    """Generate target c2w poses from traj_txt.

    1. Read traj_txt -> generate_traj_txt() -> relative c2w offsets (N, 4, 4)
    2. Optionally zero out translation (rotation_only)
    3. Compose with initial_c2w (relative_to_source) or use as absolute poses

    Args:
        traj_txt_path: path to traj txt file (3 lines: x_up, y_left, r)
        initial_c2w: (4, 4) first-frame c2w from DA3
        num_frames: how many frames to generate
        device: torch device
        relative_to_source: if True, compose relative poses on top of initial_c2w;
                            if False, treat generated poses as absolute world coords
        rotation_only: if True, zero out translation in relative poses (tripod pan/tilt)

    Returns:
        list of (4, 4) c2w tensors, length = num_frames
    """
    with open(traj_txt_path, 'r') as f:
        lines = f.readlines()
    x_up_angle = [float(i) for i in lines[0].split()]
    y_left_angle = [float(i) for i in lines[1].split()]
    r_raw = [float(i) for i in lines[2].split()]

    # generate_traj_txt returns relative c2w offsets (identity at frame 0)
    relative_c2ws = generate_traj_txt(
        x_up_angle, y_left_angle, r_raw, r_raw, num_frames, is_translation=rotation_only
    )  # (N, 4, 4) numpy, these are relative c2w transforms # Twc

    target_c2ws = []
    abs_source_c2ws = []
    for i in range(num_frames):
        rel_source = source_c2ws[i]  # already a torch tensor (4,4) on device, Twc
        abs_source_c2w = initial_c2w.inverse() @ rel_source # Tc0w @ Twc = Tc0c
        abs_source_c2ws.append(abs_source_c2w)

        rel = torch.tensor(relative_c2ws[i], dtype=torch.float32, device=device) #Twc
        abs_target_c2w = initial_c2w.inverse() @ rel # Tc0w @ Twc = Tc0c
        if relative_to_source:
            abs_target_c2w = (abs_target_c2w.inverse() @ abs_source_c2w.inverse()).inverse() # (new_Tcw_tgt = Tcw_tgt @ Tcw_src).inverse

        target_c2ws.append(abs_target_c2w)
    return target_c2ws


# ── ffmpeg streaming ──

def open_ffmpeg_writer(output_path, width, height, fps=24):
    """Open ffmpeg subprocess for streaming raw RGB24 frames to mp4."""
    ffmpeg_bin = "/usr/bin/ffmpeg" if os.path.exists("/usr/bin/ffmpeg") else "ffmpeg"
    cmd = [
        ffmpeg_bin, "-y",
        "-f", "rawvideo", "-vcodec", "rawvideo",
        "-pix_fmt", "rgb24",
        "-s", f"{width}x{height}",
        "-r", str(fps),
        "-i", "-",
        "-c:v", "libx264", "-crf", "18",
        "-pix_fmt", "yuv420p",
        "-loglevel", "warning",
        output_path,
    ]
    return subprocess.Popen(cmd, stdin=subprocess.PIPE)


class DepthWarper:
    """Batched depth-image forward splatting used by the fast render backend."""

    def __init__(self):
        self.dtype = torch.float32

    def forward_warp(self, frame1, mask1, depth1, transformation1, transformation2,
                     intrinsic1, intrinsic2=None):
        b, c, h, w = frame1.shape
        if mask1 is None:
            mask1 = torch.ones((b, 1, h, w), device=frame1.device, dtype=frame1.dtype)
        if intrinsic2 is None:
            intrinsic2 = intrinsic1

        frame1 = frame1.to(self.dtype)
        mask1 = mask1.to(self.dtype)
        depth1 = depth1.to(self.dtype)
        transformation1 = transformation1.to(self.dtype)
        transformation2 = transformation2.to(self.dtype)
        intrinsic1 = intrinsic1.to(self.dtype)
        intrinsic2 = intrinsic2.to(self.dtype)

        trans_points = self.compute_transformed_points(
            depth1, transformation1, transformation2, intrinsic1, intrinsic2
        )
        trans_coordinates = trans_points[:, :, :, :2, 0] / trans_points[:, :, :, 2:3, 0]
        trans_depth = trans_points[:, :, :, 2, 0]
        grid = self.create_grid(b, h, w).to(trans_coordinates)
        flow12 = trans_coordinates.permute(0, 3, 1, 2) - grid
        return self.bilinear_splatting(frame1, mask1, trans_depth, flow12, None, is_image=True)

    def compute_transformed_points(self, depth1, tc1w, tc2w, intrinsic1, intrinsic2):
        b, _, h, w = depth1.shape
        tc2c1 = torch.bmm(tc2w, torch.linalg.inv(tc1w))

        x1d = torch.arange(0, w, device=depth1.device)[None]
        y1d = torch.arange(0, h, device=depth1.device)[:, None]
        x2d = x1d.repeat(h, 1)
        y2d = y1d.repeat(1, w)
        ones_2d = torch.ones((h, w), device=depth1.device)
        ones_4d = ones_2d[None, :, :, None, None].repeat(b, 1, 1, 1, 1)
        pos_vectors = torch.stack([x2d, y2d, ones_2d], dim=2)[None, :, :, :, None]

        intrinsic1_inv = torch.linalg.inv(intrinsic1)[:, None, None]
        intrinsic2_4d = intrinsic2[:, None, None]
        depth_4d = depth1[:, 0][:, :, :, None, None]
        trans_4d = tc2c1[:, None, None]

        unnormalized_pos = torch.matmul(intrinsic1_inv, pos_vectors)
        world_points = depth_4d * unnormalized_pos
        world_points_homo = torch.cat([world_points, ones_4d], dim=3)
        trans_world_homo = torch.matmul(trans_4d, world_points_homo)
        trans_world = trans_world_homo[:, :, :, :3]
        return torch.matmul(intrinsic2_4d, trans_world)

    def bilinear_splatting(self, frame1, mask1, depth1, flow12, flow12_mask, is_image=False):
        b, c, h, w = frame1.shape
        if flow12_mask is None:
            flow12_mask = torch.ones((b, 1, h, w), device=flow12.device, dtype=flow12.dtype)

        grid = self.create_grid(b, h, w).to(frame1)
        trans_pos = flow12 + grid
        trans_pos_offset = trans_pos + 1
        trans_pos_floor = torch.floor(trans_pos_offset).long()
        trans_pos_ceil = torch.ceil(trans_pos_offset).long()

        trans_pos_offset = torch.stack([
            torch.clamp(trans_pos_offset[:, 0], min=0, max=w + 1),
            torch.clamp(trans_pos_offset[:, 1], min=0, max=h + 1),
        ], dim=1)
        trans_pos_floor = torch.stack([
            torch.clamp(trans_pos_floor[:, 0], min=0, max=w + 1),
            torch.clamp(trans_pos_floor[:, 1], min=0, max=h + 1),
        ], dim=1)
        trans_pos_ceil = torch.stack([
            torch.clamp(trans_pos_ceil[:, 0], min=0, max=w + 1),
            torch.clamp(trans_pos_ceil[:, 1], min=0, max=h + 1),
        ], dim=1)

        prox_weight_nw = (1 - (trans_pos_offset[:, 1:2] - trans_pos_floor[:, 1:2])) * (
            1 - (trans_pos_offset[:, 0:1] - trans_pos_floor[:, 0:1])
        )
        prox_weight_sw = (1 - (trans_pos_ceil[:, 1:2] - trans_pos_offset[:, 1:2])) * (
            1 - (trans_pos_offset[:, 0:1] - trans_pos_floor[:, 0:1])
        )
        prox_weight_ne = (1 - (trans_pos_offset[:, 1:2] - trans_pos_floor[:, 1:2])) * (
            1 - (trans_pos_ceil[:, 0:1] - trans_pos_offset[:, 0:1])
        )
        prox_weight_se = (1 - (trans_pos_ceil[:, 1:2] - trans_pos_offset[:, 1:2])) * (
            1 - (trans_pos_ceil[:, 0:1] - trans_pos_offset[:, 0:1])
        )

        sat_depth = torch.clamp(depth1, min=0, max=1000)
        log_depth = torch.log(1 + sat_depth)
        depth_weights = torch.exp(log_depth / log_depth.max().clamp_min(1e-6) * 50)
        if depth1.dim() == 3:
            valid_mask = (depth1 >= 0).to(depth1).unsqueeze(1)
            depth_weights = depth_weights.unsqueeze(1)
        else:
            valid_mask = (depth1 >= 0).to(depth1)

        def make_weight(prox_weight):
            return torch.moveaxis(
                prox_weight * mask1 * flow12_mask * valid_mask / depth_weights,
                [0, 1, 2, 3],
                [0, 3, 1, 2],
            )

        weight_nw = make_weight(prox_weight_nw)
        weight_sw = make_weight(prox_weight_sw)
        weight_ne = make_weight(prox_weight_ne)
        weight_se = make_weight(prox_weight_se)

        warped_frame = torch.zeros((b, h + 2, w + 2, c), dtype=torch.float32, device=frame1.device)
        warped_weights = torch.zeros((b, h + 2, w + 2, 1), dtype=torch.float32, device=frame1.device)

        frame1_cl = torch.moveaxis(frame1, [0, 1, 2, 3], [0, 3, 1, 2])
        batch_indices = torch.arange(b, device=frame1.device)[:, None, None]
        warped_frame.index_put_(
            (batch_indices, trans_pos_floor[:, 1], trans_pos_floor[:, 0]),
            frame1_cl * weight_nw,
            accumulate=True,
        )
        warped_frame.index_put_(
            (batch_indices, trans_pos_ceil[:, 1], trans_pos_floor[:, 0]),
            frame1_cl * weight_sw,
            accumulate=True,
        )
        warped_frame.index_put_(
            (batch_indices, trans_pos_floor[:, 1], trans_pos_ceil[:, 0]),
            frame1_cl * weight_ne,
            accumulate=True,
        )
        warped_frame.index_put_(
            (batch_indices, trans_pos_ceil[:, 1], trans_pos_ceil[:, 0]),
            frame1_cl * weight_se,
            accumulate=True,
        )
        warped_weights.index_put_(
            (batch_indices, trans_pos_floor[:, 1], trans_pos_floor[:, 0]),
            weight_nw,
            accumulate=True,
        )
        warped_weights.index_put_(
            (batch_indices, trans_pos_ceil[:, 1], trans_pos_floor[:, 0]),
            weight_sw,
            accumulate=True,
        )
        warped_weights.index_put_(
            (batch_indices, trans_pos_floor[:, 1], trans_pos_ceil[:, 0]),
            weight_ne,
            accumulate=True,
        )
        warped_weights.index_put_(
            (batch_indices, trans_pos_ceil[:, 1], trans_pos_ceil[:, 0]),
            weight_se,
            accumulate=True,
        )

        warped_frame_cf = torch.moveaxis(warped_frame, [0, 1, 2, 3], [0, 2, 3, 1])
        warped_weights_cf = torch.moveaxis(warped_weights, [0, 1, 2, 3], [0, 2, 3, 1])
        cropped_frame = warped_frame_cf[:, :, 1:-1, 1:-1]
        cropped_weights = warped_weights_cf[:, :, 1:-1, 1:-1]

        known_mask = cropped_weights > 0
        zero_value = -1 if is_image else 0
        warped_frame2 = torch.where(
            known_mask,
            cropped_frame / cropped_weights,
            torch.tensor(zero_value, dtype=frame1.dtype, device=frame1.device),
        )
        if is_image:
            warped_frame2 = torch.clamp(warped_frame2, min=-1, max=1)
        return warped_frame2, known_mask.to(frame1)

    @staticmethod
    def create_grid(b, h, w):
        x_1d = torch.arange(0, w)[None]
        y_1d = torch.arange(0, h)[:, None]
        x_2d = x_1d.repeat(h, 1)
        y_2d = y_1d.repeat(1, w)
        grid = torch.stack([x_2d, y_2d], dim=0)
        return grid[None].repeat(b, 1, 1, 1)


def read_da3_depth(path):
    """Read DA3 RGBA PNG where each pixel stores one float32 depth value."""
    img = Image.open(path)
    depth_uint8 = np.array(img)
    h, w = depth_uint8.shape[:2]
    return np.frombuffer(depth_uint8.tobytes(), dtype=np.float32).reshape(h, w).copy()


def load_rgb_depth_sequence(da3_dir, width, height):
    frame_dir = os.path.join(da3_dir, "frames")
    depth_dir = os.path.join(da3_dir, "depth")
    frame_files = sorted(
        glob.glob(os.path.join(frame_dir, "*.jpg")) + glob.glob(os.path.join(frame_dir, "*.png"))
    )
    depth_files = sorted(glob.glob(os.path.join(depth_dir, "*.png")))
    if not frame_files:
        raise FileNotFoundError(f"No RGB frames found in {frame_dir}")
    if len(frame_files) != len(depth_files):
        raise RuntimeError(
            f"Frame/depth count mismatch: {len(frame_files)} frames vs {len(depth_files)} depths"
        )

    frames, depths = [], []
    for frame_path, depth_path in zip(frame_files, depth_files):
        img_bgr = cv2.imread(frame_path, cv2.IMREAD_COLOR)
        if img_bgr is None:
            raise RuntimeError(f"Failed to read frame: {frame_path}")
        if img_bgr.shape[:2] != (height, width):
            img_bgr = cv2.resize(img_bgr, (width, height), interpolation=cv2.INTER_AREA)
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 127.5 - 1.0
        frames.append(img_rgb.transpose(2, 0, 1))

        depth = read_da3_depth(depth_path).astype(np.float32)
        if depth.shape != (height, width):
            depth = cv2.resize(depth, (width, height), interpolation=cv2.INTER_NEAREST)
        depths.append(np.clip(depth, 1e-4, 10000.0)[None])

    return np.stack(frames, axis=0), np.stack(depths, axis=0)


def write_tensor_videos(render, mask, video_path, mask_path, width, height, fps):
    """Write RGB render and known-mask tensors to the expected mp4 files."""
    video_proc = open_ffmpeg_writer(video_path, width, height, fps)
    mask_proc = open_ffmpeg_writer(mask_path, width, height, fps)
    try:
        render_np = ((render.detach().cpu().numpy().transpose(0, 2, 3, 1) + 1.0) * 127.5)
        render_np = render_np.clip(0, 255).astype(np.uint8)
        mask_np = (mask.detach().cpu().numpy().transpose(0, 2, 3, 1) * 255.0)
        mask_np = mask_np.clip(0, 255).astype(np.uint8)

        for img_rgb, mask_gray in zip(render_np, mask_np):
            mask_rgb = np.repeat(mask_gray, 3, axis=2)
            video_proc.stdin.write(img_rgb.tobytes())
            mask_proc.stdin.write(mask_rgb.tobytes())
    finally:
        video_proc.stdin.close()
        mask_proc.stdin.close()
        video_proc.wait()
        mask_proc.wait()

    if video_proc.returncode != 0:
        raise RuntimeError(f"ffmpeg failed for {video_path}")
    if mask_proc.returncode != 0:
        raise RuntimeError(f"ffmpeg failed for {mask_path}")


def render_point_cloud_warper(da3_dir, traj_txt_path, output_dir, width=832, height=480,
                              fps=24, relative_to_source=False, rotation_only=False,
                              freeze_repeat=0, freeze_frame=None):
    """Fast renderer using DA3 RGB/depth images and batched forward splatting."""
    t_start = time.perf_counter()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Device: {device}")
    logger.info("Render backend: warper")

    K_orig = load_intrinsic(da3_dir, device)
    K_render = scale_intrinsic(K_orig, width, height)
    initial_c2w, source_c2ws = load_extrinsic_c2w(da3_dir, device)
    frame_np, depth_np = load_rgb_depth_sequence(da3_dir, width, height)
    num_frames = frame_np.shape[0]
    logger.info(f"Loaded {num_frames} RGB/depth frames")

    if freeze_repeat > 0:
        if freeze_frame is None:
            freeze_frame = num_frames // 2
        freeze_frame = max(0, min(freeze_frame, num_frames - 1))
        insert_pos = freeze_frame + 1
        logger.info(f"Time-freeze: repeating frame {freeze_frame} x{freeze_repeat} "
                    f"(inserting {freeze_repeat} extra frames)")
        frame_np = np.concatenate([
            frame_np[:insert_pos],
            np.repeat(frame_np[freeze_frame:freeze_frame + 1], freeze_repeat, axis=0),
            frame_np[insert_pos:],
        ], axis=0)
        depth_np = np.concatenate([
            depth_np[:insert_pos],
            np.repeat(depth_np[freeze_frame:freeze_frame + 1], freeze_repeat, axis=0),
            depth_np[insert_pos:],
        ], axis=0)
        frozen_c2w = source_c2ws[freeze_frame]
        source_c2ws = source_c2ws[:insert_pos] + [frozen_c2w] * freeze_repeat + source_c2ws[insert_pos:]
        num_frames = frame_np.shape[0]
        logger.info(f"After freeze: {num_frames} total frames")

    target_c2ws = generate_target_c2ws(
        traj_txt_path,
        initial_c2w,
        source_c2ws,
        num_frames,
        device,
        relative_to_source=relative_to_source,
        rotation_only=rotation_only,
    )
    source_tcw = torch.stack([torch.linalg.inv(c2w) for c2w in source_c2ws], dim=0)
    target_tcw = torch.stack([torch.linalg.inv(c2w) for c2w in target_c2ws], dim=0)
    K_batch = K_render.unsqueeze(0).repeat(num_frames, 1, 1)

    frame = torch.from_numpy(frame_np).to(device=device, dtype=torch.float32)
    depth = torch.from_numpy(depth_np).to(device=device, dtype=torch.float32)
    warper = DepthWarper()

    if device.type == "cuda":
        torch.cuda.synchronize()
    t_warp = time.perf_counter()
    with torch.no_grad():
        render, mask = warper.forward_warp(
            frame,
            None,
            depth,
            source_tcw,
            target_tcw,
            K_batch,
            None,
        )
    if device.type == "cuda":
        torch.cuda.synchronize()
    warp_time = time.perf_counter() - t_warp
    logger.info(f"  Warped {num_frames} frames")

    os.makedirs(output_dir, exist_ok=True)
    video_path = os.path.join(output_dir, "render_offline.mp4")
    mask_path = os.path.join(output_dir, "mask_offline.mp4")
    write_tensor_videos(render, mask, video_path, mask_path, width, height, fps)

    logger.info(f"Saved: {video_path}")
    logger.info(f"Saved: {mask_path}")
    logger.info(f"Warper timing: warp={warp_time:.3f}s total={time.perf_counter() - t_start:.3f}s")


# ── Main rendering pipeline ──

def render_point_cloud_ply(da3_dir, traj_txt_path, output_dir, width=832, height=480,
                           point_size=2, fps=24, relative_to_source=False, rotation_only=False,
                           freeze_repeat=0, freeze_frame=None):
    """Main entry: load data, generate poses, render, save mp4s."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Device: {device}")

    # Load camera params
    K_orig = load_intrinsic(da3_dir, device)
    K_render = scale_intrinsic(K_orig, width, height)
    initial_c2w, source_c2ws = load_extrinsic_c2w(da3_dir, device)
    logger.info(f"Intrinsic (scaled to {width}x{height}):\n{K_render}")
    logger.info(f"Initial c2w:\n{initial_c2w}")
    logger.info(f"Loaded {len(source_c2ws)} source extrinsics")

    # Load PLY sequence
    points_list, colors_list = load_ply_sequence(da3_dir, device)
    num_pcds = len(points_list)
    logger.info(f"Loaded {num_pcds} point clouds")

    # Time-freeze: repeat a specific frame to create a pause effect
    if freeze_repeat > 0:
        if freeze_frame is None:
            freeze_frame = num_pcds // 2
        freeze_frame = max(0, min(freeze_frame, num_pcds - 1))
        logger.info(f"Time-freeze: repeating frame {freeze_frame} x{freeze_repeat} "
                     f"(inserting {freeze_repeat} extra frames)")
        insert_pos = freeze_frame + 1
        frozen_pts = points_list[freeze_frame]
        frozen_cols = colors_list[freeze_frame]
        frozen_c2w = source_c2ws[freeze_frame]
        points_list = points_list[:insert_pos] + [frozen_pts] * freeze_repeat + points_list[insert_pos:]
        colors_list = colors_list[:insert_pos] + [frozen_cols] * freeze_repeat + colors_list[insert_pos:]
        source_c2ws = source_c2ws[:insert_pos] + [frozen_c2w] * freeze_repeat + source_c2ws[insert_pos:]
        num_pcds = len(points_list)
        logger.info(f"After freeze: {num_pcds} total frames")

    # Generate target camera poses
    target_c2ws = generate_target_c2ws(traj_txt_path, initial_c2w, source_c2ws, num_pcds, device,
                                       relative_to_source=relative_to_source,
                                       rotation_only=rotation_only)
    num_frames = len(target_c2ws)
    logger.info(f"Generated {num_frames} target camera poses")

    # Setup output
    os.makedirs(output_dir, exist_ok=True)
    video_path = os.path.join(output_dir, "render_offline.mp4")
    mask_path = os.path.join(output_dir, "mask_offline.mp4")

    video_proc = open_ffmpeg_writer(video_path, width, height, fps)
    mask_proc = open_ffmpeg_writer(mask_path, width, height, fps)

    try:
        for idx in range(num_frames):
            # Cyclic access to point clouds (same as reference)
            pcd_idx = idx % num_pcds
            pts = points_list[pcd_idx]
            cols = colors_list[pcd_idx]
            c2w = target_c2ws[idx]

            img_bgr, mask_gray = render_batch(
                pts, cols, c2w, K_render, width, height,
                point_size=point_size, ss_ratio=2.0, bg_color=0
            )

            # BGR -> RGB for ffmpeg
            img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
            mask_rgb = cv2.cvtColor(mask_gray, cv2.COLOR_GRAY2RGB)

            video_proc.stdin.write(img_rgb.tobytes())
            mask_proc.stdin.write(mask_rgb.tobytes())

            if idx % 50 == 0:
                pos = c2w[:3, 3]
                logger.info(f"  Frame {idx}/{num_frames} | "
                            f"Pose: ({pos[0]:.2f}, {pos[1]:.2f}, {pos[2]:.2f})")
    finally:
        video_proc.stdin.close()
        mask_proc.stdin.close()
        video_proc.wait()
        mask_proc.wait()

    if video_proc.returncode != 0:
        raise RuntimeError(f"ffmpeg failed for {video_path}")
    if mask_proc.returncode != 0:
        raise RuntimeError(f"ffmpeg failed for {mask_path}")

    logger.info(f"Saved: {video_path}")
    logger.info(f"Saved: {mask_path}")


def render_point_cloud(da3_dir, traj_txt_path, output_dir, width=832, height=480,
                       point_size=2, fps=24, relative_to_source=False, rotation_only=False,
                       freeze_repeat=0, freeze_frame=None, render_backend="warper"):
    if render_backend == "warper":
        render_point_cloud_warper(
            da3_dir=da3_dir,
            traj_txt_path=traj_txt_path,
            output_dir=output_dir,
            width=width,
            height=height,
            fps=fps,
            relative_to_source=relative_to_source,
            rotation_only=rotation_only,
            freeze_repeat=freeze_repeat,
            freeze_frame=freeze_frame,
        )
    elif render_backend == "ply":
        render_point_cloud_ply(
            da3_dir=da3_dir,
            traj_txt_path=traj_txt_path,
            output_dir=output_dir,
            width=width,
            height=height,
            point_size=point_size,
            fps=fps,
            relative_to_source=relative_to_source,
            rotation_only=rotation_only,
            freeze_repeat=freeze_repeat,
            freeze_frame=freeze_frame,
        )
    else:
        raise ValueError(f"Unsupported render backend: {render_backend}")


# ── CLI ──

def main():
    parser = argparse.ArgumentParser(
        description="Offline point cloud rendering using DA3 output + traj_txt")
    parser.add_argument("--da3_dir", type=str, required=True,
                        help="DA3 output directory (contains frames_pcd/, intrinsic.txt, extrinsic.txt)")
    parser.add_argument("--traj_txt_path", type=str, required=True,
                        help="Trajectory txt file (3 lines: x_up, y_left, r)")
    parser.add_argument("--output_dir", type=str, required=True,
                        help="Output directory for render_offline.mp4 and mask_offline.mp4")
    parser.add_argument("--width", type=int, default=832)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--point_size", type=int, default=2)
    parser.add_argument("--fps", type=int, default=24)
    parser.add_argument("--relative_to_source", action="store_true",
                        help="Compose trajectory poses relative to initial view (default: off, use absolute poses)")
    parser.add_argument("--rotation_only", action="store_true",
                        help="Only apply rotation from the trajectory, ignore translation (tripod pan/tilt)")
    parser.add_argument("--freeze_repeat", type=int, default=0,
                        help="Number of times to repeat the freeze frame (0 = disabled)")
    parser.add_argument("--freeze_frame", type=int, default=None,
                        help="Frame index to freeze (default: middle frame)")
    parser.add_argument("--render_backend", choices=["warper", "ply"], default="warper",
                        help="Rendering backend. warper is faster and is the default.")
    args = parser.parse_args()

    render_point_cloud(
        da3_dir=args.da3_dir,
        traj_txt_path=args.traj_txt_path,
        output_dir=args.output_dir,
        width=args.width,
        height=args.height,
        point_size=args.point_size,
        fps=args.fps,
        relative_to_source=args.relative_to_source,
        rotation_only=args.rotation_only,
        freeze_repeat=args.freeze_repeat,
        freeze_frame=args.freeze_frame,
        render_backend=args.render_backend,
    )


if __name__ == "__main__":
    main()

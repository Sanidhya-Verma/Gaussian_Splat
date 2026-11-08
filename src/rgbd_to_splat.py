"""
src/rgbd_to_splat.py
Master runner script:
  - Mode 1 (Single Frame): Pass --rgb and --depth to convert 1 image pair.
  - Mode 2 (Multi-View):   Pass --multiview-dir to stitch all frames + trajectory.log.
"""

import os
import glob
import argparse
import numpy as np
from PIL import Image

from engine import extract_surfels, build_map
from writer import save_splat_ply


def load_trajectory_poses(log_path):
    """Reads 4x4 camera-to-world matrices from a Redwood trajectory.log file."""
    poses = []
    with open(log_path, "r") as f:
        lines = [line.strip() for line in f if line.strip()]

    # Every pose block in trajectory.log is 5 lines: 1 header line + 4 matrix rows
    for i in range(0, len(lines), 5):
        mat = np.array(
            [[float(v) for v in lines[i + r].split()] for r in range(1, 5)],
            dtype=np.float32
        )
        # Convert OpenCV camera convention (+Z forward, +Y down)
        # to our engine's OpenGL/SuperSplat convention (-Z forward, +Y up)
        flip = np.diag([1.0, -1.0, -1.0, 1.0]).astype(np.float32)
        poses.append(flip @ mat @ flip)

    return poses


def main():
    parser = argparse.ArgumentParser(description="Convert single or multi-view RGB-D data into a 3DGS .ply file")
    parser.add_argument("--rgb", type=str, help="Path to single RGB image (.png or .jpg)")
    parser.add_argument("--depth", type=str, help="Path to single 16-bit Depth image (.png)")
    parser.add_argument("--multiview-dir", type=str, help="Folder containing color/, depth/, and trajectory.log")
    parser.add_argument("--output", type=str, default="output.ply", help="Output .ply filename")
    parser.add_argument("--fx", type=float, default=525.0, help="Camera focal length fx")
    parser.add_argument("--fy", type=float, default=525.0, help="Camera focal length fy")
    parser.add_argument("--depth-scale", type=float, default=1000.0, help="Divisor to convert depth to meters")
    parser.add_argument("--voxel-size", type=float, default=0.008, help="Voxel size (m) for multi-view deduplication")
    args = parser.parse_args()

    # MODE A: Multi-View Stitching (uses build_map)
    if args.multiview_dir:
        color_files = sorted(glob.glob(os.path.join(args.multiview_dir, "color", "*.*")))
        depth_files = sorted(glob.glob(os.path.join(args.multiview_dir, "depth", "*.png")))
        log_path = os.path.join(args.multiview_dir, "trajectory.log")
        poses = load_trajectory_poses(log_path)

        frames = []
        for c_path, d_path, pose in zip(color_files, depth_files, poses):
            rgb = np.array(Image.open(c_path).convert("RGB"), dtype=np.float32)
            depth = np.array(Image.open(d_path), dtype=np.float32) / args.depth_scale
            frames.append({"rgb": rgb, "depth": depth, "pose": pose})

        rows, cols = frames[0]["depth"].shape
        intrinsics = (args.fx, args.fy, cols / 2.0, rows / 2.0)

        print(f"Stitching {len(frames)} views from {args.multiview_dir} (voxel_size={args.voxel_size}m)...")
        surfel_data = build_map(frames, intrinsics, voxel_size=args.voxel_size)
        save_splat_ply(args.output, surfel_data)

    # MODE B: Single Image Pair (uses extract_surfels)
    elif args.rgb and args.depth:
        rgb_img = np.array(Image.open(args.rgb).convert("RGB"), dtype=np.float32)
        depth_meters = np.array(Image.open(args.depth), dtype=np.float32) / args.depth_scale

        rows, cols = depth_meters.shape
        intrinsics = (args.fx, args.fy, cols / 2.0, rows / 2.0)

        surfel_data = extract_surfels(rgb_img, depth_meters, intrinsics)
        save_splat_ply(args.output, surfel_data)

    else:
        parser.error("Provide either (--rgb and --depth) for 1 view, OR --multiview-dir for multi-view stitching.")


if __name__ == "__main__":
    main()
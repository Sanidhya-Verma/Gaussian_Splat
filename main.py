"""
Main pipeline script:
Reads RGB-D image pairs from a directory, optionally filters to a single frame
or applies camera poses from an explicitly provided .log or .json file,
and saves the final 3D Gaussian Splat (.ply).
"""

import re
import json
import argparse
from pathlib import Path
import numpy as np
from PIL import Image

from src.engine import build_map
from src.writer import save_splat_ply, save_pointcloud_ply


def get_args():
    parser = argparse.ArgumentParser(description="Generate 3D Gaussian Splats from RGB-D images.")
    parser.add_argument("--images", type=str, required=True, help="Path to images directory (containing rgb/ and depth/ or flat files)")
    parser.add_argument("--frame", type=str, default=None, help="Optional single frame name/number to process (e.g., 00000)")
    parser.add_argument("--poses", type=str, default=None, help="Optional path to trajectory.log or poses.json")
    parser.add_argument("--output", type=str, default="output/gaussian_splat_model.ply", help="Output .ply file path")
    parser.add_argument("--voxel", type=float, default=0.008, help="Voxel size in meters for deduplication (default: 8mm)")
    parser.add_argument("--min_depth", type=float, default=0.25, help="Min depth threshold in meters")
    parser.add_argument("--max_depth", type=float, default=4.5, help="Max depth threshold in meters")
    parser.add_argument("--jump_limit", type=float, default=0.08, help="Depth difference limit for edge filtering")
    parser.add_argument("--pointcloud", action="store_true", help="Also save a raw RGB point cloud PLY")
    return parser.parse_args()


def load_rgb(path):
    return np.array(Image.open(path).convert("RGB"), dtype=np.float32)


def load_depth(path):
    depth = np.load(path) if str(path).endswith(".npy") else np.array(Image.open(path))
    if depth.ndim == 3:
        depth = depth[..., 0]
    depth = depth.astype(np.float32)
    if depth.max() > 50.0:
        depth /= 1000.0
    return depth


def find_image_pairs(images_dir):
    images_dir = Path(images_dir)
    rgb_dir = images_dir / "rgb"
    depth_dir = images_dir / "depth"

    if rgb_dir.is_dir() and depth_dir.is_dir():
        exts = ("*.png", "*.jpg", "*.jpeg", "*.npy")
        rgb_files = sorted([f for ext in exts for f in rgb_dir.glob(ext)])
        depth_files = sorted([f for ext in exts for f in depth_dir.glob(ext)])

        if not rgb_files or not depth_files:
            raise FileNotFoundError(f"Missing images in {rgb_dir} or {depth_dir}")

        if len(rgb_files) == len(depth_files):
            return list(zip(rgb_files, depth_files))

        pairs = []
        depth_map = {f.stem: f for f in depth_files}
        for rgb in rgb_files:
            match = depth_map.get(rgb.stem)
            if not match:
                nums = re.findall(r"\d+", rgb.stem)
                if nums:
                    for d_file in depth_files:
                        if nums[-1] in re.findall(r"\d+", d_file.stem):
                            match = d_file
                            break
            if match:
                pairs.append((rgb, match))
        return pairs

    all_files = sorted([f for f in images_dir.iterdir() if f.is_file()])
    depth_files = [f for f in all_files if "depth" in f.stem.lower() or f.suffix == ".npy"]
    rgb_files = [f for f in all_files if f not in depth_files and f.suffix.lower() in [".png", ".jpg", ".jpeg"]]

    if not rgb_files:
        raise FileNotFoundError(f"No RGB images found in {images_dir}")

    pairs = []
    for rgb in rgb_files:
        expected_stem = rgb.stem.lower()
        matched = None
        for d in depth_files:
            d_stem = d.stem.lower().replace("_depth", "").replace("-depth", "").replace("depth", "")
            if d_stem == expected_stem or (expected_stem and expected_stem in d.stem.lower()):
                matched = d
                break
        if matched:
            pairs.append((rgb, matched))

    if not pairs and len(rgb_files) == len(depth_files):
        pairs = list(zip(rgb_files, depth_files))

    return pairs


def load_poses(pose_path, pairs):
    if pose_path is None:
        return [np.eye(4, dtype=np.float32) for _ in pairs]

    p = Path(pose_path)
    if not p.is_file():
        raise FileNotFoundError(f"Pose file not found: {p}")

    if p.suffix.lower() == ".log":
        with open(p, "r") as f:
            lines = [line.strip() for line in f if line.strip()]
        F = np.diag([1.0, -1.0, -1.0, 1.0]).astype(np.float32)
        poses = []
        for i in range(0, len(lines), 5):
            T_cv = np.array(
                [[float(x) for x in lines[i + r].split()] for r in range(1, 5)],
                dtype=np.float32,
            )
            poses.append(F @ T_cv @ F)
        while len(poses) < len(pairs):
            poses.append(np.eye(4, dtype=np.float32))
        return poses[:len(pairs)]

    # 2. JSON format (list of 4x4 matrices or dict keyed by filename)
    with open(p, "r") as f:
        data = json.load(f)

    if isinstance(data, list):
        poses = [np.array(m, dtype=np.float32) for m in data]
        while len(poses) < len(pairs):
            poses.append(np.eye(4, dtype=np.float32))
        return poses[:len(pairs)]

    poses = []
    for rgb_path, _ in pairs:
        key = rgb_path.stem
        mat = data.get(key)
        if mat is None:
            nums = re.findall(r"\d+", key)
            if nums:
                target_num = int(nums[-1])
                for k, v in data.items():
                    k_nums = re.findall(r"\d+", k)
                    if k_nums and int(k_nums[-1]) == target_num:
                        mat = v
                        break
        poses.append(np.array(mat, dtype=np.float32) if mat is not None else np.eye(4, dtype=np.float32))
    return poses


def get_intrinsics(width, height):
    hfov_rad = np.deg2rad(90.0)
    fx = (width / 2.0) / np.tan(hfov_rad / 2.0)
    fy = fx
    cx = width / 2.0
    cy = height / 2.0
    return (float(fx), float(fy), float(cx), float(cy))


def main():
    args = get_args()

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    pairs = find_image_pairs(args.images)
    if args.frame:
        pairs = [(r, d) for r, d in pairs if args.frame in r.stem]

    if not pairs:
        print(f"No valid image pairs found in '{args.images}' (frame filter: {args.frame}).")
        return

    print(f"Loaded {len(pairs)} image pairs from {args.images}:")
    for rgb_p, depth_p in pairs:
        print(f"  {rgb_p.name} <-> {depth_p.name}")

    poses = load_poses(args.poses, pairs)

    frames = []
    first_rgb = load_rgb(pairs[0][0])
    height, width = first_rgb.shape[:2]

    for (rgb_path, depth_path), pose in zip(pairs, poses):
        rgb = load_rgb(rgb_path)
        depth = load_depth(depth_path)

        if rgb.shape[:2] != (height, width):
            rgb = np.array(Image.fromarray(rgb.astype(np.uint8)).resize((width, height), Image.Resampling.BILINEAR))
        if depth.shape[:2] != (height, width):
            depth = np.array(Image.fromarray(depth).resize((width, height), Image.Resampling.NEAREST))

        frames.append({
            "rgb": rgb,
            "depth": depth,
            "pose": pose,
            "name": rgb_path.stem
        })

    intrinsics = get_intrinsics(width, height)
    fx, fy, cx, cy = intrinsics
    print(f"Camera intrinsics: fx={fx:.1f}, fy={fy:.1f}, cx={cx:.1f}, cy={cy:.1f}")

    print("Generating 3D Gaussian splats...")
    surfels = build_map(
        frames=frames,
        intrinsics=intrinsics,
        voxel_size=args.voxel,
        min_depth=args.min_depth,
        max_depth=args.max_depth,
        jump_limit=args.jump_limit
    )

    total_splats = len(surfels["points"])
    print(f"Extracted {total_splats:,} surfels after voxel filtering ({args.voxel*1000:.0f}mm).")

    save_splat_ply(str(output_path), surfels)
    print(f"Saved 3D Gaussian model to: {output_path}")

    if args.pointcloud:
        pcd_path = output_path.with_name(f"{output_path.stem}_pcd.ply")
        save_pointcloud_ply(str(pcd_path), surfels)
        print(f"Saved point cloud to: {pcd_path}")


if __name__ == "__main__":
    main()
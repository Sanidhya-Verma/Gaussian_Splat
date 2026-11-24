"""
Main pipeline script:
Reads RGB-D image pairs from input/images/ (or specified directory),
applies camera poses and intrinsics from input/odometry/ (or specified pose file),
and saves the final 3D Gaussian Splat (.ply) to output/.
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
    parser = argparse.ArgumentParser(description="Generate 3D Gaussian Splats from RGB-D images and odometry.")
    parser.add_argument("--images", type=str, default="input/images", help="Path to images directory (default: input/images)")
    parser.add_argument("--poses", type=str, default=None, help="Optional path to poses.json, transforms.json, or trajectory.log")
    parser.add_argument("--output", type=str, default="output/gaussian_splat_model.ply", help="Output .ply file path")
    parser.add_argument("--frame", type=str, default=None, help="Optional single frame name/number to process")
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
    p = Path(images_dir)
    # Checking if images_dir or images_dir/images has rgb and depth subfolders
    if (p / "images" / "rgb").is_dir() and (p / "images" / "depth").is_dir():
        rgb_dir = p / "images" / "rgb"
        depth_dir = p / "images" / "depth"
    elif (p / "rgb").is_dir() and (p / "depth").is_dir():
        rgb_dir = p / "rgb"
        depth_dir = p / "depth"
    elif (Path("input/images/rgb")).is_dir() and (Path("input/images/depth")).is_dir():
        rgb_dir = Path("input/images/rgb")
        depth_dir = Path("input/images/depth")
    else:
        rgb_dir, depth_dir = None, None

    if rgb_dir and depth_dir:
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

    # Fallback to flat directory with color and depth files
    if not p.is_dir():
        raise FileNotFoundError(f"Image directory not found: {p}")

    all_files = sorted([f for f in p.iterdir() if f.is_file()])
    depth_files = [f for f in all_files if "depth" in f.stem.lower() or f.suffix == ".npy"]
    rgb_files = [f for f in all_files if f not in depth_files and f.suffix.lower() in [".png", ".jpg", ".jpeg"]]

    if not rgb_files:
        raise FileNotFoundError(f"No RGB images found in {p}")

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
        for candidate in [
            Path("input/odometry/poses.json"),
            Path("input/odometry/transforms.json"),
            Path("input/poses.json"),
            Path("input/transforms.json"),
        ]:
            if candidate.is_file():
                pose_path = candidate
                break

    if pose_path is None:
        print("No pose file found. Using identity poses.")
        return [np.eye(4, dtype=np.float32) for _ in pairs], None

    p = Path(pose_path)
    if not p.is_file():
        raise FileNotFoundError(f"Pose file not found: {p}")

    print(f"Loading camera poses from: {p}")

    # Redwood .log format
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
        return poses[:len(pairs)], None

    # JSON format (poses.json, transforms.json, or dictionary)
    with open(p, "r") as f:
        data = json.load(f)

    # Checking for embedded camera intrinsics
    intrinsics = None
    if isinstance(data, dict):
        if "camera_intrinsics" in data:
            c = data["camera_intrinsics"]
            intrinsics = (float(c["fl_x"]), float(c["fl_y"]), float(c["cx"]), float(c["cy"]))
        elif "fl_x" in data and "cx" in data:
            fl_y = data.get("fl_y", data["fl_x"])
            cy = data.get("cy", data.get("h", 800) / 2.0)
            intrinsics = (float(data["fl_x"]), float(fl_y), float(data["cx"]), float(cy))

    # Parsing the frames
    if isinstance(data, dict) and "frames" in data:
        raw_frames = data["frames"]
        extracted_poses = []
        F = np.diag([1.0, -1.0, -1.0, 1.0]).astype(np.float32)

        for item in raw_frames:
            if "c2w_nerf_opengl" in item:
                mat = np.array(item["c2w_nerf_opengl"], dtype=np.float32)
            elif "transform_matrix" in item:
                mat = np.array(item["transform_matrix"], dtype=np.float32)
            elif "c2w_opencv" in item:
                mat = F @ np.array(item["c2w_opencv"], dtype=np.float32) @ F
            else:
                mat = np.eye(4, dtype=np.float32)
            extracted_poses.append(mat)

        if len(extracted_poses) == len(pairs):
            return extracted_poses, intrinsics

        # If counts differ, try matching by file_path or stem number
        poses = []
        for rgb_path, _ in pairs:
            matched_mat = None
            nums = re.findall(r"\d+", rgb_path.stem)
            num_id = int(nums[-1]) if nums else -1
            for idx, item in enumerate(raw_frames):
                item_str = str(item.get("file_path", "")) + f" {idx}"
                item_nums = re.findall(r"\d+", item_str)
                if item_nums and int(item_nums[-1]) == num_id:
                    matched_mat = extracted_poses[idx]
                    break
            poses.append(matched_mat if matched_mat is not None else np.eye(4, dtype=np.float32))
        return poses, intrinsics

    if isinstance(data, list):
        poses = [np.array(m, dtype=np.float32) for m in data]
        while len(poses) < len(pairs):
            poses.append(np.eye(4, dtype=np.float32))
        return poses[:len(pairs)], intrinsics

    # Dict keyed by filename
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
    return poses, intrinsics


def get_intrinsics(width, height, custom=None):
    if custom is not None:
        return custom
    # Default to 90 degree horizontal field of view
    hfov_rad = np.deg2rad(90.0)
    fx = (width / 2.0) / np.tan(hfov_rad / 2.0)
    return (float(fx), float(fx), float(width / 2.0), float(height / 2.0))


def main():
    args = get_args()

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    #Discovering RGB-D image pairs
    pairs = find_image_pairs(args.images)
    if args.frame:
        pairs = [(r, d) for r, d in pairs if args.frame in r.stem]

    if not pairs:
        print(f"No valid image pairs found in '{args.images}' (frame filter: {args.frame}).")
        return

    print(f"Loaded {len(pairs)} image pairs from {args.images}:")
    for rgb_p, depth_p in pairs:
        print(f"  {rgb_p.name} <-> {depth_p.name}")

    # Loading camera poses and optional intrinsics
    poses, custom_intrinsics = load_poses(args.poses, pairs)

    # Reading image data into frames list
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

    # Camera intrinsics
    intrinsics = get_intrinsics(width, height, custom_intrinsics)
    fx, fy, cx, cy = intrinsics
    if custom_intrinsics is not None:
        print(f"Camera intrinsics (from odometry): fx={fx:.1f}, fy={fy:.1f}, cx={cx:.1f}, cy={cy:.1f}")
    else:
        print(f"Camera intrinsics (90° FOV default): fx={fx:.1f}, fy={fy:.1f}, cx={cx:.1f}, cy={cy:.1f}")

    # Building 3D Gaussian splat map
    print("Generating 3D Gaussian splats with voxel deduplication...")
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

    # Saving binary 62-property PLY for SuperSplat
    save_splat_ply(str(output_path), surfels)
    print(f"Saved 3D Gaussian model to: {output_path}")

    # Optional point cloud export
    if args.pointcloud:
        pcd_path = output_path.with_name(f"{output_path.stem}_pcd.ply")
        save_pointcloud_ply(str(pcd_path), surfels)
        print(f"Saved point cloud to: {pcd_path}")


if __name__ == "__main__":
    main()

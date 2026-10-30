"""
Universal Training-Free RGB-D to 3D Gaussian Splatting (Anisotropic Surfel) Engine.
Works on ANY RGB + Depth image pair (Single View) or Multi-View Fixed Camera Rig (JSON).

Usage (Single Image Pair - e.g. Intel RealSense D455 or TUM/Redwood):
    python3 src/rgbd_to_splat.py --rgb color.png --depth depth.png --output scene.ply

Usage (Multi-View Fixed Camera Rig Stitching):
    python3 src/rgbd_to_splat.py --config camera_rig.json --output stitched_object.ply
"""

import os
import json
import argparse
import numpy as np
from PIL import Image
from plyfile import PlyData, PlyElement


def get_fixed_ring_pose(angle_deg, radius=1.8, height=1.15, target=(0.0, 0.35, 0.0)):
    """Computes (R_cam, t_cam) for a fixed camera at `angle_deg` on a ring looking at `target`."""
    rad = np.deg2rad(angle_deg)
    t_cam = np.array([radius * np.sin(rad), height, radius * np.cos(rad)], dtype=np.float32)
    target = np.array(target, dtype=np.float32)

    forward = target - t_cam
    forward /= (np.linalg.norm(forward) + 1e-8)
    local_z = -forward  # Camera looks down -Z in SuperSplat convention

    world_up = np.array([0.0, 1.0, 0.0], dtype=np.float32)
    local_x = np.cross(forward, world_up)
    local_x /= (np.linalg.norm(local_x) + 1e-8)
    local_y = np.cross(local_z, local_x)

    R_cam = np.stack([local_x, local_y, local_z], axis=-1).astype(np.float32)
    return R_cam, t_cam


def rotation_matrices_to_quaternions(matrices):
    """Converts (N, 3, 3) orthonormal rotation matrices to [qw, qx, qy, qz] unit quaternions."""
    count = matrices.shape[0]
    quaternions = np.zeros((count, 4), dtype=np.float32)
    trace = matrices[:, 0, 0] + matrices[:, 1, 1] + matrices[:, 2, 2]

    mask1 = trace > 0.0
    if np.any(mask1):
        s = np.sqrt(trace[mask1] + 1.0) * 2.0
        quaternions[mask1, 0] = 0.25 * s
        quaternions[mask1, 1] = (matrices[mask1, 2, 1] - matrices[mask1, 1, 2]) / s
        quaternions[mask1, 2] = (matrices[mask1, 0, 2] - matrices[mask1, 2, 0]) / s
        quaternions[mask1, 3] = (matrices[mask1, 1, 0] - matrices[mask1, 0, 1]) / s

    mask2 = (~mask1) & (matrices[:, 0, 0] > matrices[:, 1, 1]) & (matrices[:, 0, 0] > matrices[:, 2, 2])
    if np.any(mask2):
        s = np.sqrt(1.0 + matrices[mask2, 0, 0] - matrices[mask2, 1, 1] - matrices[mask2, 2, 2]) * 2.0
        quaternions[mask2, 0] = (matrices[mask2, 2, 1] - matrices[mask2, 1, 2]) / s
        quaternions[mask2, 1] = 0.25 * s
        quaternions[mask2, 2] = (matrices[mask2, 0, 1] + matrices[mask2, 1, 0]) / s
        quaternions[mask2, 3] = (matrices[mask2, 0, 2] + matrices[mask2, 2, 0]) / s

    mask3 = (~mask1) & (~mask2) & (matrices[:, 1, 1] > matrices[:, 2, 2])
    if np.any(mask3):
        s = np.sqrt(1.0 + matrices[mask3, 1, 1] - matrices[mask3, 0, 0] - matrices[mask3, 2, 2]) * 2.0
        quaternions[mask3, 0] = (matrices[mask3, 0, 2] - matrices[mask3, 2, 0]) / s
        quaternions[mask3, 1] = (matrices[mask3, 0, 1] + matrices[mask3, 1, 0]) / s
        quaternions[mask3, 2] = 0.25 * s
        quaternions[mask3, 3] = (matrices[mask3, 1, 2] + matrices[mask3, 2, 1]) / s

    mask4 = (~mask1) & (~mask2) & (~mask3)
    if np.any(mask4):
        s = np.sqrt(1.0 + matrices[mask4, 2, 2] - matrices[mask4, 0, 0] - matrices[mask4, 1, 1]) * 2.0
        quaternions[mask4, 0] = (matrices[mask4, 1, 0] - matrices[mask4, 0, 1]) / s
        quaternions[mask4, 1] = (matrices[mask4, 0, 2] + matrices[mask4, 2, 0]) / s
        quaternions[mask4, 2] = (matrices[mask4, 1, 2] + matrices[mask4, 2, 1]) / s
        quaternions[mask4, 3] = 0.25 * s

    quaternions /= (np.linalg.norm(quaternions, axis=-1, keepdims=True) + 1e-8)
    return quaternions


def extract_world_surfels(rgb_img, Z, fx, fy, cx, cy, R_cam=None, t_cam=None,
                          min_depth=0.25, max_depth=4.5, jump_thresh=0.05):
    """
    Core mathematical pipeline:
      1. Pinhole de-projection (u, v, Z) -> 3D Camera Coordinates (X, Y, -Z)
      2. Smart one-sided (upwind) differences to preserve sharp obstacle edges
      3. Anisotropic neighbor spacing (dist_u, dist_v) to prevent row/column stripe gaps
      4. Rigid transformation into World Frame (R_cam, t_cam) and Quaternion orientation
    """
    H, W = Z.shape
    if rgb_img.shape[:2] != (H, W):
        rgb_img = np.array(Image.fromarray(rgb_img).resize((W, H), Image.BILINEAR))

    if R_cam is None:
        R_cam = np.eye(3, dtype=np.float32)
    if t_cam is None:
        t_cam = np.zeros(3, dtype=np.float32)

    u, v = np.meshgrid(np.arange(W, dtype=np.float32), np.arange(H, dtype=np.float32))
    X = (u - cx) * Z / fx
    Y = -(v - cy) * Z / fy
    P_cam = np.stack([X, Y, -Z], axis=-1)

    # Smart One-Sided (Upwind) Differences in u (columns) and v (rows)
    diff_u = P_cam[:, 1:, :] - P_cam[:, :-1, :]
    step_u_left  = np.pad(diff_u, ((0, 0), (1, 0), (0, 0)), mode="edge")
    step_u_right = np.pad(diff_u, ((0, 0), (0, 1), (0, 0)), mode="edge")
    jump_u_left, jump_u_right = np.abs(step_u_left[:, :, 2]), np.abs(step_u_right[:, :, 2])
    dP_du_cam = np.where((jump_u_left <= jump_u_right)[..., None], step_u_left, step_u_right)

    diff_v = P_cam[1:, :, :] - P_cam[:-1, :, :]
    step_v_up   = np.pad(diff_v, ((1, 0), (0, 0), (0, 0)), mode="edge")
    step_v_down = np.pad(diff_v, ((0, 1), (0, 0), (0, 0)), mode="edge")
    jump_v_up, jump_v_down = np.abs(step_v_up[:, :, 2]), np.abs(step_v_down[:, :, 2])
    dP_dv_cam = np.where((jump_v_up <= jump_v_down)[..., None], step_v_up, step_v_down)

    valid_2d = (
        np.isfinite(Z) & (Z > min_depth) & (Z < max_depth) &
        (np.minimum(jump_u_left, jump_u_right) < jump_thresh) &
        (np.minimum(jump_v_up, jump_v_down) < jump_thresh)
    )
    valid_2d[0, :] = valid_2d[-1, :] = valid_2d[:, 0] = valid_2d[:, -1] = False
    valid = valid_2d.reshape(-1)

    dist_u = np.clip(np.linalg.norm(dP_du_cam, axis=-1).reshape(-1)[valid], 0.001, 0.05)
    dist_v = np.clip(np.linalg.norm(dP_dv_cam, axis=-1).reshape(-1)[valid], 0.001, 0.05)

    # Transform to shared World Coordinate Frame
    pts_world = P_cam.reshape(-1, 3)[valid] @ R_cam.T + t_cam
    tu_world  = dP_du_cam.reshape(-1, 3)[valid] @ R_cam.T
    tv_world  = dP_dv_cam.reshape(-1, 3)[valid] @ R_cam.T

    e_u = tu_world / (np.linalg.norm(tu_world, axis=-1, keepdims=True) + 1e-8)
    normals = np.cross(e_u, tv_world)
    normals /= (np.linalg.norm(normals, axis=-1, keepdims=True) + 1e-8)
    e_v = np.cross(normals, e_u)
    e_v /= (np.linalg.norm(e_v, axis=-1, keepdims=True) + 1e-8)

    rot_matrices = np.stack([e_u, e_v, normals], axis=-1).astype(np.float32)
    quats = rotation_matrices_to_quaternions(rot_matrices)
    rgb = rgb_img.reshape(-1, 3)[valid].astype(np.float32)

    return pts_world, normals, quats, dist_u, dist_v, rgb


def write_3dgs_ply(output_path, pts, norms, quats, du, dv, rgb,
                   voxel_size=0.0, overlap_factor=0.95):
    """Packs world-aligned surfels into the standard 62-property binary 3DGS PLY format."""
    if voxel_size > 0.0:
        voxel_coords = np.floor(pts / voxel_size).astype(np.int32)
        _, unique_idx = np.unique(voxel_coords, axis=0, return_index=True)
        unique_idx = np.sort(unique_idx)
        pts, norms, quats = pts[unique_idx], norms[unique_idx], quats[unique_idx]
        du, dv, rgb = du[unique_idx], dv[unique_idx], rgb[unique_idx]

    n = len(pts)
    dtype = [("x", "f4"), ("y", "f4"), ("z", "f4"), ("nx", "f4"), ("ny", "f4"), ("nz", "f4"),
             ("f_dc_0", "f4"), ("f_dc_1", "f4"), ("f_dc_2", "f4")]
    dtype += [(f"f_rest_{i}", "f4") for i in range(45)]
    dtype += [("opacity", "f4"), ("scale_0", "f4"), ("scale_1", "f4"), ("scale_2", "f4"),
              ("rot_0", "f4"), ("rot_1", "f4"), ("rot_2", "f4"), ("rot_3", "f4")]

    splat = np.zeros(n, dtype=dtype)
    splat["x"], splat["y"], splat["z"] = pts[:, 0], pts[:, 1], pts[:, 2]
    splat["nx"], splat["ny"], splat["nz"] = norms[:, 0], norms[:, 1], norms[:, 2]

    SH_C0 = 0.28209479177387814
    splat["f_dc_0"] = (rgb[:, 0] / 255.0 - 0.5) / SH_C0
    splat["f_dc_1"] = (rgb[:, 1] / 255.0 - 0.5) / SH_C0
    splat["f_dc_2"] = (rgb[:, 2] / 255.0 - 0.5) / SH_C0
    splat["opacity"] = np.float32(10.0)

    splat["scale_0"] = np.log(np.maximum(du * overlap_factor, 1e-6)).astype(np.float32)
    splat["scale_1"] = np.log(np.maximum(dv * overlap_factor, 1e-6)).astype(np.float32)
    splat["scale_2"] = np.float32(np.log(0.0005))

    splat["rot_0"], splat["rot_1"], splat["rot_2"], splat["rot_3"] = (
        quats[:, 0], quats[:, 1], quats[:, 2], quats[:, 3]
    )

    PlyData([PlyElement.describe(splat, "vertex")], text=False).write(output_path)
    print(f"Saved {n} oriented 3DGS surfels -> {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Universal RGB-D to 3DGS Surfel Converter & Stitcher")
    parser.add_argument("--rgb", type=str, help="Path to single RGB image (.png/.jpg)")
    parser.add_argument("--depth", type=str, help="Path to single 16-bit Depth image (.png)")
    parser.add_argument("--config", type=str, help="Path to multi-view camera_rig.json")
    parser.add_argument("--output", type=str, default="output_splats.ply", help="Output .ply path")
    parser.add_argument("--fx", type=float, default=525.0, help="Focal length X (default: 525.0)")
    parser.add_argument("--fy", type=float, default=525.0, help="Focal length Y (default: 525.0)")
    parser.add_argument("--depth-scale", type=float, default=1000.0, help="Depth divisor to meters (1000 for RealSense mm, 5000 for TUM)")
    parser.add_argument("--voxel", type=float, default=0.006, help="Voxel deduplication size in meters for multi-view")
    args = parser.parse_args()

    if args.config:
        base_dir = os.path.dirname(os.path.abspath(args.config))
        with open(args.config, "r") as f:
            cfg = json.load(f)
        intr = cfg["intrinsics"]
        d_scale = float(intr.get("depth_scale", 1000.0))

        all_pts, all_norms, all_quats, all_du, all_dv, all_rgb = [], [], [], [], [], []
        for v_cfg in cfg["views"]:
            rgb_img = np.array(Image.open(os.path.join(base_dir, v_cfg["rgb"])).convert("RGB"), dtype=np.uint8)
            Z = np.array(Image.open(os.path.join(base_dir, v_cfg["depth"])), dtype=np.float32) / d_scale
            R_cam, t_cam = get_fixed_ring_pose(
                float(v_cfg["angle_deg"]), float(v_cfg.get("radius", 1.8)), float(v_cfg.get("height", 1.15))
            )
            pts, norms, quats, du, dv, rgb = extract_world_surfels(
                rgb_img, Z, float(intr["fx"]), float(intr["fy"]), float(intr["cx"]), float(intr["cy"]), R_cam, t_cam
            )
            all_pts.append(pts); all_norms.append(norms); all_quats.append(quats)
            all_du.append(du); all_dv.append(dv); all_rgb.append(rgb)

        write_3dgs_ply(
            args.output, np.vstack(all_pts), np.vstack(all_norms), np.vstack(all_quats),
            np.concatenate(all_du), np.concatenate(all_dv), np.vstack(all_rgb), voxel_size=args.voxel
        )

    elif args.rgb and args.depth:
        rgb_img = np.array(Image.open(args.rgb).convert("RGB"), dtype=np.uint8)
        Z = np.array(Image.open(args.depth), dtype=np.float32) / args.depth_scale
        H, W = Z.shape
        pts, norms, quats, du, dv, rgb = extract_world_surfels(
            rgb_img, Z, args.fx, args.fy, W / 2.0, H / 2.0
        )
        write_3dgs_ply(args.output, pts, norms, quats, du, dv, rgb, voxel_size=0.0)
    else:
        parser.error("Provide either (--rgb and --depth) for a single image OR --config for multi-view stitching.")


if __name__ == "__main__":
    main()

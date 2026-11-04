import os
import sys
import numpy as np

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.engine import extract_surfels, get_quaternions
from src.writer import save_splat_ply
from wave_experiments import make_wave_rgbd


if __name__ == "__main__":
    rgb, depth, intrinsics = make_wave_rgbd()
    fx, fy, cx, cy = intrinsics
    H, W = depth.shape

    # 1. THE VEIL BUG: Naive central difference (np.gradient) across object edges
    u, v = np.meshgrid(np.arange(W, dtype=np.float32), np.arange(H, dtype=np.float32))
    P = np.stack([(u - cx) * depth / fx, -(v - cy) * depth / fy, -depth], axis=-1)

    step_y, step_x = np.gradient(P, axis=(0, 1))
    vec_x = step_x.reshape(-1, 3)
    vec_y = step_y.reshape(-1, 3)

    dir_x = vec_x / (np.linalg.norm(vec_x, axis=1, keepdims=True) + 1e-8)
    normals = np.cross(dir_x, vec_y)
    normals /= (np.linalg.norm(normals, axis=1, keepdims=True) + 1e-8)
    dir_y = np.cross(normals, dir_x)

    frames = np.stack([dir_x, dir_y, normals], axis=-1)
    quats_veil = get_quaternions(frames)

    veil_data = {
        "points": P.reshape(-1, 3).astype(np.float32),
        "normals": normals.astype(np.float32),
        "quaternions": quats_veil,
        "scale_u": np.linalg.norm(vec_x, axis=1).astype(np.float32),
        "scale_v": np.linalg.norm(vec_y, axis=1).astype(np.float32),
        "colors": rgb.reshape(-1, 3),
    }
    save_splat_ply("wave_VEIL_BUG_unfiltered.ply", veil_data, overlap=0.85)

    # 2. THE UPWIND FIX: Direct call to your clean src/engine.py (Upwind one-sided differences)
    upwind_data = extract_surfels(rgb, depth, intrinsics, jump_limit=0.08)
    save_splat_ply("wave_COMPLETE_upwind_anisotropic.ply", upwind_data, overlap=0.85)

import os
import sys
import importlib.util
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.engine import extract_surfels, get_quaternions
from src.writer import save_splat_ply

_wave_script = ROOT / "adaptive splats" / "wave_broken_vs_fixed_adaptive.py"
if _wave_script.is_file():
    _spec = importlib.util.spec_from_file_location("wave_broken_vs_fixed_adaptive", str(_wave_script))
    _mod = importlib.util.module_from_spec(_spec)
    _spec.loader.exec_module(_mod)
    make_wave_rgbd = _mod.make_wave_rgbd
else:
    def make_wave_rgbd(H=120, W=160):
        fx, fy, cx, cy = 120.0, 120.0, W / 2.0, H / 2.0
        u, v = np.meshgrid(np.arange(W, dtype=np.float32), np.arange(H, dtype=np.float32))
        Z = 4.2 - (v / H) * 3.4
        X = (u - cx) * Z / fx
        Y = 0.35 * np.sin(2.5 * X) * np.cos(2.5 * Z) - 0.3
        obs = (u > 60) & (u < 100) & (v > 35) & (v < 75)
        Z[obs] = 1.3
        X[obs] = (u[obs] - cx) * Z[obs] / fx
        Y[obs] = -(v[obs] - cy) * Z[obs] / fy
        R = np.clip((X + 2.5) / 5.0 * 255, 20, 255)
        G = np.clip((Y + 0.7) * 220, 20, 255)
        B = np.clip((Z - 0.8) / 3.4 * 255, 20, 255)
        R[obs], G[obs], B[obs] = 0, 255, 220
        return np.stack([R, G, B], axis=-1).astype(np.float32), Z.astype(np.float32), (fx, fy, cx, cy)


if __name__ == "__main__":
    rgb, depth, intrinsics = make_wave_rgbd()
    fx, fy, cx, cy = intrinsics
    H, W = depth.shape

    #THE VEIL BUG: Naive central difference (np.gradient) across object edges
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

    #THE UPWIND FIX: Direct call to the clean implementation in src/engine.py (Upwind one-sided differences)
    upwind_data = extract_surfels(rgb, depth, intrinsics, jump_limit=0.08)
    save_splat_ply("wave_COMPLETE_upwind_anisotropic.ply", upwind_data, overlap=0.85)

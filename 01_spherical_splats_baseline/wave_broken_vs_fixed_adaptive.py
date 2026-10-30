"""
The main goal here is to see that how does keeping a fixed sclae differ from keeping adaptive.
The dynamic scaling (propotional to Z) helps to make size of slapt bigger at far away which 
we need to do beacause a real camera works like a flashlight cone shooting rays out from a 
tiny pinhole lens: X = ((u-cx)/Fx).Z, as Z increase the delta X also increase so we now need 
bigger splat
"""
import os
import sys
import numpy as np

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.engine import extract_surfels
from src.writer import save_splat_ply


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

    rgb = np.stack([R, G, B], axis=-1).astype(np.float32)
    return rgb, Z.astype(np.float32), (fx, fy, cx, cy)


if __name__ == "__main__":
    rgb, depth, intrinsics = make_wave_rgbd()
    fx = intrinsics[0]

    # 1. BROKEN: Constant 1.2cm radius across the entire depth cone (no jump filter)
    surfels_broken = extract_surfels(rgb, depth, intrinsics, jump_limit=999.0)
    surfels_broken["scale_u"][:] = 0.012
    surfels_broken["scale_v"][:] = 0.012
    save_splat_ply("wave_BROKEN_constant_no_filter.ply", surfels_broken, overlap=1.0)

    # 2. FIXED CAMERA CONE: Circular adaptive radius (Z / fx), leaves horizontal row gaps
    surfels_iso = extract_surfels(rgb, depth, intrinsics, jump_limit=0.08)
    z_vals = -surfels_iso["points"][:, 2]  # Recover depth Z from cam_z = -depth
    circular_scale = (z_vals / fx).astype(np.float32)
    surfels_iso["scale_u"] = circular_scale
    surfels_iso["scale_v"] = circular_scale
    save_splat_ply("wave_FIXED_adaptive_filtered.ply", surfels_iso, overlap=0.95)
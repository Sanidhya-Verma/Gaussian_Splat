"""
While scaling radius by Z/fx solved the camera cone problem for walls facing the camera 
flat-on, it introduced a new artifact on sloped floors or waves:

Main goal is to fix it here :
So idea is that instead we scale by  measuring the true 3D Euclidean distance to the column 
neighbor and row neighbor independently, and align the full 3D orthonormal rotation matrix 
(eu,ev,normal)
"""

import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.engine import extract_surfels
from src.writer import save_splat_ply
from wave_broken_vs_fixed import make_wave_rgbd


if __name__ == "__main__":
    rgb, depth, intrinsics = make_wave_rgbd()

    # Engine natively computes true 3D anisotropic neighbor distances (dist_x, dist_y)
    surfels_aniso = extract_surfels(rgb, depth, intrinsics, jump_limit=0.08)
    save_splat_ply("wave_BETTER_ADAPTIVE_anisotropic.ply", surfels_aniso, overlap=0.85)
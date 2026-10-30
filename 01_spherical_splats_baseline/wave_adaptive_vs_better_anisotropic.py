"""
Stage 01b: Simple Depth Scaling (Z / fx) vs. True 3D Neighbor Distance (Anisotropic Surfels).

In `wave_broken_vs_fixed.py`, scaling radius by Z / fx solved the camera cone problem for
surfaces facing the camera flat-on. However, because we set both pancake axes equal 
(scale_0 = scale_1 = log(Z / fx)):
- Horizontally (across columns u), neighboring pixels are ~2.0 cm apart (Z / fx).
- Vertically (across rows v), because the camera views the floor/wave at a slanted grazing 
  angle, neighboring rows are ~3.5 cm apart in 3D space!
- Result: Circular pancakes touch left-to-right (forming solid ribbons) but are too short 
  front-to-back, leaving 120 horizontal "Venetian blind" stripe gaps!

THE FIX (wave_BETTER_ADAPTIVE_anisotropic.ply):
We measure the true 3D Euclidean distance to the horizontal neighbor (dist_u) and vertical 
row neighbor (dist_v) independently, and align the full 3D rotation matrix [e_u, e_v, normal]:
  - scale_0 = log(dist_u)   <-- seals horizontal column gaps
  - scale_1 = log(dist_v)   <-- seals vertical row-to-row stripe gaps on steep slopes
  - scale_2 = log(0.0005)   <-- keeps the surfel paper-thin (0.5 mm) along the normal
"""

import numpy as np
from plyfile import PlyData, PlyElement


def rotation_matrices_to_quaternions(R):
    """
    Converts an array of (N, 3, 3) orthonormal rotation matrices into
    unit quaternions (qw, qx, qy, qz) for 3DGS rot_0..3.
    Ensures local X aligns with column tangent (u) and local Y aligns with row tangent (v).
    """
    n = R.shape[0]
    q = np.zeros((n, 4), dtype='f4')
    trace = R[:, 0, 0] + R[:, 1, 1] + R[:, 2, 2]

    # Case 1: Positive trace
    mask1 = trace > 0
    s = np.sqrt(trace[mask1] + 1.0) * 2.0
    q[mask1, 0] = 0.25 * s
    q[mask1, 1] = (R[mask1, 2, 1] - R[mask1, 1, 2]) / s
    q[mask1, 2] = (R[mask1, 0, 2] - R[mask1, 2, 0]) / s
    q[mask1, 3] = (R[mask1, 1, 0] - R[mask1, 0, 1]) / s

    # Case 2: R[0,0] is largest diagonal
    mask2 = (~mask1) & (R[:, 0, 0] > R[:, 1, 1]) & (R[:, 0, 0] > R[:, 2, 2])
    s = np.sqrt(1.0 + R[mask2, 0, 0] - R[mask2, 1, 1] - R[mask2, 2, 2]) * 2.0
    q[mask2, 0] = (R[mask2, 2, 1] - R[mask2, 1, 2]) / s
    q[mask2, 1] = 0.25 * s
    q[mask2, 2] = (R[mask2, 0, 1] + R[mask2, 1, 0]) / s
    q[mask2, 3] = (R[mask2, 0, 2] + R[mask2, 2, 0]) / s

    # Case 3: R[1,1] is largest diagonal
    mask3 = (~mask1) & (~mask2) & (R[:, 1, 1] > R[:, 2, 2])
    s = np.sqrt(1.0 + R[mask3, 1, 1] - R[mask3, 0, 0] - R[mask3, 2, 2]) * 2.0
    q[mask3, 0] = (R[mask3, 0, 2] - R[mask3, 2, 0]) / s
    q[mask3, 1] = (R[mask3, 0, 1] + R[mask3, 1, 0]) / s
    q[mask3, 2] = 0.25 * s
    q[mask3, 3] = (R[mask3, 1, 2] + R[mask3, 2, 1]) / s

    # Case 4: R[2,2] is largest diagonal
    mask4 = (~mask1) & (~mask2) & (~mask3)
    s = np.sqrt(1.0 + R[mask4, 2, 2] - R[mask4, 0, 0] - R[mask4, 1, 1]) * 2.0
    q[mask4, 0] = (R[mask4, 1, 0] - R[mask4, 0, 1]) / s
    q[mask4, 1] = (R[mask4, 0, 2] + R[mask4, 2, 0]) / s
    q[mask4, 2] = (R[mask4, 1, 2] + R[mask4, 2, 1]) / s
    q[mask4, 3] = 0.25 * s

    q /= (np.linalg.norm(q, axis=-1, keepdims=True) + 1e-8)
    return q


def write_3dgs_ply(filename, pts, normals, r, g, b, scale_0, scale_1, quats):
    """Packs splat parameters into the 62-property binary PLY format for SuperSplat."""
    n = len(pts)
    dtype = [('x', 'f4'), ('y', 'f4'), ('z', 'f4'),
             ('nx', 'f4'), ('ny', 'f4'), ('nz', 'f4'),
             ('f_dc_0', 'f4'), ('f_dc_1', 'f4'), ('f_dc_2', 'f4')]
    dtype += [(f'f_rest_{i}', 'f4') for i in range(45)]
    dtype += [('opacity', 'f4'),
              ('scale_0', 'f4'), ('scale_1', 'f4'), ('scale_2', 'f4'),
              ('rot_0', 'f4'), ('rot_1', 'f4'), ('rot_2', 'f4'), ('rot_3', 'f4')]

    splat = np.zeros(n, dtype=dtype)
    splat['x'], splat['y'], splat['z'] = pts[:, 0], pts[:, 1], pts[:, 2]
    splat['nx'], splat['ny'], splat['nz'] = normals[:, 0], normals[:, 1], normals[:, 2]

    splat['f_dc_0'] = (r / 255.0 - 0.5) / 0.28209
    splat['f_dc_1'] = (g / 255.0 - 0.5) / 0.28209
    splat['f_dc_2'] = (b / 255.0 - 0.5) / 0.28209
    splat['opacity'] = 10.0

    splat['scale_0'] = scale_0
    splat['scale_1'] = scale_1
    splat['scale_2'] = np.float32(np.log(0.0005))  # 0.5mm thin along surface normal

    splat['rot_0'] = quats[:, 0]
    splat['rot_1'] = quats[:, 1]
    splat['rot_2'] = quats[:, 2]
    splat['rot_3'] = quats[:, 3]

    PlyData([PlyElement.describe(splat, 'vertex')], text=False).write(filename)
    print(f"Saved: {filename} ({n} splats)")


# =====================================================================
# 1. Capture the Rainbow Wave Through a Real Pinhole Camera (H=120, W=160)
# =====================================================================
H, W = 120, 160
fx, fy = 120.0, 120.0
cx, cy = W / 2.0, H / 2.0

u, v = np.meshgrid(np.arange(W), np.arange(H))
Z = 4.2 - (v / H) * 3.4
X = (u - cx) * Z / fx
Y = 0.35 * np.sin(2.5 * X) * np.cos(2.5 * Z) - 0.3

# Floating cyan obstacle panel at Z = 1.3m
obstacle_mask = (u > 60) & (u < 100) & (v > 35) & (v < 75)
Z[obstacle_mask] = 1.3
X[obstacle_mask] = (u[obstacle_mask] - cx) * Z[obstacle_mask] / fx
Y[obstacle_mask] = -(v[obstacle_mask] - cy) * Z[obstacle_mask] / fy

R_col = np.clip((X + 2.5) / 5.0 * 255, 20, 255).astype('f4')
G_col = np.clip((Y + 0.7) * 220, 20, 255).astype('f4')
B_col = np.clip((Z - 0.8) / 3.4 * 255, 20, 255).astype('f4')
R_col[obstacle_mask], G_col[obstacle_mask], B_col[obstacle_mask] = 0, 255, 220

# Stack into 3D coordinates (looking down -Z in SuperSplat)
P = np.stack([X, Y, -Z], axis=-1)

# Compute 3D Grid Tangents (dP_du = column step, dP_dv = row step)
dP_dv, dP_du = np.gradient(P, axis=(0, 1))

# Naive Gradient Edge Masking (drops depth jumps > 8 cm)
dZ_dv, dZ_du = np.gradient(Z, axis=(0, 1))
depth_grad = np.sqrt(dZ_du**2 + dZ_dv**2)
valid = (depth_grad < 0.08).flatten()

pts_valid = P.reshape(-1, 3)[valid].astype('f4')
r_valid = R_col.flatten()[valid]
g_valid = G_col.flatten()[valid]
b_valid = B_col.flatten()[valid]

# =====================================================================
# FILE 1: SIMPLE ADAPTIVE (Circular Z / fx Scaling -> Row Gaps on Slopes)
# =====================================================================
normals_full = np.cross(dP_du, dP_dv)
normals_full /= (np.linalg.norm(normals_full, axis=-1, keepdims=True) + 1e-8)
norms_valid = normals_full.reshape(-1, 3)[valid].astype('f4')

nx, ny, nz = norms_valid[:, 0], norms_valid[:, 1], norms_valid[:, 2]
qw, qx, qy, qz = 1.0 + nz, -ny, nx, np.zeros(len(pts_valid), dtype='f4')
q_len = np.sqrt(qw**2 + qx**2 + qy**2 + qz**2) + 1e-8
quats_simple = np.stack([qw / q_len, qx / q_len, qy / q_len, qz / q_len], axis=-1)

z_valid = Z.flatten()[valid].astype('f4')
scale_z_fx = np.log((z_valid / fx) * 0.95).astype('f4')

write_3dgs_ply(
    'wave_FIXED_adaptive_filtered.ply',
    pts_valid, norms_valid, r_valid, g_valid, b_valid,
    scale_0=scale_z_fx,
    scale_1=scale_z_fx,
    quats=quats_simple
)

# =====================================================================
# FILE 2: BETTER ADAPTIVE (True 3D Neighbor Distance dist_u & dist_v)
# =====================================================================
dist_u = np.linalg.norm(dP_du, axis=-1).flatten()[valid]
dist_v = np.linalg.norm(dP_dv, axis=-1).flatten()[valid]

dist_u = np.clip(dist_u, 0.001, 0.08)
dist_v = np.clip(dist_v, 0.001, 0.08)

# Orthonormal 3D Basis [e_u, e_v, normal] for Exact Elliptical Alignment
tu = dP_du.reshape(-1, 3)[valid]
tv = dP_dv.reshape(-1, 3)[valid]

e_u = tu / (np.linalg.norm(tu, axis=-1, keepdims=True) + 1e-8)
normals_ortho = np.cross(e_u, tv)
normals_ortho /= (np.linalg.norm(normals_ortho, axis=-1, keepdims=True) + 1e-8)
e_v = np.cross(normals_ortho, e_u)

rot_matrices = np.stack([e_u, e_v, normals_ortho], axis=-1)
quats_aniso = rotation_matrices_to_quaternions(rot_matrices)

overlap_factor = 0.85
scale_u = np.log(dist_u * overlap_factor).astype('f4')
scale_v = np.log(dist_v * overlap_factor).astype('f4')

write_3dgs_ply(
    'wave_BETTER_ADAPTIVE_anisotropic.ply',
    pts_valid, normals_ortho.astype('f4'), r_valid, g_valid, b_valid,
    scale_0=scale_u,
    scale_1=scale_v,
    quats=quats_aniso
)

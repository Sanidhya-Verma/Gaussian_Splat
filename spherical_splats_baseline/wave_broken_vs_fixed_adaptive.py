import numpy as np
from plyfile import PlyData, PlyElement

# Capturing the rainbow wave by a pin hole camera 
H, W = 120, 160
fx, fy = 120.0, 120.0       # Camera focal length in pixels
cx, cy = W / 2.0, H / 2.0   # Optical center

# Pixel grid (u = columns, v = rows)
u, v = np.meshgrid(np.arange(W), np.arange(H))

# Depth Z ranges from 0.8m (close bottom of image) to 4.2m (far top of image)
Z = 4.2 - (v / H) * 3.4

# Calculate real-world X first using pinhole formula: X = (u - cx) * Z / fx
X = (u - cx) * Z / fx

# Apply your exact 3D sine/cosine wave to the height Y!
Y = 0.35 * np.sin(2.5 * X) * np.cos(2.5 * Z) - 0.3

# Add a floating obstacle panel in the center at Z = 1.3m (to test Edge Veils!)
obstacle_mask = (u > 60) & (u < 100) & (v > 35) & (v < 75)
Z[obstacle_mask] = 1.3
X[obstacle_mask] = (u[obstacle_mask] - cx) * Z[obstacle_mask] / fx
Y[obstacle_mask] = -(v[obstacle_mask] - cy) * Z[obstacle_mask] / fy

# Same Rainbow RGB Colors from your wave (plus bright cyan for the obstacle)
R = np.clip((X + 2.5) / 5.0 * 255, 20, 255).astype('f4')
G = np.clip((Y + 0.7) * 220, 20, 255).astype('f4')
B = np.clip((Z - 0.8) / 3.4 * 255, 20, 255).astype('f4')
R[obstacle_mask], G[obstacle_mask], B[obstacle_mask] = 0, 255, 220

# Stack into 3D points (looking down -Z in SuperSplat)
P = np.stack([X, Y, -Z], axis=-1)


# Trying to compute mpute Surface Normals & Quaternions (Same for both files)
dP_dv, dP_du = np.gradient(P, axis=(0, 1))
normals = np.cross(dP_du, dP_dv)
normals /= (np.linalg.norm(normals, axis=-1, keepdims=True) + 1e-8)

def save_3dgs_ply(filename, pts, norms, r, g, b, scale_wide_array):
    n = len(pts)
    nx, ny, nz = norms[:, 0], norms[:, 1], norms[:, 2]

    # Quaternion rotating local Z (0, 0, 1) -> normal (nx, ny, nz)
    qw, qx, qy, qz = 1.0 + nz, -ny, nx, np.zeros(n, dtype='f4')
    q_len = np.sqrt(qw**2 + qx**2 + qy**2 + qz**2) + 1e-8

    dtype = [('x', 'f4'), ('y', 'f4'), ('z', 'f4'),
             ('nx', 'f4'), ('ny', 'f4'), ('nz', 'f4'),
             ('f_dc_0', 'f4'), ('f_dc_1', 'f4'), ('f_dc_2', 'f4')]
    dtype += [(f'f_rest_{i}', 'f4') for i in range(45)]
    dtype += [('opacity', 'f4'),
              ('scale_0', 'f4'), ('scale_1', 'f4'), ('scale_2', 'f4'),
              ('rot_0', 'f4'), ('rot_1', 'f4'), ('rot_2', 'f4'), ('rot_3', 'f4')]

    splat = np.zeros(n, dtype=dtype)
    splat['x'], splat['y'], splat['z'] = pts[:, 0], pts[:, 1], pts[:, 2]
    splat['nx'], splat['ny'], splat['nz'] = nx, ny, nz
    splat['f_dc_0'] = (r / 255.0 - 0.5) / 0.28209
    splat['f_dc_1'] = (g / 255.0 - 0.5) / 0.28209
    splat['f_dc_2'] = (b / 255.0 - 0.5) / 0.28209
    splat['opacity'] = 10.0

    splat['scale_0'] = scale_wide_array
    splat['scale_1'] = scale_wide_array
    splat['scale_2'] = np.float32(np.log(0.0005))  # 0.5mm paper-thin

    splat['rot_0'], splat['rot_1'], splat['rot_2'], splat['rot_3'] = qw/q_len, qx/q_len, qy/q_len, qz/q_len
    PlyData([PlyElement.describe(splat, 'vertex')], text=False).write(filename)
    print(f"Saved: {filename} ({n} splats)")

# FILE 1: BROKEN (Constant Scale + NO Gradient Edge Masking)

pts_all = P.reshape(-1, 3).astype('f4')
norms_all = normals.reshape(-1, 3).astype('f4')

# Constant radius of 1.2 cm (log(0.012)) for every single point regardless of depth Z!
constant_scale = np.full(len(pts_all), np.log(0.012), dtype='f4')

save_3dgs_ply(
    'wave_BROKEN_constant_no_filter.ply',
    pts_all, norms_all, R.flatten(), G.flatten(), B.flatten(),
    scale_wide_array=constant_scale
)


# FILE 2: FIXED (Per-Pixel Depth Scale + Gradient Edge Masking)
dZ_dv, dZ_du = np.gradient(Z, axis=(0, 1))
depth_grad = np.sqrt(dZ_du**2 + dZ_dv**2)
valid = (depth_grad < 0.08).flatten()

# B. Per-Pixel Depth Scaling: Radius grows proportionally with Z / fx!
z_valid = Z.flatten()[valid].astype('f4')
adaptive_scale = np.log((z_valid / fx) * 0.95).astype('f4')

save_3dgs_ply(
    'wave_FIXED_adaptive_filtered.ply',
    pts_all[valid], norms_all[valid],
    R.flatten()[valid], G.flatten()[valid], B.flatten()[valid],
    scale_wide_array=adaptive_scale
)

"""
Folder 04: Real-World RGB-D Sensor Validation (Redwood Living Room Dataset)
Downloads 5 real RGB-D frames + hardware camera poses (odometry.log) and generates:
  1. redwood_1_single_view.ply      (Single frame showing occlusion holes behind furniture)
  2. redwood_5_views_stitched.ply   (Multi-view stitched 3D room scan with voxel deduplication)
"""

import os
import urllib.request
import numpy as np
from PIL import Image
from plyfile import PlyData, PlyElement


# ==============================================================================
# 1. AUTO-DOWNLOAD REAL REDWOOD RGB-D SEQUENCE (5 FRAMES + ODOMETRY POSES)
# ==============================================================================
BASE_URL = "https://raw.githubusercontent.com/isl-org/open3d_downloads/master/RGBD/odometry_data/"
DATA_DIR = "redwood_sample_data"
os.makedirs(os.path.join(DATA_DIR, "color"), exist_ok=True)
os.makedirs(os.path.join(DATA_DIR, "depth"), exist_ok=True)

files_to_fetch = ["odometry.log"]
for i in range(5):
    files_to_fetch.append(f"color/{i:05d}.jpg")
    files_to_fetch.append(f"depth/{i:05d}.png")

print("Checking / downloading real Redwood RGB-D frames...")
for rel_path in files_to_fetch:
    local_path = os.path.join(DATA_DIR, rel_path)
    if not os.path.exists(local_path):
        urllib.request.urlretrieve(BASE_URL + rel_path, local_path)
        print(f"  Downloaded {rel_path}")


def load_redwood_poses(log_path):
    """Parses Open3D/Redwood odometry.log into a list of 4x4 camera-to-world matrices."""
    with open(log_path, "r") as f:
        lines = [line.strip() for line in f if line.strip()]
    poses = []
    for i in range(0, len(lines), 5):
        mat = np.array([[float(x) for x in lines[i + r].split()] for r in range(1, 5)], dtype=np.float32)
        poses.append(mat)
    return poses


# ==============================================================================
# 2. QUATERNION & UPWIND ANISOTROPIC SURFEL ENGINE
# ==============================================================================
def rotation_matrices_to_quaternions(R):
    n = R.shape[0]
    q = np.zeros((n, 4), dtype=np.float32)
    trace = R[:, 0, 0] + R[:, 1, 1] + R[:, 2, 2]

    m1 = trace > 0.0
    if np.any(m1):
        s = np.sqrt(trace[m1] + 1.0) * 2.0
        q[m1, 0] = 0.25 * s
        q[m1, 1] = (R[m1, 2, 1] - R[m1, 1, 2]) / s
        q[m1, 2] = (R[m1, 0, 2] - R[m1, 2, 0]) / s
        q[m1, 3] = (R[m1, 1, 0] - R[m1, 0, 1]) / s

    m2 = (~m1) & (R[:, 0, 0] > R[:, 1, 1]) & (R[:, 0, 0] > R[:, 2, 2])
    if np.any(m2):
        s = np.sqrt(1.0 + R[m2, 0, 0] - R[m2, 1, 1] - R[m2, 2, 2]) * 2.0
        q[m2, 0] = (R[m2, 2, 1] - R[m2, 1, 2]) / s
        q[m2, 1] = 0.25 * s
        q[m2, 2] = (R[m2, 0, 1] + R[m2, 1, 0]) / s
        q[m2, 3] = (R[m2, 0, 2] + R[m2, 2, 0]) / s

    m3 = (~m1) & (~m2) & (R[:, 1, 1] > R[:, 2, 2])
    if np.any(m3):
        s = np.sqrt(1.0 + R[m3, 1, 1] - R[m3, 0, 0] - R[m3, 2, 2]) * 2.0
        q[m3, 0] = (R[m3, 0, 2] - R[m3, 2, 0]) / s
        q[m3, 1] = (R[m3, 0, 1] + R[m3, 1, 0]) / s
        q[m3, 2] = 0.25 * s
        q[m3, 3] = (R[m3, 1, 2] + R[m3, 2, 1]) / s

    m4 = (~m1) & (~m2) & (~m3)
    if np.any(m4):
        s = np.sqrt(1.0 + R[m4, 2, 2] - R[m4, 0, 0] - R[m4, 1, 1]) * 2.0
        q[m4, 0] = (R[m4, 1, 0] - R[m4, 0, 1]) / s
        q[m4, 1] = (R[m4, 0, 2] + R[m4, 2, 0]) / s
        q[m4, 2] = (R[m4, 1, 2] + R[m4, 2, 1]) / s
        q[m4, 3] = 0.25 * s

    q /= (np.linalg.norm(q, axis=-1, keepdims=True) + 1e-8)
    return q


def process_real_frame(rgb_img, Z, fx, fy, cx, cy, T_cam_to_world):
    H, W = Z.shape
    u, v = np.meshgrid(np.arange(W, dtype=np.float32), np.arange(H, dtype=np.float32))

    # Pinhole de-projection in OpenCV Camera Frame (+X right, +Y down, +Z forward)
    X_cv = (u - cx) * Z / fx
    Y_cv = (v - cy) * Z / fy
    P_cv = np.stack([X_cv, Y_cv, Z], axis=-1)

    # Smart One-Sided (Upwind) Differences
    diff_u = P_cv[:, 1:, :] - P_cv[:, :-1, :]
    step_u_left  = np.pad(diff_u, ((0, 0), (1, 0), (0, 0)), mode="edge")
    step_u_right = np.pad(diff_u, ((0, 0), (0, 1), (0, 0)), mode="edge")
    ju_l, ju_r = np.abs(step_u_left[:, :, 2]), np.abs(step_u_right[:, :, 2])
    dP_du = np.where((ju_l <= ju_r)[..., None], step_u_left, step_u_right)

    diff_v = P_cv[1:, :, :] - P_cv[:-1, :, :]
    step_v_up   = np.pad(diff_v, ((1, 0), (0, 0), (0, 0)), mode="edge")
    step_v_down = np.pad(diff_v, ((0, 1), (0, 0), (0, 0)), mode="edge")
    jv_u, jv_d = np.abs(step_v_up[:, :, 2]), np.abs(step_v_down[:, :, 2])
    dP_dv = np.where((jv_u <= jv_d)[..., None], step_v_up, step_v_down)

    valid_2d = (
        np.isfinite(Z) & (Z > 0.3) & (Z < 4.0) &
        (np.minimum(ju_l, ju_r) < 0.05) &
        (np.minimum(jv_u, jv_d) < 0.05)
    )
    valid_2d[0, :] = valid_2d[-1, :] = valid_2d[:, 0] = valid_2d[:, -1] = False
    valid = valid_2d.reshape(-1)

    dist_u = np.clip(np.linalg.norm(dP_du, axis=-1).reshape(-1)[valid], 0.001, 0.05)
    dist_v = np.clip(np.linalg.norm(dP_dv, axis=-1).reshape(-1)[valid], 0.001, 0.05)

    # Transform from OpenCV Camera Frame -> World Frame via T_cam_to_world
    R_cv = T_cam_to_world[:3, :3]
    t_cv = T_cam_to_world[:3, 3]

    pts_w_cv = P_cv.reshape(-1, 3)[valid] @ R_cv.T + t_cv
    tu_w_cv  = dP_du.reshape(-1, 3)[valid] @ R_cv.T
    tv_w_cv  = dP_dv.reshape(-1, 3)[valid] @ R_cv.T

    # Flip Y and Z so the scene stands upright in SuperSplat (+Y up, looking down -Z)
    flip = np.array([1.0, -1.0, -1.0], dtype=np.float32)
    pts_world = pts_w_cv * flip
    tu_world  = tu_w_cv * flip
    tv_world  = tv_w_cv * flip

    e_u = tu_world / (np.linalg.norm(tu_world, axis=-1, keepdims=True) + 1e-8)
    normals = np.cross(e_u, tv_world)
    normals /= (np.linalg.norm(normals, axis=-1, keepdims=True) + 1e-8)
    e_v = np.cross(normals, e_u)
    e_v /= (np.linalg.norm(e_v, axis=-1, keepdims=True) + 1e-8)

    quats = rotation_matrices_to_quaternions(np.stack([e_u, e_v, normals], axis=-1))
    rgb = rgb_img.reshape(-1, 3)[valid].astype(np.float32)

    return pts_world, normals, quats, dist_u, dist_v, rgb


def save_3dgs_ply(filename, pts, norms, quats, du, dv, rgb, voxel_size=0.0):
    if voxel_size > 0:
        vox = np.floor(pts / voxel_size).astype(np.int32)
        _, idx = np.unique(vox, axis=0, return_index=True)
        idx = np.sort(idx)
        pts, norms, quats, du, dv, rgb = pts[idx], norms[idx], quats[idx], du[idx], dv[idx], rgb[idx]

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
    splat["scale_0"] = np.log(du * 0.95).astype(np.float32)
    splat["scale_1"] = np.log(dv * 0.95).astype(np.float32)
    splat["scale_2"] = np.float32(np.log(0.0005))
    splat["rot_0"], splat["rot_1"], splat["rot_2"], splat["rot_3"] = quats[:, 0], quats[:, 1], quats[:, 2], quats[:, 3]

    PlyData([PlyElement.describe(splat, "vertex")], text=False).write(filename)
    print(f"Saved {filename} ({n} splats)")


# ==============================================================================
# 3. RUN SINGLE-SHOT VS. 5-VIEW STITCHED COMPARISON
# ==============================================================================
poses = load_redwood_poses(os.path.join(DATA_DIR, "odometry.log"))
fx, fy, cx, cy = 525.0 / 2.0, 525.0 / 2.0, 319.5 / 2.0, 239.5 / 2.0

all_pts, all_norms, all_quats, all_du, all_dv, all_rgb = [], [], [], [], [], []

for i in range(5):
    rgb = np.array(Image.open(os.path.join(DATA_DIR, f"color/{i:05d}.jpg")))[::2, ::2]
    Z = np.array(Image.open(os.path.join(DATA_DIR, f"depth/{i:05d}.png")), dtype=np.float32)[::2, ::2] / 1000.0

    pts, norms, quats, du, dv, c = process_real_frame(rgb, Z, fx, fy, cx, cy, poses[i])

    if i == 0:
        # Save Single-Shot baseline from Frame 0
        save_3dgs_ply("redwood_1_single_view.ply", pts, norms, quats, du, dv, c, voxel_size=0.0)

    all_pts.append(pts); all_norms.append(norms); all_quats.append(quats)
    all_du.append(du); all_dv.append(dv); all_rgb.append(c)

# Save Stitched 5-View Scan with 8mm Voxel Deduplication
save_3dgs_ply(
    "redwood_5_views_stitched.ply",
    np.vstack(all_pts), np.vstack(all_norms), np.vstack(all_quats),
    np.concatenate(all_du), np.concatenate(all_dv), np.vstack(all_rgb),
    voxel_size=0.008
)

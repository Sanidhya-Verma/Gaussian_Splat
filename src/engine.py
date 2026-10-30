"""
Here is the main engine of our Traing free RGB-D to 3D Gaussian Splatting model. It works on
ANY RGB + Depth image pair (Single View) or Multi-View Fixed Camera Rig (JSON).

Usage (Single Image Pair - e.g. Intel RealSense D455 or TUM/Redwood):
    python3 src/rgbd_to_splat.py --rgb color.png --depth depth.png --output scene.ply

Usage (Multi-View Fixed Camera Rig Stitching):
    python3 src/rgbd_to_splat.py --config camera_rig.json --output stitched_object.ply
"""
import numpy as np
from PIL import Image


def get_quaternions(rot_mats):
    qw = np.sqrt(np.maximum(0.0, 1.0 + rot_mats[:, 0, 0] + rot_mats[:, 1, 1] + rot_mats[:, 2, 2])) * 0.5
    qx = np.sqrt(np.maximum(0.0, 1.0 + rot_mats[:, 0, 0] - rot_mats[:, 1, 1] - rot_mats[:, 2, 2])) * 0.5
    qy = np.sqrt(np.maximum(0.0, 1.0 - rot_mats[:, 0, 0] + rot_mats[:, 1, 1] - rot_mats[:, 2, 2])) * 0.5
    qz = np.sqrt(np.maximum(0.0, 1.0 - rot_mats[:, 0, 0] - rot_mats[:, 1, 1] + rot_mats[:, 2, 2])) * 0.5

    qx = np.copysign(qx, rot_mats[:, 2, 1] - rot_mats[:, 1, 2])
    qy = np.copysign(qy, rot_mats[:, 0, 2] - rot_mats[:, 2, 0])
    qz = np.copysign(qz, rot_mats[:, 1, 0] - rot_mats[:, 0, 1])

    quats = np.stack([qw, qx, qy, qz], axis=-1).astype(np.float32)
    norms = np.linalg.norm(quats, axis=-1, keepdims=True) + 1e-8
    return quats / norms


def get_neighbor_step(points, axis):
    if axis == 1:
        diff = points[:, 1:, :] - points[:, :-1, :]
        left_step = np.pad(diff, ((0, 0), (1, 0), (0, 0)), mode="edge")
        right_step = np.pad(diff, ((0, 0), (0, 1), (0, 0)), mode="edge")
        step1, step2 = left_step, right_step
    else:
        diff = points[1:, :, :] - points[:-1, :, :]
        up_step = np.pad(diff, ((1, 0), (0, 0), (0, 0)), mode="edge")
        down_step = np.pad(diff, ((0, 1), (0, 0), (0, 0)), mode="edge")
        step1, step2 = up_step, down_step

    jump1 = np.abs(step1[..., 2])
    jump2 = np.abs(step2[..., 2])

    pick_first = (jump1 <= jump2)[..., None]
    best_step = np.where(pick_first, step1, step2)
    best_jump = np.minimum(jump1, jump2)
    return best_step, best_jump


def extract_surfels(rgb, depth, intrinsics, pose=None, min_depth=0.25, max_depth=4.5, jump_limit=0.05):
    depth = np.asarray(depth, dtype=np.float32)
    rows, cols = depth.shape
    fx, fy, cx, cy = intrinsics

    if rgb.shape[:2] != (rows, cols):
        img = Image.fromarray(rgb).resize((cols, rows), Image.Resampling.BILINEAR)
        rgb = np.array(img)
    rgb = rgb[..., :3].astype(np.float32)

    if pose is None:
        pose = np.eye(4, dtype=np.float32)
    else:
        pose = np.asarray(pose, dtype=np.float32)

    grid_x, grid_y = np.meshgrid(np.arange(cols, dtype=np.float32), np.arange(rows, dtype=np.float32))
    cam_x = (grid_x - cx) * depth / fx
    cam_y = -(grid_y - cy) * depth / fy
    cam_z = -depth
    cam_points = np.stack([cam_x, cam_y, cam_z], axis=-1)

    step_x, jump_x = get_neighbor_step(cam_points, axis=1)
    step_y, jump_y = get_neighbor_step(cam_points, axis=0)

    valid = (depth > min_depth) & (depth < max_depth) & (jump_x < jump_limit) & (jump_y < jump_limit)
    valid[0, :] = False
    valid[-1, :] = False
    valid[:, 0] = False
    valid[:, -1] = False
    valid = valid.reshape(-1)

    dist_x = np.linalg.norm(step_x, axis=-1).reshape(-1)[valid]
    dist_y = np.linalg.norm(step_y, axis=-1).reshape(-1)[valid]
    width = np.clip(dist_x, 0.001, 0.05)
    height = np.clip(dist_y, 0.001, 0.05)

    rot = pose[:3, :3]
    shift = pose[:3, 3]

    world_points = cam_points.reshape(-1, 3)[valid] @ rot.T + shift
    vec_x = step_x.reshape(-1, 3)[valid] @ rot.T
    vec_y = step_y.reshape(-1, 3)[valid] @ rot.T

    dir_x = vec_x / (np.linalg.norm(vec_x, axis=1, keepdims=True) + 1e-8)
    normals = np.cross(dir_x, vec_y)
    normals = normals / (np.linalg.norm(normals, axis=1, keepdims=True) + 1e-8)
    dir_y = np.cross(normals, dir_x)

    frames = np.stack([dir_x, dir_y, normals], axis=-1)
    quaternions = get_quaternions(frames)
    colors = rgb.reshape(-1, 3)[valid]

    return {
        "points": world_points.astype(np.float32),
        "normals": normals.astype(np.float32),
        "quaternions": quaternions,
        "scale_u": width.astype(np.float32),
        "scale_v": height.astype(np.float32),
        "colors": colors,
    }


def build_map(frames, intrinsics, voxel_size=0.0, min_depth=0.25, max_depth=4.5, jump_limit=0.05):
    surfel_list = []
    for f in frames:
        s = extract_surfels(
            f["rgb"],
            f["depth"],
            intrinsics,
            f.get("pose"),
            min_depth,
            max_depth,
            jump_limit
        )
        surfel_list.append(s)

    merged = {}
    for key in surfel_list[0]:
        parts = [s[key] for s in surfel_list]
        merged[key] = np.concatenate(parts, axis=0)

    if voxel_size > 0:
        voxels = np.floor(merged["points"] / voxel_size).astype(np.int32)
        _, keep_idx = np.unique(voxels, axis=0, return_index=True)
        keep_idx.sort()

        filtered = {}
        for key, val in merged.items():
            filtered[key] = val[keep_idx]
        merged = filtered

    return merged
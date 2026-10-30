import numpy as np
from plyfile import PlyData, PlyElement

# 0th-order Spherical Harmonics constant: 1 / (2 * sqrt(pi))
SH_C0 = 0.28209479


def save_splat_ply(filename, surfel_data, thickness=0.0005, overlap=0.95, opacity_logit=10.0):
    """
    Takes the dictionary output from engine.py and writes a 62-property
    binary 3D Gaussian Splatting .ply file compatible with SuperSplat.
    """
    points = surfel_data["points"]
    normals = surfel_data["normals"]
    quats = surfel_data["quaternions"]
    width = surfel_data["scale_u"]
    height = surfel_data["scale_v"]
    colors = surfel_data["colors"]

    num_pts = len(points)

    # 1. Define the strict 62-property binary header expected by 3DGS viewers
    dtype = [
        ("x", "f4"), ("y", "f4"), ("z", "f4"),
        ("nx", "f4"), ("ny", "f4"), ("nz", "f4"),
        ("f_dc_0", "f4"), ("f_dc_1", "f4"), ("f_dc_2", "f4"),
    ]
    for i in range(45):
        dtype.append((f"f_rest_{i}", "f4"))

    dtype += [
        ("opacity", "f4"),
        ("scale_0", "f4"), ("scale_1", "f4"), ("scale_2", "f4"),
        ("rot_0", "f4"), ("rot_1", "f4"), ("rot_2", "f4"), ("rot_3", "f4"),
    ]

    # Initializing with zeros automatically sets f_rest_0..44 to 0.0
    splat = np.zeros(num_pts, dtype=dtype)

    # 2. Fill 3D Positions and Surface Normals
    splat["x"] = points[:, 0]
    splat["y"] = points[:, 1]
    splat["z"] = points[:, 2]

    splat["nx"] = normals[:, 0]
    splat["ny"] = normals[:, 1]
    splat["nz"] = normals[:, 2]

    # 3. Convert RGB [0, 255] to 0th-Order Spherical Harmonics (f_dc)
    splat["f_dc_0"] = (colors[:, 0] / 255.0 - 0.5) / SH_C0
    splat["f_dc_1"] = (colors[:, 1] / 255.0 - 0.5) / SH_C0
    splat["f_dc_2"] = (colors[:, 2] / 255.0 - 0.5) / SH_C0

    # 4. Fill Opacity (in logit space: sigmoid(10.0) ~ 1.0 = solid opaque)
    splat["opacity"] = opacity_logit

    # 5. Fill Scales (3DGS stores scales in log-space: real_size = exp(scale))
    splat["scale_0"] = np.log(np.maximum(width * overlap, 1e-5)).astype(np.float32)
    splat["scale_1"] = np.log(np.maximum(height * overlap, 1e-5)).astype(np.float32)
    splat["scale_2"] = np.float32(np.log(thickness))

    # 6. Fill Quaternion Rotations [qw, qx, qy, qz]
    splat["rot_0"] = quats[:, 0]
    splat["rot_1"] = quats[:, 1]
    splat["rot_2"] = quats[:, 2]
    splat["rot_3"] = quats[:, 3]

    # 7. Write binary PLY file to disk
    element = PlyElement.describe(splat, "vertex")
    PlyData([element], text=False).write(filename)
    print(f"Saved {num_pts} Gaussian splats to {filename}")


def save_pointcloud_ply(filename, surfel_data):
    """
    Optional helper: Saves basic (x, y, z, r, g, b) point cloud for MeshLab.
    """
    points = surfel_data["points"]
    colors = np.clip(surfel_data["colors"], 0, 255).astype(np.uint8)
    num_pts = len(points)

    dtype = [
        ("x", "f4"), ("y", "f4"), ("z", "f4"),
        ("red", "u1"), ("green", "u1"), ("blue", "u1"),
    ]

    cloud = np.zeros(num_pts, dtype=dtype)
    cloud["x"] = points[:, 0]
    cloud["y"] = points[:, 1]
    cloud["z"] = points[:, 2]
    cloud["red"] = colors[:, 0]
    cloud["green"] = colors[:, 1]
    cloud["blue"] = colors[:, 2]

    element = PlyElement.describe(cloud, "vertex")
    PlyData([element], text=False).write(filename)
    print(f"Saved {num_pts} raw points to {filename}")
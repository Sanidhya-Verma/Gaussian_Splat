"""
Stage 02: True Neighbor Gradient Scaling, The 'Veil Bug' at Edges, and Upwind Anisotropic Fix.

Like in the prev 01 folder we were able to get the adaptive scaling however now we encounter 2 new problem:
Veil bug : in case of edges of 2 objects say a rock or obstacle place in  front of wall what happens is the the code computes the gradient at the edge
and finds it to be very large (like the edge of roack and nearby wall as neighbour) so it thinks that part is very importtant even though there is just 
empty air so it cause a veil bug 

Second issue is that how will we take care of the pixels at edges( thier gradient is very high) the simplest sol would be to simply delete them but they
represent the actula edge and information and doing so we will lose the accuracy we want so better and smarter fix would be to calculate the gradient
along the smoother direction (taking the minimum of the two one-sided gradients on each axis) so we follow the continuous surface of the rock/obstacle
itself rather than jumping across the empty cliff to the wall. A steep slope is still continuous and has intermediate values on both sides, whereas 
a true cliff edge has a smooth surface on one side and a massive jump on the other. Finally, for isolated 'flyover' pixels, there is no continuous
neighbor in any surrounding direction (even their minimum gradient is huge), so we can safely delete just those outliers without losing real edges. 

"""

import numpy as np
from plyfile import PlyData, PlyElement


def rotation_matrices_to_quaternions(R):
    """Converts (N, 3, 3) rotation matrices into (N, 4) unit quaternions [w, x, y, z]."""
    N = R.shape[0]
    q = np.zeros((N, 4), dtype=np.float32)
    tr = R[:, 0, 0] + R[:, 1, 1] + R[:, 2, 2]

    m = tr > 0
    s = np.sqrt(np.maximum(tr[m] + 1.0, 1e-8)) * 2.0
    q[m, 0] = 0.25 * s
    q[m, 1] = (R[m, 2, 1] - R[m, 1, 2]) / s
    q[m, 2] = (R[m, 0, 2] - R[m, 2, 0]) / s
    q[m, 3] = (R[m, 1, 0] - R[m, 0, 1]) / s

    m2 = ~m
    if np.any(m2):
        R_m = R[m2]
        idx = np.argmax(np.stack([R_m[:, 0, 0], R_m[:, 1, 1], R_m[:, 2, 2]], axis=-1), axis=-1)
        for i in range(3):
            sub = (idx == i)
            if not np.any(sub):
                continue
            j, k = (i + 1) % 3, (i + 2) % 3
            s_sub = np.sqrt(np.maximum(1.0 + R_m[sub, i, i] - R_m[sub, j, j] - R_m[sub, k, k], 1e-8)) * 2.0
            q_sub = np.zeros((np.sum(sub), 4), dtype=np.float32)
            q_sub[:, 0] = (R_m[sub, k, j] - R_m[sub, j, k]) / s_sub
            q_sub[:, i + 1] = 0.25 * s_sub
            q_sub[:, j + 1] = (R_m[sub, j, i] + R_m[sub, i, j]) / s_sub
            q_sub[:, k + 1] = (R_m[sub, k, i] + R_m[sub, i, k]) / s_sub
            q[np.where(m2)[0][sub]] = q_sub

    return q / (np.linalg.norm(q, axis=-1, keepdims=True) + 1e-8)


def save_3dgs_ply(filename, pts, normals, quats, scale_u, scale_v, rgb):
    n = len(pts)
    dtype = [("x", "f4"), ("y", "f4"), ("z", "f4"),
             ("nx", "f4"), ("ny", "f4"), ("nz", "f4"),
             ("f_dc_0", "f4"), ("f_dc_1", "f4"), ("f_dc_2", "f4")]
    dtype += [(f"f_rest_{i}", "f4") for i in range(45)]
    dtype += [("opacity", "f4"),
              ("scale_0", "f4"), ("scale_1", "f4"), ("scale_2", "f4"),
              ("rot_0", "f4"), ("rot_1", "f4"), ("rot_2", "f4"), ("rot_3", "f4")]

    splat = np.zeros(n, dtype=dtype)
    splat["x"], splat["y"], splat["z"] = pts[:, 0], pts[:, 1], pts[:, 2]
    splat["nx"], splat["ny"], splat["nz"] = normals[:, 0], normals[:, 1], normals[:, 2]
    splat["f_dc_0"] = (rgb[:, 0] / 255.0 - 0.5) / 0.28209479
    splat["f_dc_1"] = (rgb[:, 1] / 255.0 - 0.5) / 0.28209479
    splat["f_dc_2"] = (rgb[:, 2] / 255.0 - 0.5) / 0.28209479
    splat["opacity"] = 10.0
    splat["scale_0"] = np.log(np.maximum(scale_u, 1e-4)).astype(np.float32)
    splat["scale_1"] = np.log(np.maximum(scale_v, 1e-4)).astype(np.float32)
    splat["scale_2"] = np.float32(np.log(0.0005))  # 0.5mm thin surfel disc
    splat["rot_0"], splat["rot_1"], splat["rot_2"], splat["rot_3"] = (
        quats[:, 0], quats[:, 1], quats[:, 2], quats[:, 3]
    )

    PlyData([PlyElement.describe(splat, "vertex")], text=False).write(filename)
    print(f"Exported {filename} ({n} splats)")


def run_experiment():
    # Simulate a 120x120 pinhole camera looking at a curved surface with a sharp depth cliff
    H, W = 120, 120
    fx, fy, cx, cy = 120.0, 120.0, 60.0, 60.0
    u, v = np.meshgrid(np.arange(W, dtype=np.float32), np.arange(H, dtype=np.float32))

    # Foreground wavy surface at Z ~ 1.5m, dropping sharply to a background wall at Z = 3.2m
    Z = 1.5 + 0.25 * np.sin((u - cx) / 15.0) * np.cos((v - cy) / 15.0)
    cliff_mask = ((u - cx)**2 + (v - cy)**2) > (42.0**2)  # Circular object boundary
    Z[cliff_mask] = 3.2  # Sudden 1.7m depth jump (empty air between foreground & background)

    X = (u - cx) * Z / fx
    Y = -(v - cy) * Z / fy
    P = np.stack([X, Y, -Z], axis=-1)

    # Color foreground warm orange/pink and background cool blue so the veil is obvious
    rgb = np.zeros((H, W, 3), dtype=np.float32)
    rgb[~cliff_mask] = [240, 110, 90]
    rgb[cliff_mask] = [60, 130, 220]


    # CASE 1: THE VEIL BUG 
    dP_dv_naive, dP_du_naive = np.gradient(P, axis=(0, 1))
    dist_u_naive = np.linalg.norm(dP_du_naive, axis=-1).reshape(-1)
    dist_v_naive = np.linalg.norm(dP_dv_naive, axis=-1).reshape(-1)

    tu = dP_du_naive.reshape(-1, 3)
    tv = dP_dv_naive.reshape(-1, 3)
    e_u = tu / (np.linalg.norm(tu, axis=-1, keepdims=True) + 1e-8)
    norms = np.cross(e_u, tv)
    norms /= (np.linalg.norm(norms, axis=-1, keepdims=True) + 1e-8)
    e_v = np.cross(norms, e_u)
    quats_naive = rotation_matrices_to_quaternions(np.stack([e_u, e_v, norms], axis=-1))

    save_3dgs_ply(
        "01_veil_bug_central_gradient.ply",
        P.reshape(-1, 3), norms, quats_naive, dist_u_naive * 0.95, dist_v_naive * 0.95, rgb.reshape(-1, 3)
    )
  
    # CASE 2: THE FIX (One-Sided Upwind Gradient + Edge Continuity Pruning)
  
    diff_u = P[:, 1:, :] - P[:, :-1, :]
    step_u_left  = np.pad(diff_u, ((0, 0), (1, 0), (0, 0)), mode="edge")
    step_u_right = np.pad(diff_u, ((0, 0), (0, 1), (0, 0)), mode="edge")
    use_left_u = (np.abs(step_u_left[:, :, 2]) <= np.abs(step_u_right[:, :, 2]))[..., None]
    dP_du_upwind = np.where(use_left_u, step_u_left, step_u_right)

    diff_v = P[1:, :, :] - P[:-1, :, :]
    step_v_up   = np.pad(diff_v, ((1, 0), (0, 0), (0, 0)), mode="edge")
    step_v_down = np.pad(diff_v, ((0, 1), (0, 0), (0, 0)), mode="edge")
    use_up_v = (np.abs(step_v_up[:, :, 2]) <= np.abs(step_v_down[:, :, 2]))[..., None]
    dP_dv_upwind = np.where(use_up_v, step_v_up, step_v_down)

    # Prune isolated flying pixels where BOTH sides have a depth jump > 8cm
    best_jump_u = np.minimum(np.abs(step_u_left[:, :, 2]), np.abs(step_u_right[:, :, 2]))
    best_jump_v = np.minimum(np.abs(step_v_up[:, :, 2]), np.abs(step_v_down[:, :, 2]))
    valid = ((best_jump_u < 0.08) & (best_jump_v < 0.08)).reshape(-1)

    tu_fix = dP_du_upwind.reshape(-1, 3)[valid]
    tv_fix = dP_dv_upwind.reshape(-1, 3)[valid]
    dist_u_fix = np.linalg.norm(tu_fix, axis=-1)
    dist_v_fix = np.linalg.norm(tv_fix, axis=-1)

    e_u_fix = tu_fix / (np.linalg.norm(tu_fix, axis=-1, keepdims=True) + 1e-8)
    norms_fix = np.cross(e_u_fix, tv_fix)
    norms_fix /= (np.linalg.norm(norms_fix, axis=-1, keepdims=True) + 1e-8)
    e_v_fix = np.cross(norms_fix, e_u_fix)
    quats_fix = rotation_matrices_to_quaternions(np.stack([e_u_fix, e_v_fix, norms_fix], axis=-1))

    save_3dgs_ply(
        "02_fixed_upwind_anisotropic.ply",
        P.reshape(-1, 3)[valid], norms_fix, quats_fix, dist_u_fix * 0.95, dist_v_fix * 0.95, rgb.reshape(-1, 3)[valid]
    )


if __name__ == "__main__":
    run_experiment()

"""Per-vertex 6D rotation GT for BEDLAM2 + per-pixel R reconstruction.

vert_rot6d[v] = first two columns of R_v, where R_v maps the rest-pose (template) surface frame at
SMPL-X vertex v to its CAMERA-space frame on the posed, viewed body (PyTorch3D view space: +X left,
+Y up, +Z into scene). R is the polar rotation of the rest->camera deformation (normal-pinned, via a
Kabsch/SVD over the one-ring edges + a normal-pin term). Verified bit-exact to the reference generator.

Pose-matched to our correspondence: we pose with render_correspondence.pose_instance (world params from
posed3d/<job>.npz) and map to camera with world_to_cam(ext) — the EXACT functions that produced the
face_index+bary, so per-pixel R lines up with the stored correspondence.
"""
import os
import sys
import numpy as np
import torch

_BEDLAM = os.path.expanduser("~/codes/wan/bedlam")
if _BEDLAM not in sys.path:
    sys.path.insert(0, _BEDLAM)
from render_correspondence import pose_instance, world_to_cam, get_smplx  # noqa: E402

# OpenCV cam (+X right,+Y down,+Z fwd) -> PyTorch3D view (+X left,+Y up,+Z fwd).
_FLIP = np.array([-1.0, -1.0, 1.0])


def vertex_normals(V, F):
    """Area-weighted, outward, unit per-vertex normals."""
    n = np.zeros_like(V, dtype=np.float64)
    tri = V[F]
    fn = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
    for k in range(3):
        np.add.at(n, F[:, k], fn)
    return n / np.clip(np.linalg.norm(n, axis=1, keepdims=True), 1e-9, None)


def rotation_from_deformation(v_rest, v_cam, faces, n_rest, n_cam, normal_w=25.0):
    """Per-vertex polar rotation rest->camera (normal pinned). Returns (Nv,3,3) maps rest->camera."""
    Nv = v_rest.shape[0]
    e = np.concatenate([faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]],
                        faces[:, [1, 0]], faces[:, [2, 1]], faces[:, [0, 2]]], 0)
    i, j = e[:, 0], e[:, 1]
    er, ec = v_rest[j] - v_rest[i], v_cam[j] - v_cam[i]
    H = np.zeros((Nv, 3, 3))
    np.add.at(H, i, er[:, :, None] * ec[:, None, :])
    energy = np.zeros(Nv)
    np.add.at(energy, i, np.sum(er * er, axis=1))
    H += (normal_w * energy)[:, None, None] * (n_rest[:, :, None] * n_cam[:, None, :])
    U, S, Vt = np.linalg.svd(H)
    Vm, Ut = np.transpose(Vt, (0, 2, 1)), np.transpose(U, (0, 2, 1))
    D = np.broadcast_to(np.eye(3), (Nv, 3, 3)).copy()
    D[:, 2, 2] = np.sign(np.linalg.det(Vm @ Ut))      # proper rotation (det=+1)
    return Vm @ D @ Ut


def sixd_to_R_np(six):
    """(...,6) -> (...,3,3) via Gram-Schmidt. Use on GT and (numpy) net output."""
    a0, a1 = six[..., :3], six[..., 3:]
    e0 = a0 / np.clip(np.linalg.norm(a0, axis=-1, keepdims=True), 1e-9, None)
    a1 = a1 - (a1 * e0).sum(-1, keepdims=True) * e0
    e1 = a1 / np.clip(np.linalg.norm(a1, axis=-1, keepdims=True), 1e-9, None)
    return np.stack([e0, e1, np.cross(e0, e1)], axis=-1)


_TEMPLATE = None


def get_template(device="cuda"):
    """(v_template (10475,3) f64, faces (20908,3) i64, n_rest (10475,3) f64) for neutral SMPL-X."""
    global _TEMPLATE
    if _TEMPLATE is None:
        m = get_smplx("neutral", 16, device)
        v_template = m.v_template.detach().cpu().numpy().astype(np.float64)
        faces = m.faces.astype(np.int64)
        _TEMPLATE = (v_template, faces, vertex_normals(v_template, faces))
    return _TEMPLATE


def vert_rot6d_for_body(pose_world, betas, trans_world, ext, gender="neutral", device="cuda"):
    """Per-vertex 6D rotation table (10475,6) f16 for one posed body, pose-matched to rasterization."""
    v_template, faces, n_rest = get_template(device)
    vw, _ = pose_instance(pose_world, betas, trans_world, gender, device)   # world verts (V,3)
    vc = world_to_cam(vw, ext, device).detach().cpu().numpy().astype(np.float64)  # OpenCV camera
    vp = vc * _FLIP                                                          # -> PyTorch3D view
    n_cam = vertex_normals(vp, faces)
    R = rotation_from_deformation(v_template, vp, faces, n_rest, n_cam)      # (Nv,3,3)
    return np.concatenate([R[:, :, 0], R[:, :, 1]], axis=1).astype(np.float16)  # (10475,6)


# ---- GPU fast path (everything in torch on device); matches numpy core to ~1e-4 ----
_GPU = {}


def _gpu_consts(device="cuda"):
    if device not in _GPU:
        import torch
        v_template, faces, n_rest = get_template(device)
        F = torch.as_tensor(faces, device=device)
        e = torch.cat([F[:, [0, 1]], F[:, [1, 2]], F[:, [2, 0]],
                       F[:, [1, 0]], F[:, [2, 1]], F[:, [0, 2]]], 0).long()
        _GPU[device] = dict(
            vt=torch.as_tensor(v_template, device=device, dtype=torch.float32),
            F=F.long(), nrest=torch.as_tensor(n_rest, device=device, dtype=torch.float32),
            ei=e[:, 0], ej=e[:, 1], flip=torch.tensor(_FLIP, device=device, dtype=torch.float32))
    return _GPU[device]


def _vnorm_t(V, F):
    import torch
    tri = V[F]
    fn = torch.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0], dim=1)
    n = torch.zeros_like(V)
    for k in range(3):
        n.index_add_(0, F[:, k], fn)
    return torch.nn.functional.normalize(n, dim=1)


def vert_rot6d_for_body_gpu(pose_world, betas, trans_world, ext, gender="neutral",
                            device="cuda", normal_w=25.0):
    """GPU torch version (~10x faster); returns (10475,6) f16. Numerically matches the numpy core."""
    import torch
    g = _gpu_consts(device)
    vw, _ = pose_instance(pose_world, betas, trans_world, gender, device)
    vc = world_to_cam(vw, ext, device).float()
    vp = vc * g["flip"]                                                      # (Nv,3) PyTorch3D view
    n_cam = _vnorm_t(vp, g["F"])
    i, j = g["ei"], g["ej"]
    er, ec = g["vt"][j] - g["vt"][i], vp[j] - vp[i]
    Nv = g["vt"].shape[0]
    H = torch.zeros(Nv, 3, 3, device=device)
    H.index_add_(0, i, er[:, :, None] * ec[:, None, :])
    energy = torch.zeros(Nv, device=device)
    energy.index_add_(0, i, (er * er).sum(1))
    H = H + (normal_w * energy)[:, None, None] * (g["nrest"][:, :, None] * n_cam[:, None, :])
    U, S, Vt = torch.linalg.svd(H)
    Vm, Ut = Vt.transpose(1, 2), U.transpose(1, 2)
    d = torch.sign(torch.linalg.det(Vm @ Ut))
    Vm = Vm.clone(); Vm[:, :, 2] *= d[:, None]                               # fold det sign into last col
    R = Vm @ Ut                                                              # (Nv,3,3) rest->camera
    return torch.cat([R[:, :, 0], R[:, :, 1]], 1).detach().cpu().numpy().astype(np.float16)


def per_pixel_R(face_zero_based, bary3, vert_rot6d, faces, mask):
    """Reconstruct per-pixel R (H,W,3,3) from a SINGLE body's vert_rot6d table.
    face_zero_based: (H,W) int, 0-based face id (any value where ~mask, ignored).
    bary3: (H,W,3) barycentric. vert_rot6d: (10475,6). faces: (20908,3). mask: (H,W) bool."""
    tv = faces[np.where(mask, face_zero_based, 0)]                # (H,W,3) vertex ids
    w = bary3[..., None]                                          # (H,W,3,1)
    six = (w * np.stack([vert_rot6d[tv[..., k]] for k in range(3)], 2)).sum(2)  # (H,W,6)
    R = sixd_to_R_np(six.astype(np.float32))
    R[~mask] = 0
    return R

"""The body template behind the predictions: per-pixel template normal / tangent from a predicted (part, U, V),
and the colourings used by the demo.

The template is the neutral SMPL-X mesh (assets/smplx_v_template.npy, smplx_faces.npy). A (part, U, V) is located
in its part's chart with the PER-FACE-CORNER UV (assets/smplx_face_uv_v2.npy, the charts the released model
predicts) and the face's three vertex rest normals are interpolated with the barycentrics, which reproduces the
construction of the training targets and is seam-free across part boundaries.

Rotation convention (as the training targets, research/densepose/orient/rot6d.py): R maps a template direction
(rest pose, template axes) into the camera in PyTorch3D view space, +X left, +Y up, +Z into the scene. A
camera-facing normal R n has a negative z.
"""
import numpy as np
from scipy.spatial import cKDTree

from .assets import asset

UV_ASSET = "smplx_face_uv_v2.npy"          # the charts the released model predicts


def vertex_normals(V, F):
    """Area-weighted, outward, unit per-vertex normals."""
    n = np.zeros_like(V, dtype=np.float64)
    tri = V[F]
    fn = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
    for k in range(3):
        np.add.at(n, F[:, k], fn)
    return n / np.clip(np.linalg.norm(n, axis=1, keepdims=True), 1e-9, None)


def sixd_to_R_np(six):
    """(...,6) -> (...,3,3) by Gram-Schmidt (NumPy twin of pff.geometry.sixd_to_R)."""
    a0, a1 = six[..., :3], six[..., 3:]
    e0 = a0 / np.clip(np.linalg.norm(a0, axis=-1, keepdims=True), 1e-9, None)
    a1 = a1 - (a1 * e0).sum(-1, keepdims=True) * e0
    e1 = a1 / np.clip(np.linalg.norm(a1, axis=-1, keepdims=True), 1e-9, None)
    return np.stack([e0, e1, np.cross(e0, e1)], axis=-1)


def _bary(p, tri):
    a, b, c = tri[..., 0, :], tri[..., 1, :], tri[..., 2, :]
    v0, v1, v2 = b - a, c - a, p - a
    d00 = (v0 * v0).sum(-1); d01 = (v0 * v1).sum(-1); d11 = (v1 * v1).sum(-1)
    d20 = (v2 * v0).sum(-1); d21 = (v2 * v1).sum(-1)
    den = d00 * d11 - d01 * d01
    den = np.where(np.abs(den) < 1e-12, 1e-12, den)
    v = (d11 * d20 - d01 * d21) / den
    w = (d00 * d21 - d01 * d20) / den
    return np.stack([1 - v - w, v, w], -1)


class TemplateField:
    """Interpolate a per-vertex template field (normals, tangents, positions) at predicted (part, U, V)."""

    def __init__(self, per_vertex, face_uv, face_part, faces, k=12):
        self.k = k
        self.parts = {}
        for p in range(1, 25):
            fm = face_part == p
            if fm.any():
                uv_tri = face_uv[fm].astype(np.float64)                  # (T,3,2) chart triangle per face
                self.parts[p] = (uv_tri, per_vertex[faces[fm]], cKDTree(uv_tri.mean(1)))

    def __call__(self, part, U, V, normalise=True):
        """part (H,W) 0..24, U, V (H,W) -> field (H,W,C) and fg (H,W)."""
        C = next(iter(self.parts.values()))[1].shape[-1]
        out = np.zeros((*part.shape, C), np.float32)
        fg = part > 0
        for p, (uv_tri, f_tri, tree) in self.parts.items():
            s = part == p
            if not s.any():
                continue
            q = np.stack([U[s], V[s]], 1).astype(np.float64)
            kk = min(self.k, len(uv_tri))
            _, cand = tree.query(q, k=kk); cand = cand.reshape(len(q), kk)
            bc = _bary(q[:, None, :], uv_tri[cand])
            best = bc.min(-1).argmax(1); ar = np.arange(len(q))
            tsel, bsel = cand[ar, best], bc[ar, best]
            bcl = np.clip(bsel, 0, None); bcl /= np.clip(bcl.sum(1, keepdims=True), 1e-9, None)   # onto the nearest edge
            out[s] = (bcl[:, :, None] * f_tri[tsel]).sum(1)
        if normalise and fg.any():
            out[fg] /= np.clip(np.linalg.norm(out[fg], axis=-1, keepdims=True), 1e-9, None)
        return out, fg


class Template:
    """Neutral SMPL-X template: rest normals, combing tangents and positions at predicted (part, U, V)."""

    def __init__(self, uv_asset=UV_ASSET):
        self.v = np.load(asset("smplx_v_template.npy")).astype(np.float64)
        self.faces = np.load(asset("smplx_faces.npy")).astype(np.int64)
        self.n_rest = vertex_normals(self.v, self.faces)
        face_uv = np.load(asset(uv_asset))
        face_part = np.load(asset("smplx_face_to_dp24.npy")).astype(int)
        self.normals = TemplateField(self.n_rest, face_uv, face_part, self.faces)
        tang = np.load(asset("smplx_tangent_smooth.npy")).astype(np.float64)   # smooth combing field
        self.tangents = TemplateField(tang, face_uv, face_part, self.faces)
        self.positions = TemplateField(self.v, face_uv, face_part, self.faces)

    def frames(self, part, U, V, R):
        """Camera-space normal and tangent (H,W,3) = R n_template, R t_template, and fg (H,W)."""
        nt, fg = self.normals(part, U, V)
        tt, _ = self.tangents(part, U, V)
        tt -= (tt * nt).sum(-1, keepdims=True) * nt
        tt /= np.clip(np.linalg.norm(tt, axis=-1, keepdims=True), 1e-9, None)
        return np.einsum("hwij,hwj->hwi", R, nt), np.einsum("hwij,hwj->hwi", R, tt), fg


def denoise_part(part, ksize=5):
    """Local-majority filter on a part map (0 = background stays background): removes isolated part flips."""
    import cv2
    fg = part > 0
    votes = np.stack([cv2.blur((part == c).astype(np.float32), (ksize, ksize)) for c in range(1, 25)])
    out = votes.argmax(0) + 1
    out[~fg] = 0
    return out.astype(part.dtype)


# ------------------------------------------------------------------ colourings
def normal_to_rgb(n, mask, bg=1.0):
    """Camera normal -> normal-map colours, (x,y,z) -> (R,G,B) * 0.5 + 0.5 after negating y and z (as in the
    paper figures)."""
    out = np.clip(n * np.array([1.0, -1.0, -1.0], np.float32) * 0.5 + 0.5, 0, 1)
    out[~mask] = bg
    return (out * 255).astype(np.uint8)


def _part_palette():
    import matplotlib.cm as cm
    return (cm.hsv(np.linspace(0, 1, 25))[:, :3] * 255).astype(np.uint8)


PALETTE = _part_palette()            # 24 part colours, index 0 = background


def part_rgb(part, bg=255):
    out = PALETTE[np.clip(part, 0, 24)].copy()
    out[part == 0] = bg
    return out


def iuv_rgb(part, U, V, bg=255):
    """Part colour with brightness modulated by U and V, so the coordinates inside each chart show."""
    img = PALETTE[np.clip(part, 0, 24)].astype(np.float32) * (0.45 + 0.4 * U + 0.15 * V)[..., None]
    out = np.clip(img, 0, 255).astype(np.uint8)
    out[part == 0] = bg
    return out


def uv_rgb(part, U, V, bg=255):
    """U in red, V in green: the chart coordinates alone."""
    out = np.stack([U, V, np.zeros_like(U)], -1)
    out = (np.clip(out, 0, 1) * 255).astype(np.uint8)
    out[part == 0] = bg
    return out

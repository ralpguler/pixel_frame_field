"""Per-pixel template surface normal from predicted DensePose (part, U, V), matching the GT
face-index+barycentric construction and seam-free across part boundaries.

Key: use PER-FACE-CORNER UV (smplx_face_uv, (F,3,2)) + per-face part (smplx_face_to_dp24), NOT per-vertex
UV. Each face — including the ones whose vertices straddle two parts (e.g. the 221 faces across the DP
head midline 23/24) — is assigned to one part and has a valid 3-corner UV triangle in that part's chart.
Per-vertex UV (one UV per vertex) cannot represent those straddling faces, which left an uncovered strip
at part boundaries -> the visible midface seam. With per-corner UV the charts tile completely, so locating
predicted (U,V) -> face -> barycentric -> interpolate that face's 3 vertex rest-normals reproduces GT.

Point-location is hand-rolled (KDTree over UV-triangle centroids -> candidate faces -> in-triangle
barycentric test); matplotlib's TrapezoidMapTriFinder rejects the atlas (duplicate/degenerate seam UVs)."""
import numpy as np
from scipy.spatial import cKDTree


def _bary(p, tri):
    """Barycentric of p (...,2) wrt UV triangle tri (...,3,2)."""
    a, b, c = tri[..., 0, :], tri[..., 1, :], tri[..., 2, :]
    v0, v1, v2 = b - a, c - a, p - a
    d00 = (v0 * v0).sum(-1); d01 = (v0 * v1).sum(-1); d11 = (v1 * v1).sum(-1)
    d20 = (v2 * v0).sum(-1); d21 = (v2 * v1).sum(-1)
    den = d00 * d11 - d01 * d01
    den = np.where(np.abs(den) < 1e-12, 1e-12, den)
    v = (d11 * d20 - d01 * d21) / den
    w = (d00 * d21 - d01 * d20) / den
    return np.stack([1 - v - w, v, w], -1)


class TemplateNormals:
    def __init__(self, face_uv, face_part, faces, n_rest, v_template=None, k=12):
        """face_uv (F,3,2) per-face-corner UV; face_part (F,) DP part 1..24; faces (F,3); n_rest (V,3);
        v_template (V,3) optional rest-pose vertex positions (enables template-XYZ output)."""
        self.k = k
        self.parts = {}
        for p in range(1, 25):
            fm = face_part == p
            if fm.sum() == 0:
                continue
            uv_tri = face_uv[fm].astype(np.float64)               # (T,3,2) chart-UV triangle per face
            n_tri = n_rest[faces[fm]]                             # (T,3,3) the face's 3 vertex normals
            p_tri = v_template[faces[fm]] if v_template is not None else None  # (T,3,3) vtx positions
            cn = n_tri.reshape(-1, 3)
            cp = p_tri.reshape(-1, 3) if p_tri is not None else None
            self.parts[p] = (uv_tri, n_tri, p_tri, cKDTree(uv_tri.mean(1)), cKDTree(uv_tri.reshape(-1, 2)), cn, cp)

    def __call__(self, part, U, V, return_pos=False):
        nt = np.zeros((*part.shape, 3), np.float32)
        P = np.zeros((*part.shape, 3), np.float32)
        fg = part > 0
        self._n_fg = 0; self._n_fallback = 0          # instrumentation: how many pixels missed all triangles
        self._fb_mask = np.zeros(part.shape, bool)    # where fallback fired (spatial)
        for p, (uv_tri, n_tri, p_tri, cen_tree, cor_tree, cn, cp) in self.parts.items():
            s = part == p
            if not s.any():
                continue
            q = np.stack([U[s], V[s]], 1).astype(np.float64)
            kk = min(self.k, len(uv_tri))
            _, cand = cen_tree.query(q, k=kk); cand = cand.reshape(len(q), kk)
            bc = _bary(q[:, None, :], uv_tri[cand])               # (M,kk,3)
            mn = bc.min(-1); best = mn.argmax(1); ar = np.arange(len(q))
            tsel, bsel, ok = cand[ar, best], bc[ar, best], mn[ar, best] >= -1e-4
            self._n_fg += len(q); self._n_fallback += int((~ok).sum())   # missed-all-triangles count
            fbm = np.zeros(s.shape, bool); fbm[s] = ~ok; self._fb_mask |= fbm
            # clip+renormalize bary -> projects out-of-chart points onto the nearest triangle EDGE
            # (continuous), instead of snapping to a discrete nearest corner (the 1px seam artifact)
            bcl = np.clip(bsel, 0, None); bcl /= np.clip(bcl.sum(1, keepdims=True), 1e-9, None)
            nt[s] = (bcl[:, :, None] * n_tri[tsel]).sum(1)        # face-bary of vtx normals
            if return_pos and p_tri is not None:
                P[s] = (bcl[:, :, None] * p_tri[tsel]).sum(1)     # face-bary of vtx positions (template XYZ)
        if fg.any():
            nt[fg] /= np.clip(np.linalg.norm(nt[fg], axis=-1, keepdims=True), 1e-9, None)
        return (nt, P, fg) if return_pos else (nt, fg)


def denoise_part(part, ksize=5):
    """Mask-aware local-majority filter on a DP part map (H,W ints, 0=bg, 1..24). Each fg pixel is
    replaced by the most common part in a ksize window (one-hot box-filter vote). Removes the isolated
    part-misclassification speckle that, because parts are disjoint UV charts, otherwise causes
    catastrophic chart-jumps in the reconstructed XYZ/normal. Background stays background."""
    import cv2
    fg = part > 0
    votes = np.stack([cv2.blur((part == c).astype(np.float32), (ksize, ksize)) for c in range(1, 25)])
    out = votes.argmax(0) + 1
    out[~fg] = 0
    return out.astype(part.dtype)


def normal_to_rgb(nc, mask, bg=1.0):
    """Camera-space normal (H,W,3) -> conventional view-space normal-map RGB.
    Camera is OpenCV-style (+Z into scene, +Y down), so camera-FACING normals are -Z and 'up' is -Y;
    negate Y,Z so facing=blue (B high) and up=green, the standard look. (x,y,z)->(R,G,B) with *.5+.5."""
    disp = nc * np.array([1.0, -1.0, -1.0], np.float32)
    out = np.clip(disp * 0.5 + 0.5, 0, 1)
    out[~mask] = bg
    return out

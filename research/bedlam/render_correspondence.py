"""BEDLAM2 dense surface correspondence renderer.

For each label instance: pose the SMPL-X mesh in camera space, rasterize all
people in a frame together (z-buffered, so occlusion between people is correct),
and read per-pixel:
  - face index   (which SMPL-X triangle, uint16 — SMPL-X has ~20908 faces)
  - barycentric  (3 coords; we store 2x float16, third = 1 - a - b)
  - instance id  (which person)
Only visible pixels (nearest face per pixel via z-buffer).

Per visible instance we emit a tight bbox (from visible pixels only, COCO style)
and crop the face-index / barycentric maps to that box (DensePose-style compact
storage — never the full empty image).

SMPL-X 165-d pose layout: global_orient(3) body(63) jaw(3) leye(3) reye(3)
left_hand(45) right_hand(45).
"""
import os
import numpy as np
import torch

SMPLX_DIR = os.path.expanduser("~/data/smplx/models_lockedhead")
_MODELS = {}


def get_smplx(gender, num_betas=16, device="cuda"):
    key = (gender, num_betas)
    if key not in _MODELS:
        import smplx
        _MODELS[key] = smplx.create(
            SMPLX_DIR, model_type="smplx", gender=gender,
            num_betas=num_betas, use_pca=False, flat_hand_mean=True,
            num_expression_coeffs=10, ext="npz").to(device).eval()
    return _MODELS[key]


def pose_instance(pose165, betas, trans, gender, device="cuda"):
    """Return SMPL-X vertices (V,3) in camera space + faces (F,3)."""
    p = torch.as_tensor(pose165, dtype=torch.float32, device=device).view(1, -1)
    m = get_smplx(gender, betas.shape[-1], device)
    with torch.no_grad():
        out = m(
            betas=torch.as_tensor(betas, dtype=torch.float32, device=device).view(1, -1),
            global_orient=p[:, 0:3], body_pose=p[:, 3:66],
            jaw_pose=p[:, 66:69], leye_pose=p[:, 69:72], reye_pose=p[:, 72:75],
            left_hand_pose=p[:, 75:120], right_hand_pose=p[:, 120:165],
            transl=torch.as_tensor(trans, dtype=torch.float32, device=device).view(1, 3),
            return_verts=True)
    return out.vertices[0], m.faces_tensor.to(device)


def project_opencv(verts_cam, K):
    """Pinhole project camera-space verts with OpenCV intrinsics K (3x3).
    Returns pixel coords (V,2). For validation against proj_verts."""
    K = torch.as_tensor(K, dtype=torch.float32, device=verts_cam.device)
    p = verts_cam @ K.T
    return p[:, :2] / p[:, 2:3]


@torch.no_grad()
def _rasterize(verts_cam, faces, K, H, W, max_bbox_area=4096):
    """Self-contained z-buffered barycentric rasterizer (no pytorch3d).

    verts_cam: (V,3) OpenCV camera space (+Z forward, +Y down). faces: (F,3).
    Returns face_idx (H,W int32, -1 bg) and bary (H,W,3 float32), perspective-
    correct depth z-buffer. Pixel centers sampled at (x+0.5, y+0.5).
    """
    dev = verts_cam.device
    K = torch.as_tensor(K, dtype=torch.float32, device=dev)
    fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
    z = verts_cam[:, 2].clamp_min(1e-6)
    u = verts_cam[:, 0] * fx / z + cx
    v = verts_cam[:, 1] * fy / z + cy                       # (V,) pixel coords

    fa, fb, fc = faces[:, 0].long(), faces[:, 1].long(), faces[:, 2].long()
    # cull triangles with any vertex behind camera
    front = (verts_cam[:, 2] > 1e-4)
    ok = front[fa] & front[fb] & front[fc]
    ua, ub, uc = u[fa], u[fb], u[fc]
    va, vb, vc = v[fa], v[fb], v[fc]
    za, zb, zc = verts_cam[fa, 2], verts_cam[fb, 2], verts_cam[fc, 2]
    # integer bbox clamped to image
    x0 = torch.clamp(torch.floor(torch.min(torch.stack([ua, ub, uc]), 0).values), 0, W - 1).long()
    x1 = torch.clamp(torch.ceil(torch.max(torch.stack([ua, ub, uc]), 0).values), 0, W - 1).long()
    y0 = torch.clamp(torch.floor(torch.min(torch.stack([va, vb, vc]), 0).values), 0, H - 1).long()
    y1 = torch.clamp(torch.ceil(torch.max(torch.stack([va, vb, vc]), 0).values), 0, H - 1).long()
    bw = (x1 - x0 + 1).clamp_min(0)
    bh = (y1 - y0 + 1).clamp_min(0)
    area = bw * bh
    ok &= (area > 0) & (area <= max_bbox_area)
    idx = torch.nonzero(ok, as_tuple=False).squeeze(1)
    if idx.numel() == 0:
        return (torch.full((H, W), -1, dtype=torch.int32, device=dev),
                torch.zeros((H, W, 3), dtype=torch.float32, device=dev))

    bw, bh, area = bw[idx], bh[idx], area[idx]
    x0i, y0i = x0[idx], y0[idx]
    # expand triangles -> candidate pixels
    offs = torch.cumsum(area, 0)
    total = int(offs[-1].item())
    tri_of = torch.repeat_interleave(torch.arange(idx.numel(), device=dev), area)
    base = (offs - area)[tri_of]
    local = torch.arange(total, device=dev) - base
    lw = bw[tri_of]
    px = x0i[tri_of] + (local % lw)
    py = y0i[tri_of] + (local // lw)
    fxp = px.float() + 0.5
    fyp = py.float() + 0.5

    A = (ua[idx], va[idx]); B = (ub[idx], vb[idx]); C = (uc[idx], vc[idx])
    ax, ay = A[0][tri_of], A[1][tri_of]
    bx, by = B[0][tri_of], B[1][tri_of]
    ccx, ccy = C[0][tri_of], C[1][tri_of]
    # barycentric of pixel center
    v0x, v0y = bx - ax, by - ay
    v1x, v1y = ccx - ax, ccy - ay
    v2x, v2y = fxp - ax, fyp - ay
    d00 = v0x * v0x + v0y * v0y
    d01 = v0x * v1x + v0y * v1y
    d11 = v1x * v1x + v1y * v1y
    d20 = v2x * v0x + v2y * v0y
    d21 = v2x * v1x + v2y * v1y
    denom = d00 * d11 - d01 * d01
    safe = denom.abs() > 1e-12
    inv = torch.where(safe, 1.0 / denom, torch.zeros_like(denom))
    wB = (d11 * d20 - d01 * d21) * inv
    wC = (d00 * d21 - d01 * d20) * inv
    wA = 1.0 - wB - wC
    eps = -1e-4
    inside = safe & (wA >= eps) & (wB >= eps) & (wC >= eps)
    # perspective-correct depth
    zA, zB, zC = za[idx][tri_of], zb[idx][tri_of], zc[idx][tri_of]
    zinv = wA / zA + wB / zB + wC / zC
    depth = torch.where(zinv > 0, 1.0 / zinv, torch.full_like(zinv, 1e9))
    inside &= depth < 1e8

    px, py, tri_of = px[inside], py[inside], tri_of[inside]
    wA, wB, wC, depth = wA[inside], wB[inside], wC[inside], depth[inside]
    pix = py * W + px
    # z-buffer: min depth per pixel
    zbuf = torch.full((H * W,), 1e9, dtype=torch.float32, device=dev)
    zbuf.scatter_reduce_(0, pix, depth, reduce="amin", include_self=True)
    winner = depth <= (zbuf[pix] + 1e-6)
    px, py, tri_of = px[winner], py[winner], tri_of[winner]
    wA, wB, wC = wA[winner], wB[winner], wC[winner]
    pixw = py * W + px

    face_idx = torch.full((H * W,), -1, dtype=torch.int32, device=dev)
    bary = torch.zeros((H * W, 3), dtype=torch.float32, device=dev)
    gfaces = idx[tri_of].to(torch.int32)        # global face id
    face_idx[pixw] = gfaces                       # last-write-wins on ties
    bary[pixw, 0] = wA; bary[pixw, 1] = wB; bary[pixw, 2] = wC
    return face_idx.view(H, W), bary.view(H, W, 3)


def world_to_cam(verts_world, ext, device="cuda"):
    """Apply BEDLAM2 world->camera extrinsics (4x4). Returns camera-space verts
    (OpenCV: +Z forward, +Y down) ready for OpenCV-style projection."""
    E = torch.as_tensor(ext, dtype=torch.float32, device=device)
    vh = torch.cat([verts_world, torch.ones_like(verts_world[:, :1])], 1)
    return (vh @ E.T)[:, :3]


def rasterize_frame(instances, K, H, W, device="cuda"):
    """instances: list of {pose,betas,trans,gender,ext} with pose/trans in
    WORLD space and ext = that instance's OWN world->camera extrinsics (BEDLAM2
    stores a per-person camera). K is shared per frame. Pose all, transform each
    to the shared camera via its own ext, rasterize together (correct inter-
    person occlusion), split global face id -> (instance, local face).
    Returns face_idx (H,W int32,-1 bg), bary, inst_id, meta, verts_cam, faces."""
    all_v, all_f, meta, voff, foff = [], [], [], 0, 0
    for inst in instances:
        vw, f = pose_instance(inst["pose"], inst["betas"], inst["trans"],
                              inst["gender"], device)
        v = world_to_cam(vw, inst["ext"], device)
        all_v.append(v); all_f.append(f + voff)
        meta.append((voff, foff, f.shape[0], v.shape[0]))
        voff += v.shape[0]; foff += f.shape[0]
    verts = torch.cat(all_v, 0)
    faces = torch.cat(all_f, 0)
    gface, bary = _rasterize(verts, faces, K, H, W)

    face_idx = torch.full((H, W), -1, dtype=torch.int32, device=device)
    inst_id = torch.full((H, W), -1, dtype=torch.int16, device=device)
    fg = gface >= 0
    gfi = gface[fg].long()
    starts = torch.tensor([m[1] for m in meta], device=device)
    which = torch.bucketize(gfi, starts, right=True) - 1
    face_idx[fg] = (gfi - starts[which]).to(torch.int32)
    inst_id[fg] = which.to(torch.int16)
    return (face_idx.cpu().numpy(), bary.cpu().numpy(),
            inst_id.cpu().numpy(), meta, verts, faces)


def crops_from_frame(face_idx, bary, inst_id, n_inst):
    """Per visible instance: tight bbox + cropped face/bary maps."""
    out = []
    for i in range(n_inst):
        mask = inst_id == i
        if not mask.any():
            out.append(None); continue
        ys, xs = np.where(mask)
        y0, y1, x0, x1 = ys.min(), ys.max() + 1, xs.min(), xs.max() + 1
        m = mask[y0:y1, x0:x1]
        fi = np.where(m, face_idx[y0:y1, x0:x1], -1).astype(np.int32)
        bc = bary[y0:y1, x0:x1, :2].astype(np.float16)
        bc[~m] = 0
        out.append({"bbox": [int(x0), int(y0), int(x1), int(y1)],
                    "face": fi, "bary2": bc, "area": int(m.sum())})
    return out

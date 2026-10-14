#!/usr/bin/env python
"""4-panel pipeline video over a full 3DPW sequence (efficient, reusable; aarch64 GH200, single venv).

Panels per frame (same full image, different overlay), left->right, all labeled:
  [ CameraHMR (pred SMPL, red) | GT SMPL (blue) | DensePose (our IUV) | refined SMPL (ours, green) ]

Efficiency: SMPL forwards (pred / refined / GT) are batched over all frames once; DensePose is
batched in chunks of frames; the cv2 painter renderer (3 meshes x 13776 faces / frame) is the main
per-frame cost; frames stream straight to an imageio H.264 writer (no full-video RAM).

  python refine_video.py --seq outdoors_fencing_01            # full sequence
  python refine_video.py --seq outdoors_fencing_01 --max-frames 10   # quick validate

GT translation: 3DPW gives no GT camera translation, so we place each GT mesh by aligning its pelvis
(SMPL joint 0) to the CameraHMR-baseline pelvis in camera space (baseline_pelvis_local + cam_t_init).
The baseline projects onto the person by construction, so the GT mesh lands on the fencer too.
"""
import os, sys, time, argparse
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)) + "/..")
import numpy as np, inspect
for _a, _t in [('bool', np.bool_), ('int', np.int64), ('float', np.float64), ('complex', np.complex128),
               ('object', np.object_), ('str', np.str_), ('unicode', np.str_),
               ('nan', float('nan')), ('inf', float('inf'))]:
    if not hasattr(np, _a):
        setattr(np, _a, _t)
inspect.getargspec = inspect.getfullargspec
import cv2, torch, imageio.v2 as imageio
import matplotlib.cm as cm
import smplx
from model import DINOv3DPT, part_pred

# ---- paths ----
PRED = os.path.expanduser('~/codes/CameraHMR/cameramhr_3dpw_preds.npz')
REF = os.path.expanduser('~/codes/CameraHMR/refined_3dpw_locori.npz')
IMG_ROOT = os.path.expanduser('~/data/datasets/3dpw')
SMPL_DIR = os.path.expanduser('~/codes/CameraHMR/data/models/SMPL')
CKPT = os.path.expanduser('~/data/bedlam2/dp_runs/dp_pp_llrd_1e3_120k/ckpt_120000.pt')
VIZ = os.path.expanduser('~/codes/CameraHMR/viz')
DEV = 'cuda'
Z, EXP = 512, 1.3
# colors in RGB (frames are written RGB to imageio)
RED, BLUE, GREEN = (220, 60, 60), (60, 60, 220), (60, 200, 60)
PAL = (cm.hsv(np.linspace(0, 1, 25))[:, :3] * 255).astype(np.uint8)   # RGB part palette


# ---- mesh helpers (lifted from CameraHMR/viz_refine.py) ----
def cam_t_init(pred_cam, scale, center, cam_int):
    s, tx, ty = pred_cam[:, 0], pred_cam[:, 1], pred_cam[:, 2]
    B = scale * 200.0
    focal = cam_int[:, 0, 0]
    img_w = cam_int[:, 0, 2] * 2.0
    img_h = cam_int[:, 1, 2] * 2.0
    tz = 2.0 * focal / (B * s)
    cx = 2.0 * (center[:, 0] - img_w / 2.0) / (s * B)
    cy = 2.0 * (center[:, 1] - img_h / 2.0) / (s * B)
    return np.stack([tx + cx, ty + cy, tz], -1).astype(np.float32)


def render_mesh(img, verts_cam, faces, cam_int, color, alpha=0.4):
    """Painter's-algorithm translucent mesh overlay. verts_cam (V,3) camera frame."""
    fx, fy = cam_int[0, 0], cam_int[1, 1]
    cx, cy = cam_int[0, 2], cam_int[1, 2]
    z = np.clip(verts_cam[:, 2], 1e-4, None)
    u = fx * verts_cam[:, 0] / z + cx
    v = fy * verts_cam[:, 1] / z + cy
    uv = np.stack([u, v], 1)
    fz = verts_cam[faces, 2].mean(1)
    order = np.argsort(-fz)                       # far -> near
    overlay = img.copy()
    tri_uv = uv[faces].astype(np.int32)
    for fi in order:
        cv2.fillConvexPoly(overlay, tri_uv[fi], color, lineType=cv2.LINE_AA)
    return cv2.addWeighted(img, 1 - alpha, overlay, alpha, 0)


def label(img, txt):
    img = np.ascontiguousarray(img)
    cv2.rectangle(img, (0, 0), (img.shape[1], 60), (0, 0, 0), -1)
    cv2.putText(img, txt, (15, 44), cv2.FONT_HERSHEY_SIMPLEX, 1.3, (255, 255, 255), 3, cv2.LINE_AA)
    return img


# ---- DensePose (batched IUV; crop 1.3x box, paste back full frame; from evidence_3dpw.py) ----
def load_dp_model():
    sd = torch.load(CKPT, map_location='cpu')['model']
    m = DINOv3DPT(backbone='vit_large_patch16_dinov3', layers=(4, 11, 17, 23), pretrained=False,
                  decoupled=True, orient=True, rot_per_part=True).to(DEV).eval()
    m.load_state_dict(sd)
    return m


@torch.no_grad()
def infer_iuv_batch(m, crops):
    """crops: (B,Z,Z,3) uint8 -> part (B,Z,Z) int, U,V (B,Z,Z) float."""
    B = crops.shape[0]
    img = torch.from_numpy(crops).permute(0, 3, 1, 2).float().to(DEV) / 255
    with torch.autocast('cuda', dtype=torch.bfloat16):
        o = m(img)
    part = part_pred(o)                                   # (B,Z,Z)
    uv = o['uv'].float().view(B, 24, 2, Z, Z)
    sel = (part - 1).clamp(min=0)[:, None]                # (B,1,Z,Z)
    U = uv[:, :, 0].gather(1, sel).squeeze(1).clamp(0, 1)
    V = uv[:, :, 1].gather(1, sel).squeeze(1).clamp(0, 1)
    return part.cpu().numpy().astype(int), U.cpu().numpy(), V.cpu().numpy()


def iuv_overlay(full_rgb, part, U, V, M, Minv):
    """Build crop-space IUV, paste onto the full image (inverse warp). full_rgb (H,W,3) RGB."""
    H, W = full_rgb.shape[:2]
    mask = part > 0
    iuv = PAL[np.clip(part, 0, 24)].astype(np.float32) * (0.45 + 0.4 * U + 0.15 * V)[..., None]
    iuv = np.clip(iuv, 0, 255).astype(np.uint8)
    full_map = cv2.warpAffine(iuv, Minv, (W, H), flags=cv2.INTER_NEAREST, borderValue=0)
    fg = cv2.warpAffine(mask.astype(np.uint8), Minv, (W, H), flags=cv2.INTER_NEAREST) > 0
    out = full_rgb.copy()
    out[fg] = (0.35 * full_rgb[fg] + 0.65 * full_map[fg]).astype(np.uint8)
    return out


# ---- batched SMPL forwards ----
def frame_no(name):
    return int(os.path.splitext(name)[0].split('_')[-1])


def batched_verts(layer, kwargs_fn, idxs, chunk=512, want_pelvis=False):
    """Run a smplx model over idxs in chunks; return verts (n,V,3) and optionally pelvis (n,3)."""
    n = len(idxs)
    V = []
    P = [] if want_pelvis else None
    for s in range(0, n, chunk):
        e = min(s + chunk, n)
        kw = kwargs_fn(idxs[s:e])
        out = layer(**kw)
        V.append(out.vertices.detach().cpu().numpy())
        if want_pelvis:
            P.append(out.joints[:, 0].detach().cpu().numpy())
    V = np.concatenate(V, 0)
    return (V, np.concatenate(P, 0)) if want_pelvis else V


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--seq', default='outdoors_fencing_01')
    ap.add_argument('--stride', type=int, default=1)
    ap.add_argument('--max-frames', type=int, default=0)
    ap.add_argument('--fps', type=int, default=30)        # 3DPW imageFiles are 30 fps
    ap.add_argument('--dp-chunk', type=int, default=16)
    ap.add_argument('--out', default='')
    args = ap.parse_args()
    os.makedirs(VIZ, exist_ok=True)
    out_path = args.out or os.path.join(VIZ, f'refine_video_{args.seq}.mp4')

    pred = np.load(PRED, allow_pickle=True)
    ref = np.load(REF, allow_pickle=True)
    assert (pred['imgname'] == ref['imgname']).all()
    names = np.array([str(x) for x in pred['imgname']])

    # select sequence instances, sort by (imgname/frame, then instance order for multi-person)
    sel = np.array([i for i, n in enumerate(names) if n.split('/')[1] == args.seq])
    sel = sel[np.argsort([frame_no(names[i].split('/')[-1]) for i in sel], kind='stable')]
    if args.stride > 1:
        # keep all instances of every `stride`-th unique frame
        keep_imgs = sorted({names[i] for i in sel}, key=lambda n: frame_no(n.split('/')[-1]))[::args.stride]
        keep_imgs = set(keep_imgs)
        sel = np.array([i for i in sel if names[i] in keep_imgs])
    if args.max_frames:
        keep_imgs = sorted({names[i] for i in sel}, key=lambda n: frame_no(n.split('/')[-1]))[:args.max_frames]
        keep_imgs = set(keep_imgs)
        sel = np.array([i for i in sel if names[i] in keep_imgs])
    print(f'seq {args.seq}: {len(sel)} instances over '
          f'{len({names[i] for i in sel})} frames', flush=True)

    cam_int = pred['cam_int']
    base_t = cam_t_init(pred['pred_cam'], pred['scale'], pred['center'], cam_int)

    # ---- batched SMPL: baseline (rotmat), refined (aa), GT (aa, gender-specific) ----
    smpl_layer = smplx.SMPLLayer(model_path=os.path.join(SMPL_DIR, 'SMPL_NEUTRAL.pkl'),
                                 num_betas=10).to(DEV).eval()
    smpl_aa = smplx.SMPL(SMPL_DIR, gender='neutral', num_betas=10).to(DEV).eval()
    faces = smpl_layer.faces.astype(np.int64)

    def t(x):
        return torch.from_numpy(np.ascontiguousarray(x)).float().to(DEV)

    with torch.no_grad():
        v_base, pelv_base = batched_verts(
            smpl_layer,
            lambda I: dict(global_orient=t(pred['pred_global_orient'][I]),
                           body_pose=t(pred['pred_body_pose'][I]),
                           betas=t(pred['pred_betas'][I]), pose2rot=False),
            sel, want_pelvis=True)
        v_ref = batched_verts(
            smpl_aa,
            lambda I: dict(global_orient=t(ref['global_orient_aa'][I]),
                           body_pose=t(ref['body_pose_aa'][I]), betas=t(ref['betas'][I])),
            sel)
        # GT: gender-specific SMPL (pose2rot=True, axis-angle 72 = 3 + 69)
        gt_models = {}
        v_gt = np.zeros_like(v_base)
        pelv_gt = np.zeros((len(sel), 3), np.float32)
        gender = np.array([str(g) for g in pred['gender']])
        for g in np.unique(gender[sel]):
            gname = {'m': 'male', 'f': 'female'}.get(g, 'neutral')
            gt_models[g] = smplx.SMPL(SMPL_DIR, gender=gname, num_betas=10).to(DEV).eval()
            gi = np.array([k for k, i in enumerate(sel) if gender[i] == g])
            si = sel[gi]
            vg, pg = batched_verts(
                gt_models[g],
                lambda I: dict(global_orient=t(pred['gt_pose'][I][:, :3]),
                               body_pose=t(pred['gt_pose'][I][:, 3:]), betas=t(pred['gt_betas'][I])),
                si, want_pelvis=True)
            v_gt[gi] = vg
            pelv_gt[gi] = pg

    base_t_sel = base_t[sel]
    transl_ref_sel = ref['transl'][sel]
    # baseline & refined in camera frame
    v_base_cam = v_base + base_t_sel[:, None]
    v_ref_cam = v_ref + transl_ref_sel[:, None]
    # GT translation: align GT pelvis to baseline pelvis in camera space
    base_pelv_cam = pelv_base + base_t_sel
    transl_gt = base_pelv_cam - pelv_gt
    v_gt_cam = v_gt + transl_gt[:, None]

    # group instances by image (multi-person safe), ordered by frame number
    from collections import OrderedDict
    by_img = OrderedDict()
    for k, i in enumerate(sel):
        by_img.setdefault(names[i], []).append(k)   # k indexes the *_cam arrays / cam_int via sel

    m_dp = load_dp_model()
    writer = imageio.get_writer(out_path, fps=args.fps, codec='libx264',
                                quality=8, macro_block_size=None)

    img_list = list(by_img.items())
    t0 = time.time()
    written = 0
    for cs in range(0, len(img_list), args.dp_chunk):
        batch = img_list[cs:cs + args.dp_chunk]
        # read images (RGB) + build DensePose crops for the chunk (one person's box per frame for IUV)
        fulls, crops, Ms, Minvs = [], [], [], []
        for name, ks in batch:
            full = cv2.imread(os.path.join(IMG_ROOT, name))[..., ::-1]  # RGB
            fulls.append(np.ascontiguousarray(full))
            i0 = sel[ks[0]]                                  # first person for the IUV crop
            cx, cy = pred['center'][i0]
            B = EXP * pred['scale'][i0] * 200.0
            s = Z / B
            M = np.array([[s, 0, Z / 2 - cx * s], [0, s, Z / 2 - cy * s]], np.float32)
            crops.append(cv2.warpAffine(full, M, (Z, Z), flags=cv2.INTER_AREA, borderValue=0))
            Ms.append(M)
            Minvs.append(cv2.invertAffineTransform(M))
        part, U, Vv = infer_iuv_batch(m_dp, np.stack([np.ascontiguousarray(c) for c in crops]))

        for j, (name, ks) in enumerate(batch):
            full = fulls[j]
            ci = cam_int[sel[ks[0]]]
            p_cam, p_gt, p_ref = full.copy(), full.copy(), full.copy()
            for k in ks:                                     # all persons in this frame
                ci_k = cam_int[sel[k]]
                p_cam = render_mesh(p_cam, v_base_cam[k], faces, ci_k, RED)
                p_gt = render_mesh(p_gt, v_gt_cam[k], faces, ci_k, BLUE)
                p_ref = render_mesh(p_ref, v_ref_cam[k], faces, ci_k, GREEN)
            p_dp = iuv_overlay(full, part[j], U[j], Vv[j], Ms[j], Minvs[j])
            row = np.hstack([label(p_cam, 'CameraHMR (pred)'), label(p_gt, 'GT SMPL'),
                             label(p_dp, 'DensePose (ours)'), label(p_ref, 'refined (ours)')])
            writer.append_data(row)
            written += 1
        print(f'  frames {written}/{len(img_list)}', flush=True)
    writer.close()
    dt = time.time() - t0
    print(f'wrote {out_path}', flush=True)
    print(f'render: {written} frames in {dt:.1f}s = {written / dt:.2f} fps '
          f'(frame size {row.shape[1]}x{row.shape[0]})', flush=True)


if __name__ == '__main__':
    main()

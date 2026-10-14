#!/usr/bin/env python
"""Rich per-instance report for the CameraHMR refinement: top/bottom PCT% bands, each row =
  image | CameraHMR(before,red) | refined(after,green) | GT(blue) | DensePose IUV |
  LOCATION-loss heatmap | ROTATION-loss heatmap
Heatmaps are DENSE per-pixel baseline residuals over the body:
  - location = reprojection error (px) of each (I,U,V)-corresponded surface point on the CameraHMR mesh
  - rotation = geodesic angle (deg) between R_mesh(CameraHMR) and our predicted R at that pixel
This shows WHERE we gain / lose so we can target fixes. Resumable (skips existing panel jpgs).

Run in wan venv (has smplx + scipy):
  cd ~/codes/wan/densepose/orient && python report_heatmaps.py --pct 10 --n-show 12
"""
import os, sys, argparse
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)) + "/..")   # densepose/ for model
sys.path.insert(0, os.path.expanduser("~/codes/CameraHMR"))              # refine_3dpw
import numpy as np, inspect
for _a, _t in [('bool', np.bool_), ('int', np.int64), ('float', np.float64), ('complex', np.complex128),
               ('object', np.object_), ('str', np.str_), ('unicode', np.str_),
               ('nan', float('nan')), ('inf', float('inf'))]:
    if not hasattr(np, _a):
        setattr(np, _a, _t)
inspect.getargspec = inspect.getfullargspec
import cv2, torch, smplx
import matplotlib.cm as cm
os.chdir(os.path.expanduser("~/codes/CameraHMR"))   # refine_3dpw uses relative SMPL/asset paths
from model import DINOv3DPT, part_pred
from refine_3dpw import IUV2SMPL, RMesh, rpred_from_6d

H = os.path.expanduser
PRED = H('~/codes/CameraHMR/cameramhr_3dpw_preds.npz')
DIAG = H('~/codes/CameraHMR/diag_3dpw.npz')
LOC = H('~/codes/CameraHMR/refined_3dpw_loc.npz')
ORI = H('~/codes/CameraHMR/refined_3dpw_locori.npz')
IMG_ROOT = H('~/data/datasets/3dpw')
SMPL_DIR = H('~/codes/CameraHMR/data/models/SMPL')
CKPT = H('~/data/bedlam2/dp_runs/dp_pp_llrd_1e3_120k/ckpt_120000.pt')
OUT = H('~/codes/CameraHMR/viz/report_heat')
DEV = 'cuda'
Z, EXP = 512, 1.3
RED, BLUE, GREEN = (220, 60, 60), (60, 60, 220), (60, 200, 60)
PAL = (cm.hsv(np.linspace(0, 1, 25))[:, :3] * 255).astype(np.uint8)
TURBO = cm.get_cmap('turbo')
LOC_MAX, ROT_MAX = 25.0, 45.0   # heatmap saturation: px, degrees


def cam_t_init(pc, sc, ce, ci):
    s, tx, ty = pc[:, 0], pc[:, 1], pc[:, 2]; B = sc * 200.0
    f = ci[:, 0, 0]; iw = ci[:, 0, 2] * 2; ih = ci[:, 1, 2] * 2
    tz = 2 * f / (B * s); cx = 2 * (ce[:, 0] - iw / 2) / (s * B); cy = 2 * (ce[:, 1] - ih / 2) / (s * B)
    return np.stack([tx + cx, ty + cy, tz], -1).astype(np.float32)


def _proj(verts_cam, ci):
    z = np.clip(verts_cam[:, 2], 1e-4, None)
    return np.stack([ci[0, 0] * verts_cam[:, 0] / z + ci[0, 2],
                     ci[1, 1] * verts_cam[:, 1] / z + ci[1, 2]], 1)


def _face_shade(verts_cam, faces):
    """Headlight/curvature shading in [0.35,1]: sign-robust (|n_z|), shows limb structure."""
    v = verts_cam[faces]
    n = np.cross(v[:, 1] - v[:, 0], v[:, 2] - v[:, 0])
    n /= (np.linalg.norm(n, axis=1, keepdims=True) + 1e-9)
    return (0.35 + 0.65 * np.abs(n[:, 2])).astype(np.float32)


def render_mesh(img, verts_cam, faces, ci, color, alpha=0.9):
    """SHADED translucent mesh (not flat segmentation): per-face color = base * shade."""
    tuv = _proj(verts_cam, ci)[faces].astype(np.int32)
    sh = _face_shade(verts_cam, faces)
    order = np.argsort(-verts_cam[faces, 2].mean(1))
    canvas = img.copy(); mask = np.zeros(img.shape[:2], np.uint8)
    for fi in order:
        c = (int(color[0] * sh[fi]), int(color[1] * sh[fi]), int(color[2] * sh[fi]))
        cv2.fillConvexPoly(canvas, tuv[fi], c, lineType=cv2.LINE_AA)
        cv2.fillConvexPoly(mask, tuv[fi], 255)
    m = mask > 0; out = img.copy()
    out[m] = (alpha * canvas[m] + (1 - alpha) * img[m]).astype(np.uint8)
    return out


def draw_boxes(img, cx, cy, scale, exp=EXP):
    """Detection box (scale*200, yellow) + DensePose crop box (exp*scale*200, cyan)."""
    img = np.ascontiguousarray(img)
    for side, col in ((scale * 200.0, (255, 230, 0)), (exp * scale * 200.0, (0, 220, 220))):
        x0, y0 = int(cx - side / 2), int(cy - side / 2)
        cv2.rectangle(img, (x0, y0), (int(cx + side / 2), int(cy + side / 2)), col, 3)
    return img


def render_mesh_loss(img, verts_cam, faces, ci, fcol, fvalid, alpha=0.85):
    """SHADED per-face loss colors (fcol uint8 (F,3) * shade), only fvalid faces; painter order.
    Shading keeps 3D limb structure legible while hue encodes the loss."""
    tuv = _proj(verts_cam, ci)[faces].astype(np.int32)
    sh = _face_shade(verts_cam, faces)
    order = np.argsort(-verts_cam[faces, 2].mean(1))
    canvas = np.zeros_like(img); mask = np.zeros(img.shape[:2], np.uint8)
    for fi in order:
        if not fvalid[fi]:
            continue
        c = (int(fcol[fi, 0] * sh[fi]), int(fcol[fi, 1] * sh[fi]), int(fcol[fi, 2] * sh[fi]))
        cv2.fillConvexPoly(canvas, tuv[fi], c, lineType=cv2.LINE_AA)
        cv2.fillConvexPoly(mask, tuv[fi], 255, lineType=cv2.LINE_AA)
    m = mask > 0
    out = img.copy()
    out[m] = (alpha * canvas[m] + (1 - alpha) * img[m]).astype(np.uint8)
    return out


def vert_loss_to_face_colors(vids, bary, per_pt, vmax, nverts=6890):
    """Scatter per-correspondence-point loss onto SMPL vertices (bary-weighted mean),
    then -> per-face turbo colors + valid mask (faces with >=1 corresponded vertex)."""
    vsum = np.zeros(nverts); wsum = np.zeros(nverts)
    for k in range(3):
        np.add.at(wsum, vids[:, k], bary[:, k])
        np.add.at(vsum, vids[:, k], bary[:, k] * per_pt)
    vv = wsum > 1e-6
    vloss = np.zeros(nverts); vloss[vv] = vsum[vv] / wsum[vv]
    return vloss, vv


def label(img, txt, h=46):
    img = np.ascontiguousarray(img)
    cv2.rectangle(img, (0, 0), (img.shape[1], h), (0, 0, 0), -1)
    cv2.putText(img, txt, (8, h - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
    return img


def crop_box(img, cx, cy, B, sz=420):
    x0, y0 = int(cx - B / 2), int(cy - B / 2); Hh, Ww = img.shape[:2]
    px0, py0 = max(0, -x0), max(0, -y0); sx0, sy0 = max(0, x0), max(0, y0)
    sx1, sy1 = min(Ww, x0 + int(B)), min(Hh, y0 + int(B))
    canvas = np.zeros((int(B), int(B), 3), np.uint8)
    canvas[py0:py0 + (sy1 - sy0), px0:px0 + (sx1 - sx0)] = img[sy0:sy1, sx0:sx1]
    return cv2.resize(canvas, (sz, sz), interpolation=cv2.INTER_AREA)


def load_dp():
    sd = torch.load(CKPT, map_location='cpu')['model']
    m = DINOv3DPT(backbone='vit_large_patch16_dinov3', layers=(4, 11, 17, 23), pretrained=False,
                  decoupled=True, orient=True, rot_per_part=True).to(DEV).eval()
    m.load_state_dict(sd)
    return m


@torch.no_grad()
def infer_dense(m, crop):
    """-> part(Z,Z) int, U,V(Z,Z), six(Z,Z,6) gathered by predicted part."""
    img = torch.from_numpy(crop).permute(2, 0, 1).float().to(DEV)[None] / 255
    with torch.autocast('cuda', dtype=torch.bfloat16):
        o = m(img)
    part = part_pred(o)[0]
    sel = (part - 1).clamp(min=0)
    uv = o['uv'].float().view(24, 2, Z, Z)
    U = uv[:, 0].gather(0, sel[None]).squeeze(0).clamp(0, 1)
    V = uv[:, 1].gather(0, sel[None]).squeeze(0).clamp(0, 1)
    rot = o['rot6d'].float().view(24, 6, Z, Z)
    six = rot.gather(0, sel[None, None].expand(1, 6, Z, Z)).squeeze(0)   # (6,Z,Z) = pred-part rot6d
    return part.cpu().numpy().astype(int), U.cpu().numpy(), V.cpu().numpy(), six.permute(1, 2, 0)  # (Z,Z,6) tensor


def heat_overlay(full, cropvals, cropmask, M, Minv, vmax):
    Hh, Ww = full.shape[:2]
    norm = np.clip(cropvals / vmax, 0, 1)
    rgb = (TURBO(norm)[..., :3] * 255).astype(np.uint8)
    fm = cv2.warpAffine(rgb, Minv, (Ww, Hh), flags=cv2.INTER_NEAREST, borderValue=0)
    fg = cv2.warpAffine(cropmask.astype(np.uint8), Minv, (Ww, Hh), flags=cv2.INTER_NEAREST) > 0
    out = (full * 0.35).astype(np.uint8); out[fg] = fm[fg]
    return out


def iuv_overlay(full, part, U, V, M, Minv):
    Hh, Ww = full.shape[:2]; mask = part > 0
    iuv = (PAL[np.clip(part, 0, 24)].astype(np.float32) * (0.45 + 0.4 * U + 0.15 * V)[..., None])
    iuv = np.clip(iuv, 0, 255).astype(np.uint8)
    fm = cv2.warpAffine(iuv, Minv, (Ww, Hh), flags=cv2.INTER_NEAREST, borderValue=0)
    fg = cv2.warpAffine(mask.astype(np.uint8), Minv, (Ww, Hh), flags=cv2.INTER_NEAREST) > 0
    out = full.copy(); out[fg] = (0.3 * full[fg] + 0.7 * fm[fg]).astype(np.uint8)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--pct', type=float, default=10.0)
    ap.add_argument('--n-show', type=int, default=12)
    ap.add_argument('--min-count', type=int, default=80)
    ap.add_argument('--max-fg', type=int, default=30000)
    args = ap.parse_args()
    os.makedirs(OUT + '/imgs', exist_ok=True)

    pred = np.load(PRED, allow_pickle=True); diag = np.load(DIAG, allow_pickle=True)
    loc = np.load(LOC, allow_pickle=True); ori = np.load(ORI, allow_pickle=True)
    names = np.array([str(x) for x in pred['imgname']]); counts = diag['counts']
    d_overall = diag['ori_mpjpe'] - diag['base_mpjpe']
    idx_elig = np.where(counts >= args.min_count)[0]
    order = idx_elig[np.argsort(d_overall[idx_elig])]
    nb = max(1, int(round(len(order) * args.pct / 100.0)))

    def spread(tail):
        if len(tail) <= args.n_show:
            return tail
        return tail[np.linspace(0, len(tail) - 1, args.n_show).round().astype(int)]

    bands = {'top10_improve': ('Top %g%% — refinement HELPS most (ΔMPJPE)' % args.pct, spread(order[:nb])),
             'bottom10_hurt': ('Bottom %g%% — refinement HURTS most (ΔMPJPE)' % args.pct, spread(order[::-1][:nb]))}

    smpl = smplx.SMPL(SMPL_DIR, gender='neutral', num_betas=10).to(DEV).eval()
    smpl_l = smplx.SMPLLayer(model_path=os.path.join(SMPL_DIR, 'SMPL_NEUTRAL.pkl'), num_betas=10).to(DEV).eval()
    gtm = {g: smplx.SMPL(SMPL_DIR, gender={'m': 'male', 'f': 'female'}.get(g, 'neutral'), num_betas=10).to(DEV).eval()
           for g in ('m', 'f')}
    faces = smpl_l.faces.astype(np.int64)
    base_t = cam_t_init(pred['pred_cam'], pred['scale'], pred['center'], pred['cam_int'])
    m_dp = load_dp(); mapper = IUV2SMPL(); rmesh = RMesh()

    def t(x): return torch.from_numpy(np.ascontiguousarray(x)).float().to(DEV)

    html = ['<html><head><meta charset="utf-8"><style>body{font-family:sans-serif;background:#111;color:#ddd;'
            'margin:20px}h2{border-bottom:1px solid #444;margin-top:36px}img{max-width:100%;border:1px solid #333}'
            '.d{color:#5d5}.u{color:#f77}</style></head><body>']
    html.append('<h1>CameraHMR refinement — heatmap report (tuned λ=1.0, λ_rot=0.003)</h1>')
    html.append('<p>Columns: image | CameraHMR before(red) | refined after(green) | GT(blue) | DensePose IUV | '
                f'<b>location-loss on mesh</b> (reproj px, 0→{LOC_MAX:g}) | <b>rotation-loss on mesh</b> '
                f'(geodesic°, 0→{ROT_MAX:g}). Loss is painted per-face on the CameraHMR baseline mesh '
                '(turbo: blue=low, red=high); grey/unpainted = no correspondence.</p>')

    for key, (title, idxs) in bands.items():
        html.append(f'<h2>{title}</h2>')
        for rank, i in enumerate(idxs):
            fn = f'imgs/{key}_{rank:02d}.jpg'; fp = os.path.join(OUT, fn)
            b = (diag['base_pa_mpjpe'][i], diag['base_mpjpe'][i], diag['base_pve'][i])
            o = (diag['ori_pa_mpjpe'][i], diag['ori_mpjpe'][i], diag['ori_pve'][i])
            dov = o[1] - b[1]
            if not os.path.exists(fp):
                full = cv2.imread(os.path.join(IMG_ROOT, names[i]))
                if full is None:
                    continue
                full = np.ascontiguousarray(full[..., ::-1])
                cx, cy = pred['center'][i]; ci = pred['cam_int'][i]; Bx = EXP * pred['scale'][i] * 200.0
                s = Z / Bx; M = np.array([[s, 0, Z / 2 - cx * s], [0, s, Z / 2 - cy * s]], np.float32)
                Minv = cv2.invertAffineTransform(M)
                crop = cv2.warpAffine(full, M, (Z, Z), flags=cv2.INTER_AREA, borderValue=0)
                part, U, V, six = infer_dense(m_dp, np.ascontiguousarray(crop))

                # meshes (before / after / GT)
                with torch.no_grad():
                    vb = smpl_l(global_orient=t(pred['pred_global_orient'][i])[None],
                                body_pose=t(pred['pred_body_pose'][i])[None],
                                betas=t(pred['pred_betas'][i])[None], pose2rot=False)
                    Vb_body = vb.vertices; pelvb = vb.joints[0, 0]
                    vo = smpl(global_orient=t(ori['global_orient_aa'][i])[None],
                              body_pose=t(ori['body_pose_aa'][i])[None], betas=t(ori['betas'][i])[None]).vertices
                    g = str(pred['gender'][i]); gm = gtm.get(g, gtm['m'])
                    vgo = gm(global_orient=t(pred['gt_pose'][i][:3])[None], body_pose=t(pred['gt_pose'][i][3:])[None],
                             betas=t(pred['gt_betas'][i])[None])
                    vg = vgo.vertices; pelvg = vgo.joints[0, 0]
                Vb_cam = Vb_body[0].cpu().numpy() + base_t[i]
                Vo_cam = vo[0].cpu().numpy() + ori['transl'][i]
                Vg_cam = vg[0].cpu().numpy() + ((pelvb + t(base_t[i])) - pelvg).cpu().numpy()

                # dense residual heatmaps at baseline
                fgm = part > 0
                ys, xs = np.where(fgm)
                if len(xs) > args.max_fg:
                    keep = np.random.RandomState(0).choice(len(xs), args.max_fg, replace=False)
                    ys, xs = ys[keep], xs[keep]
                pk = part[ys, xs]; uvk = np.stack([U[ys, xs], V[ys, xs]], 1)
                vids, bary = mapper.map(pk.astype(np.int64), uvk)           # (M,3),(M,3)
                vids_t = torch.from_numpy(vids).to(DEV); bary_t = torch.from_numpy(bary).float().to(DEV)
                with torch.no_grad():
                    Vb = torch.from_numpy(Vb_cam).float().to(DEV)           # (6890,3) camera
                    pts = (bary_t[..., None] * Vb[vids_t]).sum(1)           # (M,3) corr pt camera
                    fx, fy, ppx, ppy = ci[0, 0], ci[1, 1], ci[0, 2], ci[1, 2]
                    zc = pts[:, 2].clamp(min=1e-4)
                    proj = torch.stack([fx * pts[:, 0] / zc + ppx, fy * pts[:, 1] / zc + ppy], 1)  # full px
                    fullxy = torch.from_numpy(np.stack([Minv[0, 0] * xs + Minv[0, 1] * ys + Minv[0, 2],
                                                        Minv[1, 0] * xs + Minv[1, 1] * ys + Minv[1, 2]], 1)).float().to(DEV)
                    loc_px = (proj - fullxy).norm(dim=1).cpu().numpy()
                    Rmesh = rmesh.rmesh_interp(Vb_body, vids_t[None], bary_t[None], flip=True)[0]   # (M,3,3)
                    six_k = six[torch.from_numpy(ys).to(DEV), torch.from_numpy(xs).to(DEV)]          # (M,6)
                    Rpred = rpred_from_6d(six_k)
                    tr = (Rmesh.transpose(-1, -2) @ Rpred).diagonal(dim1=-1, dim2=-2).sum(-1)
                    rot_deg = (torch.acos(torch.clamp((tr - 1) / 2, -1 + 1e-6, 1 - 1e-6)) * 180 / np.pi).cpu().numpy()
                # per-vertex -> per-face loss colors, painted ON the baseline mesh
                vloc, vv = vert_loss_to_face_colors(vids, bary, loc_px, LOC_MAX)
                vrot, _ = vert_loss_to_face_colors(vids, bary, rot_deg, ROT_MAX)
                fvalid = vv[faces].any(1)
                floc = np.clip(vloc[faces].mean(1) / LOC_MAX, 0, 1)
                frot = np.clip(vrot[faces].mean(1) / ROT_MAX, 0, 1)
                fcol_loc = (TURBO(floc)[:, :3] * 255).astype(np.uint8)
                fcol_rot = (TURBO(frot)[:, :3] * 255).astype(np.uint8)

                p_cam = render_mesh(full.copy(), Vb_cam, faces, ci, RED)
                p_ref = render_mesh(full.copy(), Vo_cam, faces, ci, GREEN)
                p_gt = render_mesh(full.copy(), Vg_cam, faces, ci, BLUE)
                p_dp = iuv_overlay(full, part, U, V, M, Minv)
                p_loc = render_mesh_loss(full.copy(), Vb_cam, faces, ci, fcol_loc, fvalid)
                p_rot = render_mesh_loss(full.copy(), Vb_cam, faces, ci, fcol_rot, fvalid)
                cb = Bx * 1.25
                p_img = draw_boxes(full.copy(), cx, cy, pred['scale'][i])
                panels = [crop_box(p_img, cx, cy, cb), crop_box(p_cam, cx, cy, cb), crop_box(p_ref, cx, cy, cb),
                          crop_box(p_gt, cx, cy, cb), crop_box(p_dp, cx, cy, cb),
                          crop_box(p_loc, cx, cy, cb), crop_box(p_rot, cx, cy, cb)]
                tt = ['image +bbox (det=yel, crop=cyan)', 'CameraHMR (before)', 'refined (after)', 'GT',
                      'DensePose', 'loc-loss mesh px', 'rot-loss mesh deg']
                row = np.hstack([label(p, x) for p, x in zip(panels, tt)])
                cv2.imwrite(fp, row[..., ::-1], [cv2.IMWRITE_JPEG_QUALITY, 90])
                print(f'  {key} #{rank}: {names[i]} dMPJPE {dov:+.1f} '
                      f'loc<med {np.median(loc_px):.1f}px> rot<med {np.median(rot_deg):.1f}°>', flush=True)
            cls = 'd' if dov < 0 else 'u'
            html.append(f'<div><b>{names[i]}</b> pts={int(counts[i])} '
                        f'<span class="{cls}">ΔMPJPE {dov:+.1f}</span> '
                        f'(base MPJPE {b[1]:.1f} → refined {o[1]:.1f})</div><img src="{fn}"><br>')
    with open(os.path.join(OUT, 'index.html'), 'w') as f:
        f.write('\n'.join(html))
    print('wrote', os.path.join(OUT, 'index.html'))


if __name__ == '__main__':
    main()

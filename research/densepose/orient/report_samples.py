#!/usr/bin/env python
"""Per-instance improve/hurt report for the refinement (CameraHMR base, 3DPW).

Ranks instances by how much the optimization changes MPJPE, both OVERALL (loc+ori
vs baseline) and ORIENTATION-attributable (loc+ori vs loc-only), and renders a
panel per sample:
  [ crop | CameraHMR (red) | GT (blue) | DensePose IUV | loc-only (yellow) | loc+ori (green) ]
with the per-instance metrics + deltas. Output: an HTML page + PNG strips.

Run in the wan venv (DensePose model + smplx).  Needs diag_3dpw.npz from diag_3dpw.py
and refined_3dpw_loc.npz / refined_3dpw_locori.npz from refine_3dpw.py --mode full.

  python report_samples.py --k 8
"""
import os, sys, argparse
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)) + "/..")
import numpy as np, inspect
for _a, _t in [('bool', np.bool_), ('int', np.int64), ('float', np.float64), ('complex', np.complex128),
               ('object', np.object_), ('str', np.str_), ('unicode', np.str_),
               ('nan', float('nan')), ('inf', float('inf'))]:
    if not hasattr(np, _a):
        setattr(np, _a, _t)
inspect.getargspec = inspect.getfullargspec
import cv2, torch, smplx
import matplotlib.cm as cm
from model import DINOv3DPT, part_pred

H = os.path.expanduser
PRED = H('~/codes/CameraHMR/cameramhr_3dpw_preds.npz')
DIAG = H('~/codes/CameraHMR/diag_3dpw.npz')
LOC = H('~/codes/CameraHMR/refined_3dpw_loc.npz')
ORI = H('~/codes/CameraHMR/refined_3dpw_locori.npz')
IMG_ROOT = H('~/data/datasets/3dpw')
SMPL_DIR = H('~/codes/CameraHMR/data/models/SMPL')
CKPT = H('~/data/bedlam2/dp_runs/dp_pp_llrd_1e3_120k/ckpt_120000.pt')
OUT = H('~/codes/CameraHMR/viz/report')
DEV = 'cuda'
Z, EXP = 512, 1.3
RED, BLUE, GREEN, YEL = (220, 60, 60), (60, 60, 220), (60, 200, 60), (235, 190, 40)
PAL = (cm.hsv(np.linspace(0, 1, 25))[:, :3] * 255).astype(np.uint8)


def cam_t_init(pred_cam, scale, center, cam_int):
    s, tx, ty = pred_cam[:, 0], pred_cam[:, 1], pred_cam[:, 2]
    B = scale * 200.0
    focal = cam_int[:, 0, 0]; img_w = cam_int[:, 0, 2] * 2.0; img_h = cam_int[:, 1, 2] * 2.0
    tz = 2.0 * focal / (B * s)
    cx = 2.0 * (center[:, 0] - img_w / 2.0) / (s * B)
    cy = 2.0 * (center[:, 1] - img_h / 2.0) / (s * B)
    return np.stack([tx + cx, ty + cy, tz], -1).astype(np.float32)


def render_mesh(img, verts_cam, faces, cam_int, color, alpha=0.45):
    fx, fy = cam_int[0, 0], cam_int[1, 1]; cx, cy = cam_int[0, 2], cam_int[1, 2]
    z = np.clip(verts_cam[:, 2], 1e-4, None)
    u = fx * verts_cam[:, 0] / z + cx; v = fy * verts_cam[:, 1] / z + cy
    uv = np.stack([u, v], 1)
    order = np.argsort(-verts_cam[faces, 2].mean(1))
    overlay = img.copy(); tri_uv = uv[faces].astype(np.int32)
    for fi in order:
        cv2.fillConvexPoly(overlay, tri_uv[fi], color, lineType=cv2.LINE_AA)
    return cv2.addWeighted(img, 1 - alpha, overlay, alpha, 0)


def label(img, txt, h=46):
    img = np.ascontiguousarray(img)
    cv2.rectangle(img, (0, 0), (img.shape[1], h), (0, 0, 0), -1)
    cv2.putText(img, txt, (8, h - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
    return img


def load_dp_model():
    sd = torch.load(CKPT, map_location='cpu')['model']
    m = DINOv3DPT(backbone='vit_large_patch16_dinov3', layers=(4, 11, 17, 23), pretrained=False,
                  decoupled=True, orient=True, rot_per_part=True).to(DEV).eval()
    m.load_state_dict(sd)
    return m


@torch.no_grad()
def infer_iuv(m, crop):
    img = torch.from_numpy(crop).permute(2, 0, 1).float().to(DEV)[None] / 255
    with torch.autocast('cuda', dtype=torch.bfloat16):
        o = m(img)
    part = part_pred(o)[0]
    uv = o['uv'].float().view(1, 24, 2, Z, Z)
    sel = (part - 1).clamp(min=0)[None, None]
    U = uv[:, :, 0].gather(1, sel).squeeze()[None].clamp(0, 1)[0]
    V = uv[:, :, 1].gather(1, sel).squeeze()[None].clamp(0, 1)[0]
    return part.cpu().numpy().astype(int), U.cpu().numpy(), V.cpu().numpy()


def iuv_overlay(full, part, U, V, M, Minv):
    Hh, Ww = full.shape[:2]
    mask = part > 0
    iuv = PAL[np.clip(part, 0, 24)].astype(np.float32) * (0.45 + 0.4 * U + 0.15 * V)[..., None]
    iuv = np.clip(iuv, 0, 255).astype(np.uint8)
    fm = cv2.warpAffine(iuv, Minv, (Ww, Hh), flags=cv2.INTER_NEAREST, borderValue=0)
    fg = cv2.warpAffine(mask.astype(np.uint8), Minv, (Ww, Hh), flags=cv2.INTER_NEAREST) > 0
    out = full.copy(); out[fg] = (0.3 * full[fg] + 0.7 * fm[fg]).astype(np.uint8)
    return out


def crop_box(img, cx, cy, B, sz=420):
    """Crop a square of side B centred at (cx,cy), padded, resized to sz."""
    x0, y0 = int(cx - B / 2), int(cy - B / 2)
    Hh, Ww = img.shape[:2]
    px0, py0 = max(0, -x0), max(0, -y0)
    sx0, sy0 = max(0, x0), max(0, y0); sx1, sy1 = min(Ww, x0 + int(B)), min(Hh, y0 + int(B))
    canvas = np.zeros((int(B), int(B), 3), np.uint8)
    canvas[py0:py0 + (sy1 - sy0), px0:px0 + (sx1 - sx0)] = img[sy0:sy1, sx0:sx1]
    return cv2.resize(canvas, (sz, sz), interpolation=cv2.INTER_AREA)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--pct', type=float, default=5.0, help='tail percentile for the bands')
    ap.add_argument('--n-show', type=int, default=12, help='samples rendered per band (spread across it)')
    ap.add_argument('--min-count', type=int, default=50)
    args = ap.parse_args()
    os.makedirs(OUT, exist_ok=True); os.makedirs(os.path.join(OUT, 'imgs'), exist_ok=True)

    pred = np.load(PRED, allow_pickle=True)
    diag = np.load(DIAG, allow_pickle=True)
    loc = np.load(LOC, allow_pickle=True)
    ori = np.load(ORI, allow_pickle=True)
    names = np.array([str(x) for x in pred['imgname']])
    counts = diag['counts']

    d_overall = diag['ori_mpjpe'] - diag['base_mpjpe']    # loc+ori vs baseline
    d_orient = diag['ori_mpjpe'] - diag['loc_mpjpe']       # orientation-attributable
    elig = counts >= args.min_count
    idx_elig = np.where(elig)[0]
    pct = args.pct

    def band(arr, top):
        """Return (sample idxs spread across the tail, band stats dict). top=True -> most
        improved (most negative); top=False -> most hurt (most positive). Tail = pct% of eligible."""
        order = idx_elig[np.argsort(arr[idx_elig])]          # ascending (most improved first)
        nb = max(1, int(round(len(order) * pct / 100.0)))
        tail = order[:nb] if top else order[::-1][:nb]       # the pct% band, ordered most->least extreme
        # spread n_show samples evenly across the band so they represent its whole range
        if len(tail) <= args.n_show:
            pick = tail
        else:
            pos = np.linspace(0, len(tail) - 1, args.n_show).round().astype(int)
            pick = tail[pos]
        vals = arr[tail]
        stats = dict(n=len(tail), thr=float(arr[tail[-1]]), mean=float(vals.mean()),
                     median=float(np.median(vals)), extreme=float(vals[0]))
        return pick, stats

    cats = {}
    for key, title, arr, top in [
        ('orient_improve', f'Orientation helps most — top {pct}% by orientation ΔMPJPE (loc+ori vs loc-only)', d_orient, True),
        ('orient_hurt',    f'Orientation hurts most — bottom {pct}% by orientation ΔMPJPE (loc+ori vs loc-only)', d_orient, False),
        ('overall_improve',f'Refinement helps most — top {pct}% by overall ΔMPJPE (loc+ori vs baseline)', d_overall, True),
        ('overall_hurt',   f'Refinement hurts most — bottom {pct}% by overall ΔMPJPE (loc+ori vs baseline)', d_overall, False),
    ]:
        pick, stats = band(arr, top)
        cats[key] = (title, pick, stats)

    # distribution histogram with 5% tails shaded
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        fig, axs = plt.subplots(1, 2, figsize=(13, 4))
        for ax, arr, nm in [(axs[0], d_orient[idx_elig], 'orientation ΔMPJPE (loc+ori − loc-only)'),
                            (axs[1], d_overall[idx_elig], 'overall ΔMPJPE (loc+ori − baseline)')]:
            lo, hi = np.percentile(arr, pct), np.percentile(arr, 100 - pct)
            ax.hist(arr, bins=120, color='#5577cc', range=(np.percentile(arr, 0.5), np.percentile(arr, 99.5)))
            ax.axvline(lo, color='#5d5', ls='--', label=f'top {pct}% ≤ {lo:.2f}')
            ax.axvline(hi, color='#f77', ls='--', label=f'bottom {pct}% ≥ {hi:.2f}')
            ax.axvline(0, color='k', lw=0.8)
            ax.set_title(f'{nm}\nmean {arr.mean():+.2f}  median {np.median(arr):+.2f}')
            ax.set_xlabel('mm (negative = improved)'); ax.legend(fontsize=8)
        fig.tight_layout(); fig.savefig(os.path.join(OUT, 'dist.png'), dpi=90); plt.close(fig)
    except Exception as e:
        print('hist skipped:', e)

    smpl_layer = smplx.SMPLLayer(model_path=os.path.join(SMPL_DIR, 'SMPL_NEUTRAL.pkl'), num_betas=10).to(DEV).eval()
    smpl_aa = smplx.SMPL(SMPL_DIR, gender='neutral', num_betas=10).to(DEV).eval()
    gt_models = {g: smplx.SMPL(SMPL_DIR, gender={'m': 'male', 'f': 'female'}.get(g, 'neutral'),
                               num_betas=10).to(DEV).eval() for g in ('m', 'f')}
    faces = smpl_layer.faces.astype(np.int64)
    base_t = cam_t_init(pred['pred_cam'], pred['scale'], pred['center'], pred['cam_int'])
    m_dp = load_dp_model()

    def t(x): return torch.from_numpy(np.ascontiguousarray(x)).float().to(DEV)

    @torch.no_grad()
    def verts_for(i):
        vb = smpl_layer(global_orient=t(pred['pred_global_orient'][i])[None],
                        body_pose=t(pred['pred_body_pose'][i])[None],
                        betas=t(pred['pred_betas'][i])[None], pose2rot=False)
        vl = smpl_aa(global_orient=t(loc['global_orient_aa'][i])[None],
                     body_pose=t(loc['body_pose_aa'][i])[None], betas=t(loc['betas'][i])[None])
        vo = smpl_aa(global_orient=t(ori['global_orient_aa'][i])[None],
                     body_pose=t(ori['body_pose_aa'][i])[None], betas=t(ori['betas'][i])[None])
        g = str(pred['gender'][i]); gm = gt_models.get(g, gt_models['m'])
        vg = gm(global_orient=t(pred['gt_pose'][i][:3])[None], body_pose=t(pred['gt_pose'][i][3:])[None],
                betas=t(pred['gt_betas'][i])[None])
        return (vb.vertices[0].cpu().numpy(), vb.joints[0, 0].cpu().numpy(),
                vl.vertices[0].cpu().numpy(), vo.vertices[0].cpu().numpy(),
                vg.vertices[0].cpu().numpy(), vg.joints[0, 0].cpu().numpy())

    html = ['<html><head><meta charset="utf-8"><style>body{font-family:sans-serif;background:#111;'
            'color:#ddd;margin:24px}h2{margin-top:40px;border-bottom:1px solid #444}'
            'table{border-collapse:collapse;font-size:13px;margin:4px 0}td,th{padding:2px 8px}'
            'img{max-width:100%;border:1px solid #333;margin:6px 0}.d{color:#5d5}.u{color:#f77}</style></head><body>']
    html.append('<h1>Refinement per-instance improve / hurt report (CameraHMR base, 3DPW)</h1>')
    html.append(f'<p>Config in refined_3dpw_*.npz. {len(idx_elig)} eligible instances (≥{args.min_count} corr pts). '
                f'Columns: crop | CameraHMR(red) | GT(blue) | DensePose IUV | loc-only(yellow) | loc+ori(green). '
                f'Δ vs baseline; ori-Δ vs loc-only. Each band = {pct}% tail; {args.n_show} samples spread across it.</p>')
    if os.path.exists(os.path.join(OUT, 'dist.png')):
        html.append('<img src="dist.png" style="max-width:1100px">')

    for key, (title, idxs, stats) in cats.items():
        html.append(f'<h2>{title}</h2>')
        html.append(f'<p>band: n={stats["n"]} instances &nbsp; threshold {stats["thr"]:+.2f} mm &nbsp; '
                    f'band mean {stats["mean"]:+.2f} &nbsp; median {stats["median"]:+.2f} &nbsp; '
                    f'most extreme {stats["extreme"]:+.2f} mm</p>')
        for rank, i in enumerate(idxs):
            name = names[i]
            full = cv2.imread(os.path.join(IMG_ROOT, name))
            if full is None:
                continue
            full = np.ascontiguousarray(full[..., ::-1])
            cx, cy = pred['center'][i]; ci = pred['cam_int'][i]
            B = EXP * pred['scale'][i] * 200.0
            s = Z / B
            M = np.array([[s, 0, Z / 2 - cx * s], [0, s, Z / 2 - cy * s]], np.float32)
            Minv = cv2.invertAffineTransform(M)
            crop = cv2.warpAffine(full, M, (Z, Z), flags=cv2.INTER_AREA, borderValue=0)
            part, U, V = infer_iuv(m_dp, np.ascontiguousarray(crop))

            vb, pelvb, vl, vo, vg, pelvg = verts_for(i)
            bt = base_t[i]
            vb_c = vb + bt; vl_c = vl + loc['transl'][i]; vo_c = vo + ori['transl'][i]
            vg_c = vg + ((pelvb + bt) - pelvg)             # align GT pelvis to baseline

            p_cam = render_mesh(full.copy(), vb_c, faces, ci, RED)
            p_gt = render_mesh(full.copy(), vg_c, faces, ci, BLUE)
            p_loc = render_mesh(full.copy(), vl_c, faces, ci, YEL)
            p_ori = render_mesh(full.copy(), vo_c, faces, ci, GREEN)
            p_dp = iuv_overlay(full, part, U, V, M, Minv)

            cb = B * 1.25
            panels = [crop_box(full, cx, cy, cb), crop_box(p_cam, cx, cy, cb), crop_box(p_gt, cx, cy, cb),
                      crop_box(p_dp, cx, cy, cb), crop_box(p_loc, cx, cy, cb), crop_box(p_ori, cx, cy, cb)]
            titles = ['image', 'CameraHMR', 'GT', 'DensePose', 'loc-only', 'loc+ori']
            row = np.hstack([label(p, tt) for p, tt in zip(panels, titles)])
            fn = f'imgs/{key}_{rank:02d}.jpg'
            cv2.imwrite(os.path.join(OUT, fn), row[..., ::-1], [cv2.IMWRITE_JPEG_QUALITY, 90])

            def mrow(tag):
                return (diag[f'{tag}_pa_mpjpe'][i], diag[f'{tag}_mpjpe'][i], diag[f'{tag}_pve'][i])
            b, l, o = mrow('base'), mrow('loc'), mrow('ori')
            dov = o[1] - b[1]; dor = o[1] - l[1]
            cls_ov = 'd' if dov < 0 else 'u'; cls_or = 'd' if dor < 0 else 'u'
            html.append(f'<div><b>{name}</b> &nbsp; pts={int(counts[i])} '
                        f'&nbsp; <span class="{cls_ov}">ΔMPJPE(overall) {dov:+.1f}</span> '
                        f'&nbsp; <span class="{cls_or}">ΔMPJPE(orient) {dor:+.1f}</span></div>')
            html.append('<table><tr><th></th><th>PA-MPJPE</th><th>MPJPE</th><th>PVE</th></tr>'
                        + ''.join(f'<tr><td>{n}</td><td>{m[0]:.1f}</td><td>{m[1]:.1f}</td><td>{m[2]:.1f}</td></tr>'
                                  for n, m in [('baseline', b), ('loc-only', l), ('loc+ori', o)])
                        + '</table>')
            html.append(f'<img src="{fn}">')
            print(f'  {key} #{rank}: {name} overall {dov:+.1f} orient {dor:+.1f}', flush=True)

    html.append('</body></html>')
    with open(os.path.join(OUT, 'index.html'), 'w') as f:
        f.write('\n'.join(html))
    print(f'\nwrote {os.path.join(OUT, "index.html")}')


if __name__ == '__main__':
    main()

"""Pixel frame field as oriented BOXES (paper Fig. 1), from one field_dump frame (npz + jpg).

Each box is ONE PIXEL of the prediction:
  centre      the pixel
  axes        the predicted camera-space surface frame: long axis = tangent t (the smooth template
              combing field carried by R), short axis = bitangent b = n x t, thin axis = normal n
  colour      WHERE the pixel lands on the template (canonical XYZ -> RGB)
The SAME boxes are exported on the canonical template: position = the pixel's template point pi(x),
axes = the template's own frame there. Side by side the two scenes show R(x): template frame ->
camera frame, pixel by pixel. Pixels are a hex grid over the (lightly eroded) foreground.

Frames in the dump are in the PyTorch3D camera basis (x image-LEFT, y up, z into the scene; verified on
silhouette edges 2026-09-15); Blender's camera basis is x right, y up, z towards the viewer -> diag(-1,1,-1).

  python -m orient.framefield_boxes --dump ~/data/dp_viz_compare/beauty/15731089_cscC --frame 75 \
         --out ~/data/dp_viz_compare/framefield_boxes/15731089_f075 [--step 16]
then render both scenes with orient/blender_boxes.py.
"""
import argparse, json, os, sys
import numpy as np, cv2, torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)) + "/..")
from scipy.spatial.transform import Rotation
from orient.template_normals import TemplateNormals
from orient.rot6d import get_template
from orient.pad_util import pad_white

ap = argparse.ArgumentParser()
ap.add_argument("--dump", required=True); ap.add_argument("--frame", type=int, default=75)
ap.add_argument("--step", type=float, default=16.0, help="hex-grid spacing in crop pixels")
ap.add_argument("--step-frac", type=float, default=None, help="grid spacing as a fraction of the person height (overrides --step)")
ap.add_argument("--erode", type=int, default=0, help="foreground erosion (px) before sampling (0: keep the silhouette)")
ap.add_argument("--edge-ring", type=int, default=1, help="1: add samples tracing the mask boundary (1-2 px inside), spaced ~step")
ap.add_argument("--box", type=float, nargs=3, default=[0.82, 0.50, 0.34], help="box dims (t, b, n) as a fraction of --step")
ap.add_argument("--parts", type=int, nargs="+", default=None, help="sample only these DensePose parts (1..24): a zoom pair on ONE contiguous template region")
ap.add_argument("--roi", type=float, nargs=4, default=None, help="x0 y0 x1 y1 (crop px): sample only inside -- finer zoom renders")
ap.add_argument("--sat", type=float, default=1.7, help="saturation boost of the location colours")
ap.add_argument("--uv-asset", default=None, help="chart asset of the dump's checkpoint (default: inferred from meta ckpt)")
ap.add_argument("--step-m", type=float, default=None, help="sample spacing ON THE BODY in metres (overrides --step/--step-frac; framing-independent density)")
ap.add_argument("--extra", nargs="+", type=float, default=None, help="X Y ... extra samples at exact pixels (tracked anchors), appended LAST")
ap.add_argument("--tmpl-scale", default="local", choices=["local", "height"])   # template glyph size per image px
ap.add_argument("--out", required=True)
a = ap.parse_args()
A = os.path.expanduser("~/data/bedlam2/dp_assets")
meta = json.load(open(f"{a.dump}/meta.json")); ck = os.path.basename(os.path.dirname(meta.get("ckpt", "")))
uv_asset = a.uv_asset or ("smplx_face_uv_v2.npy" if ck.startswith(("dp_csc", "dp_w_")) else "smplx_face_uv.npy")
print(f"[boxes] dump ckpt={ck} -> charts {uv_asset}", flush=True)

# ---- template: positions, normals, smooth tangent field ----
tn = lambda x: x.detach().cpu().numpy() if torch.is_tensor(x) else np.asarray(x)
vt, faces, nrest = get_template("cuda" if torch.cuda.is_available() else "cpu")
vt, faces, nrest = tn(vt).astype(np.float64), tn(faces).astype(np.int64), tn(nrest).astype(np.float64)
fuv = np.load(f"{A}/{uv_asset}"); fpart = np.load(f"{A}/smplx_face_to_dp24.npy").astype(int)
T_tmpl = np.load(f"{A}/smplx_tangent_smooth.npy").astype(np.float64)
TN_n = TemplateNormals(fuv, fpart, faces, nrest, v_template=vt)
TN_t = TemplateNormals(fuv, fpart, faces, T_tmpl)

# ---- the dumped prediction ----
d = np.load(f"{a.dump}/f_{a.frame:04d}.npz")
img = cv2.imread(f"{a.dump}/f_{a.frame:04d}.jpg")
mask = d["mask"]; part = d["part"].astype(int)
img_pw, pads = pad_white(img)                               # no samples where the crop is edge-replicated padding
if any(pads):
    _v = np.zeros(mask.shape, bool); _v[pads[0]:mask.shape[0] - pads[1], pads[2]:mask.shape[1] - pads[3]] = True
    mask = mask & _v; part = np.where(_v, part, 0)
U, V = d["uv"][0].astype(np.float32), d["uv"][1].astype(np.float32)
n_cam, t_cam = d["n_cam"].astype(np.float64), d["t_cam"].astype(np.float64)
H, W = mask.shape
if a.step_frac:                                              # same visual density for every person size
    _my = np.nonzero(mask)[0]; a.step = max(3.0, a.step_frac * float(_my.max() - _my.min())); print(f"[boxes] step {a.step:.1f}px from --step-frac", flush=True)
if a.step_m:                                                  # provisional grid -> local template m/px -> step for a fixed body spacing
    _s0 = max(3.0, 0.02 * float(np.ptp(np.nonzero(mask)[0]))); _ys, _xs = np.mgrid[int(_s0 / 2):H:int(_s0), int(_s0 / 2):W:int(_s0)]
    _ys, _xs = _ys.ravel(), _xs.ravel(); _k = mask[_ys, _xs] & (part[_ys, _xs] > 0)
    if a.roi is not None: _k &= (_xs >= a.roi[0]) & (_xs < a.roi[2]) & (_ys >= a.roi[1]) & (_ys < a.roi[3])
    if a.parts: _k &= np.isin(part[_ys, _xs], a.parts)
    _ys, _xs = _ys[_k], _xs[_k]; _, _P, _ = TN_n(part[_ys, _xs], U[_ys, _xs], V[_ys, _xs], return_pos=True)
    _d = np.hypot(_xs[:, None] - _xs[None], _ys[:, None] - _ys[None]).astype(np.float64); np.fill_diagonal(_d, np.inf)
    _d[part[_ys, _xs][:, None] != part[_ys, _xs][None]] = np.inf; _j = _d.argmin(1); _ok = _d[np.arange(len(_ys)), _j] <= 1.5 * _s0
    _mpp = float(np.median(np.linalg.norm(_P[_ok] - _P[_j[_ok]], axis=1) / _d[np.arange(len(_ys)), _j][_ok]))
    a.step = max(3.0, a.step_m / _mpp); print(f"[boxes] step {a.step:.1f}px = {100*a.step_m:.1f} cm on the body ({1000*_mpp:.2f} mm/px, n={_ok.sum()})", flush=True)
er = cv2.erode(mask.astype(np.uint8), np.ones((2 * a.erode + 1,) * 2, np.uint8)) > 0 if a.erode else mask
pts = []
for r, y in enumerate(np.arange(a.step / 2, H, a.step * np.sqrt(3) / 2)):
    for x in np.arange(a.step / 2 + (a.step / 2 if r % 2 else 0), W, a.step):
        yi, xi = int(round(y)), int(round(x))
        inroi = a.roi is None or (a.roi[0] <= xi < a.roi[2] and a.roi[1] <= yi < a.roi[3])
        inroi = inroi and (a.parts is None or int(part[yi, xi]) in a.parts) if yi < H and xi < W else inroi
        if yi < H and xi < W and er[yi, xi] and part[yi, xi] > 0 and inroi:
            pts.append((yi, xi))
if a.edge_ring:                                               # silhouette ring: the frames turn most at the edge
    dist = cv2.distanceTransform(mask.astype(np.uint8), cv2.DIST_L2, 3)
    cy, cx = np.nonzero((dist >= 1) & (dist < 2.5) & (part > 0) & (np.isin(part, a.parts) if a.parts else True))
    if a.roi is not None:
        k = (cx >= a.roi[0]) & (cx < a.roi[2]) & (cy >= a.roi[1]) & (cy < a.roi[3]); cy, cx = cy[k], cx[k]
    acc = np.array(pts, float).reshape(-1, 2)
    order = np.random.default_rng(0).permutation(len(cy)); r = 0.85 * a.step
    for j in order:                                             # greedy Poisson-disk: >= r from every accepted sample
        q = np.array([cy[j], cx[j]], float)
        if len(acc) == 0 or ((acc - q) ** 2).sum(1).min() >= r * r:
            acc = np.vstack([acc, q]); pts.append((int(cy[j]), int(cx[j])))
n_extra = 0
if a.extra:                                                   # anchors: exact pixels, grid samples too close to them dropped
    ex = np.array(a.extra, float).reshape(-1, 2)
    pts = [p_ for p_ in pts if (((ex[:, 1] - p_[0]) ** 2 + (ex[:, 0] - p_[1]) ** 2).min() >= (0.6 * a.step) ** 2)]
    pts += [(int(round(y)), int(round(x))) for x, y in ex]; n_extra = len(ex)
ys, xs = np.array(pts).T
N = len(ys)
unit = lambda v: v / np.clip(np.linalg.norm(v, axis=1, keepdims=True), 1e-9, None)

nc = unit(n_cam[ys, xs]); tc = t_cam[ys, xs]; tc = unit(tc - (tc * nc).sum(1, keepdims=True) * nc); bc = np.cross(nc, tc)
nt, P, _ = TN_n(part[ys, xs], U[ys, xs], V[ys, xs], return_pos=True)
tt, _ = TN_t(part[ys, xs], U[ys, xs], V[ys, xs])
nt = unit(nt.astype(np.float64)); tt = tt.astype(np.float64)
tt = unit(tt - (tt * nt).sum(1, keepdims=True) * nt); bt = np.cross(nt, tt)
C = np.stack([tc, bc, nc], 2); T = np.stack([tt, bt, nt], 2)          # (N,3,3) frames as columns
Rpix = C @ np.transpose(T, (0, 2, 1))                                  # the predicted rotation per box
print(f"[boxes] {N} boxes | det R in [{np.linalg.det(Rpix).min():.4f},{np.linalg.det(Rpix).max():.4f}] "
      f"| orth err {np.abs(Rpix @ np.transpose(Rpix,(0,2,1)) - np.eye(3)).max():.2e}", flush=True)

Bflip = np.diag([-1., 1., -1.])                                        # PyTorch3D cam -> Blender cam
Pm = np.array([[1., 0, 0], [0, 0, -1.], [0, 1., 0]])                   # SMPL-X y-up -> Blender z-up (front = -Y)
def quat(M, F):
    q = Rotation.from_matrix(np.einsum("ij,njk->nik", M, F)).as_quat()   # x,y,z,w
    return q[:, [3, 0, 1, 2]]
qi, qt = quat(Bflip, C), quat(Pm, T)

lo, hi = vt.min(0), vt.max(0)
plo, phi = np.percentile(P, 2, 0), np.percentile(P, 98, 0)            # stretch over the SAMPLED points
col = 0.06 + 0.94 * np.clip((P - plo) / (phi - plo), 0, 1) ** 0.85     # RGB = template (x, y, z)
import colorsys
col = np.array([colorsys.hsv_to_rgb(h, min(1.0, s_ * a.sat), v) for h, s_, v in (colorsys.rgb_to_hsv(*c) for c in col)])
ctr = vt.mean(0); Pb = (P - ctr) @ Pm.T; Vb = (vt - ctr) @ Pm.T

my, mx = np.nonzero(mask)
m_per_px = (hi[1] - lo[1]) / float(my.max() - my.min())                # template height / person height
if a.tmpl_scale == "local":                                  # median template-m per image-px between neighbouring samples:
    pxy = np.c_[xs, ys].astype(np.float64); dpx = ((pxy[:, None] - pxy[None]) ** 2).sum(-1) ** 0.5      # arrow:spacing ratio then
    np.fill_diagonal(dpx, np.inf); pp = part[ys, xs]; dpx[pp[:, None] != pp[None]] = np.inf               # matches in both panels
    j = dpx.argmin(1); ok = dpx[np.arange(N), j] <= 1.5 * a.step
    if ok.sum() >= 5:
        r_ = np.linalg.norm(P[ok] - P[j[ok]], axis=1) / dpx[np.arange(N), j][ok]
        print(f"[boxes] template scale: height {m_per_px*1000:.2f} mm/px -> local {np.median(r_)*1000:.2f} mm/px (n={ok.sum()})", flush=True)
        m_per_px = float(np.median(r_))
os.makedirs(a.out, exist_ok=True)
cv2.imwrite(f"{a.out}/bg.png", img_pw)
if any(pads): print(f"[boxes] replicated-edge padding (top,bottom,left,right) = {pads} px -> white", flush=True)
mf = cv2.GaussianBlur(mask.astype(np.float32), (0, 0), 1.3)[..., None]                # feathered: no 8x staircase in zooms
seg = img.astype(np.float32) * (0.25 + 0.90 * mf)                                     # body bright, background dimmed
cv2.imwrite(f"{a.out}/bg_seg.png", np.clip(seg, 0, 255).astype(np.uint8))
dout = cv2.distanceTransform((~mask.astype(bool)).astype(np.uint8), cv2.DIST_L2, 3)[..., None]
af = mf + (1 - mf) * 0.30 * np.exp(-dout / 10.0)                                     # white-page teaser: a faint ghost (hair) near
wht = img.astype(np.float32) * af + 255.0 * (1 - af)                                  # the body, PURE white by ~30 px (no rectangle)
cv2.imwrite(f"{a.out}/bg_white.png", np.clip(wht, 0, 255).astype(np.uint8))
np.savetxt(f"{a.out}/img_boxes.csv", np.c_[(xs + .5) / W, 1 - (ys + .5) / H, qi, col], delimiter=",",
           header="u,v,qw,qx,qy,qz,r,g,b", comments="", fmt="%.6f")
np.savetxt(f"{a.out}/tmpl_boxes.csv", np.c_[Pb, qt, col], delimiter=",",
           header="x,y,z,qw,qx,qy,qz,r,g,b", comments="", fmt="%.6f")
with open(f"{a.out}/tmpl_body.obj", "w") as f:
    f.writelines(f"v {v[0]:.5f} {v[1]:.5f} {v[2]:.5f}\n" for v in Vb)
    f.writelines(f"f {t[0]+1} {t[1]+1} {t[2]+1}\n" for t in faces)
box_px = [s * a.step for s in a.box]
json.dump(dict(parts=a.parts, roi=(list(a.roi) if a.roi else None), dump=a.dump, frame=a.frame, ckpt=meta.get("ckpt"), charts=uv_asset, W=W, H=H, step=a.step, n=int(N), n_extra=n_extra, valid=[int(pads[2]), int(pads[0]), int(W - pads[3]), int(H - pads[1])],
               box_img=[s / W for s in box_px], box_tmpl=[s * m_per_px for s in box_px], m_per_px=m_per_px,
               person_bbox=[int(mx.min()), int(my.min()), int(mx.max()), int(my.max())],
               tmpl_bbox=[float(Vb[:, 0].min()), float(Vb[:, 2].min()), float(Vb[:, 0].max()), float(Vb[:, 2].max())]),
          open(f"{a.out}/meta.json", "w"), indent=1)
print(f"[boxes] wrote {a.out} | box {box_px[0]:.1f}x{box_px[1]:.1f}x{box_px[2]:.1f}px = "
      f"{box_px[0]*m_per_px*100:.1f}x{box_px[1]*m_per_px*100:.1f}x{box_px[2]*m_per_px*100:.1f} cm on the template", flush=True)

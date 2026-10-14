"""Overlay the model's PREDICTED framefield on a real image: at equidistant body pixels, draw a shaded
3D arrow-tripod oriented by the predicted per-pixel SO(3) frame (primary axis = outward surface normal),
projected into the image and composited BRIGHT over a darkened photo so the arrows glow in alignment
with the body. Short shaft + big head. Uses the best (+MAMMA-120k) model.

Usage: framefield_overlay.py CLIP --frame 75 [--n 600 --dark 0.28 --arrow 26 --ckpt ...]
"""
import os  # noqa: E402
import sys  # noqa: E402
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import _env  # noqa: E402  (repository paths: viz/_env.py)
import os, sys, argparse, numpy as np, cv2, torch
import matplotlib.cm as cm
from model import DINOv3DPT, part_pred
from orient.template_normals import TemplateNormals
from orient.rot6d import get_template, sixd_to_R_np
from orient.detect_track import detect_persons

ap = argparse.ArgumentParser()
D = _env.data("bedlam2/dp_runs")
ap.add_argument("clip"); ap.add_argument("--frame", type=int, default=75)
ap.add_argument("--ckpt", default=f"{D}/dp_pp_llrd_1e3_mamma/ckpt_120000.pt")
ap.add_argument("--n", type=int, default=600, help="equidistant arrows")
ap.add_argument("--dark", type=float, default=0.28, help="image darkening factor")
ap.add_argument("--arrow", type=float, default=26, help="arrow size in pixels")
ap.add_argument("--single", action="store_true", help="single outward arrow (default: full RGB tripod)")
ap.add_argument("--tilt-az", type=float, default=32, help="off-axis view azimuth (deg) so arrows show 3D form")
ap.add_argument("--tilt-el", type=float, default=20, help="off-axis view elevation (deg)")
ap.add_argument("--uvthr", type=float, default=0.06, help="UV match radius: fixed template point -> pixel")
ap.add_argument("--video", action="store_true", help="render the whole clip (tracked arrows) to mp4")
ap.add_argument("--maxframes", type=int, default=0)
ap.add_argument("--smooth", type=int, default=15, help="temporal window (frames) for bbox smoothing; <=1 = off (jittery)")
ap.add_argument("--export-blender", default=None, help="dir to export bg image(s) + tracked point/quaternion CSV(s) for Blender")
ap.add_argument("--export-template", default=None, help="dir to export canonical template body + same fixed points w/ surface frames (reference)")
ap.add_argument("--export-paired", default=None, help="dir: export image + template FRONT, only points visible in BOTH (paired comparison)")
ap.add_argument("--out", default=_env.out("framefield/overlay.png"))
a = ap.parse_args()
dev = "cuda"; Z = 512; EXP = 1.3
A = _env.ASSETS
v_template, faces, n_rest = get_template("cuda")
fuv = np.load(f"{A}/smplx_face_uv.npy"); fpart = np.load(f"{A}/smplx_face_to_dp24.npy").astype(int)
TN = TemplateNormals(fuv, fpart, np.asarray(faces), np.asarray(n_rest))
sd = torch.load(a.ckpt, map_location="cpu")["model"]
rpp = sd["rot_head.2.weight"].shape[0] != 6
m = DINOv3DPT(backbone="vit_large_patch16_dinov3", layers=(4, 11, 17, 23), pretrained=False,
              decoupled=True, orient=True, rot_per_part=rpp).to(dev).eval(); m.load_state_dict(sd)


@torch.no_grad()
def predict(crop, cs):
    img = torch.from_numpy(cv2.resize(crop, (Z, Z))).permute(2, 0, 1)[None].float().to(dev) / 255
    with torch.autocast("cuda", dtype=torch.bfloat16):
        o = m(img)
    part = cv2.resize(part_pred(o)[0].cpu().numpy().astype(np.uint8), (cs, cs), interpolation=cv2.INTER_NEAREST).astype(int)
    r, c = np.indices(part.shape); idx = np.clip(part - 1, 0, 23)
    uv = o["uv"].float()[0].view(24, 2, Z, Z).clamp(0, 1).cpu().numpy()
    uvv = np.stack([[cv2.resize(uv[p, k], (cs, cs)) for k in range(2)] for p in range(24)])
    US = uvv[idx, 0, r, c]; VS = uvv[idx, 1, r, c]
    rot = o["rot6d"].float()[0].view(24, 6, Z, Z).cpu().numpy()
    rotv = np.stack([[cv2.resize(rot[p, k], (cs, cs)) for k in range(6)] for p in range(24)])
    six = rotv[idx, :, r, c]; R = sixd_to_R_np(six)
    nt, mask = TN(part, US, VS)
    return part, US, VS, nt, mask, R


def arrow(shaft=0.34, head=0.5, rs=0.11, rc=0.30, nseg=12):
    """chunky arrow: fat shaft + big head, along +Z. returns verts, tris."""
    ang = np.linspace(0, 2 * np.pi, nseg, endpoint=False); csn, sn = np.cos(ang), np.sin(ang)
    r0 = np.stack([rs * csn, rs * sn, np.zeros(nseg)], 1)
    r1 = np.stack([rs * csn, rs * sn, np.full(nseg, shaft)], 1)
    r2 = np.stack([rc * csn, rc * sn, np.full(nseg, shaft)], 1)
    v = np.concatenate([r0, r1, r2, [[0, 0, shaft + head]], [[0, 0, 0.]]])
    T, B = 3 * nseg, 3 * nseg + 1; f = []
    for i in range(nseg):
        j = (i + 1) % nseg
        f += [[i, j, nseg + j], [i, nseg + j, nseg + i], [2 * nseg + i, 2 * nseg + j, T], [B, nseg + j, i]]
    return v, np.array(f)


def rot_z_to(axis):
    return {0: np.array([[0, 0, 1], [0, 1, 0], [-1, 0, 0.]]), 1: np.array([[1, 0, 0], [0, 0, -1], [0, 1, 0.]]),
            2: np.eye(3)}[axis]


# canonical instance geometry: full RGB tripod by default (chunky, shaded); z-arrow(normal, blue) is longest
AV, AF = arrow()
if not a.single:
    parts = [(0, [1, .28, .28], .6), (1, [.35, 1, .4], .6), (2, [.4, .6, 1], 1.0)]
    TV, TF, TC = [], [], []; off = 0
    for ax, col, ln in parts:
        vv = (AV * [1, 1, ln]) @ rot_z_to(ax).T
        TV.append(vv); TF.append(AF + off); TC.append(np.tile(col, (len(vv), 1))); off += len(vv)
    TV, TF, TC = np.concatenate(TV), np.concatenate(TF), np.concatenate(TC)
else:
    TV, TF, TC = AV.copy(), AF.copy(), None            # single outward arrow; color set per-instance by twist

# ---- FIXED, equidistant TEMPLATE points (tracked across frames) ----
from scipy.spatial import cKDTree
vert_part = np.load(f"{A}/smplx_vert_to_dp24.npy").astype(int)     # (10475,) DP part per template vertex
vert_uv = np.load(f"{A}/smplx_vert_uv.npy")                        # (10475,2) UV per template vertex
vt_np = v_template.detach().cpu().numpy() if torch.is_tensor(v_template) else np.asarray(v_template)
nrest_np = n_rest.detach().cpu().numpy() if torch.is_tensor(n_rest) else np.asarray(n_rest)


def fps_template(k, seed=0):
    d = ((vt_np - vt_np[seed]) ** 2).sum(1); sel = [seed]
    for _ in range(k - 1):
        j = int(np.argmax(d)); sel.append(j); d = np.minimum(d, ((vt_np - vt_np[j]) ** 2).sum(1))
    return np.array(sel)


S = fps_template(a.n)                                              # FIXED template vertex indices, equidistant
Spart = vert_part[S]; Suv = vert_uv[S]; Sn = nrest_np[S]
# template surface frame per fixed point (t,b,n)
ref = np.tile([0., 1., 0.], (len(Sn), 1)); deg = np.abs((ref * Sn).sum(1)) > 0.95; ref[deg] = [1, 0, 0]
St = ref - (ref * Sn).sum(1, keepdims=True) * Sn; St /= np.clip(np.linalg.norm(St, axis=1, keepdims=True), 1e-9, None)
STf = np.stack([St, np.cross(Sn, St), Sn], 2)                      # (N,3,3) template frame per fixed point
# per-part KD lookup: which template point owns each pixel -> so each fixed S point maps to a pixel
part_trees = {}
for p in np.unique(Spart):
    ip = np.where(Spart == p)[0]
    if len(ip): part_trees[p] = (ip, cKDTree(Suv[ip]))
L = np.array([0.3, -0.5, -0.8]); L = L / np.linalg.norm(L)
# fixed off-axis view so normal-pointing arrows show their 3D form (not collapsed head-on)
_az, _el = np.radians(a.tilt_az), np.radians(a.tilt_el)
Ry = np.array([[np.cos(_az), 0, np.sin(_az)], [0, 1, 0], [-np.sin(_az), 0, np.cos(_az)]])
Rx = np.array([[1, 0, 0], [0, np.cos(_el), -np.sin(_el)], [0, np.sin(_el), np.cos(_el)]])
Rview = Ry @ Rx


Bflip = np.diag([-1., 1., -1.])                                  # PyTorch3D-view cam -> Blender-ortho-cam basis


def R_to_quat(M):
    tr = np.trace(M)
    if tr > 0:
        s = np.sqrt(tr + 1) * 2; q = [.25*s, (M[2,1]-M[1,2])/s, (M[0,2]-M[2,0])/s, (M[1,0]-M[0,1])/s]
    else:
        i = int(np.argmax([M[0,0], M[1,1], M[2,2]]))
        if i == 0: s = np.sqrt(1+M[0,0]-M[1,1]-M[2,2])*2; q = [(M[2,1]-M[1,2])/s,.25*s,(M[0,1]+M[1,0])/s,(M[0,2]+M[2,0])/s]
        elif i == 1: s = np.sqrt(1+M[1,1]-M[0,0]-M[2,2])*2; q = [(M[0,2]-M[2,0])/s,(M[0,1]+M[1,0])/s,.25*s,(M[1,2]+M[2,1])/s]
        else: s = np.sqrt(1+M[2,2]-M[0,0]-M[1,1])*2; q = [(M[1,0]-M[0,1])/s,(M[0,2]+M[2,0])/s,(M[1,2]+M[2,1])/s,.25*s]
    q = np.array(q); return q / max(np.linalg.norm(q), 1e-9)


def detect_box(rgb):
    """Largest person as (cx, cy, s) with s = half the (expanded) square side. None if no detection."""
    dets = sorted(detect_persons(rgb, thr=0.5), key=lambda b: (b[2]-b[0])*(b[3]-b[1]), reverse=True)
    if not dets:
        return None
    x0, y0, x1, y1 = dets[0][:4]
    return np.array([(x0+x1)/2, (y0+y1)/2, max(x1-x0, y1-y0)*EXP/2])


def smooth_boxes(B, win):
    """B: (F,3) of (cx,cy,s) with NaN rows for missed detections. Fill gaps, then Gaussian-smooth the
    trajectory and freeze ONE side length for the whole clip -- a per-frame box makes the crop jitter and
    rescale, which reads as the whole scene shaking even though the overlay stays glued to the body."""
    F = len(B)
    for c in range(3):                                          # linear-interpolate over missed frames
        col = B[:, c]; m = np.isnan(col)
        if m.all():
            return None
        col[m] = np.interp(np.flatnonzero(m), np.flatnonzero(~m), col[~m])
    if win > 1:
        r = int(win) // 2
        k = np.exp(-0.5 * (np.arange(-r, r + 1) / (win / 4.0)) ** 2); k /= k.sum()
        Bp = np.pad(B, ((r, r), (0, 0)), mode="edge")
        B = np.stack([np.convolve(Bp[:, c], k, mode="valid")[:F] for c in range(3)], 1)
    B[:, 2] = np.median(B[:, 2])                                # constant crop size -> constant scale
    return B


def compute(rgb, box=None):
    """Return (crop, records) where records = [(px, py, D0, twist)] for visible TRACKED template points.
    D0 = Rpix @ template_frame = the true camera-space frame (col2 = outward normal). No view tilt.
    box = pre-smoothed (cx, cy, s); if None, detect on this frame alone (jittery)."""
    H, W = rgb.shape[:2]
    if box is None:
        box = detect_box(rgb)
        if box is None:
            return None, []
    cx, cy, s = box
    side = int(2*s); X0, Y0 = int(cx-s), int(cy-s)
    pad = ((max(0, -Y0), max(0, Y0+side-H)), (max(0, -X0), max(0, X0+side-W)), (0, 0))
    crop = np.pad(rgb, pad, mode="edge")[max(0, Y0):Y0+side+pad[0][0], max(0, X0):X0+side+pad[1][0]]
    part, US, VS, nt, mask, R = predict(crop, crop.shape[0])
    ys, xs = np.nonzero(mask); ppart = part[ys, xs]; puv = np.stack([US[ys, xs], VS[ys, xs]], 1)
    recs = []
    for p, (ip, tree) in part_trees.items():
        pm = ppart == p
        if pm.sum() < 3:
            continue
        ptree = cKDTree(puv[pm]); pix_x = xs[pm]; pix_y = ys[pm]
        dist, nn = ptree.query(Suv[ip])
        for q, si in enumerate(ip):
            if dist[q] < a.uvthr:
                px, py = int(pix_x[nn[q]]), int(pix_y[nn[q]]); Rpix = R[py, px]
                tw = (np.arctan2(Rpix[1, 0], Rpix[0, 0]) + np.pi) / (2*np.pi)
                recs.append((si, px, py, Rpix @ STf[si], tw))   # si = fixed-point index (for pairing)
    recs.sort(key=lambda r: r[2])
    recs = [r for r in recs if float(r[3][2, 2]) < 0.30]         # keep only points whose normal faces camera
    return crop, recs


def draw_overlay(crop, recs):                                   # flat matplotlib-style preview (limited!)
    canvas = crop.astype(np.float32) * a.dark
    for si, px, py, D0, tw in recs:
        Vi = (Rview @ D0 @ (TV * a.arrow).T).T
        scr = np.stack([px + Vi[:, 0], py - Vi[:, 1]], 1)
        base = TC if not a.single else np.tile(cm.plasma(tw)[:3], (len(TV), 1))
        tri = Vi[TF]; fn = np.cross(tri[:, 1]-tri[:, 0], tri[:, 2]-tri[:, 0])
        fn /= np.clip(np.linalg.norm(fn, axis=1, keepdims=True), 1e-9, None)
        shade = 0.5 + 0.5*np.clip(np.abs((fn*L).sum(1)), 0, 1)
        for t in np.argsort(-tri[:, :, 2].mean(1)):
            col = np.clip(base[TF[t][0]] * shade[t] * 255 * 1.7 + 30, 0, 255)
            cv2.fillConvexPoly(canvas, scr[TF[t]].astype(np.int32), (float(col[0]), float(col[1]), float(col[2])), lineType=cv2.LINE_AA)
    return np.clip(canvas, 0, 255).astype(np.uint8)


if a.export_paired:
    # image + template FRONT, keeping ONLY points visible in BOTH -> apples-to-apples comparison.
    outdir = a.export_paired; os.makedirs(outdir, exist_ok=True)
    cap = cv2.VideoCapture(a.clip); cap.set(1, a.frame); ok, bgr = cap.read(); cap.release()
    crop, recs = compute(np.ascontiguousarray(bgr[..., ::-1]))
    # TRUE visibility: reconstructed camera-space normal (D0[:,2]) must face the camera. The surface a
    # front-facing person shows is predominantly facing the camera, so keep points whose normal agrees
    # with that dominant facing sign; the minority (back points wrongly UV-matched) are dropped.
    Pm = np.array([[1., 0, 0], [0, 0, -1.], [0, 1., 0]])
    # DIAG: of the matched fixed points, how many sit on the template FRONT (n_rest_z>0) vs BACK?
    nz_tmpl = np.array([float(nrest_np[S[int(si)], 2]) for si, *_ in recs])
    print(f"[DIAG] matched pts: front(nz>0)={int((nz_tmpl>0).sum())} back(nz<0)={int((nz_tmpl<0).sum())}", flush=True)
    zc = np.array([float(D0[2, 2]) for si, px, py, D0, tw in recs])
    print(f"[DIAG] recon-normal z: mean={zc.mean():+.3f} front-pts-meanz={zc[nz_tmpl>0].mean():+.3f} back-pts-meanz={zc[nz_tmpl<0].mean():+.3f}", flush=True)
    # A point is truly visible iff its TEMPLATE-front side faces the image camera. Use the template normal
    # sign directly: keep the side whose template points dominate the match AND agree with recon normal.
    face_sign = 1.0 if (nz_tmpl > 0).sum() >= (nz_tmpl < 0).sum() else -1.0
    recs = [r for r, nz in zip(recs, nz_tmpl) if np.sign(nz) == face_sign]
    keep = set(int(si) for si, *_ in recs)
    Nz = (Pm @ nrest_np[S].T).T                                  # z-up template normals per fixed point
    front = Nz[sorted(keep)].mean(0); front = front / (np.linalg.norm(front) + 1e-9)   # template view dir
    print(f"[paired] kept {len(keep)} (template-{'front' if face_sign>0 else 'back'}) of {len(nz_tmpl)} matched", flush=True)
    # image bg + pts (shared points only)
    cv2.imwrite(f"{outdir}/bg_{a.frame:04d}.png", crop[..., ::-1])
    rows = []
    for si, px, py, D0, tw in recs:
        if int(si) not in keep:
            continue
        q = R_to_quat(Bflip @ D0)
        rows.append([px / crop.shape[1], 1 - py / crop.shape[0], q[0], q[1], q[2], q[3], tw])
    np.savetxt(f"{outdir}/pts_{a.frame:04d}.csv", np.array(rows), delimiter=",",
               header="u,v,qw,qx,qy,qz,twist", comments="")
    # template body + pts (shared only) + front-facing camera position
    Vt = (vt_np - vt_np.mean(0)) @ Pm.T
    with open(f"{outdir}/tmpl_body.obj", "w") as f:
        for v in Vt: f.write(f"v {v[0]:.5f} {v[1]:.5f} {v[2]:.5f}\n")
        for t in faces + 1: f.write(f"f {int(t[0])} {int(t[1])} {int(t[2])}\n")
    trows = []
    for k in sorted(keep):
        pos = Vt[S[k]]; q = R_to_quat(Pm @ STf[k])
        trows.append([pos[0], pos[1], pos[2], q[0], q[1], q[2], q[3], 0.0])
    np.savetxt(f"{outdir}/tmpl_pts.csv", np.array(trows), delimiter=",",
               header="x,y,z,qw,qx,qy,qz,twist", comments="")
    campos = front * 2.6 + np.array([0, 0, 0.15])
    np.savetxt(f"{outdir}/tmpl_cam.txt", campos[None], delimiter=",")
    print(f"PAIRED EXPORT DONE -> {outdir} ({len(keep)} shared points, frame {a.frame})", flush=True)
elif a.export_template:
    # canonical template body + the SAME fixed FPS points with their template surface frames (reference:
    # normal MUST point out of the template surface). Y-up SMPL -> Z-up for viewing.
    outdir = a.export_template; os.makedirs(outdir, exist_ok=True)
    Pm = np.array([[1., 0, 0], [0, 0, -1.], [0, 1., 0]])
    Vt = (vt_np - vt_np.mean(0)) @ Pm.T
    with open(f"{outdir}/tmpl_body.obj", "w") as f:
        for v in Vt: f.write(f"v {v[0]:.5f} {v[1]:.5f} {v[2]:.5f}\n")
        for t in faces + 1: f.write(f"f {int(t[0])} {int(t[1])} {int(t[2])}\n")
    rows = []
    for k, si in enumerate(S):
        pos = Vt[si]; q = R_to_quat(Pm @ STf[k])
        rows.append([pos[0], pos[1], pos[2], q[0], q[1], q[2], q[3], 0.0])
    np.savetxt(f"{outdir}/tmpl_pts.csv", np.array(rows), delimiter=",",
               header="x,y,z,qw,qx,qy,qz,twist", comments="")
    print(f"TEMPLATE EXPORT DONE -> {outdir} ({len(S)} points)", flush=True)
elif a.export_blender:
    # Export what Blender needs to render the tracked overlay properly (real 3D shaded tripods, aligned).
    outdir = a.export_blender; os.makedirs(outdir, exist_ok=True)
    cap = cv2.VideoCapture(a.clip)
    frames_idx = list(range(0, (a.maxframes or int(cap.get(7))), 1)) if a.video else [a.frame]
    boxes = {}
    if a.video and a.smooth > 1:
        # PASS 1: detect on every frame, then smooth the box trajectory. Per-frame boxes make the crop
        # translate and rescale, which reads as the whole scene shaking.
        B = np.full((len(frames_idx), 3), np.nan)
        for j, fi in enumerate(frames_idx):
            cap.set(1, fi); ok, bgr = cap.read()
            if not ok:
                break
            b = detect_box(np.ascontiguousarray(bgr[..., ::-1]))
            if b is not None:
                B[j] = b
        miss = int(np.isnan(B[:, 0]).sum())
        Bs = smooth_boxes(B, a.smooth)
        if Bs is None:
            raise SystemExit("no person detected anywhere in the clip")
        boxes = {fi: Bs[j] for j, fi in enumerate(frames_idx)}
        print(f"[smooth] win={a.smooth} | {miss} missed detections interpolated | "
              f"constant crop side={2*Bs[0,2]:.0f}px", flush=True)
    meta = []
    for fi in frames_idx:
        cap.set(1, fi); ok, bgr = cap.read()
        if not ok:
            break
        crop, recs = compute(np.ascontiguousarray(bgr[..., ::-1]), boxes.get(fi))
        if crop is None:
            continue
        cv2.imwrite(f"{outdir}/bg_{fi:04d}.png", crop[..., ::-1])
        rows = []
        for si, px, py, D0, tw in recs:
            q = R_to_quat(Bflip @ D0)                            # frame in Blender-camera basis
            # idx = which FIXED template point this is -> lets the viewer light up the same point on the template
            rows.append([int(si), px / crop.shape[1], 1 - py / crop.shape[0], q[0], q[1], q[2], q[3], tw])
        np.savetxt(f"{outdir}/pts_{fi:04d}.csv", np.array(rows),
                   delimiter=",", header="idx,u,v,qw,qx,qy,qz,twist", comments="")
        meta.append((fi, crop.shape[1], crop.shape[0], len(recs)))
        print(f"exported frame {fi}: {len(recs)} tracked arrows", flush=True)
    cap.release()
    print("BLENDER EXPORT DONE ->", outdir, flush=True)
elif a.video:
    import imageio
    cap = cv2.VideoCapture(a.clip); fps = cap.get(5) or 16; frames = []; n = 0
    while True:
        ok, bgr = cap.read()
        if not ok or (a.maxframes and n >= a.maxframes):
            break
        frames.append(np.ascontiguousarray(bgr[..., ::-1])); n += 1
    cap.release()
    wr = imageio.get_writer(a.out.replace(".png", ".mp4"), fps=fps, codec="libx264", quality=8,
                            pixelformat="yuv420p", macro_block_size=8)
    for k, fr in enumerate(frames):
        crop, recs = compute(fr)
        wr.append_data(draw_overlay(crop, recs) if crop is not None else (fr * a.dark).astype(np.uint8))
        if k % 20 == 0: print(f"frame {k}/{len(frames)}", flush=True)
    wr.close(); print("wrote", a.out.replace(".png", ".mp4"), flush=True)
else:
    cap = cv2.VideoCapture(a.clip); cap.set(1, a.frame); ok, bgr = cap.read(); cap.release()
    crop, recs = compute(np.ascontiguousarray(bgr[..., ::-1]))
    cv2.imwrite(a.out, draw_overlay(crop, recs)[..., ::-1])
    print("wrote", a.out, f"| {len(recs)} tracked arrows visible", flush=True)

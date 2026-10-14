"""Pixel frame field over TIME: how the predicted geometry changes while the template frames stay fixed.
  python -m orient.fig_sequence scan  DUMP [DUMP ...]                     # rank clips by torso rotation, pick frames
  python -m orient.fig_sequence build DUMP OUT.png [--k 6] [--frames F ...] [--region chest|head]
DUMP = a field_dump --video directory (f_XXXX.npz per frame, shipped model).
Frame picks: the torso's global orientation per frame = chordal mean of the per-pixel R = C T^T over torso pixels (visible
normals always face the camera, so a mean normal cannot measure a turn); take the frame pair with the largest relative
rotation and K frames evenly spaced in cumulative rotation between them.
Figure: column 0 = the canonical template (fixed) | columns 1..K = frames. Row 1 full person, row 2 a same-size zoom on the
region. Anchors = a few surface points tracked FORWARD by (part, uv) nearest neighbour, exported as exact-pixel glyphs and
ringed in the same colour in every frame and on the template: one surface point, one fixed template frame, a rotating
camera frame."""
import argparse, json, os, subprocess, sys, glob, numpy as np, cv2, torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)) + "/..")
from orient.template_normals import TemplateNormals
from orient.rot6d import get_template

HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(HERE)
PY = os.path.join(os.path.dirname(ROOT), ".venv/bin/python")
FB = os.path.expanduser("~/data/dp_viz_compare/framefield_boxes"); A = os.path.expanduser("~/data/bedlam2/dp_assets")
BL = ["apptainer", "exec", "-B", "/lus/lfs1aip2,/projects", os.path.expanduser("~/data/containers/blender.sif"),
      "blender", "--background", "--python", f"{HERE}/blender_boxes.py", "--"]
IMG = "--glyph tripod --samples 96 --bg bg.png --dark 0.45 --key 22 --emit 0.02".split()
REGION = {"chest": [1, 2], "head": [23, 24]}
ACOL = ["#f08c00", "#7c3aed", "#e8559a"]
_TN = {}


def tn_lookups():
    if not _TN:
        t = lambda x: x.detach().cpu().numpy() if torch.is_tensor(x) else np.asarray(x)
        vt, faces, nrest = get_template("cpu"); vt, faces, nrest = t(vt).astype(np.float64), t(faces).astype(np.int64), t(nrest).astype(np.float64)
        fuv = np.load(f"{A}/smplx_face_uv_v2.npy"); fpart = np.load(f"{A}/smplx_face_to_dp24.npy").astype(int)
        _TN["n"] = TemplateNormals(fuv, fpart, faces, nrest, v_template=vt)
        _TN["t"] = TemplateNormals(fuv, fpart, faces, np.load(f"{A}/smplx_tangent_smooth.npy").astype(np.float64))
    return _TN["n"], _TN["t"]


def unit(v): return v / np.clip(np.linalg.norm(v, axis=1, keepdims=True), 1e-9, None)


def torso_R(d, rng, nmax=400):
    part = d["part"].astype(int); ys, xs = np.nonzero(np.isin(part, [1, 2]))
    if len(ys) < 50: return None, len(ys)
    k = rng.choice(len(ys), min(nmax, len(ys)), replace=False); ys, xs = ys[k], xs[k]
    TNn, TNt = tn_lookups(); U, V = d["uv"][0][ys, xs].astype(np.float32), d["uv"][1][ys, xs].astype(np.float32)
    nc = unit(d["n_cam"][ys, xs].astype(np.float64)); tc = d["t_cam"][ys, xs].astype(np.float64); tc = unit(tc - (tc * nc).sum(1, keepdims=True) * nc)
    nt, _ = TNn(part[ys, xs], U, V); tt, _ = TNt(part[ys, xs], U, V); nt = unit(nt.astype(np.float64)); tt = tt.astype(np.float64)
    tt = unit(tt - (tt * nt).sum(1, keepdims=True) * nt)
    C = np.stack([tc, np.cross(nc, tc), nc], 2); T = np.stack([tt, np.cross(nt, tt), nt], 2)
    u, _, vt = np.linalg.svd((C @ np.transpose(T, (0, 2, 1))).mean(0)); R = u @ vt
    if np.linalg.det(R) < 0: u[:, -1] *= -1; R = u @ vt
    return R, int(np.isin(part, [1, 2]).sum())


def ang(R1, R2): return float(np.degrees(np.arccos(np.clip((np.trace(R1.T @ R2) - 1) / 2, -1, 1))))


def scan_one(dump, k=6, stride=1):
    meta = json.load(open(f"{dump}/meta.json")); frames = meta["frames"][::stride]; rng = np.random.default_rng(0); Rs, cnt, fg = [], [], []
    for f in frames:
        d = np.load(f"{dump}/f_{f:04d}.npz"); R, c = torso_R(d, rng); Rs.append(R); cnt.append(c); fg.append(float(d["mask"].mean()))
    cnt = np.array(cnt); ok = np.array([R is not None for R in Rs]) & (cnt >= 0.25 * cnt.max())
    idx = np.flatnonzero(ok)
    if len(idx) < k: return dict(dump=dump, score=0.0, picks=[])
    best = (0.0, idx[0], idx[-1])
    for i in idx[::2]:
        for j in idx[::2]:
            if j - i >= k and (a_ := ang(Rs[i], Rs[j])) > best[0]: best = (a_, i, j)
    _, i, j = best; seg = [q for q in idx if i <= q <= j]
    s = np.concatenate([[0], np.cumsum([ang(Rs[p], Rs[q]) for p, q in zip(seg[:-1], seg[1:])])])
    picks = sorted({seg[int(np.argmin(np.abs(s - t)))] for t in np.linspace(0, s[-1], k)})
    return dict(dump=dump, score=round(best[0], 1), path=round(float(s[-1]), 1), fps=meta.get("fps", 16),
                picks=[int(frames[q]) for q in picks], side=meta.get("side"))


def region_box(part, region):
    ys, xs = np.nonzero(np.isin(part, REGION[region]))
    if len(ys) < 50: return None
    if region == "chest": k = ys <= ys.min() + 0.6 * np.ptp(ys); ys, xs = ys[k], xs[k]
    return xs.min(), ys.min(), xs.max(), ys.max()


def track(anchors, d):
    """forward (part, uv) nearest neighbour; None when the point is not visible."""
    part = d["part"].astype(int); U, V = d["uv"][0].astype(np.float32), d["uv"][1].astype(np.float32); out = []
    for p, u, v in anchors:
        ys, xs = np.nonzero(part == p)
        if len(ys) == 0: out.append(None); continue
        dd = (U[ys, xs] - u) ** 2 + (V[ys, xs] - v) ** 2; j = int(np.argmin(dd))
        out.append((float(xs[j]), float(ys[j])) if dd[j] < 0.02 ** 2 else None)
    return out


def build(a):
    dump = os.path.expanduser(a.dump); meta = json.load(open(f"{dump}/meta.json")); fps = meta.get("fps", 16)
    frames = a.frames or scan_one(dump, a.k)["picks"]; D = {f: np.load(f"{dump}/f_{f:04d}.npz") for f in frames}
    from orient.pad_util import valid_region
    parts = {f: np.where(valid_region(cv2.imread(f"{dump}/f_{f:04d}.jpg")), D[f]["part"].astype(int), 0) for f in frames}; H, W = parts[frames[0]].shape
    boxes = {f: region_box(parts[f], a.region) for f in frames}
    S = 1.15 * max(max(b[2] - b[0], b[3] - b[1]) for b in boxes.values() if b)
    pb = np.array([[*np.nonzero(parts[f].any(0))[0][[0, -1]], *np.nonzero(parts[f].any(1))[0][[0, -1]]] for f in frames])
    mg = 0.03 * H; union = [max(0, pb[:, 0].min() - mg), max(0, pb[:, 2].min() - mg), min(W, pb[:, 1].max() + mg), min(H, pb[:, 3].max() + mg)]
    ref = max(frames, key=lambda f: np.isin(parts[f], REGION[a.region]).sum())
    x0, y0, x1, y1 = boxes[ref]; cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
    ys, xs = np.nonzero(np.isin(parts[ref], REGION[a.region]) & (np.abs(np.arange(W)[None] - cx) < 0.28 * (x1 - x0)) & (np.abs(np.arange(H)[:, None] - cy) < 0.28 * (y1 - y0)))
    sel = [int(np.argmin((xs - cx) ** 2 + (ys - cy) ** 2))]                               # farthest-point anchors from the centre
    for _ in range(a.anchors - 1):
        dmin = np.min([(xs - xs[s]) ** 2 + (ys - ys[s]) ** 2 for s in sel], 0); sel.append(int(np.argmax(dmin)))
    U, V = D[ref]["uv"][0], D[ref]["uv"][1]
    anchors = [(int(parts[ref][ys[s], xs[s]]), float(U[ys[s], xs[s]]), float(V[ys[s], xs[s]])) for s in sel]
    tag = f"{os.path.basename(dump.rstrip('/'))}_{a.region}"; jobs, info = [], {}
    for f in frames:
        b = boxes[f]; tr = track(anchors, dict(part=parts[f], uv=D[f]["uv"])); vis = [k for k, t in enumerate(tr) if t is not None]
        if b is None: bx = by = None
        else: bx, by = (b[0] + b[2]) / 2, (b[1] + b[3]) / 2
        if b is None and vis: bx, by = np.mean([tr[k] for k in vis], 0)
        roi = None if bx is None else [max(0, bx - S / 2), max(0, by - S / 2), min(W, bx + S / 2), min(H, by + S / 2)]
        full, zoom = f"{FB}/seq_{tag}/f{f:04d}", f"{FB}/seq_{tag}/f{f:04d}_zoom"
        jobs.append((full, ["--step-m", str(a.step_full), "--roi", *map(str, union)], a.scale_full, []))
        if roi:
            ex = [v for k in vis for v in tr[k]]
            jobs.append((zoom, ["--step-m", str(a.step_zoom), "--roi", *map(str, roi), "--parts", *map(str, REGION[a.region])] + (["--extra", *map(str, ex)] if ex else []),
                         a.scale_zoom, ["--frame", "glyphs"] if f == ref else None))
        info[f] = dict(full=full, zoom=zoom if roi else None, roi=roi, vis=vis)
    print(f"[seq] {tag} frames {frames} ref {ref} anchors {anchors} S {S:.0f}px", flush=True)
    procs = []
    for d, args, sc, targs in jobs:
        if not a.skip_export:
            r = subprocess.run([PY, "-m", "orient.framefield_boxes", "--dump", dump, "--frame", os.path.basename(d)[1:5].lstrip("0") or "0",
                                "--box", "0.8", "0.5", "0.3", *args, "--out", d], cwd=ROOT, capture_output=True, text=True)
            if r.returncode: print(r.stdout[-600:], r.stderr[-1500:]); sys.exit(f"export failed {d}")
        if a.skip_render: continue
        procs.append(subprocess.Popen(BL + ["image", d, "--scale", str(sc), "--extra-scale", str(a.extra_scale)] + IMG, stdout=open(f"{d}/log_image_tri.txt", "w"), stderr=subprocess.STDOUT))
        if targs is not None and (d.endswith("_zoom") or f"f{ref:04d}" in d):
            procs.append(subprocess.Popen(BL + ["template", d, "--scale", "2", "--glyph", "tripod", "--samples", "96", "--extra-scale", str(a.extra_scale)] + targs,
                                          stdout=open(f"{d}/log_template_tri.txt", "w"), stderr=subprocess.STDOUT))
    if any(p.wait() for p in procs): sys.exit("a render failed")
    compose(a, frames, ref, info, fps, anchors)


def compose(a, frames, ref, info, fps, anchors):
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle, Circle
    from PIL import Image
    K = len(frames); G, LAB = 22, 60; Pw = (a.width - G * K) / (K + 1)
    rd = lambda p: np.asarray(Image.open(p).convert("RGBA"))
    def onwhite(im):
        bg = np.full(im.shape[:2] + (3,), 255.0); al = im[..., 3:4] / 255.0; return (im[..., :3] * al + bg * (1 - al)).astype(np.uint8)
    def fit(im, w, h):
        s = min(w / im.shape[1], h / im.shape[0]); return np.asarray(Image.fromarray(im).resize((max(1, int(im.shape[1] * s)), max(1, int(im.shape[0] * s))), Image.LANCZOS)), s
    def tight(im, pad=16):
        ys, xs = np.nonzero(im[..., 3] > 8); b = (max(0, xs.min() - pad), max(0, ys.min() - pad), min(im.shape[1], xs.max() + pad), min(im.shape[0], ys.max() + pad))
        return im[b[1]:b[3], b[0]:b[2]], b
    F0 = rd(f"{info[frames[0]]['full']}/render_image_tripod.png"); h1 = Pw * F0.shape[0] / F0.shape[1]; h2 = Pw
    Ht = int(LAB + h1 + G + h2 + LAB); Wt = a.width
    fig = plt.figure(figsize=(Wt / 100, Ht / 100), dpi=100, facecolor="white")
    def put(im, x, y, w, h, edge=None):
        ax = fig.add_axes([x / Wt, (Ht - y - h) / Ht, w / Wt, h / Ht]); ax.imshow(im); ax.set_xlim(-0.5, im.shape[1] - 0.5); ax.set_ylim(im.shape[0] - 0.5, -0.5)
        ax.set_xticks([]); ax.set_yticks([])
        for sp in ax.spines.values(): sp.set_visible(edge is not None); sp.set_edgecolor(edge or "k"); sp.set_linewidth(2.5)
        return ax
    # column 0: the template (fixed)
    tz_dir = info[ref]["zoom"]; tf, _ = tight(rd(f"{info[ref]['full']}/render_template_tripod.png"))
    im, _ = fit(onwhite(tf), Pw, h1); put(im, (Pw - im.shape[1]) / 2, LAB + (h1 - im.shape[0]) / 2, im.shape[1], im.shape[0])
    tz_raw = rd(f"{tz_dir}/render_template_tripod.png"); tz, tb = tight(tz_raw); im, s = fit(onwhite(tz), Pw, h2)
    ox, oy = (Pw - im.shape[1]) / 2, LAB + h1 + G + (h2 - im.shape[0]) / 2; ax = put(im, ox, oy, im.shape[1], im.shape[0], "#9ca3af")
    zm = json.load(open(f"{tz_dir}/meta.json")); XY = np.array(json.load(open(f"{tz_dir}/screen_template_tripod.json"))["xy"])
    n, ne = zm["n"], zm.get("n_extra", 0); vis = info[ref]["vis"]
    fig.text(Pw / 2 / Wt, 1 - 30 / Ht, "template (fixed)", ha="center", va="center", fontsize=22)
    fig.lines.append(plt.Line2D([(Pw + G / 2) / Wt] * 2, [0.03, 0.97], transform=fig.transFigure, color="#d1d5db", lw=2))
    t0 = frames[0]
    for c, f in enumerate(frames):
        x = (c + 1) * (Pw + G); I = info[f]
        im, s1 = fit(rd(f"{I['full']}/render_image_tripod.png")[..., :3], Pw, h1); ax = put(im, x, LAB, im.shape[1], im.shape[0])
        fm = json.load(open(f"{I['full']}/meta.json")); v = fm.get("valid") or [0, 0, 1e9, 1e9]; r0 = [max(fm["roi"][0], v[0]), max(fm["roi"][1], v[1]), min(fm["roi"][2], v[2]), min(fm["roi"][3], v[3])]; k1 = im.shape[1] / (r0[2] - r0[0])
        if I["roi"]:
            r = I["roi"]; ax.add_patch(Rectangle(((r[0] - r0[0]) * k1, (r[1] - r0[1]) * k1), (r[2] - r[0]) * k1, (r[3] - r[1]) * k1, fill=False, ec="#6b7280", lw=2))
        fig.text((x + Pw / 2) / Wt, 1 - 30 / Ht, f"t = {(f - t0) / fps:.2f} s", ha="center", va="center", fontsize=22)
        if not I["zoom"]: continue
        zr = rd(f"{I['zoom']}/render_image_tripod.png")[..., :3]; im, s2 = fit(zr, Pw, h2)
        ax = put(im, x + (Pw - im.shape[1]) / 2, LAB + h1 + G + (h2 - im.shape[0]) / 2, im.shape[1], im.shape[0], "#9ca3af")
        zm = json.load(open(f"{I['zoom']}/meta.json")); XY = np.array(json.load(open(f"{I['zoom']}/screen_image_tripod.json"))["xy"]); n, ne = zm["n"], zm.get("n_extra", 0)
    fig.text(0.5, 18 / Ht, "enlarged frames = the same surface points over time: fixed on the template, rotating in the image", ha="center", va="center", fontsize=20, color="#374151")
    fig.savefig(a.out, dpi=100, facecolor="white"); print("saved", a.out, (Wt, Ht), flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); sp = ap.add_subparsers(dest="cmd", required=True)
    s1 = sp.add_parser("scan"); s1.add_argument("dumps", nargs="+"); s1.add_argument("--k", type=int, default=6); s1.add_argument("--stride", type=int, default=2)
    s1.add_argument("--json", default=None)
    s2 = sp.add_parser("build"); s2.add_argument("dump"); s2.add_argument("out"); s2.add_argument("--k", type=int, default=6)
    s2.add_argument("--frames", type=int, nargs="+"); s2.add_argument("--region", default="chest", choices=list(REGION))
    s2.add_argument("--anchors", type=int, default=3); s2.add_argument("--width", type=int, default=3400)
    s2.add_argument("--step-full", type=float, default=0.13); s2.add_argument("--step-zoom", type=float, default=0.055); s2.add_argument("--extra-scale", type=float, default=2.4)   # metres on the body
    s2.add_argument("--scale-full", type=float, default=2); s2.add_argument("--scale-zoom", type=float, default=6)
    s2.add_argument("--skip-export", action="store_true"); s2.add_argument("--skip-render", action="store_true")
    a = ap.parse_args()
    if a.cmd == "scan":
        res = []
        for dmp in a.dumps:
            try: r = scan_one(os.path.expanduser(dmp), a.k, a.stride)
            except Exception as e: r = dict(dump=dmp, score=-1, err=str(e)[:200])
            print(json.dumps(r), flush=True); res.append(r)
        if a.json: json.dump(sorted(res, key=lambda r: -r["score"]), open(a.json, "w"), indent=1)
    else:
        build(a)

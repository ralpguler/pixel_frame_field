"""Template-vs-image teaser for ANY field-dump frame, end to end: auto zoom ROIs -> framefield_boxes exports ->
Blender renders (tripods, white page) -> fig_teaser_pairs.
  python -m orient.teaser_auto --dump DIR --frame F --tag NAME --fig OUT.png [--zooms head chest joint] [--skip-render]
Zoom picks (from the dump's own part map): head = parts 23,24; chest = torso parts 1,2, top 60% of their height;
joint = the knee/elbow (upper+lower limb parts of one side) with the most visible pixels around the part contact."""
import argparse, json, os, subprocess, sys, numpy as np, cv2

HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(HERE)
PY = os.path.join(os.path.dirname(ROOT), ".venv/bin/python")
FB = os.path.expanduser("~/data/dp_viz_compare/framefield_boxes")
BL = ["apptainer", "exec", "-B", "/lus/lfs1aip2,/projects", os.path.expanduser("~/data/containers/blender.sif"),
      "blender", "--background", "--python", f"{HERE}/blender_boxes.py", "--"]
IMG = "--glyph tripod --samples 96 --bg bg.png --dark 0.45 --key 22 --emit 0.02".split()
JOINTS = {"right knee": ([7, 9], [11, 13]), "left knee": ([8, 10], [12, 14])}   # no elbows: on the T-pose template an elbow zoom is a strip


def box(ys, xs, W, H, pad=0.10, min_aspect=0.55):
    x0, x1, y0, y1 = xs.min(), xs.max(), ys.min(), ys.max(); w, h = x1 - x0, y1 - y0
    if w < min_aspect * h: c = (x0 + x1) / 2; w = min_aspect * h; x0, x1 = c - w / 2, c + w / 2
    if h < min_aspect * w: c = (y0 + y1) / 2; h = min_aspect * w; y0, y1 = c - h / 2, c + h / 2
    p = pad * max(w, h)
    return [round(float(max(0, x0 - p)), 1), round(float(max(0, y0 - p)), 1), round(float(min(W, x1 + p)), 1), round(float(min(H, y1 + p)), 1)]


def pick_zooms(part, which):
    H, W = part.shape; ys_all = np.nonzero(part > 0)[0]; ph = float(np.ptp(ys_all)); out = []
    for z in which:
        if z == "head":
            ys, xs = np.nonzero(np.isin(part, [23, 24]))
            if len(ys) > 300: out.append(("head", [23, 24], box(ys, xs, W, H)))
        elif z == "chest":
            ys, xs = np.nonzero(np.isin(part, [1, 2]))
            if len(ys) > 600:
                k = ys <= ys.min() + 0.6 * np.ptp(ys); out.append(("chest", [1, 2], box(ys[k], xs[k], W, H, min_aspect=0.7)))
        elif z == "joint":
            best = None
            for name, (up, lo) in JOINTS.items():
                U, L = np.isin(part, up), np.isin(part, lo)
                c = cv2.dilate(U.astype(np.uint8), np.ones((7, 7), np.uint8)).astype(bool) & L
                if c.sum() < 10: continue
                cy, cx = np.median(np.nonzero(c), 1); r = 0.10 * ph
                ys, xs = np.nonzero(U | L); k = (np.abs(ys - cy) < r) & (np.abs(xs - cx) < r)
                if best is None or k.sum() > best[0]: best = (int(k.sum()), name, up + lo, ys[k], xs[k])
                if name == "left knee" and best and best[0] > 300: break
            if best and best[0] > 300: out.append((best[1], best[2], box(best[3], best[4], W, H, min_aspect=0.6)))
    return out


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--dump", required=True); ap.add_argument("--frame", type=int, required=True)
    ap.add_argument("--tag", required=True); ap.add_argument("--fig", required=True)
    ap.add_argument("--zooms", nargs="+", default=["head", "chest", "joint"]); ap.add_argument("--skip-render", action="store_true")
    ap.add_argument("--step-full", type=float, default=0.055); ap.add_argument("--step-zoom", type=float, default=0.035)   # metres on the body
    a = ap.parse_args(); dump = os.path.expanduser(a.dump)
    part = np.load(f"{dump}/f_{a.frame:04d}.npz")["part"].astype(int)
    from orient.pad_util import valid_region
    part = np.where(valid_region(cv2.imread(f"{dump}/f_{a.frame:04d}.jpg")), part, 0); zooms = pick_zooms(part, a.zooms)
    full = f"{FB}/{a.tag}"; jobs = [(full, ["--step-m", str(a.step_full)], 3, [])]
    for name, parts, roi in zooms:
        d = f"{full}_{name.replace(' ', '_')}"
        jobs.append((d, ["--step-m", str(a.step_zoom), "--roi", *map(str, roi), "--parts", *map(str, parts)], 8, ["--frame", "glyphs"]))
    print(f"[{a.tag}] zooms:", [(n, r) for n, _, r in zooms], flush=True)
    procs = []
    for d, args, sc, targs in jobs:
        r = subprocess.run([PY, "-m", "orient.framefield_boxes", "--dump", dump, "--frame", str(a.frame), "--box", "0.8", "0.5", "0.3",
                            *args, "--out", d], cwd=ROOT, capture_output=True, text=True)
        if r.returncode: print(r.stdout[-800:], r.stderr[-1500:]); sys.exit(f"export failed: {d}")
        if a.skip_render: continue
        procs.append(subprocess.Popen(BL + ["image", d, "--scale", str(sc)] + IMG, stdout=open(f"{d}/log_image_tri.txt", "w"), stderr=subprocess.STDOUT))
        procs.append(subprocess.Popen(BL + ["template", d, "--scale", "2", "--glyph", "tripod", "--samples", "96"] + targs,
                                      stdout=open(f"{d}/log_template_tri.txt", "w"), stderr=subprocess.STDOUT))
    bad = [p.args[10:12] for p in procs if p.wait()]
    if bad: sys.exit(f"render failed: {bad}")
    cmd = [PY, "-m", "orient.fig_teaser_pairs", full, a.fig]
    for (name, _, _), (d, _, _, _) in zip(zooms, jobs[1:]): cmd += ["--zoom", d, name]
    r = subprocess.run(cmd + ["--glyph", "tripod"], cwd=ROOT, capture_output=True, text=True); print(r.stdout.strip()[-300:], r.stderr[-800:], flush=True)


if __name__ == "__main__":
    main()

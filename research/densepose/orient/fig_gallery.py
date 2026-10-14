"""Gallery of pixel frame fields: per example the input crop and the RGB-cube render side by side, in a grid.
  python -m orient.fig_gallery OUT.{png,jpg} DIR1 DIR2 ... [--rows 2] [--height 700] [--glyph cube]
Each DIR is an orient/framefield_boxes.py export with render_image[_<glyph>].png from orient/blender_boxes.py.
"""
import argparse, json, os, numpy as np
from PIL import Image

ap = argparse.ArgumentParser(); ap.add_argument("out"); ap.add_argument("dirs", nargs="+")
ap.add_argument("--rows", type=int, default=2); ap.add_argument("--height", type=int, default=700); ap.add_argument("--glyph", default="cube")
a = ap.parse_args(); suf = "" if a.glyph == "box" else f"_{a.glyph}"
rs = lambda im, h: np.asarray(Image.fromarray(im).resize((max(1, int(im.shape[1] * h / im.shape[0])), h), Image.LANCZOS))
tiles = []
for D in a.dirs:
    D = os.path.expanduser(D); m = json.load(open(f"{D}/meta.json")); W, H = m["W"], m["H"]; x0, y0, x1, y1 = m["person_bbox"]; mg = 0.04 * H
    bx = (int(max(0, x0 - mg)), int(max(0, y0 - mg)), int(min(W, x1 + mg)), int(min(H, y1 + mg)))   # same window as the render border
    ph = rs(np.asarray(Image.open(f"{D}/bg.png").convert("RGB").crop(bx)), a.height)
    rr = rs(np.asarray(Image.open(f"{D}/render_image{suf}.png").convert("RGB")), a.height)
    tiles.append(np.concatenate([ph, np.full((a.height, 8, 3), 255, np.uint8), rr], 1))
# justified rows: split the tiles into --rows groups (order kept inside a row) with the most equal total widths,
# then scale every row to the same width (row heights differ slightly; no empty space)
import itertools
wd = [t.shape[1] for t in tiles]; n = len(tiles); nr = max(1, min(a.rows, n))
best = None
for lab in itertools.product(range(nr), repeat=n):
    if len(set(lab)) < nr: continue
    tot = [sum(w for w, l in zip(wd, lab) if l == r) + 36 * (lab.count(r) - 1) for r in range(nr)]
    sc = max(tot) - min(tot)
    if best is None or sc < best[0]: best = (sc, lab)
groups = [[t for t, l in zip(tiles, best[1]) if l == r] for r in range(nr)]
rows = []
for g in groups:
    parts = []
    for k, t in enumerate(g):
        parts += ([np.full((a.height, 36, 3), 255, np.uint8)] if k else []) + [t]
    rows.append(np.concatenate(parts, 1))
Wm = max(r.shape[1] for r in rows)
rows = [np.asarray(Image.fromarray(r).resize((Wm, int(r.shape[0] * Wm / r.shape[1])), Image.LANCZOS)) for r in rows]
out = []
for k, r in enumerate(rows):
    out += ([np.full((36, Wm, 3), 255, np.uint8)] if k else []) + [r]
Image.fromarray(np.concatenate(out, 0)).save(a.out, **({"quality": 92} if a.out.endswith(".jpg") else {})); print("saved", a.out)

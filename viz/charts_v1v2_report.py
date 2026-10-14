"""v1 vs v2 chart-space visual comparison -> self-contained HTML.

For each part: (a) CHART SPACE -- triangles rasterised into the UV unit square, coloured by
template XYZ, with a fold-severity overlay (|XYZ first claimant - XYZ last claimant| where >1
triangle claims a texel); (b) ON THE BODY -- part vertices in a front/back split (PCA thickness
axis), coloured by an (u,v) checker colourmap, so a fold shows as the SAME colours/checker cells
appearing on both sides of the limb.

Outputs panels + stats to ~/data/dp_viz_compare/charts_html/ and writes charts_v1v2.html there.

  python -m orient.charts_v1v2_report
"""
import os  # noqa: E402
import sys  # noqa: E402
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import _env  # noqa: E402  (repository paths: viz/_env.py)
_env.take_out_arg()
import base64
import os

import cv2
import numpy as np
from scipy.spatial import cKDTree

A = _env.ASSETS
OUTD = _env.out("charts_html")
os.makedirs(OUTD, exist_ok=True)
K = 640
PARTS = [(1, "torso front (control)"), (4, "right hand"), (3, "left hand"),
         (6, "right foot"), (5, "left foot"), (23, "head half")]

UV1 = np.load(f"{A}/smplx_face_uv.npy").astype(np.float64)          # (F,3,2) official v1
UV2 = np.load(f"{A}/smplx_face_uv_v2.npy").astype(np.float64)       # (F,3,2) fold-free v2
fpart = np.load(f"{A}/smplx_face_to_dp24.npy").astype(int)
faces = np.load(f"{A}/smplx_faces.npy").astype(int)
vt = np.load(f"{A}/smplx_v_template.npy").astype(np.float64)
hidden = np.load(f"{A}/smplx_hidden_faces.npy")

# per-vertex normals (area-weighted)
fn = np.cross(vt[faces[:, 1]] - vt[faces[:, 0]], vt[faces[:, 2]] - vt[faces[:, 0]])
vn = np.zeros_like(vt)
for c in range(3):
    np.add.at(vn, faces[:, c], fn)
vn /= np.clip(np.linalg.norm(vn, axis=1, keepdims=True), 1e-12, None)


def xyz_color(fm):
    """Per-face RGB from template centroid, normalised inside the part's bbox."""
    c = vt[faces[fm]].mean(1)
    mn, mx = c.min(0), c.max(0)
    return ((c - mn) / np.clip(mx - mn, 1e-9, None) * 220 + 25).astype(np.uint8)


def chart_panel(uvset, fm, col):
    """Rasterise the chart. Returns (colour image, fold-severity mm image, count)."""
    img = np.full((K, K, 3), 18, np.uint8)
    cnt = np.zeros((K, K), np.int16)
    first = np.full((K, K, 3), np.nan, np.float32)
    last = np.full((K, K, 3), np.nan, np.float32)
    cxyz = vt[faces[fm]].mean(1).astype(np.float32)
    for i, f in enumerate(fm):
        pts = np.round(uvset[f] * (K - 1)).astype(np.int32)
        m = np.zeros((K, K), np.uint8)
        cv2.fillPoly(m, [pts], 1)
        mb = m.astype(bool)
        img[mb] = col[i]
        cnt[mb] += 1
        fmask = mb & np.isnan(first[..., 0])
        first[fmask] = cxyz[i]
        last[mb] = cxyz[i]
    sev = np.linalg.norm(np.nan_to_num(first) - np.nan_to_num(last), axis=-1) * 1000  # mm
    sev[cnt < 2] = 0
    return img, sev, cnt


def body_panels(uvset, p, fm):
    """Front/back scatter of the part's surface, coloured by (u,v) checker map."""
    vs = np.unique(faces[fm])
    # mean uv per vertex over its face-corners in this part
    acc = np.zeros((len(vt), 2)); n = np.zeros(len(vt))
    for c in range(3):
        np.add.at(acc, faces[fm][:, c], uvset[fm][:, c]); np.add.at(n, faces[fm][:, c], 1)
    uvv = acc[vs] / n[vs, None]
    P = vt[vs] - vt[vs].mean(0)
    _, _, W = np.linalg.svd(P, full_matrices=False)
    a1, a2, a3 = W                                       # a3 = thickness axis
    x, y = P @ a1, P @ a2
    side = (vn[vs] @ a3) >= 0
    hsv = np.stack([uvv[:, 0] * 179, np.full(len(vs), 230.0),
                    70 + 165 * uvv[:, 1]], 1).astype(np.uint8)
    rgb = cv2.cvtColor(hsv[None], cv2.COLOR_HSV2RGB)[0].astype(np.float32)
    check = ((np.floor(uvv[:, 0] * 8) + np.floor(uvv[:, 1] * 8)) % 2).astype(bool)
    rgb[check] *= 0.55
    rgb = rgb.astype(np.uint8)
    S = 560
    xs = ((x - x.min()) / max(x.ptp(), 1e-9) * (S - 70) + 35).astype(int)
    ys = ((y - y.min()) / max(y.ptp(), 1e-9) * (S - 70) + 35).astype(int)
    out = []
    for sgn in (side, ~side):
        im = np.full((S, S, 3), 18, np.uint8)
        for xi, yi, ci in zip(xs[sgn], S - 1 - ys[sgn], rgb[sgn]):
            cv2.circle(im, (xi, yi), 5, tuple(int(v) for v in ci), -1)
        out.append(im)
    return out


def fold_stats(uvset, fm, nsamp=6000, seed=3):
    """% of surface samples whose (u,v) is claimed by another NON-ADJACENT triangle, and the
    median 3D collision distance among folded samples (the validated fold-census method)."""
    rng = np.random.RandomState(seed)
    tri = vt[faces[fm]]
    area = np.linalg.norm(np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0]), axis=1)
    pick = rng.choice(len(fm), nsamp, p=area / area.sum())
    r = np.sort(rng.rand(nsamp, 2), 1)
    bc = np.stack([r[:, 0], r[:, 1] - r[:, 0], 1 - r[:, 1]], 1)
    q = (bc[:, :, None] * uvset[fm][pick]).sum(1)
    x3 = (bc[:, :, None] * tri[pick]).sum(1)
    tuv = uvset[fm]
    tree = cKDTree(tuv.mean(1))
    kk = min(40, len(fm))
    _, cand = tree.query(q, k=kk)
    a, b, c = tuv[cand][:, :, 0], tuv[cand][:, :, 1], tuv[cand][:, :, 2]
    v0, v1v, v2v = b - a, c - a, q[:, None] - a
    d00 = (v0 * v0).sum(-1); d01 = (v0 * v1v).sum(-1); d11 = (v1v * v1v).sum(-1)
    d20 = (v2v * v0).sum(-1); d21 = (v2v * v1v).sum(-1)
    den = np.where(np.abs(d00 * d11 - d01 * d01) < 1e-14, 1e-14, d00 * d11 - d01 * d01)
    w1 = (d11 * d20 - d01 * d21) / den
    w2 = (d00 * d21 - d01 * d20) / den
    inside = (w1 >= -0.01) & (w2 >= -0.01) & (w1 + w2 <= 1.01)
    sh = np.sort(faces[fm], 1)
    own = sh[pick]
    fold = np.zeros(nsamp, bool); dist = np.zeros(nsamp)
    for i in range(nsamp):
        cl = cand[i][inside[i]]
        cl = cl[cl != pick[i]]
        if not len(cl):
            continue
        adj = (np.isin(sh[cl], own[i]).sum(1)) >= 1        # shares a vertex -> mesh-adjacent
        cl = cl[~adj]
        if not len(cl):
            continue
        w = np.clip(np.stack([1 - w1[i] - w2[i], w1[i], w2[i]])[None], 0, 1)  # rough
        oxyz = tri[cl].mean(1)
        d = np.linalg.norm(oxyz - x3[i], axis=1)
        if d.max() > 0.02:
            fold[i] = True; dist[i] = d.max()
    return 100 * fold.mean(), (np.median(dist[fold]) * 1000 if fold.any() else 0.0)


def save(name, im):
    cv2.imwrite(f"{OUTD}/{name}.png", im[..., ::-1])
    return name


def b64(name):
    return "data:image/png;base64," + base64.b64encode(open(f"{OUTD}/{name}.png", "rb").read()).decode()


rows = []
for p, label in PARTS:
    fm = np.where(fpart == p)[0]
    fmv = fm[~hidden[fm]] if p in (23, 24) else fm       # head: show skin-only too
    col1 = xyz_color(fm)
    r = {"p": p, "label": label}
    for tag, uvset in (("v1", UV1), ("v2", UV2)):
        img, sev, cnt = chart_panel(uvset, fm, col1)
        heat = np.zeros((K, K, 3), np.uint8)
        heat[..., 0] = np.clip(sev * 1.4, 0, 255).astype(np.uint8)
        over = img.copy()
        m = sev > 20
        over[m] = (0.35 * over[m] + 0.65 * np.array([255, 40, 40])).astype(np.uint8)
        r[f"chart_{tag}"] = save(f"p{p}_{tag}_chart", img)
        r[f"fold_{tag}"] = save(f"p{p}_{tag}_fold", over)
        f, b = body_panels(uvset, p, fm)
        r[f"front_{tag}"] = save(f"p{p}_{tag}_front", f)
        r[f"back_{tag}"] = save(f"p{p}_{tag}_back", b)
        r[f"pct_{tag}"], r[f"mm_{tag}"] = fold_stats(uvset, fm)
    rows.append(r)
    print(f"part {p:2d} {label:22s} v1 fold {r['pct_v1']:5.1f}% / {r['mm_v1']:5.0f}mm   "
          f"v2 fold {r['pct_v2']:5.1f}% / {r['mm_v2']:5.0f}mm", flush=True)

CSS = """
:root{--bg:#0c0e13;--panel:#141822;--line:#232838;--tx:#d5d2ca;--mut:#8a8f9c;--acc:#b16cea;
--gold:#d9a441;--bad:#c96a5a;--good:#7fb069}
*{box-sizing:border-box;margin:0}
body{background:var(--bg);color:var(--tx);font:16px/1.6 system-ui,sans-serif;padding:0 20px 90px}
.wrap{max-width:1080px;margin:0 auto}
header{padding:56px 0 18px;border-bottom:1px solid var(--line)}
.eyebrow{font:600 12px/1 ui-monospace,monospace;letter-spacing:.22em;color:var(--acc);text-transform:uppercase}
h1{font-family:Palatino,Georgia,serif;font-size:clamp(28px,4.5vw,42px);margin:10px 0 8px}
.sub{color:var(--mut);max-width:70ch}
section{margin-top:52px}
h2{font-family:Palatino,Georgia,serif;font-size:24px;margin-bottom:4px}
.no{font:600 11px/1 ui-monospace,monospace;color:var(--mut);letter-spacing:.2em}
p{max-width:76ch}
.grid{display:grid;grid-template-columns:repeat(4,1fr);gap:10px;margin-top:14px}
@media(max-width:800px){.grid{grid-template-columns:repeat(2,1fr)}}
.cell{background:var(--panel);border:1px solid var(--line);border-radius:8px;overflow:hidden}
.cell img{width:100%;display:block}
.cell .cap{font:11px/1.5 ui-monospace,monospace;color:var(--mut);padding:6px 9px}
.stats{display:flex;gap:12px;margin-top:12px;flex-wrap:wrap}
.stat{background:var(--panel);border:1px solid var(--line);border-radius:8px;padding:10px 16px}
.stat .k{font:600 10.5px/1 ui-monospace,monospace;color:var(--mut);letter-spacing:.12em;text-transform:uppercase}
.stat .n{font-family:Palatino,serif;font-size:24px;margin-top:3px}
.bad{color:var(--bad)}.good{color:var(--good)}
.note{color:var(--mut);font-size:14px;max-width:76ch;margin-top:10px}
"""

body = f"""<div class="wrap">
<header>
<div class="eyebrow">DensePose charts · v1 vs v2</div>
<h1>What the fold looks like</h1>
<p class="sub">v1 = the official DensePose charts transferred to SMPL-X. v2 = our re-flattening
(Tutte/mean-value embedding) of the four parts where v1 is non-injective: both hands and both
feet. Everything else is bit-identical between conventions. Chart squares are coloured by
template XYZ; <b>red overlay = texels claimed by 2+ non-adjacent triangles</b> (fold), with
severity = 3D distance between claimants. Body views split each part along its thickness axis
and paint the surface by an (u,v) hue&times;brightness checker — a fold makes the SAME colours and
checker cells appear on BOTH sides of the limb.</p>
</header>"""

for r in rows:
    v1c, v2c = ("bad", "good") if r["pct_v1"] > 1 else ("good", "good")
    body += f"""
<section>
<div class="no">PART {r['p']}</div><h2>{r['label']}</h2>
<div class="stats">
<div class="stat"><div class="k">v1 folded surface</div><div class="n {v1c}">{r['pct_v1']:.1f}%</div></div>
<div class="stat"><div class="k">v1 median collision</div><div class="n {v1c}">{r['mm_v1']:.0f} mm</div></div>
<div class="stat"><div class="k">v2 folded surface</div><div class="n {v2c}">{r['pct_v2']:.1f}%</div></div>
<div class="stat"><div class="k">v2 median collision</div><div class="n {v2c}">{r['mm_v2']:.0f} mm</div></div>
</div>
<div class="grid">
<div class="cell"><img src="{b64(r['fold_v1'])}"><div class="cap">v1 chart · fold in red</div></div>
<div class="cell"><img src="{b64(r['fold_v2'])}"><div class="cap">v2 chart · fold in red</div></div>
<div class="cell"><img src="{b64(r['front_v1'])}"><div class="cap">v1 on body · side A</div></div>
<div class="cell"><img src="{b64(r['back_v1'])}"><div class="cap">v1 on body · side B</div></div>
<div class="cell"><img src="{b64(r['chart_v1'])}"><div class="cap">v1 chart · XYZ colour</div></div>
<div class="cell"><img src="{b64(r['chart_v2'])}"><div class="cap">v2 chart · XYZ colour</div></div>
<div class="cell"><img src="{b64(r['front_v2'])}"><div class="cap">v2 on body · side A</div></div>
<div class="cell"><img src="{b64(r['back_v2'])}"><div class="cap">v2 on body · side B</div></div>
</div>
</section>"""

body += """
<section>
<div class="no">WHY IT MATTERS</div><h2>Consequences</h2>
<p>1 · <b>GT labels</b>: on a folded chart two surface points share one (part,u,v) — a regression
target that is a coin flip. This capped hand/foot accuracy until the v2 re-flattening
(training on v2 = the A/C/D/E runs). 2 · <b>Garment texture atlases</b> (MCLD-style): the
appearance texture is built by warping pixels into chart space — under v1, palm and back-of-hand
pixels land on the SAME texels and overwrite each other; no correspondence model, however good,
can undo that. 3 · <b>COCO eval</b>: annotations are v1-convention, so v2 models are scored
through the exact v2&rarr;v1 remap (worth 4.7 AP). The head keeps its official chart in v2; its
residual red is SMPL-X hidden geometry (eyeballs, mouth interior) — handled separately by the
hidden-faces training mask, not by re-flattening.</p>
<p class="note">Generated by orient/charts_v1v2_report.py · assets smplx_face_uv{,_v2}.npy ·
fold census: 6k area-weighted surface samples/part, non-adjacent claimant &gt;20mm.</p>
</section>
</div>"""

html = f"<title>DensePose charts: v1 vs v2</title><style>{CSS}</style>{body}"
with open(f"{OUTD}/charts_v1v2.html", "w") as f:
    f.write(html)
print(f"wrote {OUTD}/charts_v1v2.html ({os.path.getsize(f'{OUTD}/charts_v1v2.html')/1e6:.1f} MB)")

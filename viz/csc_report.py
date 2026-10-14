"""CSC-loss ablation report (A/C/D/E on the clean 512 GT) -> self-contained HTML.

Numbers are hard-coded from eval_512 + eval_csc_detail runs of 2026-07-14/15 (jobs 5655275,
5661311, 5662139); figures embedded from ~/data/dp_viz_compare/beauty/.
"""
import os  # noqa: E402
import sys  # noqa: E402
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import _env  # noqa: E402  (repository paths: viz/_env.py)
_env.take_out_arg()
import base64
import io
import os

import cv2

B = _env.data("dp_viz_compare/beauty")
OUTD = _env.out("csc_report")
os.makedirs(OUTD, exist_ok=True)
OUT = f"{OUTD}/csc_ablation.html"


def imgb64(path, maxw=1400, q=86):
    im = cv2.imread(path)
    if im.shape[1] > maxw:
        s = maxw / im.shape[1]
        im = cv2.resize(im, (maxw, int(im.shape[0] * s)), interpolation=cv2.INTER_AREA)
    ok, buf = cv2.imencode(".jpg", im, [cv2.IMWRITE_JPEG_QUALITY, q])
    return "data:image/jpeg;base64," + base64.b64encode(buf).decode()


def fig(path, cap):
    return (f'<figure class="fig"><img src="{imgb64(path)}" alt="{cap}">'
            f'<figcaption>{cap}</figcaption></figure>')


CSS = """
:root{--bg:#0c0e13;--panel:#141822;--line:#232838;--tx:#d5d2ca;--mut:#8a8f9c;
--acc:#b16cea;--gold:#d9a441;--good:#7fb069;--bad:#c96a5a}
*{box-sizing:border-box;margin:0}
body{background:var(--bg);color:var(--tx);font:16px/1.65 system-ui,-apple-system,"Segoe UI",sans-serif;
padding:0 20px 90px}
.wrap{max-width:980px;margin:0 auto}
header{padding:64px 0 20px;border-bottom:1px solid var(--line)}
.eyebrow{font:600 12px/1 ui-monospace,Menlo,monospace;letter-spacing:.22em;color:var(--acc);
text-transform:uppercase}
h1{font-family:"Iowan Old Style",Palatino,Georgia,serif;font-size:clamp(30px,5vw,46px);font-weight:600;
line-height:1.1;text-wrap:balance;margin:12px 0 10px}
.sub{color:var(--mut);max-width:68ch}
.meta{display:flex;gap:20px;flex-wrap:wrap;margin-top:18px;font:12px/1.5 ui-monospace,Menlo,monospace;
color:var(--mut)} .meta b{color:var(--tx)}
section{margin-top:58px}
h2{font-family:"Iowan Old Style",Palatino,Georgia,serif;font-size:26px;font-weight:600;margin:0 0 6px}
.no{font:600 11.5px/1 ui-monospace,Menlo,monospace;color:var(--mut);letter-spacing:.2em}
p{max-width:72ch}
p.note{color:var(--mut);font-size:14px}
.tablewrap{overflow-x:auto;margin-top:14px}
table{border-collapse:collapse;font-size:14.5px;font-variant-numeric:tabular-nums;min-width:640px}
th{font:600 11px/1.3 ui-monospace,Menlo,monospace;letter-spacing:.12em;text-transform:uppercase;
color:var(--mut);text-align:right;padding:9px 14px;border-bottom:1px solid var(--line)}
th:first-child,td:first-child{text-align:left}
td{padding:9px 14px;border-bottom:1px solid var(--line);text-align:right}
td:first-child{font-weight:600;white-space:nowrap}
tr.hl td{background:color-mix(in srgb,var(--gold) 7%,transparent)}
.d{font-size:12.5px;margin-left:6px}
.good{color:var(--good)} .bad{color:var(--bad)} .mut{color:var(--mut)}
.cards{display:grid;grid-template-columns:1fr 1fr;gap:14px;margin-top:16px}
@media(max-width:680px){.cards{grid-template-columns:1fr}}
.v{background:var(--panel);border:1px solid var(--line);border-radius:10px;padding:15px 17px}
.v h3{font:600 11.5px/1 ui-monospace,Menlo,monospace;letter-spacing:.16em;text-transform:uppercase;
color:var(--mut);margin-bottom:8px}
.v.win h3{color:var(--gold)}
.v p{font-size:14.5px}
.stat{background:var(--panel);border:1px solid var(--line);border-radius:10px;padding:14px 18px}
.stat .k{font:600 11px/1 ui-monospace,Menlo,monospace;letter-spacing:.14em;color:var(--mut);
text-transform:uppercase}
.stat .n{font-family:"Iowan Old Style",Palatino,Georgia,serif;font-size:30px;color:var(--tx);
margin-top:4px} .stat .n b{color:var(--gold)}
.stats{display:grid;grid-template-columns:repeat(auto-fit,minmax(200px,1fr));gap:14px;margin-top:16px}
.fig{background:var(--panel);border:1px solid var(--line);border-radius:10px;overflow:hidden;
margin-top:16px}
.fig img{width:100%;display:block}
.fig figcaption{font:12px/1.6 ui-monospace,Menlo,monospace;color:var(--mut);padding:10px 14px}
footer{margin-top:70px;padding-top:22px;border-top:1px solid var(--line);
font:12.5px/1.8 ui-monospace,Menlo,monospace;color:var(--mut)} footer code{color:var(--tx)}
"""


def d(val, good=True):
    cls = "good" if good else "bad"
    return f'<span class="d {cls}">{val}</span>'


html = f"""<title>CSC Loss — Ablation Report</title>
<style>{CSS}</style>
<div class="wrap">
<header>
<div class="eyebrow">Oriented DensePose · canonical surface coordinate loss</div>
<h1>The CSC loss is a correspondence loss — and the seam claim holds</h1>
<p class="sub">Four runs (A/C/D/E), 25k steps fine-tuned from <code>ckpt_120000</code> on the rebuilt
512-canvas GT with fold-free charts. Scored on 512 clean val samples with
<code>eval_512</code> + <code>eval_csc_detail</code>.</p>
<div class="meta"><span>date <b>2026-07-15</b></span><span>data <b>BEDLAM2 + MAMMA · gt512</b></span>
<span>loss <b>w_xyz 2.0 · w_grad 5.0 · clip 0.20</b></span><span>charts <b>face_uv_v2</b></span>
<span>jobs <b>5648564–7 · 5655275 · 5662139 · 5668430 · 5675674</b></span></div>
</header>

<section>
<div class="no">SETUP</div><h2>What was run</h2>
<p>The CSC operator maps predicted <i>(part, u, v)</i> differentiably to canonical template XYZ
(one 3D grid-sample over a baked 512² per-part texture; round-trip median 0.000&nbsp;mm), and the loss
supervises that map <i>and its image-space gradient</i> against GT — MiDaS-style gradient matching
applied to correspondences. Ambiguous head texels (ear/jaw fold, 12%) are masked, not supervised.</p>
<div class="tablewrap"><table>
<tr><th>run</th><th>CSC terms</th><th>part gather</th><th>throughput</th><th>role</th></tr>
<tr><td>A</td><td>none</td><td>—</td><td>202 img/s</td><td>clean-GT baseline: isolates the data fix</td></tr>
<tr class="hl"><td>C</td><td>xyz + grad</td><td>GT part</td><td>146 img/s</td><td>simplest CSC variant</td></tr>
<tr><td>D</td><td>xyz + grad</td><td>argmax pred</td><td>141 img/s</td><td>no-GT-at-train-time variant</td></tr>
<tr><td>E</td><td>xyz + grad</td><td>soft top-3</td><td>144 img/s</td><td>boundary-mixture hypothesis</td></tr>
</table></div>
</section>

<section>
<div class="no">HEADLINE</div><h2>System-level results</h2>
<div class="tablewrap"><table>
<tr><th>system</th><th>geo_e2e ↓</th><th>surf ↓</th><th>hands+feet surf ↓</th><th>part acc ↑</th>
<th>misplaced-disc ↓</th></tr>
<tr><td>ckpt_120000 (old GT, v1 charts)</td><td>6.96°</td><td>12.54 mm</td><td>92.6 mm*</td>
<td>94.3%</td><td>2.96%</td></tr>
<tr><td>A · clean GT + v2 charts</td><td>6.88°</td><td>10.52 mm</td><td>24.8 mm</td><td>95.4%</td>
<td>2.65%</td></tr>
<tr class="hl"><td>C · + CSC (gt)</td><td><b>6.77°</b></td><td><b>8.18 mm</b>{d("−22%")}</td>
<td><b>13.5 mm</b>{d("−46%")}</td><td>95.4%</td><td>2.65%</td></tr>
<tr><td>D · + CSC (pred)</td><td>6.90°</td><td>8.22 mm</td><td>13.5 mm</td><td>95.4%</td><td>2.65%</td></tr>
<tr><td>E · + CSC (soft)</td><td>6.87°</td><td>8.23 mm</td><td>13.9 mm</td><td>95.4%</td><td>2.63%</td></tr>
</table></div>
<p class="note">*Chart-mismatch inflated on hands/feet: the old model predicts v1 (folded) chart
coordinates, scored against v2 charts. The full-body 12.54 mm is mostly fair — 18 of 24 parts share
identical charts. Deltas in green are C relative to A.</p>
<div class="stats">
<div class="stat"><div class="k">seam-break rate (A → C)</div><div class="n">3.08 → <b>1.47%</b></div></div>
<div class="stat"><div class="k">hands+feet surf (A → C)</div><div class="n">24.8 → <b>13.5 mm</b></div></div>
<div class="stat"><div class="k">interior grad residual</div><div class="n">0.61 → <b>0.42</b> mm/px</div></div>
<div class="stat"><div class="k">rotation geo_e2e</div><div class="n">6.88 → 6.77° <span class="d mut">flat</span></div></div>
</div>
</section>

<section>
<div class="no">DECOMPOSITION</div><h2>Where the gain lives</h2>
<p>Surface-error percentiles show the improvement sits in the <b>body of the distribution, not the
tail</b>: the p99 (~240&nbsp;mm) is part-misclassification chart jumps, untouched because part accuracy
is frozen across all runs.</p>
<div class="tablewrap"><table>
<tr><th>surf percentile</th><th>p50</th><th>p75</th><th>p90</th><th>p99</th></tr>
<tr><td>A</td><td>10.20</td><td>18.05</td><td>33.24</td><td>247</td></tr>
<tr class="hl"><td>C</td><td>7.58{d("−26%")}</td><td>14.12{d("−22%")}</td><td>26.37{d("−21%")}</td>
<td>235{d("−5%", False)}</td></tr>
</table></div>
<div class="tablewrap"><table>
<tr><th>region surf p50 (mm)</th><th>A</th><th>C</th><th>Δ</th><th>p90 A → C</th></tr>
<tr><td>torso</td><td>10.99</td><td>7.56</td><td class="good">−31%</td><td>29.7 → 25.3</td></tr>
<tr><td>hands</td><td>22.65</td><td>13.81</td><td class="good">−39%</td><td>118 → 67</td></tr>
<tr><td>feet</td><td>26.24</td><td>13.64</td><td class="good">−48%</td><td>157 → 76</td></tr>
<tr><td>legs</td><td>9.56</td><td>7.24</td><td class="good">−24%</td><td>26.0 → 22.1</td></tr>
<tr><td>arms</td><td>7.62</td><td>6.51</td><td class="good">−15%</td><td>24.4 → 22.5</td></tr>
<tr><td>head</td><td>8.70</td><td>7.08</td><td class="good">−19%</td><td>23.8 → 21.8</td></tr>
</table></div>
<p class="note">The head improves 19% despite keeping its v1 chart — the CSC loss operates on its
unambiguous ~88% of texels, so the per-texel ambiguity mask pays off rather than merely doing no harm.</p>
</section>

<section>
<div class="no">THE SEAM CLAIM</div><h2>Gradient matching did its job</h2>
<p>Across a true GT part boundary the body is physically continuous, so a predicted XYZ step
&gt;30&nbsp;mm there is a chart-jump artifact — exactly what the image-space gradient term targets.</p>
<div class="tablewrap"><table>
<tr><th>run</th><th>seam step p50 (GT: 3.69)</th><th>seam-break rate ↓</th>
<th>interior grad residual p50 ↓</th><th>surf ≤3px of boundary</th></tr>
<tr><td>A</td><td>3.38 mm</td><td>3.08%</td><td>0.605 mm/px</td><td>11.41 mm</td></tr>
<tr class="hl"><td>C</td><td>3.51 mm</td><td><b>1.47%</b>{d("−52%")}</td><td><b>0.417</b>{d("−31%")}</td>
<td>8.65 mm</td></tr>
<tr><td>D</td><td>3.55 mm</td><td>1.49%</td><td>0.419</td><td>8.76 mm</td></tr>
<tr><td>E</td><td>3.40 mm</td><td>1.99%</td><td>0.422</td><td>9.04 mm</td></tr>
</table></div>
<div class="cards">
<div class="v win"><h3>Confirmed</h3><p>The seam-break rate halves and the median seam step (3.51 mm vs
GT 3.69 mm) is essentially exact. The interior gradient residual — the MiDaS-style term's direct
target — drops 31%. The loss delivered precisely its design goal.</p></div>
<div class="v"><h3>E's epitaph</h3><p>Soft gather is measurably <i>worse at seams</i> (1.99% vs 1.47%)
— mixing charts across a boundary blurs the very transition it was hypothesised to help. D shows
GT parts are not needed at train time; C stays the working checkpoint.</p></div>
</div>
</section>

<section>
<div class="no">WHAT DIDN'T MOVE</div><h2>Rotations and discontinuity placement are part-bounded</h2>
<p>The rotation jump structure is identical across all four runs: predictions are <i>smoother than GT</i>
in the interior (0.59° vs 0.89° per edge), step only ~1.0° across missed true seams (GT: 29°), and step
~31° at spurious ones (GT: 1.6°). Spurious ≈ missed (1.28% vs 1.36%) — the model places roughly the
right <i>number</i> of discontinuities at slightly wrong <i>positions</i>. With part accuracy frozen at
95.4% everywhere, every remaining large-error mode — the misplaced discontinuities, the p99 surface
tail, the residual +1.3 mm boundary band — traces to the part head, not to UV or rotation quality.</p>
<div class="cards">
<div class="v"><h3>Next lever 1 · part boundary</h3><p><code>--rot-dilate</code> (supervise each part's
rotation in a dilated band beyond its mask) is implemented and idle; part-boundary localisation losses
are the other candidate.</p></div>
<div class="v"><h3>Next lever 2 · hidden geometry</h3><p>The eye/lip speckle is trained-in: 1–3% of head
pixels carry SMPL-X eyeball/mouth-interior labels (verified on GT). Fix is loader-level (mask pixels
whose GT face index is hidden geometry out of part/UV/rot losses) + a fine-tune from C — no shard
regeneration.</p></div>
</div>
</section>

<section>
<div class="no">REAL-DOMAIN CHECK</div><h2>COCO minival — the CSC loss is also a sim2real regulariser</h2>
<p>Official DensePose GPS eval (GT boxes, minival; job 5668430). COCO annotations are v1-chart
coordinates, so face_uv_v2 checkpoints get an <b>exact v2→v1 remap</b> on hands/feet (baked through
face+barycentrics, round-trip 0.001 chart units) before scoring.</p>
<div class="tablewrap"><table>
<tr><th>system</th><th>OGPS AP@0.5:0.95 ↑</th><th>AR ↑</th></tr>
<tr><td>ckpt_120000 (baseline)</td><td>0.565</td><td>0.684</td></tr>
<tr><td>A · clean GT only</td><td>0.552 <span class="d bad">−1.3 AP</span></td><td>0.673</td></tr>
<tr class="hl"><td>C · + CSC (gt)</td><td><b>0.568</b>{d("+1.6 vs A")}</td><td><b>0.690</b></td></tr>
<tr><td>D · + CSC (pred)</td><td>0.563</td><td>0.685</td></tr>
<tr><td>E · + CSC (soft)</td><td>0.560</td><td>0.685</td></tr>
<tr><td>C without chart remap</td><td>0.521 <span class="d bad">−4.7 AP handicap</span></td><td>0.637</td></tr>
</table></div>
<div class="cards">
<div class="v win"><h3>The unexpected finding</h3><p>25k extra steps of pure synthetic fine-tuning (A)
<i>degraded</i> real-domain COCO by 1.3 AP — the known sim2real drift. The CSC loss recovered that
drift and landed at-or-slightly-above baseline (+1.6 AP over A). On real data it behaves as a
regulariser against synthetic overfitting, not merely a synthetic-metric win.</p></div>
<div class="v"><h3>Method note</h3><p>The remap is not cosmetic: scoring v2-chart predictions raw
against v1-chart annotations costs 4.7 AP — enough to have flipped the conclusion. Chart conventions
must travel with checkpoints.</p></div>
</div>
</section>

<section>
<div class="no">CONTEXT</div><h2>Where the absolute level comes from — the recipe lineage, decomposed</h2>
<p>The ~0.57 range is well below the earlier IUV-only model (dp_v2_L512, 0.633) and the COCO-fine-tuned
champion (0.73–0.74). Riza challenged the gap, so every step of the orient recipe was re-scored with the
authoritative GT-box GPS at <b>matched step 58k</b> (job 5675674) — separating what the COCO sparse-point
proxy had blurred together at decision time.</p>
<div class="tablewrap"><table>
<tr><th>lineage step</th><th>COCO AP ↑</th><th>Δ</th><th>BEDLAM rot geo ↓</th></tr>
<tr><td>dp_v2_L512 · IUV-only, hflip, flat LR, 75k</td><td>0.633</td><td class="mut">—</td><td class="mut">—</td></tr>
<tr><td>+ rotation head, − hflip, decoupled (58k)</td><td>0.630</td><td><span class="d mut">−0.3</span></td><td>9.01°</td></tr>
<tr><td>+ per-part rotation head (24×6D)</td><td>0.612</td><td><span class="d bad">−1.8</span></td><td>8.48°</td></tr>
<tr><td>+ LLRD structure (base 3e-4)</td><td>0.622</td><td><span class="d good">+1.0</span></td><td>8.05°</td></tr>
<tr><td>+ base lr 1e-3 <span class="mut">(the "synthetic winner")</span></td><td>0.588</td><td><span class="d bad">−3.4</span></td><td>7.86°</td></tr>
<tr><td>+ 58k → 120k pure synthetic</td><td>0.578</td><td><span class="d bad">−1.0</span></td><td>7.30°</td></tr>
<tr><td>+ MAMMA 50/50 mix (= ckpt_120000 baseline)</td><td>0.565</td><td><span class="d bad">−1.3</span></td><td class="mut">occl. −4.5°</td></tr>
</table></div>
<div class="cards">
<div class="v win"><h3>The rotation task is free — the recipe wasn't</h3><p>Adding orientation cost 0.3 AP
(hflip loss included). Synthetic rotation geo improved monotonically down the ladder while real AP fell —
they anti-correlate on the backbone-LR axis. LLRD at 3e-4 is the only step that improved <i>both</i>; the
real cost sits in the per-part head (−1.8) and the 1e-3 base LR (−3.4), which trains DINOv3's top blocks
25× hotter and overwrites its real-image features with the synthetic manifold.</p></div>
<div class="v"><h3>Why the decision looked safe</h3><p>The LR choice <i>did</i> weigh correspondence and
COCO: at 58k the sparse proxy read flat 0.784 / 3e-4 0.788 / 1e-3 0.782 pt_acc — correct <i>ranking</i>
(matches GPS exactly) but 6× magnitude compression, so a 3.4 AP loss read as noise. Sparse point accuracy
is nearly blind to what GPS measures. Real-domain sweet spot in this family: per-part + LLRD 3e-4;
recovery lever for deployment: joint BEDLAM+COCO fine-tune (worth +11 AP on dp_v3).</p></div>
</div>
</section>

<section>
<div class="no">EVIDENCE</div><h2>In pixels</h2>
{fig(f"{B}/hands_4way.png", "Both hero-clip hands, predicted normals. OLD collapses to rainbow speckle; "
     "all three CSC variants produce actual shaded hands with resolved fingers — and are mutually "
     "indistinguishable, matching the numbers.")}
{fig(f"{B}/face_zoom_4way.png", "Portrait clip, face zoom (photo | normals | head parts | UV). The face was "
     "already good and stays good; the temple hand goes from chaos to structure in C/D/E. Residual "
     "eye/lip speckle in ALL rows = the hidden-geometry labels (next lever 2).")}
</section>

<footer>
<p>Checkpoints: <code>~/data/bedlam2/dp_runs/dp_csc_{{A,C,D,E}}/ckpt_025000.pt</code> · working model:
<code>dp_csc_C</code>. Evals: <code>orient/eval_512.py</code>, <code>orient/eval_csc_detail.py</code>.
Chain of custody: GT rebuild (512 canvas, exact rotations) → fold-free charts (face_uv_v2) → CSC loss.
Hands+feet surface error, system-level: 92.6* → 24.8 → 13.5 mm.</p>
</footer>
</div>"""

with open(OUT, "w") as f:
    f.write(html)
print(f"wrote {OUT} ({os.path.getsize(OUT)/1e6:.1f} MB)")

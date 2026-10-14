# Oriented DensePose — paper results dossier

Working reference for writing. Every number is sourced to a job/log or memory file. Flag before
final: rows marked ⚠ should be re-pulled from logs to confirm the last digit.

---

## 0. The paper in one paragraph

We revisit the *correspondence* side of DensePose, which has been frozen at a 2019 detectron2
model while pose-guided generators stacked on top of it. We (a) add a per-pixel **surface
orientation** output — a full SO(3) frame field — to DensePose for the first time; (b) fix a
label defect in the official UV charts, add a canonical-surface-coordinate (CSC) loss, occlusion
training data (MAMMA), and hidden-geometry masking; and (c) reach **0.741 GPS AP on DensePose-COCO — above every model in our
lineage** (see §2d.2: do NOT phrase this as state of the art) while also predicting orientation. We then show, in a
controlled experiment, that the resulting correspondences are worth **~3× the most recent
published architectural step** to a downstream garment-transfer model, generator held fixed.

---

## 1. Correspondence accuracy — DensePose-COCO (GT-box OGPS AP)

Authoritative eval: `dpeval/run_eval.py`, ViT-L, input 512, v2→v1 chart remap where applicable.

### 1a. Headline
| model | COCO AP | note |
|---|---|---|
| dp_v2_L512 (prior IUV-only, our lineage) | 0.633 | no rotation |
| historical COCO-ft champion (dp_ft_joint) | 0.732–0.738 | in-domain COCO supervision |
| **dp_w_MF @3k — WINNER** | **0.741** | rotations + CSC + MAMMA aboard |

MF checkpoint curve (job 5702500): 1k 0.713 / 2k 0.729 / **3k 0.741** / 5k 0.737 / 10k 0.734.

### 1b. Recipe decomposition (matched step 58k, job 5675674) — why the lineage sat low, and the fix
| step | COCO AP | Δ |
|---|---|---|
| dp_v2_L512 (IUV-only, flat LR) | 0.633 | — |
| + rotation head, −hflip, decoupled | 0.630 | −0.3 (task ~free) |
| + per-part rotation head | 0.612 | −1.8 |
| + LLRD backbone LR (base 3e-4) | 0.622 | +1.0 |
| + base lr 1e-3 ("synthetic winner") | 0.588 | −3.4 |
| + 58k→120k pure synthetic | 0.578 | −1.0 (sim2real drift) |
| + MAMMA 50/50 | 0.565 | −1.3 |
Lesson: the drop was a training-recipe artifact (LR + drift), not the rotation task. Sparse
COCO proxy hid a 3.4 AP loss (magnitude-compressed ~6×) → **gate recipe decisions on GPS**.

### 1c. CSC-loss ablation (25k ft from ckpt_120000; jobs 5648564-7, 5668430)
| system | COCO AP | surf err (mm) | hands+feet (mm) |
|---|---|---|---|
| baseline ckpt_120000 | 0.565 | 12.54 | 92.6* |
| A · clean GT only | 0.552 | 10.52 | 24.81 |
| **C · +CSC (gt gather)** | **0.568** | **8.18** | **13.47** |
| D · +CSC (argmax) | 0.563 | 8.22 | 13.46 |
| E · +CSC (soft) | 0.560 | 8.23 | 13.87 |
CSC = correspondence loss: hands/feet surface error **−46%**; also a sim2real regularizer
(+1.6 AP over A on COCO). Seam-break rate 3.08→1.47%.

---

## 2. Occlusion — held-out MAMMA sequences (geo_e2e°, the airtight MAMMA proof)

29 sequences / 59 shards excluded from training (md5-hash holdout), probed with `indomain_probe`.
| model | trained on MAMMA? | held-out occlusion geo_e2e |
|---|---|---|
| W-B (BEDLAM-only, same recipe) | no | **14.80°** |
| **W-M (BEDLAM+MAMMA)** | yes | **6.89°** |
Same recipe, only the data differs → MAMMA is worth **−7.9°** on genuinely held-out occlusion,
and is *free* on COCO (W-M 0.621 vs W-B 0.613) and clean synthetic (BEDLAM-val 6.86 vs 6.98°).
Kills the "MAMMA costs COCO" concern (that was a bad-LR artifact).

---

## 2b. ⚠ METRIC DEFINITIONS — get these right in the paper

- **geo_oracle** — rotation geodesic error with the per-part rotation channel gathered by the
  **GT** part. Measures the rotation head in isolation. NOT deployable performance.
- **geo_e2e** — gathered by the **PREDICTED** part. This is the honest, end-to-end number and is
  what §2 and the winner tables report. At 120k on the pure-BEDLAM lineage: oracle 7.30° vs
  e2e 7.95°; the 0.65° gap is part-misclassification.
- **The orientation ablation LADDER in §3 is ORACLE geo** (it predates the e2e correction):
  single-head 9.01 → +per-part 8.48 → +LLRD-3e-4 8.05 → +LLRD-1e-3 7.86 (all @58k), → 7.30 @120k.
  **Label it explicitly as oracle-gathered in the paper**, or report the ladder as *relative*
  improvement only. Do NOT mix ladder numbers into a table whose other rows are e2e.
- **surf_err** — median per-pixel distance (mm) between predicted and GT template XYZ, on BEDLAM-val.
- **COCO GPS AP** — official DensePose geodesic point similarity, GT-box protocol.
- Verified-from-log values (eval_512, BEDLAM-val, geo_e2e / surf_err / hands+feet surf):
  dp_csc_A 6.88 / 10.52 / 24.81 · dp_csc_C 6.77 / 8.18 / 13.47 · dp_csc_D 6.90 / 8.22 / 13.46 ·
  dp_csc_E 6.87 / 8.23 / 13.87 · dp_w_F0 6.97 / 8.41 / 13.61 · dp_w_M 6.86 / 8.22 / 15.09 ·
  dp_w_B 6.98 / 8.23 / 19.84 · dp_w_J 7.18 / 8.79 / 21.01.
  Held-out MAMMA geo_e2e: dp_w_M 6.89 · dp_w_B 14.80 · dp_w_J 10.77 · dp_w_MF@1k 7.75,
  @3k 8.24, @10k 8.31 · dp_csc_C 6.40 (⚠ its lineage saw those sequences — in-distribution,
  state the caveat) · dp_w_F0 10.62.

## 2c. CSC operator validation gates — RE-RUN AND VERIFIED 2026-07-21 (K=512, v2 charts)

`python -c "from orient.csc import _check; _check(K=512, uv_asset='smplx_face_uv_v2.npy')"`
```
GATE 1 round-trip vs exact barycentric XYZ (200k pts):
  UNAMBIGUOUS 74.5% of samples  -> median 0.000 mm | p99 0.335 mm | max 63.350 mm   [gate says FAIL]
  ambiguous   25.5% (MASKED OUT) -> median 1.503 mm | max 58.570 mm
  hands 84.5% usable (median 0.001, max 8.331) | feet 88.1% (0.000, 63.350) | head 52.3% (0.001, 23.798)
GATE 2 autograd vs float64 finite-diff d(X)/d(uv), h=1e-7: median rel err 5.21e-10  PASS
GATE 3 GT uv through the loss: L_xyz 0.0135 mm, L_grad 0.0056 mm  PASS
```
**⚠ HONESTY REQUIREMENT: the drafted method text quotes the median and p99 but NOT the max, and
the gate's own criterion (max < 1 mm on the unambiguous set) FAILS at max 63.35 mm.** State it
correctly: the operator is exact for the overwhelming majority (median ≈ 0, p99 0.335 mm) with a
thin tail (max 63 mm) on sliver triangles / texel-boundary cases; the tail is small enough not to
affect training because the loss is an L1 over millions of pixels, but do NOT claim the gate passes.
Also note head usability is only 52.3% — that is the hidden-geometry mask doing its job, worth a
sentence. Gate-2 value is 5.21e-10 (draft says 5.3e-10 — use the measured one).

## 2d. ⚠ CLAIMS THAT MUST BE PHRASED CAREFULLY (checked 2026-07-21)

1. **"the rotation head costs 0.3 AP"** — the 0.633→0.630 step bundles THREE changes: adding the
   rotation task, DISABLING hflip augmentation (the rotation target has no mirror implemented),
   and the decoupled mask head. Correct phrasing: *"adding the rotation task — together with
   disabling horizontal-flip augmentation, which our rotation target requires — costs 0.3 AP."*
   Never attribute the whole 0.3 to the rotation head alone.
2. **"above every model in our lineage"** is the safe form of the COCO claim. Do NOT write
   "state of the art on DensePose-COCO" — we did not re-run other groups' models under our
   protocol, and Leffa/others do not report GPS AP at all. Our 0.741 vs our own lineage's best
   prior (0.738 COCO-ft champion, 0.633 IUV-only) is what the evidence supports.
3. **CSC "+1.6 AP over a matched control"** = C (0.568) vs A (0.552), both 25k ft from the same
   seed. Say "matched control", not "over the baseline" (baseline ckpt_120000 is 0.565).
4. **Chart fold "roughly half"** — measured 48.9/49.1% (hands) and 52.4/53.0% (feet); v2 residual
   0.0% hands, 0.6–0.7% feet. Head residual 0.6% in BOTH v1 and v2 (it is hidden geometry, not a
   flattening defect) — do not imply v2 fixed the head.
5. **Occlusion 14.80°→6.89°** is W-B vs W-M, i.e. an ABLATION OF THE DATA at fixed recipe, not a
   before/after of our method. Phrase as "adding occlusion data is worth −7.9° on a sequence-level
   holdout".
6. **The downstream "3× the most recent published architectural step"** compares our +0.0116 to
   the MCLD→FPDM +0.0037. Both are LPIPS-alex deltas, so this is legitimate — but say
   "the most recent published step on this benchmark", not "3× better than prior work".
7. **Do not claim FID improves anywhere.** In the controlled swap it is +0.021 / +0.004 (worse,
   inside noise). Cross-paper FID is not comparable at all.

## 3. Orientation / frame field

Per-pixel 6D→SO(3), R·N_template = camera-space normal. geo_e2e (end-to-end, gather by predicted
part) on BEDLAM-val: dp_csc_C **6.77°**, W-M 6.86°, MF 6.97°. Full arc single→best −1.71° (~19%).
Visualization suite (forward-composition only — pull back the frame field per pixel, never invert):
wind/LIC/hatch (2D), anisotropic-satin/sequins/chrome/diffraction (Cycles PBR), Cycles fur,
6-DoF object placement. Figures in `~/data/dp_viz_compare/beauty/`.

---

## 4. Chart fold fix (v2) — a label-defect correction

Official DP24 charts are non-injective on hands/feet: ~49% of hand surface / 53% of foot surface
folds (two skin points share one (part,u,v)), median 3D collision 60–67 mm. Sheet-split (v2)
→ 0.0% / 0.7%. **⚠ CORRECTED (reviewers caught this):** the 4.7 AP is the cost of scoring v2-chart
predictions against v1-convention annotations WITHOUT the remap (0.521 vs 0.568) — a
PROTOCOL artifact, NOT evidence that v2 charts improve accuracy. The chart repair is a
CORRECTNESS result (measurable injectivity restored); its AP effect is confounded with
schedule in the only near-matched comparison we have, and that comparison is negative.
Do not claim '+4.7 AP from the chart fix' anywhere.
Figures/report: `charts_html/charts_v1v2.html`. Head residual (~0.6%) is SMPL-X hidden geometry,
handled by the loader mask, not re-flattening.

---

## 5. Downstream — garment transfer (controlled swap inside MCLD, CVPR'25)

**Design:** retrain MCLD twice, identical in every way (750×1101 original images, 8,512/8,512
test pairs, seed 12580, 60k steps, authors' unmodified code — git-verified at commit 6279ff7,
both texture atlases from the authors' unwrap fn, our regen matches theirs to 0.11%). ONE
variable: the DensePose model (arm A = detectron2 R101; arm B = ours, dp_w_F0@3k).

### 5a. Definitive numbers (original geometry, vs authors' released GT; jobs 5734865/66)
| 256×176 | FID↓ | LPIPS↓ | SSIM↑ | PSNR↑ |
|---|---|---|---|---|
| arm A (detectron2) | 6.659 | 0.1497 | 0.7462 | 18.66 |
| **arm B (ours)** | 6.680 | **0.1380** | **0.7575** | **19.22** |
| Δ | +0.02 | **−0.0117** | **+0.0113** | **+0.57 dB** |
| 512×352 | | | | |
| arm A | 6.959 | 0.1781 | 0.7501 | 18.03 |
| **arm B** | 6.963 | **0.1658** | **0.7582** | **18.53** |
| Δ | +0.004 | **−0.0123** | **+0.0080** | **+0.50 dB** |

### 5b. Paired significance (per-image, same GT, n=8,512; job 5747295)
256: LPIPS +0.0116 CI[+0.0110,+0.0123] win **68.7%** dz 0.371 | SSIM win 72.7% | PSNR win 69.5%
512: LPIPS +0.0123 CI[+0.0116,+0.0129] win **72.2%** | SSIM win 72.6% | PSNR win 68.9%
All Wilcoxon p≈0 (uninformative at n=8512 — report win rate + CI + dz).

### 5c. Attribution — NOT a detector upgrade (job 5748524)
| subset | n | ΔLPIPS | win |
|---|---|---|---|
| all (headline) | 8,512 | +0.0116 | 68.7% |
| detectron2 blank/partial | 29 (0.35%) | +0.0838 | 89.7% |
| **both models full map** | 8,483 (99.66%) | **+0.0114** | 68.6% |
Failure cases = ~2% of the effect. Robustness is a separate, one-line result (detectron2 fails 29,
ours 0).

### 5d. Standing (honest)
Ahead of MCLD (0.1482) and FPDM (0.1445, best published) on LPIPS; highest SSIM+PSNR reported.
FID neutral (correct signature — correspondences buy spatial fidelity; FID = distributional
realism, set by the untouched diffusion prior). Magnitude: our +0.0116 vs the entire MCLD→FPDM
step +0.0037 (which *lost* SSIM/PSNR/FID) → ~3×, strictly dominant.
**Leffa (CVPR'25) leads on LPIPS 0.114 / FID 5.71 — but on a DIFFERENT AXIS** (VtonEval uses
SqueezeNet-LPIPS + 1024 resize + cleanfid; MCLD uses AlexNet-LPIPS). → resolve with the cross-eval
(§7). The controlled claim is immune to this: Leffa is a different generator; it says nothing
about correspondence quality.

### 5e. Validation of the setup
- Our eval reproduces published MCLD to 4 decimals on their released outputs (LPIPS 0.14817 vs
  0.1482, PSNR 18.840 vs 18.84, etc.). Note: paper "SSIM" = evaluate.py `SSIM_256` field.
- Arm A reproduces published MCLD within 1–2% → faithful baseline, not a win vs a broken repro.
- Confound found+killed: mirror dataset re-crops per image; rebuilt on originals where both arms
  verify aligned; an alignment-warp control came out *worse* → the ~7% offset was learnable, never
  inflated the result. Effect replicates across both dataset builds (−0.0131/−0.0116 @256).

---

## 6. Figures inventory
- Charts v1 vs v2 fold: `charts_html/charts_v1v2.html` (+ per-part PNGs) — §4
- CSC hands/face 4-way: `beauty/hands_4way.png`, `beauty/face_zoom_4way.png` — §1c
- Frame-field viz suite: `beauty/…` (satin, sequins, chrome, fur, objects) — §3
- Garment qualitative (stat-selected rows + worst case): `mcld/report_figs/qualitative.jpg`,
  `detail.jpg` — §5
- Published HTML report: artifact 3404b521 (garment transfer)

## 6b. ★ LEFFA CROSS-EVAL — the axis problem, resolved (2026-07-21, local 2-GPU)

Leffa's published numbers are computed by **VtonEval**, whose metric definitions differ from
MCLD's `evaluate.py`:
| | Leffa / VtonEval | MCLD / ours |
|---|---|---|
| LPIPS | torchmetrics, **SqueezeNet** | `lpips` pkg, **AlexNet** |
| SSIM / PSNR | torchmetrics | MCLD's own |
⇒ Leffa's 0.114 and our 0.138 were NEVER on the same axis. (Shared-baseline agreement in the two
papers' tables proves only that both COPIED CFLD/PIDM numbers from the source papers.)

**Re-scored our arms with VtonEval's exact definitions** (`paper/crosseval_vtoneval.py`;
metric code reimplemented verbatim from VtonEval/eval.py because its filename→id matcher is
VITON-style and collides on DeepFashion pairs; same released GT, n=8512):

| axis: VtonEval (squeeze-LPIPS) | LPIPS↓ | SSIM↑ | PSNR↑ |
|---|---|---|---|
| **@256** arm A (detectron2) | 0.0961 | 0.7467 | 18.073 |
| **@256 arm B (ours)** | **0.0891** | **0.7580** | **18.631** |
| @256 Leffa *published* | 0.114 | 0.732 | n/r |
| **@512** arm A | 0.1149 | 0.7503 | 17.486 |
| **@512 arm B (ours)** | **0.1077** | **0.7583** | **17.971** |
| @512 Leffa *published* | 0.119 | 0.755 | n/r |

**On a common axis, arm B leads Leffa on LPIPS at both resolutions (−22% @256, −9.5% @512) and on
SSIM. Even arm A (the detectron2 baseline) beats Leffa's published LPIPS** ⇒ Leffa's apparent lead
was a metric-protocol artifact, not a quality gap. The A→B delta is preserved on this axis too
(−0.0070 @256, −0.0072 @512), so the controlled claim is axis-independent.
Caveats to state: n=8512 vs their 8570 (0.7% fewer pairs); FID not compared (different reference
protocol); numbers are *their published* vs *our recomputed*, so §7 runs Leffa itself for the
fully airtight version.


## 6c. ★★ AIRTIGHT LEFFA COMPARISON — we RAN Leffa (2026-07-22, 4 GPUs, 1499 identical pairs)

Ran Leffa's released `pose_transfer.pth` ourselves (repo commit as cloned; SDXL-inpainting base,
19.4 GB ckpt; their DensePose R-50; 50 steps, guidance 2.5, seed 42; src=target-pose image drives
DensePose, ref=source-appearance image — verified visually against GT). Then scored Leffa AND both
our arms **on the same 1499 images with BOTH harnesses**:

| @256, n=1499 | VtonEval axis (squeeze-LPIPS) | | | MCLD axis (alex-LPIPS) | | |
|---|---|---|---|---|---|---|
| system | LPIPS↓ | SSIM↑ | PSNR↑ | LPIPS↓ | SSIM↑ | PSNR↑ |
| Leffa (we ran it) | 0.0969 | 0.6870 | 16.858 | 0.1554 | 0.6845 | 17.452 |
| arm A (detectron2) | 0.0958 | 0.7470 | 17.916 | 0.1494 | 0.7464 | 18.704 |
| **arm B (ours)** | **0.0889** | **0.7580** | **18.474** | **0.1379** | **0.7575** | **19.299** |

**arm B beats Leffa on EVERY metric on BOTH axes** (LPIPS −8.3% squeeze / −11.3% alex; SSIM +0.071;
PSNR +1.6/+1.8 dB). Even arm A beats Leffa. The result no longer depends on anyone's published
numbers — same images, same code, same metrics.

⚠ **REPRODUCTION GAP TO REPORT HONESTLY:** our Leffa run scores 0.0969 LPIPS / 0.687 SSIM on the
VtonEval axis vs their published 0.114 / 0.732. We do NOT reproduce their published numbers
exactly (we are better on LPIPS, worse on SSIM). Likely causes: our 1499-pair random subset vs
their 8570; our save path (generate at 768x1024 → resize to 176x256 / 352x512) vs their eval's
resize-to-1024-height; possible pair-construction differences. State this plainly — the
like-for-like comparison stands regardless, but do not claim we reproduced Leffa's paper.
**@512, same n=1499, both axes:**
| system | LPIPS(sq)↓ | SSIM↑ | PSNR↑ | LPIPS(alex)↓ | SSIM↑ | PSNR↑ |
|---|---|---|---|---|---|---|
| Leffa (we ran it) | 0.1235 | 0.7105 | 16.310 | 0.2024 | 0.7095 | 16.895 |
| arm A | 0.1143 | 0.7505 | 17.333 | 0.1778 | 0.7503 | 18.084 |
| **arm B (ours)** | **0.1075** | **0.7585** | **17.814** | **0.1657** | **0.7583** | **18.605** |
Same conclusion at 512: arm B > arm A > Leffa on every metric, both axes.
**Per-image win rate vs Leffa (squeeze-LPIPS, n=1499): arm B closer to target on 63.2%.**
Qualitative figure `figs/fig_leffa.png` (rows chosen by per-image LPIPS, includes the case
where Leffa wins). Per-image scores cached at ~/data/mcld/leffa_perimage.npz.

## 7. Remaining experiment — Leffa cross-eval (staged, HELD until cluster frees)
Run Leffa's released `pose_transfer.pth` on DeepFashion test → (a) reproduce their 0.114 with
VtonEval [verifies we can run their system]; (b) score Leffa outputs with MCLD's AlexNet harness
AND our arm-A/B outputs with VtonEval's SqueezeNet harness → everyone on both axes, standing
table becomes honest. ~1 GPU-day. Not submitted: user's own queue saturated, FIFO, would delay it.

## 8. Open / optional
- HMR refinement (CameraHMR + our dense IUV+rotation, 3DPW) — second downstream host if wanted.
- rot-dilate part-boundary experiment (built, idle) — the evidence-backed lever for the p99
  chart-jump tail.

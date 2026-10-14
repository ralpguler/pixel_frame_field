# Figure inventory — everything available in `figs/`

USE ALL OF THESE. Three were added after the first drafting round and MUST be integrated:
`fig_method`, `fig_frontier`, `fig_effect`.

| file | width | what it shows | belongs in |
|---|---|---|---|
| `fig_teaser.png` | figure* | photo \| IUV \| camera normals $\mathbf{R}\mathbf{n}_{tpl}$ \| anisotropic-satin render \| attached objects | Intro (page 1 teaser) |
| **`fig_method.pdf`** | figure* | pipeline: input crop → DINOv3 ViT-L/16 + DPT → four heads (mask 1, part 25, UV 24×2, rot 24×6) → per-pixel $(i,u,v,\mathbf{R})$, with the three rendered outputs; caption line names the CSC loss | **Method §3 (overview figure)** |
| `fig_charts.png` | figure* | v1 vs v2 UV charts: chart square with fold overlay in red + the same chart painted on the body, both sides | Method §3.4 (chart fix) |
| `fig_csc_hands.png` | figure | hands, 4-way: baseline / A / C / D predicted normals — rainbow speckle → resolved fingers | Experiments (CSC ablation) |
| `fig_csc_face.png` | figure | face zoom 4-way (photo \| normals \| head parts \| UV) | Experiments (optional, or supplement) |
| **`fig_frontier.pdf`** | figure | scatter: DensePose-COCO GPS AP (x) vs held-out occlusion error (y, inverted) for W-B / W-M / W-J / W-MF@1k / W-MF@3k, with an arrow marking the two-stage joint fine-tune | **Experiments — the data/recipe ablation §4** |
| **`fig_effect.pdf`** | figure | horizontal bars comparing LPIPS improvement: the published MCLD→FPDM step (+0.0037) vs our correspondence swap (+0.0116), with CI whiskers | **Downstream §6 (effect-size context)** |
| `fig_garment.png` | figure* | garment transfer, 5 rows: source \| target pose (detectron2) \| target pose (ours) \| arm A \| arm B \| GT. Rows are chosen BY the per-image statistics: 3 largest gains, the median, and the worst regression | Downstream §6 |
| `fig_garment_detail.png` | figure | detail crop of the largest-gain case: arm A \| arm B \| GT | Downstream §6 |

Notes for captions:
- `fig_frontier`: the point is that W-J (COCO mixed in during training) is *dominated* — the
  two-stage W-MF reaches higher AP at lower occlusion error. Seed quality, not mix ratio, bends
  the trade-off.
- `fig_effect`: whiskers are the bootstrap 95% CI of our paired estimate; the FPDM bar has no CI
  because it is a published point estimate. Say so in the caption.
- `fig_garment`: state explicitly that the worst-case row is included deliberately — in the ~31%
  of images where arm B loses, the failure is an appearance error (wrong garment texture), not a
  pose error.
- `fig_method`: `.pdf` is vector — prefer it over `.png`.

## Added 2026-07-22 — after running Leffa ourselves

| file | width | what it shows | belongs in |
|---|---|---|---|
| **`fig_leffa.png`** | figure | 5 cols: source \| **Leffa** \| arm A \| arm B \| GT. Rows chosen BY per-image LPIPS: the two largest margins over Leffa, the median case, and the case where **Leffa wins most** (our arm B emits a background artifact — include it, it is honest) | Downstream §6, next to the Leffa table |

Caption must state: rows are statistically selected, not curated; arm B is closer to the target on
**63.2%** of the 1499 images than Leffa is; and that the last row is a failure case for us.

# Oriented DensePose — CVPR 2026 submission: STATUS

**Build:** `export PATH=$HOME/.TinyTeX/bin/aarch64-linux:$PATH; pdflatex main; bibtex main; pdflatex main; pdflatex main`

**Current state (2026-07-22):** compiles clean — 0 errors, 0 undefined refs, 0 overfull boxes.
- Content pages 1–8 (CVPR limit — references excluded) ✓
- References pp. 9–12
- Supplementary pp. 13–14

**Title:** Oriented DensePose: From Surface Points to Surface Frames

**Main paper contains** (per Riza's directive that the paper must be self-contained):
- Fig 1 teaser · Fig 2 pipeline · Fig 3 frame field (6 effects, wide strip) · Fig 4 garment transfer
- Tab 1 DensePose-COCO (incl. W-B/W-M/W-J/W-MF data ablation + held-out occlusion)
- Tab 2 recipe diagnosis · Tab 3 CSC ablation · Tab 4 MCLD controlled swap · Tab 5 Leffa cross-eval
- Paired statistics, attribution split, chart-fold numbers, full Limitations, Conclusion

**Supplementary contains only detail:** CSC baking + validation gates, orientation recipe ladder
(oracle metric, flagged), rejected chart alternatives + chart figure, GT-canvas affine, hidden
geometry, occlusion holdout construction, Leffa qualitative figure.

**Headline claims and where they are sourced (all in RESULTS.md):**
- 0.741 GPS AP on DensePose-COCO minival, GT-box protocol — above every model in our lineage
  (NOT phrased as state of the art; no external model re-scored under our harness)
- Oriented output costs 2.1 AP at matched training step (rot head + per-part head + no hflip)
- CSC: −46% hand/foot surface error, +1.6 AP over a matched control
- Occlusion data: 14.80° → 6.89° on a sequence-level holdout
- Downstream MCLD swap: +0.0116 LPIPS @256, +0.0123 @512, 68.7% per-image win rate; FID does not move
- Leffa, run by us on 1499 identical pairs, both harnesses: arm B ahead on every metric;
  reproduction gap to their published numbers stated explicitly

**Known weaknesses the paper itself declares:** single training run per model (no seed variance);
reported checkpoint is the max of a sweep on the reported split; chart repair's AP effect is
confounded and the near-matched comparison is negative; frame field has no baseline; downstream is
one host, one domain; our Leffa run does not reproduce their published numbers.

# Length and what belongs where — RIZA'S DIRECTIVE (2026-07-22)

> "The paper should be complete and contain all relevant info. Sup mat should be only for
> supplementary info."

The first compression pass got this WRONG: it pushed core evidence (the garment-transfer
qualitative figure, the frame-field figure, the Leffa comparison table, the orientation results
table) into the appendix, leaving the main paper with a single figure. **That is not acceptable.**

## The main paper (8 pages of content, references excluded) MUST contain
Every claim the abstract makes, with its evidence:
- **Figures**: teaser; the method/pipeline figure (`fig_method.pdf`); the frame-field suite
  (`fig_framefield.png`) — this is the payoff of the paper's central contribution and cannot live
  in an appendix; the garment-transfer qualitative (`fig_garment.png`).
- **Tables**: DensePose-COCO comparison; the recipe ladder; the CSC ablation; the data ablation
  (W-B / W-M / W-J / W-MF — this is what proves the occlusion claim); the downstream MCLD table;
  **the Leffa cross-evaluation table** (a headline result — we ran Leffa ourselves).
- **Numbers**: orientation accuracy (geo_e2e), the paired statistics (win rates + CIs), the
  attribution split (29/8512), the chart-fold statistics, the occlusion holdout result.
- Limitations, in full and honest.

## The supplementary may ONLY contain
Detail that a reader can skip without losing the argument or being unable to judge the claims:
- How the CSC texture is baked (KD-tree, barycentrics, ambiguity dilation) and the validation-gate
  numbers.
- Chart re-flattening alternatives we rejected and why.
- The fixed GT-canvas affine derivation; hidden-geometry face-set construction; how the occlusion
  holdout was hashed.
- Extended qualitative grids, the face-zoom panel, per-checkpoint curves, dataset conversion notes.
- The full visualization-suite description beyond what the main figure shows.

## How to make room
Cut PROSE, never evidence. Specific targets: long derivations already in the supplementary,
repeated explanations (the forward-composition principle is stated more than once), narrative
sentences that restate a table, and any implementation minutiae. Tighten with `\small`,
`\setlength{\tabcolsep}`, and by merging short paragraphs — not by deleting results.

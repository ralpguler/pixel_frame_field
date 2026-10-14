# Reframe: Pixel Frame Fields (direction set by Riza, 2026-09-15)

> "The new DensePose portrayal is not impressive, as we do not have anything novel there. Just works better
> thanks to BEDLAM. The actual novelty is the orientation." — "pixel framefields", "so we predict framefields in
> pixels", "we should show it through boxes or frames. This is the main contribution of the paper."

The paper stops being "DensePose, better" and becomes **pixel frame fields**. DensePose-COCO accuracy (0.741) is
demoted to a supporting table ("the correspondence half is competitive"); Figure 1 shows the frame field itself.

## Thesis
For every pixel of a person we predict a rotation R(x) in SO(3) that carries the body template's own surface frame at
the corresponding surface point into the camera. Together with the correspondence pi(x) this is a **pixel frame
field**: where each pixel lies on the body *and* how the surface is turned there, including the twist about the normal
that a normal map cannot represent. Anchored to the template, the frame is chart-independent and continuous across
seams, so anything defined once on the template (a direction such as hair, weave or fur; an object; an anisotropic
material) is transported into the image pixel by pixel, forward, with no body fitting and no tracking.

## Title candidates
1. Pixel Frame Fields: Dense Surface Frames of the Human Body from a Single Image
2. Pixel Frame Fields for Humans
3. From Surface Points to Surface Frames: Pixel Frame Fields for Humans (keeps the current subtitle)

## Contributions (new order)
1. **The pixel frame field** as a task and representation: template-anchored gauge, chart-independent, a per-pixel
   6D/SO(3) head (per-part), ARAP/Kabsch ground truth from synthetic bodies.
2. **What learning a frame field demands of the correspondence** (enabling, not headline): fold-free hand/foot charts,
   metric (CSC) supervision, occlusion data, and never resampling labels.
3. **An evaluation of frame fields** against every other way of obtaining a frame (NEW, required — see E1–E6),
   per component (normal angle, twist angle, full geodesic), on synthetic and real data.
4. **Uses of the field**: forward rendering (anisotropic satin, fur, sequins, diffraction), 6-DoF attachment, and the
   controlled downstream swap (MCLD; correspondence half).

## Figures
- **Fig. 1 (figure\*, page 1)**: (a) input | (b) pixel frame field drawn as oriented boxes, one per sampled pixel,
  coloured by template location | (c) the same boxes on the canonical template | (d) zooms. Generators (repo):
  `densepose/orient/framefield_boxes.py` -> `blender_boxes.py` (image|template) -> `fig1_compose.py`.
  Outputs `~/data/dp_viz_compare/framefield_boxes/<clip>_f075[_wMF]/`. Poses: 37828157 f75 (kick, profile torso),
  15731089 f75 (frontal walk). Shipped model dp_w_MF@3k, v2 charts.
- Fig. 2 method (existing `fig_method.pdf`, re-labelled around R(x) and the gauge).
- Fig. 3 frame accuracy vs baselines, qualitative error maps (new, from E1–E4).
- Fig. 4 uses (existing `fig_framefield.png` strip).
- Supplement: chart folds, CSC hands, garment qualitative, Leffa.

## Section plan (8 content pages)
| section | pages | content |
|---|---|---|
| 1 Intro | 1.0 | with Fig. 1 |
| 2 Related | 0.7 | dense correspondence (DensePose, CSE); human normal estimation (PIFuHD, ICON/ECON, Sapiens, DSINE); HMR (a fitted mesh gives frames); frame/cross fields in geometry processing; rotation representations |
| 3 Pixel frame fields | 2.2 | definition + gauge; network; rotation supervision; correspondence prerequisites condensed |
| 4 Evaluating frame fields | 2.5 | metrics, data, baselines, main table, occlusion holdout, ablations, correspondence table (COCO) |
| 5 Uses | 1.1 | rendering, attachment, MCLD swap (brief) |
| 6 Limitations + conclusion | 0.5 | |

## Experiments required before rewriting
| # | experiment | why | cost | status |
|---|---|---|---|---|
| E1 | **UV-Jacobian frame**: differentiate the predicted (u,v) map, push through the chart->template Jacobian, orthonormalise -> a frame with no rotation head | the null hypothesis for contribution 1; the current paper names it and never ran it | eval only, hours | not run |
| E2 | **Body-fit frames**: CameraHMR and NLF SMPL fits rendered to per-pixel frames with the same ARAP definition | the standard route to a frame | CameraHMR 3DPW preds cached; BEDLAM-val inference needed | partial |
| E3 | **Normal estimator** (Sapiens-normal or DSINE) for the normal component + template tangent for twist | off-the-shelf normals | weights NOT on disk (download) | not run |
| E4 | **Real-data frames on 3DPW test** (GT boxes; pseudo-GT frames from its SMPL fits) | a real-domain number; current 6.77 deg is synthetic only | frames from GT SMPL with rot6d.py; SMPL<->SMPL-X gauge via smplx2smpl_deftrafo_setup.pkl | not run |
| E5 | **Checkpoint selection on a held-out split** | current checkpoint is the max of a sweep on the reported split | eval only | not run |
| E6 | **Normal / twist decomposition** of the geodesic error | shows the twist is predicted, not just the normal | eval only | not run |

Recommended order: E1, E6, E4 first (cheap; they decide whether the head is justified), then E2, then E3.

## What moves out of the main paper
- DensePose-COCO 0.741: one row group in a "correspondence half" table, no headline claim.
- Recipe diagnosis (LR, over-training): supplement.
- Leffa cross-eval: supplement. MCLD controlled swap: short paragraph in Uses.
- Chart fold: kept, compressed, as a prerequisite for frames on hands and feet.

## Open decisions (Riza)
- Title.
- Downstream: keep MCLD (consumes only IUV) or add a frame-consuming host (e.g. normal-guided animation, relighting) —
  the latter supports the thesis directly but is new work.
- Which of E1–E6 to run before the rewrite.

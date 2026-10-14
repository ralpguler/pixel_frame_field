# Visualisations

Every HTML page made during the project is in `pages/`, with the scripts that generate them alongside in this
folder.

## Opening the pages

The pages need no server and no network. Open them directly in a browser:

- the single files (`framefield.html`, `framefield_multi.html`, `framefield_beauty.html`, `csc_ablation.html`,
  `charts_v1v2.html`) embed all their images, video and data;
- the folders (`*/index.html`) load only the images in their own `imgs/` or beside them, so keep each folder
  together.

**Checked:** every local file a page references exists, and no page loads anything from the internet.

## The pages

| page | what it shows | made from | generator |
|---|---|---|---|
| `framefield.html` | the frame field on one clip: the canonical template (left) beside the predicted per-pixel frames reprojected onto 150 video frames (right), with fixed tracked points or a sweeping band | Pexels clip 15731089, exported by `framefield_overlay.py` with its default checkpoint at the time (`dp_pp_llrd_1e3_mamma/ckpt_120000`) | `framefield_pair_html.py` |
| `framefield_multi.html` | the same viewer over four clips (dance, slow, two-person, crowd) | Pexels clips 15731089, 15153663, 28513779, 37828157, exported by `framefield_overlay.py` | `framefield_multi_html.py` |
| `framefield_beauty.html` | the "beauty contest": ways of drawing the frame field (line-integral convolution, satin, hatching, fur, objects and others) on video | per-pixel fields from `field_dump.py`, rendered by `research/densepose/orient/viz_*.py` and `blender_*.py` | `beauty_report_assets.py`, then `beauty_report.py` |
| `csc_ablation.html` | the canonical-surface-coordinate (CSC) loss ablation: arms A, C, D, E at 25k steps | `dp_csc_{A,C,D,E}/ckpt_025000` and the beauty renders | `csc_report.py` |
| `charts_v1v2.html` | the official DensePose UV charts (v1) against our fold-free charts (v2) on hands and feet | the lookup tables in `assets/` | `charts_v1v2_report.py` |
| `compare_coco/` | two checkpoints side by side on COCO people: part/UV, surface normal, and their rotation disagreement | `dp_scratch_bedlam_orient_pp_llrd_1e3/ckpt_050000` (BEDLAM 50k) against `dp_pp_llrd_1e3_mamma/ckpt_050000` (+MAMMA 50k), COCO val2014 minival | `compare_models_coco.py` |
| `rot_pred_viz/` | predicted against ground-truth rotations on held-out BEDLAM2 | a 120k-step checkpoint, BEDLAM2 rotation shards (val) | `viz_rot_pred.py` |
| `rot6d_gt_viz/` | the per-pixel 6D rotation ground truth, channel by channel | BEDLAM2 rotation shards | `viz_rot6d.py` |
| `dl_gallery/` | the dense ground truth the data loader produces (input, parts, UV), without label resampling | BEDLAM2 shards | `make_dl_gallery.py` |
| `part_lut_check/` | check of the DensePose-24 part and UV lookup on SMPL-X renders | BEDLAM2 renders, SMPL-X | `viz_part_lut.py` |
| `threedpw_report/` | 3DPW: per-instance improvement or harm when our correspondences refine CameraHMR | CameraHMR 3DPW predictions and refinements, `dp_pp_llrd_1e3_120k/ckpt_120000` | `report_samples.py` |
| `threedpw_report_heat/` | the same study as error heat maps (λ = 1.0, λ_rot = 0.003) | as above | `report_heatmaps.py` |

The run names are research training runs (`research/README`); only the final model is released.

**Other generators here:**
- `framefield_html.py` and `framefield_anim_html.py` are earlier versions of `framefield.html`, built on 3DPW poses.
- `framefield_reproj_html.py` is an alternative reprojection page.
- `framefield_studio.py` writes still images, not a page.

## The two stages: export, then page

Pages that show model output do not run the model. A model-forward **export** step writes per-frame files, and the
page generators read them.

1. **Point export** for `framefield*.html`: `framefield_overlay.py CLIP.mp4 --n 340 --export-blender DIR --video
   --maxframes 150`, and `--export-template DIR_TMPL` once. Per frame it writes:
   - `bg_NNNN.png`: the crop;
   - `pts_NNNN.csv`: one row per tracked point visible in that frame, with columns `idx,u,v,qw,qx,qy,qz,twist`
     (point index, position in the crop in [0,1], the frame as a quaternion, twist).
   The template folder holds `tmpl_body.obj` and `tmpl_pts.csv` (`x,y,z,qw,qx,qy,qz,twist`, template metres).
2. **Field dump** for the beauty renders: `field_dump.py CLIP --out DIR --video --maxframes 150`. Per frame it writes
   an NPZ with the keys `mask`, `part` (1 to 24, 0 = background), `uv` (2,H,W), `n_cam` and `t_cam` (camera-space
   normal and combing tangent, R · template direction) and `twist`, plus a JPG and `meta.json`.

`demo.py --dump` (repository root) is meant to write the same field-dump keys, so that its output can replace
step 2. Both export scripts take `--ckpt`; the released weights work there too, being the same state dict
(`--ckpt pff_dp_w_MF_3k.pt`). They need a GPU: the model, the RT-DETR person detector and the template frames are
placed on `cuda`.

## Running a generator

From the repository root:

```bash
python viz/csc_report.py --out /tmp/csc_report            # default output: viz/out/<page>
python viz/framefield_pair_html.py --dir EXPORT_DIR --tmpl TEMPLATE_DIR --out framefield.html
python viz/framefield_multi_html.py --tmpl TEMPLATE_DIR \
    --clips dance=DIR_15731089,slow=DIR_15153663,two-person=DIR_28513779,crowd=DIR_37828157
python viz/beauty_report_assets.py --beauty BEAUTY_DIR --out ASSETS_DIR && python viz/beauty_report.py
```

`viz/_env.py`, imported first by every script, makes them run from this repository:

- **Research code:** `research/densepose` and `research/bedlam` go on `sys.path`, for the research model, data
  loader and `orient/` helpers.
- **Lookup tables:** read from `assets/` (`pff.assets`; override with `PFF_ASSETS`).
- **Research data paths:** `~/data/<path>` becomes `$PFF_DATA/<path>` (default `~/data`, as on the cluster). This
  covers inputs, checkpoints and shards.
- **CameraHMR files:** `$CAMERAHMR` (default `~/codes/CameraHMR`).
- **SMPL-X body model:** `$SMPLX_DIR` (default `~/data/smplx/models_lockedhead`; SMPL-X licence, not included).
- **`--out`:** every script accepts it; without it the output goes to `viz/out/` (git-ignored).

**What else they need:**
- `threedpw_*`: the CameraHMR repository with its 3DPW predictions, refinements and SMPL model, and 3DPW images.
- `dl_gallery`, `part_lut_check`, `rot6d_gt_viz`, `rot_pred_viz`: the BEDLAM2 shards and renders.
- `compare_coco`: COCO val2014 minival.
- Anything that runs the model: a GPU.

## What was checked (14 October 2026, on the cluster, CPU only)

- **Regenerated byte for byte**, from the original inputs: `framefield.html` (from the original export),
  `csc_ablation.html`, `charts_v1v2.html` and `framefield_beauty.html` (through `beauty_report_assets.py` then
  `beauty_report.py`).
- **`framefield_multi.html`:** regenerates with the same clips, JavaScript and structure, but each clip's embedded
  media are 3 to 4 % larger than in the July page. This is most likely a newer image or video encoder; the content
  is the same.
- **`framefield_reproj_html.py`:** runs on the original export.
- **Every generator and both exporters:** compile, and every import resolves to this repository's code.
- **Not run:**
  - the generators that need the model, the BEDLAM2 shards, SMPL-X or CameraHMR (`framefield_html`,
    `framefield_anim_html`, `framefield_studio`, `compare_models_coco`, `viz_rot_pred`, `viz_rot6d`,
    `make_dl_gallery`, `viz_part_lut`, `report_samples`, `report_heatmaps`);
  - the two exporters, which need a GPU.

On the cluster, `research/densepose/orient/rot6d.py` adds `~/codes/wan/bedlam` to `sys.path` itself, so there it
loads that identical copy of `render_correspondence.py`; elsewhere the path does not exist and this repository's copy
is used.

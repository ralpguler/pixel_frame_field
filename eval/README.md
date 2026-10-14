# Evaluation

Two evaluations, both ports of the research scripts that produced the paper's numbers.

## DensePose-COCO GPS (`coco_gps.py`)

The official DensePose geodesic point similarity (GPS) evaluation on COCO 2014 minival, ground-truth-box protocol:
every DensePose-annotated person box is a detection of score 1. Port of `research/densepose/dpeval/run_eval.py`;
the evaluator `densepose_cocoeval.py` is the research copy, unchanged.

The defaults are the paper's settings (`research/slurm/mf_gate.sbatch`):

- input 512;
- the v2 -> v1 chart remap (`assets/uv_v2_to_v1.npy`). The model predicts the fold-free v2 charts on the hands and
  feet, while the COCO annotations use the official v1 charts; pass `--uv-remap none` only for a v1-chart model.

**Data in `eval/data/`.** `SMPL_subdiv.mat` and `SMPL_SUBDIV_TRANSFORM.mat` are included. Two files are not (they
are git-ignored); copy them in from the DensePose release:

- `Pdist_matrix.mat` (1.3 GB): the geodesic distance table of the official DensePose evaluation data;
- `densepose_coco_2014_minival.json` (31 MB): the DensePose-COCO minival annotations.

The images are the COCO val2014 files listed in minival (`--images`).

```bash
python eval/coco_gps.py --ckpt checkpoints/pff_dp_w_MF_3k.pt --images /data/coco/val2014
```

Scoring holds the geodesic table in memory (it peaked at 3.1 GB here). On a machine with little memory, predict and score in
two processes; the scoring one never loads the model:

```bash
python eval/coco_gps.py --ckpt checkpoints/pff_dp_w_MF_3k.pt --images /data/coco/val2014 \
    --save-dets dets.pkl --predict-only
python eval/coco_gps.py --load-dets dets.pkl
```

`--boxes dets.json` ({image_id: [[x, y, w, h, score], ...]}) scores given detections instead of the ground-truth
boxes; `--limit N` scores the first N images only.

**Reference** (the first line of the summary, AP at OGPS 0.50:0.95):

| checkpoint | GPS AP |
|---|---|
| **dp_w_MF step 3,000 (released)** | **0.741** |
| dp_w_MF steps 1k / 2k / 5k / 10k | 0.713 / 0.729 / 0.737 / 0.734 |

## Synthetic rotation and surface accuracy (`bedlam_val.py`)

Port of `research/densepose/orient/eval_512.py`, on the fixed 512 x 512 ground-truth canvas of the synthetic shards
(the BEDLAM2 `val` split of `dp_shards_512`, or the held-out MAMMA shards). The prediction is warped into the canvas
as in training, and the rotation and chart coordinates are read at the PREDICTED part (end to end). It reports:

- `geo_e2e`: geodesic rotation error (degrees);
- `surf_err`: distance on the template between the predicted and true surface point (mm, through the CSC table,
  which must be built first: `python -m pff.csc --build -K 512`);
- part accuracy, hands-and-feet and head breakdowns;
- the share of misplaced rotation discontinuities at part boundaries (spurious or missed), which a mean over pixels
  cannot show.

`geo_e2e` and `surf_err` are, as in the research script and the paper, the **mean over batches of per-batch
medians**; the pooled mean and median over all pixels are printed as well.

```bash
python eval/bedlam_val.py --ckpt checkpoints/pff_dp_w_MF_3k.pt --shards '/data/dp_shards_512/val/shard-*.tar'
```

Defaults: 512 samples, batch 16, 6 loader workers (on a GPU). On a CPU with little memory use `--batch 1 --workers 0`.

**Reference** (research logs, BEDLAM2 val, `geo_e2e` / `surf_err` / hands-and-feet `surf_err`): dp_w_M (stage 1)
6.86 deg / 8.22 mm / 15.09 mm. Held-out MAMMA `geo_e2e`: dp_w_M 6.89 deg, dp_w_MF at 3k 8.24 deg.

## Checked against the research code

On a CPU (no GPU was available), with the released weights:

- `coco_gps.py`: the five predictions of the first three minival images (part, U, V pasted into the box) are
  identical to those of the research `run_eval.py` loop run on the same images; prediction and scoring run end to
  end.
- `bedlam_val.py`: on the first four BEDLAM2 val samples it prints exactly the numbers of the research
  `eval_512.py` (geo_e2e 5.76 deg, surf_err 6.65 mm, part accuracy 96.7 %, misplaced discontinuities 2.97 %).

The full evaluations were not rerun.

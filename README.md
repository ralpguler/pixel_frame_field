# Pixel Frame Fields (Oriented DensePose)

For every pixel of a person, the DensePose correspondence (body part and chart coordinates U,V on the SMPL-X based
DensePose surface) **and a surface frame**: the rotation that takes the body template's local surface axes to the
camera. Paper: "Oriented DensePose: From Surface Points to Surface Frames" (`research/paper/`).

| path | what |
|---|---|
| `pff/` | the model (`model.py`), the read-out of rotations and parts (`geometry.py`), losses, the canonical-surface-coordinate operator (`csc.py`), training data loaders (`data/`), inference |
| `train.py`, `configs/` | the two-stage training recipe of the released model |
| `eval/` | DensePose-COCO GPS evaluation and the synthetic rotation evaluation |
| `demo.py` | images and videos: part, UV and surface-normal panels, raw per-frame dumps |
| `viz/` | the interactive HTML visualisations (`viz/pages/`) and their generators |
| `assets/` | SMPL-X and DensePose lookup tables (`assets/README.md`) |
| `research/` | the code as it was run, laid out as on the cluster under `~/codes/wan` (model lineage, ablations, paper, job scripts, evaluation reports) |

## The model

DINOv3 ViT-L/16 backbone, a compact DPT decoder and four dense heads on a 512 x 512 person crop: body mask, 24 part
logits, 24 x (U,V), and 24 x 6D rotations (each read at the pixel's own part). Released checkpoint: **dp_w_MF, step
3,000** (1.26 GB, weights only), 0.741 GPS AP on DensePose-COCO minival with ground-truth boxes.

```python
from pff.model import load_model
from pff.geometry import read_out
model = load_model("checkpoints/pff_dp_w_MF_3k.pt")
maps = read_out(model(crop))     # crop (B,3,512,512) RGB in [0,1] -> part (B,H,W) 0..24, uv (B,2,H,W), R (B,H,W,3,3)
```

**Checked against the research code** (CPU, 14 October 2026); each of these is identical bit for bit:

- the clean model with the released weights: all four outputs on the same input (maximum difference 0.0);
- the synthetic data loader on real shard samples, and the COCO loader on minival instances;
- both losses on a real batch, every term;
- the optimiser parameter groups (26 layer-wise groups in stage 1, 2 in stage 2) and the learning-rate schedule;
- the CSC table rebuilt from `assets/`.

No training step was run here (no GPU was available).

## Training

```bash
python -m pff.csc --build -K 512                                                    # once
torchrun --standalone --nproc_per_node=4 train.py --config configs/stage1_synthetic.yaml
torchrun --standalone --nproc_per_node=4 train.py --config configs/stage2_joint_coco.yaml
```

Rerunning a command resumes from the newest checkpoint in its `run_dir`.

| stage | data | steps | optimiser | research run |
|---|---|---|---|---|
| 1 | BEDLAM2 + MAMMA renders, about 50/50 | 120k, 4 GPUs x 32 | AdamW, lr 3e-4, layer-wise decay 0.75, warm-up 500, cosine | `dp_w_M` |
| 2 | the same + DensePose-COCO on every second step | 10k (released: 3k) | AdamW, lr 5e-5, backbone x 0.1, warm-up 200, cosine | `dp_w_MF` |

**Losses.**

- Synthetic batches: body-mask BCE + part cross-entropy + UV smooth-L1 + rotation (squared chordal distance, weight 1)
  + canonical surface coordinates (template-space L1 in metres, weight 2, and its multi-scale gradient, weight 5).
  All are computed in a fixed 512 x 512 ground-truth canvas that is never resampled; the prediction is warped into it.
- COCO batches: body-mask BCE + part cross-entropy + UV smooth-L1 at the annotated points. There is no UV term on the
  hands and feet, whose v1 charts in COCO differ from the model's fold-free v2 charts.

**Data** (paths in the configs):

- Synthetic WebDataset shards (one person instance per sample: RGB canvas, per-pixel SMPL-X face and barycentrics in
  a fixed 512 x 512 canvas, per-vertex rotations):
  - `dp_shards_512`: BEDLAM2, 711 GB, from `research/bedlam/render_gt512.py` then `research/densepose/reshard_gt512.py`;
  - `dp_shards_512_mamma`: MAMMA, 907 GB, from `research/densepose/orient/mamma_to_dp_shards.py`
    (`research/slurm/mamma512.sbatch`).
- DensePose-COCO: the official annotations (`densepose_coco_2014_train.json`, `densepose_coco_2014_minival.json`)
  with the COCO 2014 images.

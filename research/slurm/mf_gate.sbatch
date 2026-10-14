#!/bin/bash
#SBATCH --job-name=mf-gate
#SBATCH --account=brics.u6ga
#SBATCH --partition=workq
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=120G
#SBATCH --time=05:00:00
#SBATCH --output=/home/u6ga/raguler.u6ga/codes/wan/slurm/logs/%x-%j.out
# MF gate: the UNCONTAMINATED two-stage frontier — joint ft from W-M (clean seed that never saw
# the holdout, unlike F0's dp_csc_C lineage). COCO GPS on early ckpts + holdout forgetting curve.
set -uo pipefail
export HF_HOME=$HOME/data/hf_home TORCH_COMPILE_DISABLE=1 CUDA_VISIBLE_DEVICES=0
PY=$HOME/codes/wan/.venv/bin/python
D=$HOME/data/bedlam2/dp_runs
ARGS="--backbone vit_large_patch16_dinov3 --layers 4,11,17,23 --input 512"
RM="--uv-remap $HOME/data/bedlam2/dp_assets/uv_v2_to_v1.npy"
HOLD=$HOME/data/datasets/dp_shards_rot_mamma/holdout_links

cd ~/codes/wan/densepose/dpeval
for CK in $D/dp_w_MF/ckpt_001000.pt $D/dp_w_MF/ckpt_002000.pt $D/dp_w_MF/ckpt_003000.pt \
          $D/dp_w_MF/ckpt_005000.pt $D/dp_w_MF/ckpt_010000.pt; do
  [ -e "$CK" ] || continue
  echo "===== COCO: $CK ====="
  $PY run_eval.py $ARGS --ckpt $CK $RM
done

cd ~/codes/wan/densepose
echo "===== MAMMA HELD-OUT curve: dp_w_MF ====="
$PY -m orient.indomain_probe --run dp_w_MF --ckpts 1000 2000 3000 5000 7000 10000 \
  --shards "$HOLD/*.tar" --nshards 59 --batches 40
echo "F0B GATE DONE"

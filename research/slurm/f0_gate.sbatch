#!/bin/bash
#SBATCH --job-name=f0-gate
#SBATCH --account=brics.u6ga
#SBATCH --partition=workq
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=120G
#SBATCH --time=03:00:00
#SBATCH --output=/home/u6ga/raguler.u6ga/codes/wan/slurm/logs/%x-%j.out
# The F0 gate: did the joint COCO ft (0.568 -> 0.730 AP) COST anything on the synthetic side?
# eval_512 (BEDLAM-val: geo_e2e/surf/seams) + indomain_probe on the HELD-OUT MAMMA sequences
# (occlusion), F0-final vs its seed dp_csc_C.
set -uo pipefail
cd ~/codes/wan/densepose
export HF_HOME=$HOME/data/hf_home TORCH_COMPILE_DISABLE=1 CUDA_VISIBLE_DEVICES=0
PY=$HOME/codes/wan/.venv/bin/python
D=$HOME/data/bedlam2/dp_runs
HOLD=$HOME/data/datasets/dp_shards_rot_mamma/holdout_links

echo "===== eval_512: dp_csc_C (seed) ====="
$PY -m orient.eval_512 --ckpt $D/dp_csc_C/ckpt_025000.pt --n 512
echo "===== eval_512: dp_w_F0 (joint ft) ====="
$PY -m orient.eval_512 --ckpt $D/dp_w_F0/ckpt_010000.pt --n 512

echo "===== MAMMA HELD-OUT probe: dp_csc_C ====="
$PY -m orient.indomain_probe --run dp_csc_C --ckpts 25000 --shards "$HOLD/*.tar" --nshards 59 --batches 40
echo "===== MAMMA HELD-OUT probe: dp_w_F0 ====="
$PY -m orient.indomain_probe --run dp_w_F0 --ckpts 10000 --shards "$HOLD/*.tar" --nshards 59 --batches 40
echo "F0 GATE DONE"

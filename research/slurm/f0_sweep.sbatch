#!/bin/bash
#SBATCH --job-name=f0-sweep
#SBATCH --account=brics.u6ga
#SBATCH --partition=workq
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=120G
#SBATCH --time=04:00:00
#SBATCH --output=/home/u6ga/raguler.u6ga/codes/wan/slurm/logs/%x-%j.out
# F0 forgetting curve: held-out MAMMA occlusion probe across ALL F0 ckpts (1k..10k).
# COCO already hits 0.730 @5k; find the knee where occlusion (MAMMA skill) is still intact.
set -uo pipefail
cd ~/codes/wan/densepose
export HF_HOME=$HOME/data/hf_home TORCH_COMPILE_DISABLE=1 CUDA_VISIBLE_DEVICES=0
PY=$HOME/codes/wan/.venv/bin/python
HOLD=$HOME/data/datasets/dp_shards_rot_mamma/holdout_links
$PY -m orient.indomain_probe --run dp_w_F0 \
  --ckpts 1000 2000 3000 4000 5000 6000 7000 8000 9000 10000 \
  --shards "$HOLD/*.tar" --nshards 59 --batches 40
echo "F0 SWEEP DONE"

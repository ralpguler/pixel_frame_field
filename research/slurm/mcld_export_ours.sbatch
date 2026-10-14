#!/bin/bash
#SBATCH --job-name=mcld-export
#SBATCH --account=brics.u6ga
#SBATCH --partition=workq
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=120G
#SBATCH --time=16:00:00
#SBATCH --output=/home/u6ga/raguler.u6ga/codes/wan/slurm/logs/%x-%j.out
# Arm B inputs: OUR DensePose (dp_w_F0 @3k — 0.730 COCO AP, speaks the COCO convention the MCLD
# ecosystem was trained on) exported in MCLD's canvas format over the full DeepFashion split.
# Resumable (skips existing files).
set -uo pipefail
cd ~/codes/wan/densepose
export HF_HOME=$HOME/data/hf_home TORCH_COMPILE_DISABLE=1 CUDA_VISIBLE_DEVICES=0
PY=$HOME/codes/wan/.venv/bin/python
CK=$HOME/data/bedlam2/dp_runs/dp_w_F0/ckpt_003000.pt
F=$HOME/data/mcld/fashion

for SPLIT in test train; do
  echo "===== exporting $SPLIT ====="
  $PY -m orient.export_mcld_densepose --images $F/$SPLIT --out $F/${SPLIT}_densepose_ours --ckpt $CK
done
echo "EXPORT DONE"

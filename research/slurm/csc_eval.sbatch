#!/bin/bash
#SBATCH --job-name=csc-eval
#SBATCH --account=brics.u6ga
#SBATCH --partition=workq
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=120G
#SBATCH --time=03:00:00
#SBATCH --output=/home/u6ga/raguler.u6ga/codes/wan/slurm/logs/%x-%j.out
# eval_512 on the A/C/D/E final checkpoints: geo_e2e + surf_mm + misplaced-discontinuity rate.
set -uo pipefail
cd ~/codes/wan/densepose
export HF_HOME=$HOME/data/hf_home TORCH_COMPILE_DISABLE=1 CUDA_VISIBLE_DEVICES=0
PY=$HOME/codes/wan/.venv/bin/python
D=$HOME/data/bedlam2/dp_runs
for R in dp_csc_A dp_csc_C dp_csc_D dp_csc_E; do
  CK=$(ls -v $D/$R/ckpt_*.pt 2>/dev/null | tail -1)
  [ -n "$CK" ] && echo "=== $R ($CK) ===" && $PY -m orient.eval_512 --ckpt $CK --n 512
done
echo "CSC EVAL DONE"

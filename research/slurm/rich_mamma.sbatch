#!/bin/bash
#SBATCH --job-name=rich-mamma
#SBATCH --account=brics.u6ga
#SBATCH --partition=workq
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH --mem=200G
#SBATCH --time=12:00:00
#SBATCH --output=/home/u6ga/raguler.u6ga/codes/wan/slurm/logs/%x-%j.out
# RICH test with the +MAMMA-120k DensePose model providing the correspondence, vs the pure-BEDLAM-120k
# result (baseline CameraHMR + old corr are cached & reused). Extract corr with the MAMMA ckpt into a
# SEPARATE npz, refine with --tag _mamma (no clobber), print the 3-way table.
set -uo pipefail
CHMR=$HOME/codes/CameraHMR
CHMR_PY=$CHMR/.venv/bin/python
WAN_PY=$HOME/codes/wan/.venv/bin/python
export CUDA_VISIBLE_DEVICES=0 HF_HOME=$HOME/data/hf_home TORCH_COMPILE_DISABLE=1
MAMMA_CKPT=$HOME/data/bedlam2/dp_runs/dp_pp_llrd_1e3_mamma/ckpt_120000.pt
CORR=$CHMR/dp_corr_rich_mamma.npz

echo "==================== STEP 1: DensePose corr with +MAMMA-120k ===================="
cd "$HOME/codes/wan/densepose"
$WAN_PY -u extract_dp_corr_rich.py --batch 16 --ckpt "$MAMMA_CKPT" --out "$CORR" || { echo "EXTRACT FAILED"; exit 1; }
[ -s "$CORR" ] || { echo "no corr npz"; exit 1; }

echo "==================== STEP 2: refinement + 3-way table (+MAMMA corr) ===================="
cd "$CHMR"
$CHMR_PY -u refine_rich.py --mode full --only both --lam_rot 0.003 --corr "$CORR" --tag _mamma || { echo "REFINE FAILED"; exit 1; }
echo "==================== RICH +MAMMA PIPELINE DONE ===================="

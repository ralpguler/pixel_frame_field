#!/bin/bash
#SBATCH --job-name=rich-pipe
#SBATCH --account=brics.u6ga
#SBATCH --partition=workq
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH --mem=200G
#SBATCH --time=12:00:00
#SBATCH --output=/home/u6ga/raguler.u6ga/codes/wan/slurm/logs/%x-%j.out
# Full RICH-test pipeline: CameraHMR baseline -> DensePose correspondences ->
# oriented-DensePose refinement (loc / loc+ori) -> 3-way PA-MPJPE/MPJPE/PVE table.
# Mirrors the completed 3DPW pipeline. Two venvs: CameraHMR (.venv) for the SMPL
# forward + refinement, wan (.venv) for the DensePose correspondence extraction.
set -uo pipefail
CHMR=$HOME/codes/CameraHMR
CHMR_PY=$CHMR/.venv/bin/python
WAN_PY=$HOME/codes/wan/.venv/bin/python
export CUDA_VISIBLE_DEVICES=0

PREDS=$CHMR/cameramhr_rich_preds.npz
CORR=$CHMR/dp_corr_rich.npz

echo "==================== STEP 1: CameraHMR baseline (21287) ===================="
cd "$CHMR"
$CHMR_PY -u eval_rich_standalone.py --out cameramhr_rich_preds.npz || { echo "BASELINE FAILED"; exit 1; }
[ -s "$PREDS" ] || { echo "no preds npz"; exit 1; }

echo "==================== STEP 2: DensePose correspondences ===================="
cd "$HOME/codes/wan/densepose"
$WAN_PY -u extract_dp_corr_rich.py --batch 16 || { echo "EXTRACT FAILED"; exit 1; }
[ -s "$CORR" ] || { echo "no corr npz"; exit 1; }

echo "==================== STEP 3a: geometry sanity (GT-mesh reproj) ===================="
cd "$CHMR"
$CHMR_PY -u refine_rich.py --mode sanity --n 200
echo "==================== STEP 3b: rotation frame-consistency gate ===================="
$CHMR_PY -u refine_rich.py --mode rotsanity --n 200

echo "==================== STEP 4: full refinement + 3-way table ===================="
$CHMR_PY -u refine_rich.py --mode full --only both --lam_rot 0.003 || { echo "REFINE FAILED"; exit 1; }

echo "==================== RICH PIPELINE DONE ===================="

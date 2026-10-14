#!/bin/bash
#SBATCH --job-name=report-heat
#SBATCH --account=brics.u6ga
#SBATCH --partition=workq
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=72
#SBATCH --mem=120G
#SBATCH --time=01:00:00
#SBATCH --output=/home/u6ga/raguler.u6ga/codes/wan/slurm/logs/%x-%j.out
# CameraHMR refinement heatmap report (top/bottom 10%): image|before|after|GT|DensePose|loc-loss|rot-loss.
# Resumable (skips existing panel jpgs) so a time limit / requeue is harmless.
set -uo pipefail
cd ~/codes/wan/densepose/orient
export CUDA_VISIBLE_DEVICES=0
export TMPDIR=/local/user/$(id -u)/heat_scratch; mkdir -p "$TMPDIR"
~/codes/wan/.venv/bin/python -u report_heatmaps.py --pct 10 --n-show 12
echo "[$(date +%F\ %T)] report-heat done -> ~/codes/CameraHMR/viz/report_heat/index.html"

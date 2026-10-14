#!/bin/bash
#SBATCH --job-name=nlf-joint
#SBATCH --account=brics.u6ga
#SBATCH --partition=workq
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=72
#SBATCH --mem=120G
#SBATCH --time=03:00:00
#SBATCH --output=/home/u6ga/raguler.u6ga/codes/wan/slurm/logs/%x-%j.out
# NLF +fit JOINT orientation integration: reimplement NLF's fit (uncertainty-weighted nonparam
# verts+joints) + our orientation term jointly; sweep lam_rot to find the peak. Runs on a COMPUTE
# node (login-node watchdog kills long interactive GPU jobs).
set -uo pipefail
cd ~/codes/CameraHMR
export CUDA_VISIBLE_DEVICES=0
export TMPDIR=/local/user/$(id -u)/nlf_scratch
mkdir -p "$TMPDIR"
~/codes/CameraHMR/.venv/bin/python -u refine_nlf_joint.py \
  --npz ~/codes/nlf/nlf_nonparam_3dpw_sub.npz --iters 300 --lr 0.02 --lam 0.05 \
  --sweep_rot 0.0,0.001,0.003,0.006,0.01,0.02,0.04
echo "[$(date +%F\ %T)] nlf-joint done"

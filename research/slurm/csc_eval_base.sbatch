#!/bin/bash
#SBATCH --job-name=csc-eval-base
#SBATCH --account=brics.u6ga
#SBATCH --partition=workq
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=120G
#SBATCH --time=01:00:00
#SBATCH --output=/home/u6ga/raguler.u6ga/codes/wan/slurm/logs/%x-%j.out
# The missing baseline row: the pre-ablation +MAMMA ckpt_120000 on the SAME clean 512 val,
# plus a field dump of the hero clip with the winning C checkpoint for the qualitative check.
set -uo pipefail
cd ~/codes/wan/densepose
export HF_HOME=$HOME/data/hf_home TORCH_COMPILE_DISABLE=1 CUDA_VISIBLE_DEVICES=0 MPLBACKEND=Agg
PY=$HOME/codes/wan/.venv/bin/python
$PY -m orient.eval_512 --ckpt $HOME/data/bedlam2/dp_runs/dp_pp_llrd_1e3_mamma/ckpt_120000.pt --n 512
$PY -m orient.field_dump.py 2>/dev/null || true
$PY -u orient/field_dump.py $HOME/data/datasets/pexels_hifps/clips/15731089_conformed16.mp4 \
    --out $HOME/data/dp_viz_compare/beauty/15731089_cscC --video --maxframes 150 \
    --ckpt $HOME/data/bedlam2/dp_runs/dp_csc_C/ckpt_025000.pt
echo "BASE EVAL + C DUMP DONE"

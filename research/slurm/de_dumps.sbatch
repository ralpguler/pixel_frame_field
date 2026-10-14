#!/bin/bash
#SBATCH --job-name=de-dumps
#SBATCH --account=brics.u6ga
#SBATCH --partition=workq
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=80G
#SBATCH --time=00:45:00
#SBATCH --output=/home/u6ga/raguler.u6ga/codes/wan/slurm/logs/%x-%j.out
set -uo pipefail
cd ~/codes/wan/densepose
export HF_HOME=$HOME/data/hf_home TORCH_COMPILE_DISABLE=1 CUDA_VISIBLE_DEVICES=0 MPLBACKEND=Agg
PY=$HOME/codes/wan/.venv/bin/python
CLIPS=$HOME/data/datasets/pexels_hifps/clips
B=$HOME/data/dp_viz_compare/beauty
for R in D E; do
  CK=$HOME/data/bedlam2/dp_runs/dp_csc_$R/ckpt_025000.pt
  $PY -u orient/field_dump.py $CLIPS/15731089_conformed16.mp4 --out $B/15731089_csc$R \
      --frame 75 --ckpt $CK --uv-asset smplx_face_uv_v2.npy
  $PY -u orient/field_dump.py $CLIPS/28513779_conformed16.mp4 --out $B/28513779_csc$R \
      --frame 75 --ckpt $CK --uv-asset smplx_face_uv_v2.npy
done
echo "DE DUMPS DONE"

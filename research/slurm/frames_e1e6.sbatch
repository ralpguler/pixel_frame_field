#!/bin/bash
#SBATCH --job-name=frames-e1e6
#SBATCH --account=brics.u6ga
#SBATCH --partition=workq
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=120G
#SBATCH --time=02:30:00
#SBATCH --output=/home/u6ga/raguler.u6ga/data/dp_frames_eval/logs/%x-%j.out
# E1/E6 (paper/REFRAME.md), evaluation only. Protocol: ~/data/dp_frames_eval/PROTOCOL.txt
#   sbatch --export=ALL,SPLIT=S1 frames_e1e6.sbatch     (and SPLIT=S2)
set -uo pipefail
export HF_HOME=$HOME/data/hf_home TORCH_COMPILE_DISABLE=1 CUDA_VISIBLE_DEVICES=0
PY=$HOME/codes/wan/.venv/bin/python
cd ~/codes/wan/densepose
HOLD=$HOME/data/datasets/dp_shards_rot_mamma/holdout_links
if [ "$SPLIT" = "S1" ]; then
  echo "===== ORIGINAL eval_512 (unmodified) dp_w_M ====="
  $PY -m orient.eval_512 --ckpt $HOME/data/bedlam2/dp_runs/dp_w_M/ckpt_120000.pt --n 512
else
  echo "===== ORIGINAL indomain_probe (unmodified) dp_w_M ====="
  $PY -m orient.indomain_probe --run dp_w_M --ckpts 120000 --shards "$HOLD/*.tar" --nshards 59 --batches 40
fi
for CK in dp_w_M/ckpt_120000.pt dp_csc_C/ckpt_025000.pt dp_w_MF/ckpt_003000.pt dp_v21_M/ckpt_120000.pt dp_w_B/ckpt_120000.pt; do
  echo "===== E1/E6 $SPLIT $CK ====="
  $PY -m orient.eval_frames_e1e6 run --split $SPLIT --ckpt $CK 2>&1 | grep -v -i warning
done
echo "===== FLOOR $SPLIT ====="
$PY -m orient.eval_frames_e1e6 floor --split $SPLIT --r 3 2>&1 | grep -v -i warning
echo "FRAMES E1E6 $SPLIT DONE"

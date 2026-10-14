#!/bin/bash
#SBATCH --job-name=mcld-evB
#SBATCH --account=brics.u6ga
#SBATCH --partition=workq
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=200G
#SBATCH --time=14:00:00
#SBATCH --output=/home/u6ga/raguler.u6ga/codes/wan/slurm/logs/%x-%j.out
# ARM B: generation with OUR-densepose conditioning (fashion_armB root via train_armB.yaml)
# on the common 8512-pair test set, then metrics vs the common-pair released GT.
# THE causal comparison vs arm A: same recipe, same pairs, only the correspondences differ.
set -uo pipefail
cd ~/codes/wan/mcld
export HF_HOME=$HOME/data/hf_home TORCH_COMPILE_DISABLE=1 CUDA_VISIBLE_DEVICES=0
PY=$HOME/data/mcld/venv/bin/python
OUT=$HOME/data/mcld/results_armB/

$PY test.py --save_path $OUT --ckpt_dir $HOME/data/mcld/exp/armB_final/ \
  --config_path ./configs/train/train_armB.yaml
$PY evaluate.py --img_path ${OUT}mcld_armB/ \
  --gt_path $HOME/data/mcld/released_results/gt_common/ \
  --training_path ./dataset/fashion/
echo "ARM B EVAL DONE"

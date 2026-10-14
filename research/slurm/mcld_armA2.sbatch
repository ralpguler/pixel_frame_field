#!/bin/bash
#SBATCH --job-name=mcld-A2
#SBATCH --account=brics.u6ga
#SBATCH --partition=workq
#SBATCH --nodes=1
#SBATCH --gres=gpu:2
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=200G
#SBATCH --time=16:00:00
#SBATCH --output=/home/u6ga/raguler.u6ga/codes/wan/slurm/logs/%x-%j.out
# ARM A' — the confound control. Identical to arm A in every way EXCEPT the pose-guider
# conditioning is warped into the images' own (mirror) geometry (align_their_densepose.py).
# Textures/face are the authors' originals unchanged: a UV atlas is geometry-independent, so
# exactly ONE variable differs from arm A.
#   A' vs A  = how much the misaligned conditioning actually cost (confound size)
#   A' vs B  = the clean correspondence comparison on mirror data
set -uo pipefail
cd ~/codes/wan/mcld
export HF_HOME=$HOME/data/hf_home TORCH_COMPILE_DISABLE=1
PY=$HOME/data/mcld/venv/bin/python
F=$HOME/data/mcld/fashion
R=$HOME/data/mcld/fashion_armA2

# 1. aligned conditioning (resumable; skips existing)
for s in test train; do
  [ -d "$F/${s}_densepose_aligned" ] && [ "$(ls $F/${s}_densepose_aligned | wc -l)" -ge "$(ls $F/${s}_densepose | wc -l)" ] && continue
  $PY align_their_densepose.py --split $s || exit 1
done

# 2. arm-A' root: mirror images + ALIGNED their-densepose + THEIR textures + their face
mkdir -p $R
for s in train test; do
  ln -sfn $F/$s                     $R/$s
  ln -sfn $F/${s}_densepose_aligned $R/${s}_densepose
  ln -sfn $F/${s}_texture           $R/${s}_texture
  ln -sfn $F/${s}_face              $R/${s}_face
done
ln -sfn $R dataset/fashion_armA2
sed -e "s|exp_name: 'mcld_armA'|exp_name: 'mcld_armA2'|" \
    -e "s|root_dir: \"dataset/fashion/\"|root_dir: \"dataset/fashion_armA2/\"|" \
    configs/train/train_armA.yaml > configs/train/train_armA2.yaml

# 3. train (resume-chained)
SAVE=$HOME/data/mcld/exp/mcld_armA2
CFG=configs/train/train_armA2.yaml
if ls $SAVE/checkpoint-* >/dev/null 2>&1; then
  RUN=configs/train/train_armA2_resume.yaml
  sed "s|resume_from_checkpoint: ''|resume_from_checkpoint: 'latest'|" $CFG > $RUN
  echo "[A2] resuming"
else
  RUN=$CFG; echo "[A2] fresh start"
fi
$HOME/data/mcld/venv/bin/accelerate launch --num_processes 2 --main_process_port 12420 \
  train.py --config $RUN
echo "[A2 done $(date +%F\ %T)]"

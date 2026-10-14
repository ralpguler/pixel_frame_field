#!/bin/bash
#SBATCH --job-name=mcld-tex
#SBATCH --account=brics.u6ga
#SBATCH --partition=workq
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=120G
#SBATCH --time=12:00:00
#SBATCH --output=/home/u6ga/raguler.u6ga/codes/wan/slurm/logs/%x-%j.out
# Arm B texture maps from OUR densepose (CPU pool; resumable).
set -uo pipefail
cd ~/codes/wan/mcld
~/data/mcld/venv/bin/python regen_textures_ours.py

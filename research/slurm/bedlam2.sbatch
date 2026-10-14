#!/bin/bash
#SBATCH --job-name=bedlam2-dl
#SBATCH --account=brics.u6ga
#SBATCH --partition=workq
#SBATCH --cpus-per-task=4
#SBATCH --time=24:00:00
#SBATCH --output=/home/u6ga/raguler.u6ga/codes/wan/slurm/logs/%x-%j.out

# Download BEDLAM2.0 movies (159 GB) + everything needed to reproject SMPL-X
# meshes onto the frames:
#   - mp4: the movies (66 tars)
#   - gt:  per-sequence camera intrinsics/extrinsics + scene GT (CSV+JSON, 3.7 GB)
#   - ground_truth/motions:   SMPL-X motion sequences (npz) + index csv
#   - ground_truth/camerahmr: processed per-image SMPL-X labels (6fps npz)
#   - ground_truth/database:  render database (sequence <-> assets mapping)
# All resumable; re-submitting continues where it stopped.
# NOTE: the SMPL-X body model itself (needed to pose meshes from the npz
# parameters) requires separate registration at https://smpl-x.is.tue.mpg.de/

set -uo pipefail
cd ~/data/bedlam2/downloader

U=$(grep '^BEDLAM_USER=' ~/codes/wan/.env | cut -d= -f2-)
P=$(grep '^BEDLAM_PASS=' ~/codes/wan/.env | cut -d= -f2-)
[[ -n "$U" && -n "$P" ]] || { echo "missing BEDLAM credentials in ~/codes/wan/.env"; exit 1; }

echo "[$(date +%F\ %T)] stage: gt (camera ground truth, 3.7 GB)"
printf '%s\n%s\n' "$U" "$P" | bash b2_download.sh gt

echo "[$(date +%F\ %T)] stage: mp4 movies (159 GB)"
printf '%s\n%s\n' "$U" "$P" | bash b2_download.sh mp4

echo "[$(date +%F\ %T)] stage: SMPL-X ground truth extras"
urle () { local LANG=C i x; for (( i = 0; i < ${#1}; i++ )); do x="${1:i:1}"; [[ "${x}" == [a-zA-Z0-9.~-] ]] && echo -n "${x}" || printf '%%%02X' "'${x}"; done; }
UU=$(urle "$U"); PP=$(urle "$P")
mkdir -p ../ground_truth
for f in \
  "ground_truth/camerahmr_6fps_npz/bedlam2_labels_processed.zip" \
  "ground_truth/database/bedlam2_render_database-20250318.zip" \
  "ground_truth/motions/b2_bedlam2amass_motion_training.csv" \
  "ground_truth/motions/b2_motions_npz_training.tar"; do
  name=$(basename "$f")
  echo "  fetching $name"
  wget --post-data "username=$UU&password=$PP" \
    "https://download.is.tue.mpg.de/download.php?domain=bedlam2&resume=1&sfile=$f" \
    -O "../ground_truth/$name" --continue --quiet || echo "  FAILED: $name"
done

echo "[$(date +%F\ %T)] all stages done"
du -sh ~/data/bedlam2/

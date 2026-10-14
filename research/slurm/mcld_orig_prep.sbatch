#!/bin/bash
#SBATCH --job-name=mo-prep
#SBATCH --account=brics.u6ga
#SBATCH --partition=workq
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=120G
#SBATCH --time=06:00:00
#SBATCH --output=/home/u6ga/raguler.u6ga/codes/wan/slurm/logs/%x-%j.out
# ORIGINAL-GEOMETRY REBUILD, step 1/5. Waits for Riza's rsync of img_highres.zip to land and
# go quiet, then decrypts+extracts (parallel, password never in argv) and builds
# fashion_orig/{train,test}. Fixes the confound: the mirror images (768x1024) did NOT match the
# authors' released densepose (750x1101), so arm A trained on MISALIGNED conditioning while arm
# B (our densepose, exported from the mirror) was aligned. With original images BOTH arms get
# aligned conditioning and the A-vs-B comparison becomes clean.
set -uo pipefail
cd ~/codes/wan/mcld_orig
Z=$HOME/data/datasets/deepfashion_inshop/img_highres.zip
PY=$HOME/data/mcld/venv/bin/python

echo "[prep] waiting for $Z to appear and stabilise..."
for i in $(seq 1 360); do            # up to 3h
  if [ -f "$Z" ]; then
    s1=$(stat -c %s "$Z"); sleep 30; s2=$(stat -c %s "$Z")
    # rsync writes a hidden temp file then renames; require size stable AND no temp left
    if [ "$s1" = "$s2" ] && ! ls $HOME/data/datasets/deepfashion_inshop/.img_highres.zip.* >/dev/null 2>&1; then
      echo "[prep] stable at $(numfmt --to=iec $s2)"; break
    fi
  fi
  sleep 30
done
[ -f "$Z" ] || { echo "FATAL: zip never arrived"; exit 1; }

$PY -c "
import zipfile,sys
try:
    z=zipfile.ZipFile('$Z'); n=len(z.infolist()); print(f'[prep] zip OK, {n} members')
except Exception as e:
    print('FATAL: bad zip:',e); sys.exit(1)
" || exit 1

$PY extract_orig.py || exit 1

# ---- arm-A root: original images + the AUTHORS' released conditions (already 750x1101) ----
F=$HOME/data/mcld/fashion
O=$HOME/data/mcld/fashion_orig
for s in train test; do
  ln -sfn $F/${s}_densepose $O/${s}_densepose
  ln -sfn $F/${s}_texture   $O/${s}_texture        # authors' (reference; cond re-points arm A)
  ln -sfn $F/${s}_face      $O/${s}_face
done
$PY - <<'EOF'
import os
from PIL import Image
O=os.path.expanduser("~/data/mcld/fashion_orig/")
n=sorted(os.listdir(O+"test"))[0]; b=os.path.splitext(n)[0]
img=Image.open(O+"test/"+n).size; dp=Image.open(O+"test_densepose/"+b+".png").size
print(f"[prep] GEOMETRY CHECK image {img} vs their densepose {dp} -> {'ALIGNED' if img==dp else 'STILL MISMATCHED'}")
EOF
echo "PREP DONE"

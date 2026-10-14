#!/bin/bash
#SBATCH --job-name=mo-cond
#SBATCH --account=brics.u6ga
#SBATCH --partition=workq
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=200G
#SBATCH --time=16:00:00
#SBATCH --output=/home/u6ga/raguler.u6ga/codes/wan/slurm/logs/%x-%j.out
# ORIGINAL-GEOMETRY REBUILD, step 2/5: arm-B conditions from the ORIGINAL images —
# our DensePose (dp_w_F0@3k) + textures regenerated from it. Then pair csvs + configs, and
# it submits the two training arms itself so the chain needs no human.
set -uo pipefail
export HF_HOME=$HOME/data/hf_home TORCH_COMPILE_DISABLE=1 CUDA_VISIBLE_DEVICES=0
O=$HOME/data/mcld/fashion_orig
OB=$HOME/data/mcld/fashion_origB
PY=$HOME/data/mcld/venv/bin/python
DPY=$HOME/codes/wan/.venv/bin/python

cd ~/codes/wan/densepose
for SPLIT in test train; do
  echo "===== our densepose on ORIGINAL $SPLIT ====="
  $DPY -m orient.export_mcld_densepose --images $O/$SPLIT --out $O/${SPLIT}_densepose_ours \
       --ckpt $HOME/data/bedlam2/dp_runs/dp_w_F0/ckpt_003000.pt || exit 1
done

cd ~/codes/wan/mcld_orig
# SYMMETRY (Riza's check, 2026-07-20): the texture atlas is DERIVED from densepose, so it must
# differ between arms — but it must not differ in PROVENANCE. Regenerate BOTH arms' atlases with
# the same code/params, so the only upstream difference is the densepose model itself.
echo "===== textures from THEIR densepose (arm A, our code) ====="
MCLD_ROOT=$O DP_SUFFIX=densepose TEX_SUFFIX=texture_repro $PY regen_textures_ours.py || exit 1
echo "===== textures from OUR densepose (arm B, same code) ====="
MCLD_ROOT=$O DP_SUFFIX=densepose_ours TEX_SUFFIX=texture_ours $PY regen_textures_ours.py || exit 1
# record how our regeneration compares with the authors' released atlases (provenance check)
$PY - <<'EOF'
import os, numpy as np, cv2
O=os.path.expanduser("~/data/mcld/fashion_orig/")
a,b=O+"test_texture/", O+"test_texture_repro/"
ns=[n for n in sorted(os.listdir(b))[:200] if os.path.exists(a+n)]
d=[np.abs(cv2.imread(a+n).astype(float)-cv2.imread(b+n).astype(float)).mean() for n in ns]
print(f"[cond] provenance check: our regen vs authors' released texture, "
      f"mean|diff| {np.mean(d):.2f}/255 over {len(ns)} atlases", flush=True)
EOF
# arm A now reads the REGENERATED atlas (same code as arm B); authors' stays for reference
for s in train test; do ln -sfn $O/${s}_texture_repro $O/${s}_texture_armA; done

# ---- arm-B root ----
mkdir -p $OB
for s in train test; do
  ln -sfn $O/$s                    $OB/$s
  ln -sfn $O/${s}_densepose_ours   $OB/${s}_densepose
  ln -sfn $O/${s}_texture_ours     $OB/${s}_texture
  ln -sfn $HOME/data/mcld/fashion/${s}_face $OB/${s}_face
done

# ---- pair csvs: intersection over BOTH arms' conditions, identical for A and B ----
$PY - <<'EOF'
import os, shutil, pandas as pd
O=os.path.expanduser("~/data/mcld/fashion_orig/")
for split in ("train","test"):
    bak=f"./data/{split}_armA.csv"; cur=f"./data/{split}_orig.csv"
    df=pd.read_csv(bak)
    img=set(os.listdir(O+split))
    dpA=set(os.listdir(O+f"{split}_densepose"));  texA=set(os.listdir(O+f"{split}_texture_repro"))   # arm A reads the REGENERATED atlas
    dpB=set(os.listdir(O+f"{split}_densepose_ours")); texB=set(os.listdir(O+f"{split}_texture_ours"))
    def ok(r):
        pf=os.path.splitext(r["from"])[0]+".png"; pt=os.path.splitext(r["to"])[0]+".png"
        return (r["from"] in img and r["to"] in img and
                pt in dpA and pf in texA and pt in dpB and pf in texB)
    keep=df[df.apply(ok,axis=1)]
    keep.to_csv(cur,index=False)
    print(f"{split}: {len(keep)}/{len(df)} pairs common to BOTH arms")
EOF

# configs: same recipe, only root_dir differs; csv switch handled by test/train scripts
for ARM in A B; do
  ROOT=$([ $ARM = A ] && echo "dataset/fashion_origA/" || echo "dataset/fashion_origB/")
  sed -e "s|exp_name: 'mcld_armA'|exp_name: 'mcld_orig$ARM'|" \
      -e "s|root_dir: \"dataset/fashion/\"|root_dir: \"$ROOT\"|" \
      configs/train/train_armA.yaml > configs/train/train_orig$ARM.yaml
done
OA=$HOME/data/mcld/fashion_origA
mkdir -p $OA
for s in train test; do
  ln -sfn $O/$s $OA/$s; ln -sfn $O/${s}_densepose $OA/${s}_densepose
  ln -sfn $O/${s}_texture_repro $OA/${s}_texture
  ln -sfn $HOME/data/mcld/fashion/${s}_face $OA/${s}_face
done
ln -sfn $O  dataset/fashion_orig
ln -sfn $OA dataset/fashion_origA
ln -sfn $OB dataset/fashion_origB
cp data/test_orig.csv data/test.csv && cp data/train_orig.csv data/train.csv

cd ~/codes/wan/slurm
A1=$(sbatch --parsable --job-name=mo-trainA mcld_orig_train.sbatch)
A2=$(sbatch --parsable --job-name=mo-trainA --dependency=afterany:$A1 mcld_orig_train.sbatch)
B1=$(sbatch --parsable --job-name=mo-trainB mcld_orig_train.sbatch)
B2=$(sbatch --parsable --job-name=mo-trainB --dependency=afterany:$B1 mcld_orig_train.sbatch)
echo "[cond] submitted armA $A1->$A2   armB $B1->$B2"
sbatch --dependency=afterany:$A2 --job-name=mo-evalA mcld_orig_eval.sbatch
sbatch --dependency=afterany:$B2 --job-name=mo-evalB mcld_orig_eval.sbatch
echo "COND DONE"

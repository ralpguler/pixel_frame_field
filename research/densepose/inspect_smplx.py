"""Inspect SMPL-X NEUTRAL model arrays to design the DensePose-24 construction:
canonical template orientation, skinning weights, joints, kinematic tree."""
import numpy as np
import torch
import sys
sys.path.insert(0, "/home/u6ga/raguler.u6ga/codes/wan/bedlam")
from render_correspondence import get_smplx, pose_instance

dev = "cpu"
m = get_smplx("neutral", 16, dev)
print("=== model attrs ===")
for a in ("lbs_weights", "v_template", "faces_tensor", "J_regressor", "parents"):
    v = getattr(m, a, None)
    if v is not None and hasattr(v, "shape"):
        print(f"  {a}: shape={tuple(v.shape)} dtype={v.dtype}")
lbs = m.lbs_weights.detach().cpu().numpy()
print("  lbs_weights argmax range:", lbs.argmax(1).min(), lbs.argmax(1).max(),
      "n joints used:", len(np.unique(lbs.argmax(1))))

# canonical template (rest pose) vertices
vt = m.v_template.detach().cpu().numpy()
print("\n=== v_template coord ranges (canonical T-pose) ===")
for i, ax in enumerate("xyz"):
    print(f"  {ax}: [{vt[:,i].min():.3f}, {vt[:,i].max():.3f}]  span={vt[:,i].ptp():.3f}")

# joints in rest pose
J = (m.J_regressor.detach().cpu().numpy() @ vt)
print("\n=== rest-pose joints (first 25) x,y,z ===")
for j in range(min(25, J.shape[0])):
    print(f"  j{j:2d}: {J[j,0]:+.3f} {J[j,1]:+.3f} {J[j,2]:+.3f}")
print("  total joints:", J.shape[0])

# parents / kintree
par = m.parents.detach().cpu().numpy() if hasattr(m.parents, "detach") else np.asarray(m.parents)
print("\n=== parents (kinematic tree) ===")
print(" ", par.tolist())

# which way does the face point? nose ~ head region; check head vertices anterior axis
# Use skinning to find head-dominated verts (joint 15 in SMPL-X = head)
head_j = 15
head_verts = vt[lbs.argmax(1) == head_j]
print(f"\n=== head-dominated verts (joint {head_j}): n={len(head_verts)} ===")
if len(head_verts):
    for i, ax in enumerate("xyz"):
        print(f"  {ax} mean={head_verts[:,i].mean():+.3f} range=[{head_verts[:,i].min():+.3f},{head_verts[:,i].max():+.3f}]")
print("\n(anterior axis = the one where the face/nose sticks out; expect z+ for SMPL-X)")

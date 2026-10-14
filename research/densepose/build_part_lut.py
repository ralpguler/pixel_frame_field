"""Build a DensePose-compatible 24-part labeling on the SMPL-X mesh, directly
from skinning weights (limb identity + side) and canonical-pose geometry
(front/back, head L/R). No external SMPL<->SMPL-X correspondence needed.

Outputs to ~/data/bedlam2/dp_assets/ :
  smplx_vert_to_dp24.npy   (10475,) int8   part id in [1..24] for every vertex
  smplx_face_to_dp24.npy   (20908,) int16  per-face majority part id in [1..24]
  dp24_lr_swap.npy         (25,)    int8    part-id permutation under hflip (L<->R)
  dp24_names.json          {id: name}       0=background, 1..24 parts

Part id convention (canonical DensePose-24 semantics):
  0  background
  1  torso_back        2  torso_front
  3  hand_right        4  hand_left
  5  foot_left         6  foot_right
  7  upperleg_right_back   8  upperleg_left_back
  9  upperleg_right_front 10  upperleg_left_front
 11  lowerleg_right_back  12  lowerleg_left_back
 13  lowerleg_right_front 14  lowerleg_left_front
 15  upperarm_left_front  16  upperarm_right_front
 17  upperarm_left_back   18  upperarm_right_back
 19  lowerarm_left_front  20  lowerarm_right_front
 21  lowerarm_left_back   22  lowerarm_right_back
 23  head_right          24  head_left

Front = anterior = larger z (per-segment median split). +x = anatomical left.
"""
import json
import os
import sys

import numpy as np

sys.path.insert(0, "/home/u6ga/raguler.u6ga/codes/wan/bedlam")
from render_correspondence import get_smplx

OUT = os.path.expanduser("~/data/bedlam2/dp_assets")

# SMPL-X 55-joint -> coarse limb segment + side. (verified joint ordering)
TORSO = [0, 3, 6, 9, 12, 13, 14]            # pelvis, spine1-3, neck, collars
HEAD = [15, 22, 23, 24]                       # head, jaw, eyes
SEG = {
    "torso": TORSO, "head": HEAD,
    "uarm_L": [16], "uarm_R": [17],
    "larm_L": [18], "larm_R": [19],
    "hand_L": [20] + list(range(25, 40)), "hand_R": [21] + list(range(40, 55)),
    "uleg_L": [1], "uleg_R": [2],
    "lleg_L": [4], "lleg_R": [5],
    "foot_L": [7, 10], "foot_R": [8, 11],
}
# which segments get a front/back split (others are single parts)
SPLIT_FB = {"torso", "uarm_L", "uarm_R", "larm_L", "larm_R",
            "uleg_L", "uleg_R", "lleg_L", "lleg_R"}

# (segment, fb) -> dp24 id.  fb in {"", "front", "back"}
DP24 = {
    ("torso", "back"): 1, ("torso", "front"): 2,
    ("hand_R", ""): 3, ("hand_L", ""): 4,
    ("foot_L", ""): 5, ("foot_R", ""): 6,
    ("uleg_R", "back"): 7, ("uleg_L", "back"): 8,
    ("uleg_R", "front"): 9, ("uleg_L", "front"): 10,
    ("lleg_R", "back"): 11, ("lleg_L", "back"): 12,
    ("lleg_R", "front"): 13, ("lleg_L", "front"): 14,
    ("uarm_L", "front"): 15, ("uarm_R", "front"): 16,
    ("uarm_L", "back"): 17, ("uarm_R", "back"): 18,
    ("larm_L", "front"): 19, ("larm_R", "front"): 20,
    ("larm_L", "back"): 21, ("larm_R", "back"): 22,
    ("head_R", ""): 23, ("head_L", ""): 24,
}
NAMES = {0: "background"}
for (seg, fb), idx in DP24.items():
    NAMES[idx] = seg + ("_" + fb if fb else "")

# L<->R swap pairs (for hflip augmentation). torso parts map to themselves.
SWAP_PAIRS = [(3, 4), (5, 6), (7, 8), (9, 10), (11, 12), (13, 14),
              (15, 16), (17, 18), (19, 20), (21, 22), (23, 24)]


def main():
    m = get_smplx("neutral", 16, "cpu")
    lbs = m.lbs_weights.detach().cpu().numpy()          # (10475,55)
    vt = m.v_template.detach().cpu().numpy()            # (10475,3) canonical
    faces = m.faces_tensor.detach().cpu().numpy()       # (20908,3)
    vert_joint = lbs.argmax(1)                          # (10475,) in [0,54]

    # joint -> segment name
    joint2seg = {}
    for seg, joints in SEG.items():
        for j in joints:
            joint2seg[j] = seg
    # head verts split L/R by x sign below; assign provisional 'head'
    for j in HEAD:
        joint2seg[j] = "head"

    vseg = np.array([joint2seg[j] for j in vert_joint])  # (10475,) seg names

    vert_part = np.zeros(len(vt), np.int16)
    for seg in set(vseg):
        sel = np.where(vseg == seg)[0]
        if seg == "head":
            # split head into L/R by x sign (+x = left)
            for sub, side in (("head_L", vt[sel, 0] >= 0), ("head_R", vt[sel, 0] < 0)):
                ids = sel[side]
                vert_part[ids] = DP24[(sub, "")]
            continue
        if seg in SPLIT_FB:
            zmed = np.median(vt[sel, 2])
            front = vt[sel, 2] >= zmed       # anterior = larger z
            vert_part[sel[front]] = DP24[(seg, "front")]
            vert_part[sel[~front]] = DP24[(seg, "back")]
        else:
            vert_part[sel] = DP24[(seg, "")]

    assert (vert_part > 0).all(), "every vertex must get a body part"

    # per-face majority of its 3 verts (3-element mode)
    a, b, c = vert_part[faces].T
    face_part = np.where(a == b, a, np.where(a == c, a, np.where(b == c, b, a)))

    # L/R swap permutation over [0..24]
    swap = np.arange(25, dtype=np.int8)
    for i, j in SWAP_PAIRS:
        swap[i], swap[j] = j, i

    os.makedirs(OUT, exist_ok=True)
    np.save(f"{OUT}/smplx_vert_to_dp24.npy", vert_part.astype(np.int8))
    np.save(f"{OUT}/smplx_face_to_dp24.npy", face_part.astype(np.int16))
    np.save(f"{OUT}/dp24_lr_swap.npy", swap)
    json.dump(NAMES, open(f"{OUT}/dp24_names.json", "w"), indent=1)

    # report distribution
    print("=== per-part vertex counts ===")
    for idx in range(1, 25):
        print(f"  {idx:2d} {NAMES[idx]:22s} verts={int((vert_part==idx).sum()):5d}"
              f"  faces={int((face_part==idx).sum()):5d}")
    print(f"\nwrote LUTs to {OUT}")
    print("swap:", swap.tolist())


if __name__ == "__main__":
    main()

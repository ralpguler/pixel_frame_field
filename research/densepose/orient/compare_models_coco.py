"""Side-by-side comparison of TWO oriented-DensePose checkpoints on in-the-wild COCO persons.
No GT -> we compare the two models to each other. Per detected person (EXP=1.3 crop):
  RGB | A: IUV | A: normal | B: IUV | B: normal | disagreement (geodesic deg between A,B rotations)
Purpose: see whether model B's predicted rotations are BROKEN vs just softer than model A.
-> dpeval/compare_coco/index.html

Usage: compare_models_coco.py --a RUN_A/ckpt.pt --b RUN_B/ckpt.pt --la LABEL_A --lb LABEL_B --n 12
"""
import os, glob, argparse, numpy as np, cv2, torch
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
import matplotlib.cm as cm
from scipy.spatial.transform import Rotation
import sys; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)) + "/..")
from model import DINOv3DPT, part_pred
from orient.template_normals import TemplateNormals, normal_to_rgb
from orient.rot6d import get_template, sixd_to_R_np
from orient.detect_track import detect_persons

ap = argparse.ArgumentParser()
D = os.path.expanduser("~/data/bedlam2/dp_runs")
ap.add_argument("--a", default=f"{D}/dp_scratch_bedlam_orient_pp_llrd_1e3/ckpt_050000.pt")
ap.add_argument("--b", default=f"{D}/dp_pp_llrd_1e3_mamma/ckpt_050000.pt")
ap.add_argument("--la", default="BEDLAM 50k")
ap.add_argument("--lb", default="+MAMMA 50k")
ap.add_argument("--backbone", default="vit_large_patch16_dinov3")
ap.add_argument("--layers", default="4,11,17,23")
ap.add_argument("--n", type=int, default=12)
ap.add_argument("--thr", type=float, default=0.5)
ap.add_argument("--seed", type=int, default=0)
a = ap.parse_args()
dev = "cuda"; Z = 512; EXP = 1.3
OUT = os.path.expanduser("~/codes/wan/densepose/dpeval/compare_coco"); os.makedirs(OUT + "/imgs", exist_ok=True)
A = os.path.expanduser("~/data/bedlam2/dp_assets")
v_template, faces, n_rest = get_template("cuda")
fuv = np.load(f"{A}/smplx_face_uv.npy"); fpart = np.load(f"{A}/smplx_face_to_dp24.npy").astype(int)
TN = TemplateNormals(fuv, fpart, np.asarray(faces), np.asarray(n_rest))


def load(ckpt):
    sd = torch.load(ckpt, map_location="cpu")["model"]
    rpp = sd["rot_head.2.weight"].shape[0] != 6
    m = DINOv3DPT(backbone=a.backbone, layers=tuple(int(x) for x in a.layers.split(",")),
                  pretrained=False, decoupled=True, orient=True, rot_per_part=rpp).to(dev).eval()
    m.load_state_dict(sd); print("loaded", ckpt, "per_part" if rpp else "single", flush=True)
    return m


mA, mB = load(a.a), load(a.b)


@torch.no_grad()
def predict(m, crop_rgb, cs):
    img = torch.from_numpy(cv2.resize(crop_rgb, (Z, Z))).permute(2, 0, 1)[None].float().to(dev) / 255
    with torch.autocast("cuda", dtype=torch.bfloat16):
        o = m(img)
    part = part_pred(o)[0].cpu().numpy().astype(int)
    partS = cv2.resize(part.astype(np.uint8), (cs, cs), interpolation=cv2.INTER_NEAREST).astype(int)
    r, c = np.indices(partS.shape); idx = np.clip(partS - 1, 0, 23)
    uv = o["uv"].float()[0].view(24, 2, Z, Z).clamp(0, 1).cpu().numpy()
    uvv = np.stack([[cv2.resize(uv[p, k], (cs, cs)) for k in range(2)] for p in range(24)])
    US = uvv[idx, 0, r, c]; VS = uvv[idx, 1, r, c]
    rot = o["rot6d"].float()[0]
    if rot.shape[0] == 6:
        rt = rot.permute(1, 2, 0).cpu().numpy()
        six = np.concatenate([cv2.resize(rt[..., :3], (cs, cs)), cv2.resize(rt[..., 3:], (cs, cs))], -1)
    else:
        rot = rot.view(24, 6, Z, Z).cpu().numpy()
        rotv = np.stack([[cv2.resize(rot[p, k], (cs, cs)) for k in range(6)] for p in range(24)])
        six = rotv[idx, :, r, c]
    R = sixd_to_R_np(six)
    nt, mask = TN(partS, US, VS)
    return partS, nt, mask, R


def iuv_rgb(part, mask):
    out = cm.hsv(np.clip(part, 0, 24) / 24.0)[..., :3].copy(); out[~mask] = 1.0; return out


def normal_rgb(R, nt, mask):
    return normal_to_rgb(np.einsum("hwij,hwj->hwi", R, nt), mask)


def disagree(RA, RB, mask):
    rel = np.matmul(np.transpose(RA, (0, 1, 3, 2)), RB)
    tr = rel[..., 0, 0] + rel[..., 1, 1] + rel[..., 2, 2]
    d = np.degrees(np.arccos(np.clip((tr - 1) / 2, -1, 1))); d[~mask] = np.nan
    return d


imgs = sorted(glob.glob(os.path.expanduser("~/data/coco_val2014_minival/*.jpg")))
rng = np.random.RandomState(a.seed); rng.shuffle(imgs)
rows = []
for path in imgs:
    if len(rows) >= a.n:
        break
    bgr = cv2.imread(path)
    if bgr is None:
        continue
    rgb = bgr[..., ::-1]
    boxes = detect_persons(rgb, thr=a.thr)
    if len(boxes) == 0:
        continue
    # largest person box
    boxes = sorted(boxes, key=lambda b: (b[2] - b[0]) * (b[3] - b[1]), reverse=True)
    x0, y0, x1, y1 = boxes[0][:4]
    cx, cy = (x0 + x1) / 2, (y0 + y1) / 2; s = max(x1 - x0, y1 - y0) * EXP / 2
    X0, Y0, X1, Y1 = int(cx - s), int(cy - s), int(cx + s), int(cy + s)
    H, W = rgb.shape[:2]
    pad = ((max(0, -Y0), max(0, Y1 - H)), (max(0, -X0), max(0, X1 - W)), (0, 0))
    crop = np.pad(rgb, pad, mode="edge")[max(0, Y0):Y1 + pad[0][0], max(0, X0):X1 + pad[1][0]]
    cs = crop.shape[0]
    if cs < 64 or crop.shape[0] != crop.shape[1]:
        crop = cv2.resize(crop, (max(cs, 64), max(cs, 64))); cs = crop.shape[0]
    pA, nA, mA_, RA = predict(mA, crop, cs)
    pB, nB, mB_, RB = predict(mB, crop, cs)
    both = mA_ & mB_
    dis = disagree(RA, RB, both)
    md = float(np.nanmean(dis)) if both.any() else float("nan")
    fig, ax = plt.subplots(1, 6, figsize=(20, 3.6))
    ax[0].imshow(crop); ax[0].set_title("RGB", fontsize=9)
    ax[1].imshow(iuv_rgb(pA, mA_)); ax[1].set_title(f"{a.la}: IUV", fontsize=9)
    ax[2].imshow(normal_rgb(RA, nA, mA_)); ax[2].set_title(f"{a.la}: normal", fontsize=9)
    ax[3].imshow(iuv_rgb(pB, mB_)); ax[3].set_title(f"{a.lb}: IUV", fontsize=9)
    ax[4].imshow(normal_rgb(RB, nB, mB_)); ax[4].set_title(f"{a.lb}: normal", fontsize=9)
    hd = ax[5].imshow(dis, cmap="turbo", vmin=0, vmax=45); ax[5].set_title(f"A vs B disagree° ({md:.1f})", fontsize=9)
    fig.colorbar(hd, ax=[ax[5]], fraction=0.046, pad=0.02)
    for x in ax: x.axis("off")
    fn = f"imgs/{len(rows):02d}.png"; fig.savefig(OUT + "/" + fn, dpi=70, bbox_inches="tight"); plt.close(fig)
    rows.append((fn, md, os.path.basename(path)))
    print("rendered", fn, "disagree %.1f" % md, flush=True)

H = ['<html><head><meta charset=utf-8><style>body{font-family:sans-serif;background:#fff}'
     'img{width:100%;max-width:1500px}.it{border-bottom:1px solid #ccc;padding:6px 0}</style></head><body>']
H.append(f"<h2>Model comparison on COCO — {a.la} vs {a.lb}</h2>")
H.append(f"<p>RGB | <b>{a.la}</b> IUV, normal | <b>{a.lb}</b> IUV, normal | <b>A vs B disagreement</b> "
         "(geodesic deg, turbo 0-45). No GT: this shows whether B's rotations are broken vs softer.</p>")
mD = np.nanmean([r[1] for r in rows]) if rows else float("nan")
H.append(f"<p>mean A-vs-B rotation disagreement over shown persons: <b>{mD:.1f}°</b></p>")
for fn, d, nm in rows:
    H.append(f"<div class=it>{nm} — A/B disagree {d:.1f}°<br><img src='{fn}'></div>")
H.append("</body></html>")
open(OUT + "/index.html", "w").write("\n".join(H))
print(f"wrote {OUT}/index.html ({len(rows)} persons, mean disagree {mD:.1f} deg)", flush=True)

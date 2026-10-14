"""Diagnose hand/feet normal artifacts: for given video frames, run TWO checkpoints and show, per frame,
[ crop | A: IUV | A: normal | B: IUV | B: normal ]. Overlays a red outline where the PART map disagrees
with a temporally/spatially smoothed part (a proxy for part-misclassification = chart-jump). Lets us see
whether the rainbow normal speckle on hands/feet coincides with wrong PART labels (dense-localization error)
rather than rotation error. Also crops a zoom on the lowest (feet) and the hand regions.

Usage: diag_hands_feet.py CLIP --frames 75,90,110 [--a A.pt --b B.pt --la .. --lb ..]
"""
import os, sys, argparse, numpy as np, cv2, torch
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt, matplotlib.cm as cm
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)) + "/..")
from model import DINOv3DPT, part_pred
from orient.template_normals import TemplateNormals, normal_to_rgb
from orient.rot6d import get_template, sixd_to_R_np
from orient.detect_track import detect_persons

ap = argparse.ArgumentParser()
D = os.path.expanduser("~/data/bedlam2/dp_runs")
ap.add_argument("clip")
ap.add_argument("--frames", default="75")
ap.add_argument("--a", default=f"{D}/dp_pp_llrd_1e3_120k/ckpt_120000.pt")
ap.add_argument("--b", default=f"{D}/dp_pp_llrd_1e3_mamma/ckpt_120000.pt")
ap.add_argument("--la", default="BEDLAM 120k"); ap.add_argument("--lb", default="+MAMMA 120k")
ap.add_argument("--out", default=os.path.expanduser("~/data/dp_viz_compare/diag_handsfeet"))
a = ap.parse_args()
dev = "cuda"; Z = 512; EXP = 1.3
os.makedirs(a.out, exist_ok=True)
A = os.path.expanduser("~/data/bedlam2/dp_assets")
v_template, faces, n_rest = get_template("cuda")
fuv = np.load(f"{A}/smplx_face_uv.npy"); fpart = np.load(f"{A}/smplx_face_to_dp24.npy").astype(int)
TN = TemplateNormals(fuv, fpart, np.asarray(faces), np.asarray(n_rest))


def load(ck):
    sd = torch.load(ck, map_location="cpu")["model"]
    rpp = sd["rot_head.2.weight"].shape[0] != 6
    m = DINOv3DPT(backbone="vit_large_patch16_dinov3", layers=(4, 11, 17, 23),
                  pretrained=False, decoupled=True, orient=True, rot_per_part=rpp).to(dev).eval()
    m.load_state_dict(sd); return m


mA, mB = load(a.a), load(a.b)


@torch.no_grad()
def predict(m, crop, cs):
    img = torch.from_numpy(cv2.resize(crop, (Z, Z))).permute(2, 0, 1)[None].float().to(dev) / 255
    with torch.autocast("cuda", dtype=torch.bfloat16):
        o = m(img)
    part = cv2.resize(part_pred(o)[0].cpu().numpy().astype(np.uint8), (cs, cs), interpolation=cv2.INTER_NEAREST).astype(int)
    r, c = np.indices(part.shape); idx = np.clip(part - 1, 0, 23)
    uv = o["uv"].float()[0].view(24, 2, Z, Z).clamp(0, 1).cpu().numpy()
    uvv = np.stack([[cv2.resize(uv[p, k], (cs, cs)) for k in range(2)] for p in range(24)])
    US = uvv[idx, 0, r, c]; VS = uvv[idx, 1, r, c]
    rot = o["rot6d"].float()[0].view(24, 6, Z, Z).cpu().numpy()
    rotv = np.stack([[cv2.resize(rot[p, k], (cs, cs)) for k in range(6)] for p in range(24)])
    six = rotv[idx, :, r, c]; R = sixd_to_R_np(six)
    nt, mask = TN(part, US, VS)
    return part, nt, mask, R


def iuv_rgb(part, mask):
    out = cm.hsv(np.clip(part, 0, 24) / 24.0)[..., :3].copy(); out[~mask] = 1.0; return out


def nrm_rgb(R, nt, mask):
    return normal_to_rgb(np.einsum("hwij,hwj->hwi", R, nt), mask)


cap = cv2.VideoCapture(a.clip)
for fi in [int(x) for x in a.frames.split(",")]:
    cap.set(1, fi); ok, bgr = cap.read()
    if not ok:
        print("no frame", fi); continue
    rgb = bgr[..., ::-1]
    dets = sorted(detect_persons(rgb, thr=0.5), key=lambda b: (b[2]-b[0])*(b[3]-b[1]), reverse=True)
    if not dets:
        print("no person", fi); continue
    x0, y0, x1, y1 = dets[0][:4]; cx, cy = (x0+x1)/2, (y0+y1)/2; s = max(x1-x0, y1-y0)*EXP/2
    side = int(2*s); X0, Y0 = int(cx-s), int(cy-s); H, W = rgb.shape[:2]
    pad = ((max(0, -Y0), max(0, Y0+side-H)), (max(0, -X0), max(0, X0+side-W)), (0, 0))
    crop = np.pad(rgb, pad, mode="edge")[max(0, Y0):Y0+side+pad[0][0], max(0, X0):X0+side+pad[1][0]]
    cs = crop.shape[0]
    pA, ntA, mkA, RA = predict(mA, crop, cs)
    pB, ntB, mkB, RB = predict(mB, crop, cs)
    fig, ax = plt.subplots(1, 5, figsize=(20, 4.4))
    ax[0].imshow(crop); ax[0].set_title("input")
    ax[1].imshow(iuv_rgb(pA, mkA)); ax[1].set_title(f"{a.la}: IUV (part)")
    ax[2].imshow(nrm_rgb(RA, ntA, mkA)); ax[2].set_title(f"{a.la}: normal")
    ax[3].imshow(iuv_rgb(pB, mkB)); ax[3].set_title(f"{a.lb}: IUV (part)")
    ax[4].imshow(nrm_rgb(RB, ntB, mkB)); ax[4].set_title(f"{a.lb}: normal")
    for x in ax: x.axis("off")
    fn = f"{a.out}/frame_{fi:04d}.png"; fig.savefig(fn, dpi=85, bbox_inches="tight"); plt.close(fig)
    print("wrote", fn, flush=True)
cap.release()

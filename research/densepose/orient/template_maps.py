"""Dense template-space maps from predicted DensePose (part, U, V): the canonical 3D surface POSITION
(XYZ) and the rest-pose template NORMAL, GT vs predicted. No rotation R involved — this isolates the
IUV -> template mapping itself. Speckle here = the model's (I,U,V) prediction error (wrong chart).
  RGB | GT template-XYZ | PRED template-XYZ | GT template-normal | PRED template-normal
-> dpeval/template_maps.png
"""
import os, io, glob, argparse, numpy as np, cv2, torch
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
import sys; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)) + "/..")
from data import unpack_rgba, STORED_EXPAND
from model import DINOv3DPT, part_pred
from orient.rot6d import get_template
from orient.template_normals import TemplateNormals, denoise_part
import webdataset as wds
from PIL import Image

ap = argparse.ArgumentParser()
ap.add_argument("--ckpt", default=os.path.expanduser("~/data/bedlam2/dp_runs/dp_pp_llrd_1e3_120k/ckpt_120000.pt"))
ap.add_argument("--idx", type=int, default=6)
a = ap.parse_args()
dev = "cuda"; Z = 512
A = os.path.expanduser("~/data/bedlam2/dp_assets")
v_template, faces, n_rest = get_template("cpu"); faces = np.asarray(faces); n_rest = np.asarray(n_rest)
v_template = np.asarray(v_template)
fuv = np.load(f"{A}/smplx_face_uv.npy"); fpart = np.load(f"{A}/smplx_face_to_dp24.npy").astype(int)
TN = TemplateNormals(fuv, fpart, faces, n_rest, v_template)
LO = np.percentile(v_template, 1, 0); HI = np.percentile(v_template, 99, 0)   # XYZ color normalization

_sd = torch.load(a.ckpt, map_location="cpu")["model"]
_rpp = _sd["rot_head.2.weight"].shape[0] != 6
m = DINOv3DPT(backbone="vit_large_patch16_dinov3", layers=(4, 11, 17, 23), pretrained=False,
              decoupled=True, orient=True, rot_per_part=_rpp).to(dev).eval()
m.load_state_dict(_sd)


@torch.no_grad()
def predict(crop, cs):
    """Return predicted part (cs,cs) and the FULL per-part UV volume (24,2,cs,cs) so we can gather U,V
    by ANY part (predicted or GT) — the UV head is per-part (24x2 channels)."""
    img = torch.from_numpy(cv2.resize(crop, (Z, Z))).permute(2, 0, 1)[None].float().to(dev) / 255
    with torch.autocast("cuda", dtype=torch.bfloat16):
        o = m(img)
    part = part_pred(o)[0].cpu().numpy().astype(int)
    uv = o["uv"].float()[0].view(24, 2, Z, Z).clamp(0, 1).cpu().numpy()
    uvv = np.stack([[cv2.resize(uv[p, c], (cs, cs)) for c in range(2)] for p in range(24)])  # (24,2,cs,cs)
    return cv2.resize(part.astype(np.uint8), (cs, cs), interpolation=cv2.INTER_NEAREST).astype(int), uvv


def gather_uv(uvv, part):
    """uvv (24,2,cs,cs), part (cs,cs) 1..24 -> U,V (cs,cs) from that part's channel."""
    idx = np.clip(part - 1, 0, 23); r, c = np.indices(part.shape)
    return uvv[idx, 0, r, c], uvv[idx, 1, r, c]


def xyz_rgb(P, mask):
    out = np.clip((P - LO) / (HI - LO), 0, 1); out[~mask] = 1; return out


def nrm_rgb(nt, mask):
    out = np.clip(nt * 0.5 + 0.5, 0, 1); out[~mask] = 1; return out


shards = sorted(glob.glob(os.path.expanduser("~/data/bedlam2/dp_shards_rot/val/*.tar")))
k = 0
for s in wds.WebDataset(shards, shardshuffle=False, empty_check=False, handler=wds.warn_and_continue):
    if "corr.png" not in s:
        continue
    rgb = cv2.imdecode(np.frombuffer(s["rgb.jpg"], np.uint8), cv2.IMREAD_COLOR)[..., ::-1]
    face, bary = unpack_rgba(np.array(Image.open(io.BytesIO(s["corr.png"]))))
    S0 = rgb.shape[0]; cs = max(8, min(int(round(1.3 / STORED_EXPAND * S0)), S0)); o = (S0 - cs) // 2
    rc = np.ascontiguousarray(rgb[o:o + cs, o:o + cs]); fw = np.ascontiguousarray(face[o:o + cs, o:o + cs]); bw = np.ascontiguousarray(bary[o:o + cs, o:o + cs])
    mask = fw >= 0
    if mask.mean() < 0.03:
        continue
    if k != a.idx:
        k += 1; continue
    # GT template XYZ straight from rasterized face+bary, and GT (part,U,V)
    tv = faces[np.where(mask, fw, 0)]
    gP = np.zeros((cs, cs, 3), np.float32); gP[mask] = (bw[mask][:, :, None] * v_template[tv[mask]]).sum(1)
    gpart = np.where(mask, fpart[np.where(mask, fw, 0)], 0)
    gU = np.zeros((cs, cs), np.float32); gV = np.zeros((cs, cs), np.float32)
    guv = (bw[mask][:, :, None] * fuv[fw[mask]]).sum(1); gU[mask] = guv[:, 0]; gV[mask] = guv[:, 1]
    # predicted part (raw) + denoised, full per-part UV volume
    pS, uvv = predict(rc, cs)
    pD = denoise_part(pS, 5)                                 # cleaned part
    Ur, Vr = gather_uv(uvv, pS)            # U,V by RAW pred part
    Ud, Vd = gather_uv(uvv, pD)            # U,V by DENOISED part
    _, P_raw, m_raw = TN(pS, Ur, Vr, return_pos=True)        # full raw prediction
    NTd, P_den, m_den = TN(pD, Ud, Vd, return_pos=True)      # denoised prediction (normal + XYZ)
    # numbers
    mm = mask
    pmm = 100 * (pS[mm] != gpart[mm]).mean(); pmd = 100 * (pD[mm] != gpart[mm]).mean()
    fig, ax = plt.subplots(1, 5, figsize=(26, 6.5))
    ax[0].imshow(rc); ax[0].set_title("RGB")
    ax[1].imshow(xyz_rgb(gP, mask)); ax[1].set_title("GT XYZ")
    ax[2].imshow(xyz_rgb(P_raw, m_raw)); ax[2].set_title(f"PRED XYZ raw (part-err {pmm:.1f}%)")
    ax[3].imshow(xyz_rgb(P_den, m_den)); ax[3].set_title(f"PRED XYZ denoised (part-err {pmd:.1f}%)")
    ax[4].imshow(nrm_rgb(NTd, m_den)); ax[4].set_title("PRED template-normal (denoised)")
    for x in ax: x.axis("off")
    out = os.path.expanduser("~/codes/wan/densepose/dpeval/template_maps.png")
    fig.savefig(out, dpi=130, bbox_inches="tight")
    print("wrote", out, "| part-mismatch raw %.1f%% -> denoised %.1f%%" % (pmm, pmd))
    break

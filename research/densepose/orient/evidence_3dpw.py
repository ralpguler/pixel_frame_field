"""Stage-1 evidence panels for the CameraHMR-refinement viz: run our oriented-DensePose on chosen 3DPW
frames and paste the predicted IUV + surface-normal maps back onto the FULL image (same 1.3x box our
correspondences used). Saves <out>/evi_<idx>_iuv.png and _nrm.png per instance for the composer (Stage 2)."""
import os, sys, argparse, numpy as np, cv2, torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)) + "/..")
from model import DINOv3DPT, part_pred, gather_rot6d, sixd_to_R
from orient.rot6d import get_template
from orient.template_normals import TemplateNormals, normal_to_rgb
import matplotlib.cm as cm

ap = argparse.ArgumentParser()
ap.add_argument("--preds", default=os.path.expanduser("~/codes/CameraHMR/cameramhr_3dpw_preds.npz"))
ap.add_argument("--img-root", default=os.path.expanduser("~/data/datasets/3dpw"))
ap.add_argument("--ckpt", default=os.path.expanduser("~/data/bedlam2/dp_runs/dp_pp_llrd_1e3_120k/ckpt_120000.pt"))
ap.add_argument("--out", default=os.path.expanduser("~/codes/CameraHMR/viz/evidence"))
ap.add_argument("--idx", type=int, nargs="+", required=True)
a = ap.parse_args()
dev = "cuda"; Z = 512; EXP = 1.3
os.makedirs(a.out, exist_ok=True)

P = np.load(a.preds, allow_pickle=True)
A = os.path.expanduser("~/data/bedlam2/dp_assets")
v_template, faces, n_rest = get_template("cpu"); faces = np.asarray(faces); n_rest = np.asarray(n_rest)
fuv = np.load(f"{A}/smplx_face_uv.npy"); fpart = np.load(f"{A}/smplx_face_to_dp24.npy").astype(int)
TN = TemplateNormals(fuv, fpart, faces, n_rest)
PAL = (cm.hsv(np.linspace(0, 1, 25))[:, :3] * 255).astype(np.uint8)

sd = torch.load(a.ckpt, map_location="cpu")["model"]
m = DINOv3DPT(backbone="vit_large_patch16_dinov3", layers=(4, 11, 17, 23), pretrained=False,
              decoupled=True, orient=True, rot_per_part=True).to(dev).eval()
m.load_state_dict(sd)


@torch.no_grad()
def infer(crop):
    img = torch.from_numpy(crop).permute(2, 0, 1)[None].float().to(dev) / 255
    with torch.autocast("cuda", dtype=torch.bfloat16):
        o = m(img)
    part = part_pred(o)[0].cpu().numpy().astype(int)
    uv = o["uv"].float()[0].view(24, 2, Z, Z); sel = (torch.from_numpy(part).to(dev) - 1).clamp(0)
    U = uv[:, 0].gather(0, sel[None])[0].clamp(0, 1).cpu().numpy()
    V = uv[:, 1].gather(0, sel[None])[0].clamp(0, 1).cpu().numpy()
    six = gather_rot6d(o["rot6d"], torch.from_numpy(part).to(dev)[None]).float()[0].permute(1, 2, 0).cpu().numpy()
    return part, U, V, six


def iuv_rgb(part, U, V):
    mask = part > 0
    img = PAL[np.clip(part, 0, 24)].astype(np.float32) * (0.45 + 0.4 * U + 0.15 * V)[..., None]
    return np.clip(img, 0, 255).astype(np.uint8), mask


for idx in a.idx:
    name = str(P["imgname"][idx]); cx, cy = P["center"][idx]; B = EXP * P["scale"][idx] * 200.0
    full = cv2.imread(os.path.join(a.img_root, name))[..., ::-1]
    H, W = full.shape[:2]
    s = Z / B; M = np.array([[s, 0, Z / 2 - cx * s], [0, s, Z / 2 - cy * s]], np.float32)
    crop = cv2.warpAffine(full, M, (Z, Z), flags=cv2.INTER_AREA, borderValue=0)
    part, U, V, six = infer(np.ascontiguousarray(crop))
    iuv, mask = iuv_rgb(part, U, V)
    R = sixd_to_R(torch.from_numpy(six).reshape(-1, 6)).reshape(Z, Z, 3, 3).numpy()
    nt, _ = TN(part, U, V); ncam = np.einsum("hwij,hwj->hwi", R, nt)
    nrm = (normal_to_rgb(ncam, mask) * 255).astype(np.uint8)
    # paste crop-space maps back to full image (inverse warp)
    Minv = cv2.invertAffineTransform(M)
    for tag, cropmap in [("iuv", iuv), ("nrm", nrm)]:
        full_map = cv2.warpAffine(cropmap, Minv, (W, H), flags=cv2.INTER_NEAREST, borderValue=0)
        fg_full = cv2.warpAffine(mask.astype(np.uint8), Minv, (W, H), flags=cv2.INTER_NEAREST) > 0
        out = full.copy(); out[fg_full] = (0.35 * full[fg_full] + 0.65 * full_map[fg_full]).astype(np.uint8)
        cv2.imwrite(os.path.join(a.out, f"evi_{idx}_{tag}.png"), out[..., ::-1])
    print(f"idx {idx} {name}: fg {mask.mean():.2f} -> wrote evi_{idx}_iuv/nrm.png", flush=True)
print("done", flush=True)

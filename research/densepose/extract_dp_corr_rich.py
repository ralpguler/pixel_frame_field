"""STAGE A: extract dense DensePose correspondences from the oriented-DensePose model on
3DPW test crops (CameraHMR boxes), for later SMPL mesh refinement.

Crop transform (full <-> 512 crop), B = scale*200, c = center:
    full_x = c_x + (x/512 - 0.5)*B ,  full_y = c_y + (y/512 - 0.5)*B
We crop with an exact affine (full->crop): s = 512/B,
    M = [[s,0, 256 - c_x*s],[0,s, 256 - c_y*s]]  (cv2.warpAffine, zero pad).
This maps the box center -> crop pixel (256,256) and is the exact inverse of the formula above.
"""
import os, sys, time, argparse
import numpy as np, cv2, torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from model import DINOv3DPT, part_pred, gather_rot6d

Z = 512
K = 400
MIN_FG = 30
# The model was trained on context-padded crops; the tight CameraHMR box (side=scale*200) fills the
# frame and the mask head collapses to all-background. We expand the box side by EXP and back-project
# with the SAME expanded side, so full-image coords stay exact (box center still maps to crop (256,256)).
EXP = 1.3
DEV = "cuda"
DATA = os.path.expanduser("~/codes/CameraHMR/data/test-images/RICH")
PREDS = os.path.expanduser("~/codes/CameraHMR/cameramhr_rich_preds.npz")
OUT = os.path.expanduser("~/codes/CameraHMR/dp_corr_rich.npz")
DBG = os.path.expanduser("~/codes/CameraHMR/dp_corr_rich_debug.png")
CKPT = os.path.expanduser("~/data/bedlam2/dp_runs/dp_pp_llrd_1e3_120k/ckpt_120000.pt")


def build_model():
    m = DINOv3DPT(backbone="vit_large_patch16_dinov3", layers=(4, 11, 17, 23),
                  pretrained=False, decoupled=True, orient=True, rot_per_part=True).to(DEV).eval()
    sd = torch.load(CKPT, map_location="cpu")["model"]
    m.load_state_dict(sd)
    print("loaded", CKPT, flush=True)
    return m


_cache = {"name": None, "img": None}


def load_full(imgname):
    if _cache["name"] != imgname:
        bgr = cv2.imread(os.path.join(DATA, imgname))
        if bgr is None:
            raise FileNotFoundError(imgname)
        _cache["name"] = imgname
        _cache["img"] = bgr[..., ::-1]  # RGB
    return _cache["img"]


def make_crop(rgb, cx, cy, B):
    s = Z / B
    M = np.array([[s, 0, Z / 2 - cx * s], [0, s, Z / 2 - cy * s]], np.float32)
    return cv2.warpAffine(rgb, M, (Z, Z), flags=cv2.INTER_LINEAR, borderValue=0)


@torch.no_grad()
def infer_batch(m, crops):
    img = torch.from_numpy(np.stack(crops)).permute(0, 3, 1, 2).float().to(DEV) / 255
    with torch.autocast("cuda", dtype=torch.bfloat16):
        o = m(img)
    part = part_pred(o)                                   # (B,Z,Z) 0..24
    uv = o["uv"].float().view(-1, 24, 2, Z, Z)
    sel = (part - 1).clamp(0)                             # (B,Z,Z)
    U = uv[:, :, 0].gather(1, sel[:, None]).squeeze(1).clamp(0, 1)
    V = uv[:, :, 1].gather(1, sel[:, None]).squeeze(1).clamp(0, 1)
    six = gather_rot6d(o["rot6d"], part).float().permute(0, 2, 3, 1)  # (B,Z,Z,6)
    return (part.cpu().numpy().astype(np.int16), U.cpu().numpy(),
            V.cpu().numpy(), six.cpu().numpy())


def sample_instance(part, U, V, six, cx, cy, B, rng):
    ys, xs = np.nonzero(part > 0)
    nfg = ys.size
    if nfg < MIN_FG:
        return None, nfg
    k = min(K, nfg)
    pick = rng.choice(nfg, size=k, replace=False)
    yy = ys[pick]; xx = xs[pick]
    p = part[yy, xx].astype(np.int16)
    u = U[yy, xx].astype(np.float32)
    v = V[yy, xx].astype(np.float32)
    r6 = six[yy, xx].astype(np.float32)
    fx = (cx + (xx / Z - 0.5) * B).astype(np.float32)
    fy = (cy + (yy / Z - 0.5) * B).astype(np.float32)
    xy = np.stack([fx, fy], 1)
    uvp = np.stack([u, v], 1)
    return (p, uvp, xy, r6), nfg


_PAL = None
def palette():
    global _PAL
    if _PAL is None:
        import matplotlib.cm as cm
        _PAL = (cm.hsv(np.linspace(0, 1, 25))[:, :3] * 255).astype(np.uint8)
    return _PAL


def save_debug(crop, samp):
    pal = palette()
    ov = crop[..., ::-1].copy()  # to BGR for cv2
    p, uvp, xy, r6 = samp
    # need crop-pixel coords; recompute from part sampling is not stored, so redraw from a fresh sample
    # (debug receives crop + recomputed pixel coords) -> handled by caller passing pix
    return ov


def main():
    global CKPT, OUT
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", type=int, default=0, help="run only first N instances (smoke test)")
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--ckpt", default=CKPT, help="DensePose checkpoint providing the correspondence")
    ap.add_argument("--out", default=OUT, help="output corr npz")
    a = ap.parse_args()
    CKPT = os.path.expanduser(a.ckpt); OUT = os.path.expanduser(a.out)
    print(f"[extract] ckpt={CKPT}\n[extract] out={OUT}", flush=True)

    d = np.load(PREDS, allow_pickle=True)
    imgname = d["imgname"]; center = d["center"].astype(np.float64); scale = d["scale"].astype(np.float64)
    N = len(imgname)
    if a.smoke:
        N = min(N, a.smoke)
    rng = np.random.default_rng(0)
    m = build_model()

    pt_part = []; pt_uv = []; pt_xy = []; pt_r6 = []
    index = np.zeros(N + 1, np.int64)
    skipped = 0
    dbg_done = False
    t0 = time.time()

    bs = a.batch
    for s in range(0, N, bs):
        e = min(N, s + bs)
        crops = []; meta = []
        for i in range(s, e):
            cx, cy = center[i]; B = EXP * scale[i] * 200.0
            rgb = load_full(imgname[i])
            crops.append(np.ascontiguousarray(make_crop(rgb, cx, cy, B)))
            meta.append((cx, cy, B))
        part, U, V, six = infer_batch(m, crops)
        for j, i in enumerate(range(s, e)):
            cx, cy, B = meta[j]
            samp, nfg = sample_instance(part[j], U[j], V[j], six[j], cx, cy, B, rng)
            if samp is None:
                skipped += 1
                index[i + 1] = index[i]
                continue
            p, uvp, xy, r6 = samp
            pt_part.append(p); pt_uv.append(uvp); pt_xy.append(xy); pt_r6.append(r6)
            index[i + 1] = index[i] + p.shape[0]
            if a.smoke and i < 3:
                print(f"  inst {i} {imgname[i]} nfg={nfg} fg_frac={nfg/(Z*Z):.3f} npts={p.shape[0]}", flush=True)
                for q in range(min(4, p.shape[0])):
                    print(f"    part={int(p[q])} u={uvp[q,0]:.3f} v={uvp[q,1]:.3f} "
                          f"xy_full=({xy[q,0]:.1f},{xy[q,1]:.1f})", flush=True)
            # debug overlay on first valid instance
            if a.smoke and not dbg_done:
                pal = palette()
                ov = crops[j][..., ::-1].copy()
                ys, xs = np.nonzero(part[j] > 0)
                kk = min(K, ys.size)
                pk = rng.choice(ys.size, size=kk, replace=False)
                for yy, xx in zip(ys[pk], xs[pk]):
                    col = pal[int(part[j][yy, xx])]
                    cv2.circle(ov, (int(xx), int(yy)), 2, (int(col[2]), int(col[1]), int(col[0])), -1)
                cv2.imwrite(DBG, ov)
                dbg_done = True
                print("  wrote debug overlay:", DBG, flush=True)
        if (e % 2000 < bs) or e == N:
            print(f"[{e}/{N}] M={int(index[e])} skipped={skipped} "
                  f"{e/(time.time()-t0):.1f} inst/s", flush=True)

    pt_part = np.concatenate(pt_part) if pt_part else np.zeros((0,), np.int16)
    pt_uv = np.concatenate(pt_uv) if pt_uv else np.zeros((0, 2), np.float32)
    pt_xy = np.concatenate(pt_xy) if pt_xy else np.zeros((0, 2), np.float32)
    pt_r6 = np.concatenate(pt_r6) if pt_r6 else np.zeros((0, 6), np.float32)
    M = pt_part.shape[0]
    print(f"DONE N={N} M={M} skipped={skipped} time={time.time()-t0:.1f}s", flush=True)

    if not a.smoke:
        np.savez_compressed(OUT,
            pt_part=pt_part.astype(np.float32), pt_uv=pt_uv.astype(np.float32),
            pt_xy_full=pt_xy.astype(np.float32), pt_rot6d=pt_r6.astype(np.float32),
            index=index.astype(np.int64), imgname=imgname[:N])
        print("saved", OUT, f"{os.path.getsize(OUT)/1e6:.1f} MB", flush=True)


if __name__ == "__main__":
    main()

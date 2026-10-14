"""Qualitative results gallery on COCO minival: detect people (fasterrcnn), run DensePose per box,
composite the predicted 24-part IUV. Shows v3 (synth-only) vs the COCO-fine-tuned model side by side
so the sim2real improvement is visible, plus the fine-tuned UV. -> results_report/index.html
"""
import os, sys, json, html, numpy as np, cv2, torch
sys.path.insert(0, ".."); sys.path.insert(0, ".")
from model import DINOv3DPT, part_pred
from data import _PAL
from pycocotools.coco import COCO
from torchvision.models.detection import fasterrcnn_resnet50_fpn_v2, FasterRCNN_ResNet50_FPN_V2_Weights

dev = "cuda"; Z = 512; EXP = 1.3
IMG = os.path.expanduser("~/data/coco_val2014_minival")
GT = "data/densepose_coco_2014_minival.json"
OUT = os.path.expanduser("~/codes/wan/densepose/dpeval/results_report"); os.makedirs(OUT + "/imgs", exist_ok=True)
BB = "vit_large_patch16_dinov3"; LAYERS = (4, 11, 17, 23)
CKPTS = {"v3 (synth-only)": os.path.expanduser("~/data/bedlam2/dp_runs/dp_v3_L512_dec/final.pt"),
         "fine-tuned (COCO)": os.path.expanduser("~/data/bedlam2/dp_runs/dp_ft_coco/ckpt_002000.pt")}
PAL = _PAL


def load(ckpt):
    m = DINOv3DPT(backbone=BB, layers=LAYERS, pretrained=False, decoupled=True).to(dev).eval()
    m.load_state_dict(torch.load(ckpt, map_location="cpu")["model"]); return m


models = {k: load(v) for k, v in CKPTS.items()}
det = fasterrcnn_resnet50_fpn_v2(weights=FasterRCNN_ResNet50_FPN_V2_Weights.DEFAULT).eval().to(dev)
coco = COCO(GT)


@torch.no_grad()
def boxes(rgb, thr=0.8):
    t = torch.from_numpy(rgb).permute(2, 0, 1).float().to(dev) / 255
    o = det([t])[0]; out = []
    for b, l, s in zip(o["boxes"], o["labels"], o["scores"]):
        if int(l) == 1 and float(s) > thr:
            out.append(b.tolist())
    return out


@torch.no_grad()
def infer(m, rgb, box):
    x0, y0, x1, y1 = box; bw, bh = x1 - x0, y1 - y0
    cx, cy = x0 + bw / 2, y0 + bh / 2; side = EXP * max(bw, bh); S = int(round(side))
    H, W = rgb.shape[:2]; ix0, iy0 = int(round(cx - side / 2)), int(round(cy - side / 2))
    canv = np.zeros((S, S, 3), np.uint8)
    ax0, ay0 = max(0, -ix0), max(0, -iy0); bx0, by0 = max(0, ix0), max(0, iy0)
    bx1, by1 = min(W, ix0 + S), min(H, iy0 + S)
    canv[ay0:ay0 + (by1 - by0), ax0:ax0 + (bx1 - bx0)] = rgb[by0:by1, bx0:bx1]
    img = torch.from_numpy(cv2.resize(canv, (Z, Z))).permute(2, 0, 1)[None].float().to(dev) / 255
    with torch.autocast("cuda", dtype=torch.bfloat16):
        o = m(img)
    part = part_pred(o)[0].cpu().numpy().astype(np.uint8)              # (Z,Z) 0..24
    uv = o["uv"].float()[0].view(24, 2, Z, Z)
    sel = (torch.from_numpy(part).to(dev).long() - 1).clamp(0)
    U = uv[:, 0].gather(0, sel[None])[0].clamp(0, 1).cpu().numpy()
    V = uv[:, 1].gather(0, sel[None])[0].clamp(0, 1).cpu().numpy()
    partS = cv2.resize(part, (S, S), interpolation=cv2.INTER_NEAREST)
    US = cv2.resize(U, (S, S)); VS = cv2.resize(V, (S, S))
    return partS, US, VS, (ix0, iy0, S)


def composite(rgb, perbox, want_uv=False):
    """perbox: list of (partS,US,VS,(ix0,iy0,S)). Returns part-overlay (and UV-overlay if want_uv)."""
    H, W = rgb.shape[:2]
    ov = rgb.copy()
    uvov = rgb.copy()
    for partS, US, VS, (ix0, iy0, S) in perbox:
        x0, y0 = max(0, ix0), max(0, iy0); x1, y1 = min(W, ix0 + S), min(H, iy0 + S)
        sx, sy = x0 - ix0, y0 - iy0
        pp = partS[sy:sy + (y1 - y0), sx:sx + (x1 - x0)]
        fg = pp > 0
        col = PAL[pp]
        reg = ov[y0:y1, x0:x1]; reg[fg] = (0.45 * reg[fg] + 0.55 * col[fg]).astype(np.uint8)
        ov[y0:y1, x0:x1] = reg
        if want_uv:
            uu = (US[sy:sy + (y1 - y0), sx:sx + (x1 - x0)] * 255).astype(np.uint8)
            vv = (VS[sy:sy + (y1 - y0), sx:sx + (x1 - x0)] * 255).astype(np.uint8)
            uvimg = np.stack([uu, vv, np.full_like(uu, 128)], -1)
            r = uvov[y0:y1, x0:x1]; r[fg] = uvimg[fg]; uvov[y0:y1, x0:x1] = r
    return (ov, uvov) if want_uv else (ov, None)


# pick a varied sample: spread across the set + the most-crowded images
ids = coco.getImgIds()
dp_ids = [i for i in ids if any("dp_x" in a for a in coco.loadAnns(coco.getAnnIds(imgIds=i)))]
ncount = {i: sum("dp_x" in a for a in coco.loadAnns(coco.getAnnIds(imgIds=i))) for i in dp_ids}
crowded = sorted(dp_ids, key=lambda i: -ncount[i])[:12]
spread = dp_ids[::max(1, len(dp_ids) // 45)][:45]
sample = list(dict.fromkeys(crowded + spread))
print(f"{len(sample)} images ({len(crowded)} crowded + spread)", flush=True)

rows = []
for k, ii in enumerate(sample):
    info = coco.loadImgs(ii)[0]; p = os.path.join(IMG, info["file_name"])
    if not os.path.exists(p):
        continue
    rgb = cv2.imread(p)[..., ::-1].copy()
    bxs = boxes(rgb)
    if not bxs:
        continue
    panels = {}
    for name, m in models.items():
        per = [infer(m, rgb, b) for b in bxs]
        ov, uvov = composite(rgb, per, want_uv=(name.startswith("fine")))
        panels[name] = (ov, uvov)
    # build a row image: input | v3 parts | ft parts | ft UV
    cells = [rgb, panels["v3 (synth-only)"][0], panels["fine-tuned (COCO)"][0], panels["fine-tuned (COCO)"][1]]
    h = min(c.shape[0] for c in cells); cells = [cv2.resize(c, (int(c.shape[1] * h / c.shape[0]), h)) for c in cells]
    strip = np.concatenate(cells, 1)
    fn = f"imgs/r{k:03d}.png"
    cv2.imwrite(OUT + "/" + fn, strip[..., ::-1])
    rows.append((fn, ii, info["file_name"], len(bxs)))
    if k % 10 == 0:
        print(f"  {k}/{len(sample)}", flush=True)

H = ['<html><head><meta charset=utf-8><style>body{font-family:sans-serif;background:#111;color:#ddd}'
     'img{max-width:1400px;width:100%}.it{border-bottom:1px solid #333;padding:8px 0}</style></head><body>']
H.append("<h2>DensePose fine-tuned vs synth-only — COCO minival</h2>")
H.append("<p>Columns: <b>input</b> | <b>v3 (synthetic-only, GPS 0.62)</b> parts | "
         "<b>fine-tuned (COCO, GPS 0.73)</b> parts | <b>fine-tuned UV</b> (U=red, V=green). "
         "Detector boxes (fasterrcnn). Top rows = most crowded.</p>")
for fn, ii, fname, nb in rows:
    H.append(f"<div class=it>img {ii} ({html.escape(fname)}) — {nb} detections<br><img src='{fn}'></div>")
H.append("</body></html>")
open(OUT + "/index.html", "w").write("\n".join(H))
print(f"wrote {OUT}/index.html with {len(rows)} rows", flush=True)

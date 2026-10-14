"""DensePose-COCO dataloader (SPARSE supervision) for real-image fine-tuning.

Each annotated person instance provides:
  - ~100-150 surface points: (dp_x,dp_y) in [0,255] bbox space, dp_I in 1..24, dp_U/dp_V in [0,1].
  - dp_masks: 14 coarse-part RLEs (256x256 bbox space); their union = the body silhouette.

Our 24-part + per-part-UV convention IS the official DensePose chart (we transferred UV_Processed.mat
onto SMPL-X), so dp_I/dp_U/dp_V map straight onto our heads with NO remapping.

Two-stage crop recipe (matches data.py / inference): centered on the GT bbox, mild scale jitter only
(jitter_expand around val_expand=1.3), NO translation. hflip is OFF by default (UV mirror needs the
official per-part flip; not worth a subtle bug on sparse UV — revisit later). Color jitter optional.

Emits per sample: img[3,Z,Z] in [0,1], and a sparse-GT dict consumed by dp_losses_sparse:
  pts  [N,2]  grid_sample coords in [-1,1] (x,y) on the ZxZ prediction
  I    [N]    long, 0..23 (part id - 1)
  uv   [N,2]  GT (U,V) in [0,1]
  mask [Z,Z]  float body silhouette (1=target body, 0=bg), for the BCE mask head
"""
import os
import json
import numpy as np
import cv2
import torch
import scipy.io as sio
from torch.utils.data import Dataset, DataLoader
from pycocotools.coco import COCO
from pycocotools import mask as mask_util

_ASSETS = os.path.expanduser("~/data/bedlam2/dp_assets")


def _load_uv_symmetry():
    """Official DensePose hflip tables: per-part 256x256 (U,V)->mirrored(U,V), indexed [V_loc,U_loc]
    with the PRE-swap part; then the part index itself is swapped via dp24_lr_swap."""
    sym = sio.loadmat(f"{_ASSETS}/UV_symmetry_transforms.mat")
    Ut = np.stack([sym["U_transforms"][0, i] for i in range(24)]).astype(np.float32)  # (24,256,256)
    Vt = np.stack([sym["V_transforms"][0, i] for i in range(24)]).astype(np.float32)
    swap = np.load(f"{_ASSETS}/dp24_lr_swap.npy")                                      # (25,) 0..24
    return Ut, Vt, swap


def _color_jitter(rgb, rng):
    x = rgb.astype(np.float32)
    x *= rng.uniform(0.8, 1.2)                       # brightness
    m = x.mean(axis=(0, 1), keepdims=True)
    x = (x - m) * rng.uniform(0.8, 1.2) + m          # contrast
    return np.clip(x, 0, 255).astype(np.uint8)


class COCODP(Dataset):
    def __init__(self, ann_json, img_dir, input_size=512, train=True,
                 jitter_expand=(1.2, 1.4), val_expand=1.3, color=True, hflip=True, seed=0):
        self.coco = COCO(ann_json)
        self.img_dir = img_dir
        self.z = input_size
        self.train = train
        self.je = jitter_expand
        self.ve = val_expand
        self.color = color
        self.hflip = hflip
        self.seed = seed
        self._Ut, self._Vt, self._swap = _load_uv_symmetry() if hflip else (None, None, None)
        ids = []
        for a in self.coco.loadAnns(self.coco.getAnnIds()):
            if "dp_x" in a and len(a["dp_x"]) > 0:
                ids.append(a["id"])
        self.ann_ids = ids

    def __len__(self):
        return len(self.ann_ids)

    def _body_mask(self, ann, sx0, sy0, side):
        """Union of the 14 coarse-part RLEs (256x256 bbox space) -> ZxZ body mask in crop frame."""
        bx, by, bw, bh = ann["bbox"]
        Z = self.z
        body256 = np.zeros((256, 256), np.uint8)
        for rle in ann["dp_masks"]:
            if rle:
                m = mask_util.decode(rle)            # 256x256 {0,1}
                body256 |= m.astype(np.uint8)
        # 256x256 covers the tight bbox -> resize to bbox px, paste into ZxZ crop frame
        canvas = np.zeros((Z, Z), np.float32)
        # bbox extent within the crop, in Z units
        x0 = (bx - sx0) / side * Z
        y0 = (by - sy0) / side * Z
        w = bw / side * Z
        h = bh / side * Z
        tw, th = max(1, int(round(w))), max(1, int(round(h)))
        bm = cv2.resize(body256 * 255, (tw, th), interpolation=cv2.INTER_LINEAR)
        ix0, iy0 = int(round(x0)), int(round(y0))
        # source/destination clipping
        dx0, dy0 = max(0, ix0), max(0, iy0)
        dx1, dy1 = min(Z, ix0 + tw), min(Z, iy0 + th)
        if dx1 > dx0 and dy1 > dy0:
            sx, sy = dx0 - ix0, dy0 - iy0
            canvas[dy0:dy1, dx0:dx1] = bm[sy:sy + (dy1 - dy0), sx:sx + (dx1 - dx0)] / 255.0
        return canvas

    def __getitem__(self, i):
        ann = self.coco.loadAnns(self.ann_ids[i])[0]
        info = self.coco.loadImgs(ann["image_id"])[0]
        p = os.path.join(self.img_dir, info["file_name"])
        bgr = cv2.imread(p)
        if bgr is None:
            # degenerate fallback: black crop, empty supervision
            Z = self.z
            return (torch.zeros(3, Z, Z),
                    {"pts": torch.zeros(0, 2), "I": torch.zeros(0, dtype=torch.long),
                     "uv": torch.zeros(0, 2), "mask": torch.zeros(Z, Z)})
        rgb = bgr[..., ::-1]
        H, W = rgb.shape[:2]
        Z = self.z
        # train: FRESH OS-entropy seed every call so aug (flip/scale/color) varies across epochs &
        # workers (a fixed (seed,i) seed froze each sample's aug -> no augmentation, just relabeling).
        rng = np.random.RandomState(None) if self.train else \
            np.random.RandomState((self.seed * 1000003 + i) & 0x7fffffff)

        bx, by, bw, bh = ann["bbox"]
        cx, cy = bx + bw / 2, by + bh / 2
        e = rng.uniform(*self.je) if self.train else self.ve
        side = e * max(bw, bh)
        sx0, sy0 = cx - side / 2, cy - side / 2
        S = int(round(side))

        # crop (zero-pad out-of-image) then resize to Z
        canv = np.zeros((S, S, 3), np.uint8)
        ix0, iy0 = int(round(sx0)), int(round(sy0))
        ax0, ay0 = max(0, -ix0), max(0, -iy0)
        bx0, by0 = max(0, ix0), max(0, iy0)
        bx1, by1 = min(W, ix0 + S), min(H, iy0 + S)
        if bx1 > bx0 and by1 > by0:
            canv[ay0:ay0 + (by1 - by0), ax0:ax0 + (bx1 - bx0)] = rgb[by0:by1, bx0:bx1]
        if self.train and self.color:
            canv = _color_jitter(canv, rng)
        img = cv2.resize(canv, (Z, Z))

        # points: bbox-normalized [0,255] -> image px -> crop frac -> grid_sample [-1,1]
        dpx = np.asarray(ann["dp_x"], np.float32) / 255.0 * bw + bx
        dpy = np.asarray(ann["dp_y"], np.float32) / 255.0 * bh + by
        gx = (dpx - sx0) / side * 2 - 1
        gy = (dpy - sy0) / side * 2 - 1
        dI = np.asarray(ann["dp_I"], np.int64)                    # 1..24
        dU = np.asarray(ann["dp_U"], np.float32)
        dV = np.asarray(ann["dp_V"], np.float32)
        mask = self._body_mask(ann, sx0, sy0, side)               # (Z,Z) float

        # horizontal flip with the OFFICIAL DensePose UV mirror (image, points, parts, U/V, mask)
        if self.train and self.hflip and rng.rand() < 0.5:
            img = img[:, ::-1]
            mask = mask[:, ::-1]
            gx = -gx
            Ul = np.clip((dU * 255).astype(np.int64), 0, 255)
            Vl = np.clip((dV * 255).astype(np.int64), 0, 255)
            dU = self._Ut[dI - 1, Vl, Ul]                         # remap UV in the pre-swap chart
            dV = self._Vt[dI - 1, Vl, Ul]
            dI = self._swap[dI]                                   # swap L/R part (1..24)

        img_t = torch.from_numpy(np.ascontiguousarray(img)).permute(2, 0, 1).float() / 255.0
        pts = torch.from_numpy(np.ascontiguousarray(np.stack([gx, gy], 1))).float()
        I = torch.from_numpy(np.ascontiguousarray((dI - 1).astype(np.int64)))   # 0..23 (Long for CE)
        uv = torch.from_numpy(np.ascontiguousarray(np.stack([dU, dV], 1).astype(np.float32)))
        mask = torch.from_numpy(np.ascontiguousarray(mask))
        return img_t, {"pts": pts, "I": I, "uv": uv, "mask": mask}


def collate_coco(batch):
    imgs = torch.stack([b[0] for b in batch], 0)
    gt = [b[1] for b in batch]
    return {"img": imgs, "gt_sparse": gt}


def build_coco_loader(ann_json, img_dir, batch=32, workers=8, input_size=512,
                      train=True, seed=0, distributed=False):
    ds = COCODP(ann_json, img_dir, input_size=input_size, train=train, hflip=train, seed=seed)
    sampler = None
    if distributed:
        from torch.utils.data.distributed import DistributedSampler
        sampler = DistributedSampler(ds, shuffle=train, drop_last=train)
    loader = DataLoader(ds, batch_size=batch, shuffle=(train and sampler is None),
                        sampler=sampler, num_workers=workers, pin_memory=True,
                        drop_last=train, persistent_workers=workers > 0,
                        collate_fn=collate_coco, prefetch_factor=4 if workers > 0 else None)
    loader.sampler_ref = sampler   # so the train loop can call set_epoch
    return loader


if __name__ == "__main__":
    import sys
    ann = sys.argv[1] if len(sys.argv) > 1 else \
        os.path.expanduser("~/codes/wan/densepose/dpeval/data/densepose_coco_2014_minival.json")
    imgd = sys.argv[2] if len(sys.argv) > 2 else os.path.expanduser("~/data/coco_val2014_minival")
    ld = build_coco_loader(ann, imgd, batch=4, workers=0, train=True)
    b = next(iter(ld))
    print("img", tuple(b["img"].shape), b["img"].dtype)
    for j, g in enumerate(b["gt_sparse"]):
        print(f"  inst{j}: N={g['pts'].shape[0]} I[min,max]={int(g['I'].min())},{int(g['I'].max())} "
              f"pts[min,max]={g['pts'].min():.2f},{g['pts'].max():.2f} "
              f"uv[min,max]={g['uv'].min():.2f},{g['uv'].max():.2f} mask_fg={g['mask'].mean():.3f}")

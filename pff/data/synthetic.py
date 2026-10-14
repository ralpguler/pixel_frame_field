"""Synthetic training data (BEDLAM2 and MAMMA renders) from WebDataset shards, the fixed-canvas path of the research
loader (research/densepose/data.py, DPDecode._gt512 and build_loader with gt512=True).

A shard sample is one person instance:
  rgb.jpg    the square RGB canvas around the person (`meta.box` = [x0, y0, side] in source pixels);
  corr.png   per pixel of a FIXED 512 x 512 ground-truth canvas (`meta.gt_box`): SMPL-X face id (16 bits, R<<8|G,
             0xFFFF = background) and two barycentrics (B, A, /255);
  meta.json  box, gt_box;
  rot.npy    (10475, 6) float16: per SMPL-X vertex, the 6D rotation from the template to the camera.
The ground truth is decoded in its own canvas and never resampled: part = face -> DensePose part, uv = barycentric
blend of the face's chart corners, xyz = barycentric blend of the template vertices (for the CSC loss), rot6d =
barycentric blend of the vertex 6Ds. Only the IMAGE is cropped (a concentric scale jitter around the box) and the
loss warps the prediction into the canvas with `theta`.
"""
import glob
import io
import json
import os
import re

import cv2
import numpy as np
import torch
import webdataset as wds
from PIL import Image

from ..assets import asset

FACE_BG = 0xFFFF
STORED_EXPAND = 2.0          # the stored RGB canvas is the person box expanded 2x


def unpack_corr(rgba):
    """corr.png -> face (H,W) int32 (-1 = background) and barycentrics (H,W,3)."""
    fid = (rgba[..., 0].astype(np.uint16) << 8) | rgba[..., 1]
    fg = fid != FACE_BG
    face = np.where(fg, fid, -1).astype(np.int32)
    u = rgba[..., 2].astype(np.float32) / 255
    v = rgba[..., 3].astype(np.float32) / 255
    bary = np.stack([u, v, 1 - u - v], -1)
    bary[~fg] = 0
    return face, bary


def color_jitter(rgb, rng):
    x = rgb.astype(np.float32)
    x *= rng.uniform(0.8, 1.2)                                        # brightness
    m = x.mean((0, 1), keepdims=True)
    x = (x - m) * rng.uniform(0.8, 1.2) + m                           # contrast
    return np.clip(x, 0, 255).astype(np.uint8)


class Decode:
    """Shard sample -> img (3,Z,Z) in [0,1] and the 512 x 512 ground truth (part, uv, fg, xyz, rot6d, theta)."""

    def __init__(self, input_size=512, train=True, jitter_expand=(1.25, 1.4), val_expand=1.3,
                 uv_asset="smplx_face_uv_v2.npy", hidden_mask=True, color=True):
        # jitter_expand starts at 1.25, above the ground-truth canvas's 1.2 expansion of the box: the image crop is
        # always a little wider than the canvas, so every supervised pixel is seen by the network.
        self.z, self.train, self.je, self.ve, self.color = input_size, train, jitter_expand, val_expand, color
        self.face2part = np.load(asset("smplx_face_to_dp24.npy"))
        self.face_uv = np.load(asset(uv_asset))
        self.faces = np.load(asset("smplx_faces.npy"))
        self.v_template = np.load(asset("smplx_v_template.npy")).astype(np.float32)
        # SMPL-X hidden geometry (eyeballs, mouth interior): their labels are unlearnable, so they leave the
        # part / UV / rotation / CSC supervision; the body-mask BCE still sees the whole person.
        self.hidden = np.load(asset("smplx_hidden_faces.npy")) if hidden_mask else None

    def __call__(self, sample):
        jpg, png, meta, rotnpy = sample
        vert_rot6d = np.load(io.BytesIO(rotnpy))
        rgb = cv2.imdecode(np.frombuffer(jpg, np.uint8), cv2.IMREAD_COLOR)[..., ::-1]
        face, bary = unpack_corr(np.array(Image.open(io.BytesIO(png))))
        meta = json.loads(meta)
        S0 = rgb.shape[0]
        rng = np.random if self.train else np.random.RandomState(0)

        G = face.shape[0]
        fg = face >= 0
        part = np.zeros((G, G), np.int64)
        uv = np.zeros((2, G, G), np.float32)
        xyz = np.zeros((G, G, 3), np.float32)
        rot6 = np.zeros((G, G, 6), np.float32)
        if fg.any():
            ff, bw = face[fg], bary[fg]
            part[fg] = self.face2part[ff]
            q = (bw[:, :, None] * self.face_uv[ff]).sum(1)
            uv[0][fg] = q[:, 0]; uv[1][fg] = q[:, 1]
            tv = self.faces[ff]
            xyz[fg] = (bw[:, :, None] * self.v_template[tv]).sum(1)
            rot6[fg] = (bw[:, :, None] * vert_rot6d[tv]).sum(1)
        if self.hidden is not None and fg.any():
            hid = np.zeros_like(fg)
            hid[fg] = self.hidden[face[fg]]
            fg = fg & ~hid

        # the image crop: concentric with the box (no translation jitter), scale jitter only
        e = rng.uniform(*self.je) if self.train else self.ve
        cs = int(round((e / STORED_EXPAND) * S0)); cs = max(8, min(cs, S0))
        x0 = int(np.clip(S0 // 2 - cs // 2, 0, S0 - cs))
        y0 = int(np.clip(S0 // 2 - cs // 2, 0, S0 - cs))
        img = cv2.resize(np.ascontiguousarray(rgb[y0:y0 + cs, x0:x0 + cs]), (self.z, self.z),
                         interpolation=cv2.INTER_AREA)
        if self.train and self.color:
            img = color_jitter(img, rng)

        # affine: ground-truth canvas coordinates -> prediction coordinates (both normalised to [-1,1])
        bx0, by0, side = meta["box"]
        gx0, gy0, gside = meta["gt_box"]
        sps = side / S0
        ix0 = bx0 + x0 * sps; iy0 = by0 + y0 * sps; isz = cs * sps
        sx = gside / isz
        tx = 2.0 * (gx0 - ix0) / isz + sx - 1.0
        ty = 2.0 * (gy0 - iy0) / isz + sx - 1.0
        theta = np.array([[sx, 0.0, tx], [0.0, sx, ty]], np.float32)

        return {"img": torch.from_numpy(np.ascontiguousarray(img)).permute(2, 0, 1).float() / 255,
                "part": torch.from_numpy(part), "uv": torch.from_numpy(uv), "fg": torch.from_numpy(fg),
                "xyz": torch.from_numpy(xyz.astype(np.float16)), "theta": torch.from_numpy(theta),
                "rot6d": torch.from_numpy(rot6.astype(np.float16))}


def collate(samples):
    return {k: torch.stack([s[k] for s in samples]) for k in samples[0]}


def shard_list(bedlam_root, mamma_root=None, split="train", mamma_holdout=None):
    """BEDLAM2 shards, plus (train only) MAMMA shards with the held-out sequences removed, mixed about 50/50 by
    repeating the smaller BEDLAM2 list."""
    shards = sorted(glob.glob(f"{bedlam_root}/{split}/shard-*.tar"))
    if mamma_root and split == "train" and os.path.isdir(f"{mamma_root}/train"):
        ms = sorted(glob.glob(f"{mamma_root}/train/*.tar"))
        if mamma_holdout:
            ho = {l.strip() for l in open(os.path.expanduser(mamma_holdout)) if l.strip()}
            n0 = len(ms)
            ms = [s for s in ms if re.sub(r"_\d+\.tar$", "", os.path.basename(s)) not in ho]
            print(f"[data] MAMMA holdout: {n0 - len(ms)} shards ({len(ho)} sequences) excluded", flush=True)
        rep = max(1, round(len(ms) / max(len(shards), 1)))
        shards = shards * rep + ms
        print(f"[data] {len(shards) - len(ms)} BEDLAM2 (x{rep}) + {len(ms)} MAMMA = {len(shards)} shards", flush=True)
    assert shards, f"no shards under {bedlam_root}/{split}"
    return shards


def build_loader(shards, batch=32, workers=8, input_size=512, train=True, uv_asset="smplx_face_uv_v2.npy",
                 hidden_mask=True):
    """Infinite shuffled stream of collated batches for training (train=True), one ordered pass otherwise."""
    dec = Decode(input_size=input_size, train=train, uv_asset=uv_asset, hidden_mask=hidden_mask)
    ds = wds.WebDataset(shards, shardshuffle=1000 if train else False, nodesplitter=wds.split_by_node,
                        workersplitter=wds.split_by_worker, empty_check=False, handler=wds.warn_and_continue)
    if train:
        ds = ds.shuffle(2000).repeat()
    ds = (ds.to_tuple("rgb.jpg", "corr.png", "meta.json", "rot.npy", handler=wds.warn_and_continue)
            .map(dec, handler=wds.warn_and_continue)
            .batched(batch, collation_fn=collate, partial=False))
    return wds.WebLoader(ds, batch_size=None, num_workers=workers, pin_memory=True, persistent_workers=workers > 0,
                         prefetch_factor=4 if workers > 0 else None)

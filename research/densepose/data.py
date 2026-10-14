"""Efficient WebDataset dataloader for DensePose-from-bbox training.

Faithful to the original DensePose loss: the GT is NEVER resampled. The model input
RGB is resized (bilinear, fine), but the GT (part labels + UV) is kept at the native
crop-window resolution. At train time the predicted dense logits/UV fields are
interpolated UP to each sample's GT resolution and the loss is computed over all
foreground pixels. No nearest/naive interpolation of labels anywhere.

Per sample the loader yields (GT size = native window cs, varies per sample):
  img  (3,Z,Z) float           model input (crop window resized)
  part (cs,cs) long            part id 0..24 (0=bg), exact native-res
  uv   (2,cs,cs) float         GT (U,V), exact native-res (valid where part>0)
  fg   (cs,cs) bool            part>0 (UV valid)
A custom collate stacks `img` and keeps the variable-size GT as a per-sample list.
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

FACE_BG = 0xFFFF
ROOT = os.path.expanduser("~/data/bedlam2")
SHARDS = f"{ROOT}/dp_shards"
SHARDS_ROT = f"{ROOT}/dp_shards_rot"     # shards augmented with rot.npy (orientation GT)
SHARDS_ROT_MAMMA = os.path.expanduser("~/data/datasets/dp_shards_rot_mamma")  # MAMMA-syn multi-person/contact, train-only
SHARDS_512 = f"{ROOT}/dp_shards_512"                      # fixed 512^2 GT canvas per instance (never resampled)
SHARDS_512_MAMMA = os.path.expanduser("~/data/datasets/dp_shards_512_mamma")
ASSETS = f"{ROOT}/dp_assets"
ROT_CAP = 384                            # cap rotation-GT res (smooth field) -> bounded CPU memory
CSC_CAP = 384                            # same for the canonical-surface-coordinate GT
STORED_EXPAND = 2.0     # context factor used in preprocess_shards (native-res crops)

_PAL = np.array([
    [0, 0, 0], [178, 34, 34], [255, 99, 71], [30, 144, 255], [0, 191, 255],
    [50, 205, 50], [34, 139, 34], [255, 215, 0], [218, 165, 32],
    [255, 140, 0], [255, 200, 0], [148, 0, 211], [186, 85, 211],
    [199, 21, 133], [255, 105, 180], [0, 206, 209], [64, 224, 208],
    [0, 128, 128], [72, 209, 204], [255, 0, 255], [238, 130, 238],
    [139, 0, 139], [218, 112, 214], [240, 230, 140], [189, 183, 107],
], np.uint8)


def unpack_rgba(rgba):
    """RGBA pack -> face (H,W int32, -1 bg), bary (H,W,3). Matches encode_correspondence."""
    fid = (rgba[..., 0].astype(np.uint16) << 8) | rgba[..., 1]
    fg = fid != FACE_BG
    face = np.where(fg, fid, -1).astype(np.int32)
    u = rgba[..., 2].astype(np.float32) / 255
    v = rgba[..., 3].astype(np.float32) / 255
    bary = np.stack([u, v, 1 - u - v], -1)
    bary[~fg] = 0
    return face, bary


class DPDecode:
    """Decode + augment -> input image (resized) + EXACT native-res dense GT.
    Runs in DataLoader workers. GT (part/uv) is never resampled."""

    def __init__(self, input_size=384, train=True, jitter_expand=(1.25, 1.4),
                 val_expand=1.3, trans_jitter=0.0, color=True, hflip=True, orient=False,
                 csc=False, uv_asset="smplx_face_uv.npy", gt512=False, hidden_mask=False):
        # NOTE jitter_expand[0] = 1.25 > GT_EXPAND = 1.2 (bedlam/render_gt512.py). It used to be 1.2,
        # i.e. EXACTLY the GT canvas -- and then integer rounding of the crop window
        # (cs = round(e/2 * S0)) could make the image crop a hair SMALLER than the GT canvas, so a ring
        # of GT pixels had no image support and the loss would supervise pixels the network never saw.
        # Measured: 2/300 samples. A 4% margin makes it rounding-proof.
        # two-stage: box is assumed ~correct & centered on the TARGET, so centering IS the target
        # signal. NO translation jitter (it lets the model pick any person in a crowd), and only
        # mild scale jitter around the inference val_expand=1.3. hflip/color (appearance) kept.
        self.z = input_size
        self.train = train
        self.je = jitter_expand
        self.ve = val_expand
        self.tj = trans_jitter
        self.color = color
        self.orient = orient
        # hflip needs the per-vertex R mirror for the orientation target (subtle); disable when orient.
        self.hflip = hflip and not orient
        self.csc = csc
        self.gt512 = gt512
        # SMPL-X hidden geometry (eyeballs + mouth/socket interior): those pixels' labels are
        # unlearnable coin flips (1% of head px). Drop them from part/UV/rot/CSC supervision by
        # clearing fg -- but AFTER part is filled, so the body-mask BCE still sees the full person.
        self.hidden = np.load(f"{ASSETS}/smplx_hidden_faces.npy") if hidden_mask else None
        self.face2part = np.load(f"{ASSETS}/smplx_face_to_dp24.npy")
        self.face_uv = np.load(f"{ASSETS}/{uv_asset}")     # v2 = fold-free hands/feet (build_part_uv_v2.py)
        mir = np.load(f"{ASSETS}/smplx_mirror.npz")
        self.face_mirror = mir["face_mirror"]
        self.corner_perm = mir["corner_perm"].astype(np.int64)
        if orient:
            self.faces = np.load(f"{ASSETS}/smplx_faces.npy")          # (20908,3) for rot6d interp
        if csc or gt512:
            self.faces = np.load(f"{ASSETS}/smplx_faces.npy")
            self.v_template = np.load(f"{ASSETS}/smplx_v_template.npy").astype(np.float32)  # (10475,3)
        if gt512:
            # The GT canvas is a FIXED expansion of the tight body box; every train crop is a WIDER
            # expansion of the SAME box, so the GT is always inside the network's view -- but only if the
            # crops stay concentric. Translation jitter would break that.
            assert trans_jitter == 0.0, "gt512 needs concentric crops: trans_jitter must be 0"
            # check the RESOLVED flag: orient already forces hflip off (self.hflip above)
            assert not self.hflip, "gt512: hflip would need the GT canvas mirrored too"
        assert self.hidden is None or gt512, "hidden_mask is only wired into the gt512 path"

    def _rng(self):
        return np.random if self.train else np.random.RandomState(0)

    def __call__(self, sample):
        if self.orient:
            jpg, png, meta, rotnpy = sample
            vert_rot6d = np.load(io.BytesIO(rotnpy))                   # (10475,6) f16 for this body
        else:
            jpg, png, meta = sample
        rgb = cv2.imdecode(np.frombuffer(jpg, np.uint8), cv2.IMREAD_COLOR)[..., ::-1]
        face, bary = unpack_rgba(np.array(Image.open(io.BytesIO(png))))
        S0 = rgb.shape[0]
        rng = self._rng()

        if self.gt512:
            return self._gt512(rgb, face, bary, json.loads(meta), vert_rot6d, S0, rng)

        # ---- choose crop window (box-jitter) in native coords ----
        e = rng.uniform(*self.je) if self.train else self.ve
        cs = int(round((e / STORED_EXPAND) * S0)); cs = max(8, min(cs, S0))
        if self.train and self.tj > 0:
            mx = int(self.tj * cs); ox = rng.randint(-mx, mx + 1); oy = rng.randint(-mx, mx + 1)
        else:
            ox = oy = 0
        x0 = int(np.clip(S0 // 2 + ox - cs // 2, 0, S0 - cs))
        y0 = int(np.clip(S0 // 2 + oy - cs // 2, 0, S0 - cs))
        fw = np.ascontiguousarray(face[y0:y0 + cs, x0:x0 + cs])
        bw = np.ascontiguousarray(bary[y0:y0 + cs, x0:x0 + cs])

        # ---- input image: crop window resized to Z (bilinear; RGB may interpolate) ----
        img = cv2.resize(np.ascontiguousarray(rgb[y0:y0 + cs, x0:x0 + cs]),
                         (self.z, self.z), interpolation=cv2.INTER_AREA)
        flip = self.train and self.hflip and rng.rand() < 0.5
        if flip:
            img = np.ascontiguousarray(img[:, ::-1])
            fw = np.ascontiguousarray(fw[:, ::-1]); bw = np.ascontiguousarray(bw[:, ::-1])
            m = fw >= 0
            of = fw[m]
            nb = np.zeros((of.shape[0], 3), bw.dtype)
            np.put_along_axis(nb, self.corner_perm[of], bw[m], axis=1)
            fw[m] = self.face_mirror[of]; bw[m] = nb
        if self.train and self.color:
            img = self._color(img, rng)

        # ---- EXACT native-res dense GT (no resampling) ----
        fg = fw >= 0
        part = np.zeros((cs, cs), np.int64)
        uv = np.zeros((2, cs, cs), np.float32)
        if fg.any():
            ff = fw[fg]
            part[fg] = self.face2part[ff]
            uvp = (bw[fg][:, :, None] * self.face_uv[ff]).sum(1)   # (n,2)
            uv[0][fg] = uvp[:, 0]; uv[1][fg] = uvp[:, 1]
        out = {
            "img": torch.from_numpy(np.ascontiguousarray(img)).permute(2, 0, 1).float() / 255,
            "part": torch.from_numpy(part),
            "uv": torch.from_numpy(uv),
            "fg": torch.from_numpy(fg),
        }
        if self.orient:
            # ROTATION GT at a capped res. NEVER interpolate rotations: bilinearly resizing the 6D field
            # AVERAGES THE ROTATIONS OF TWO DIFFERENT SURFACES wherever a limb occludes the torso, and
            # Gram-Schmidt on that blend yields a frame belonging to neither. Where the blend lands near
            # zero (a pixel NEAREST calls fg but whose bilinear value came mostly from background) the
            # Gram-Schmidt runs on noise and the rotation is arbitrary. Measured on MAMMA (832->384):
            # 4.4% of pixels wrong by >10 deg, 3.2% by >30 deg, up to a full 180 deg flip -- concentrated
            # exactly at the occlusion boundaries MAMMA exists to teach.
            # Instead resample the CORRESPONDENCE (face + barycentrics, both NEAREST -- a face id is a
            # label, not a quantity) and RECOMPUTE the rotation from it. Every output pixel then carries
            # the exact rotation of exactly one surface point. Also cheaper: no native-res 6ch buffer.
            R = min(cs, ROT_CAP)
            if R != cs:
                fwr = cv2.resize(fw.astype(np.float32), (R, R),
                                 interpolation=cv2.INTER_NEAREST).astype(np.int32)
                bwr = cv2.resize(bw, (R, R), interpolation=cv2.INTER_NEAREST)
            else:
                fwr, bwr = fw, bw
            rfg = fwr >= 0
            r6 = np.zeros((R, R, 6), np.float32)
            if rfg.any():
                r6[rfg] = (bwr[rfg][:, :, None] * vert_rot6d[self.faces[fwr[rfg]]]).sum(1)
            out["rot6d"] = torch.from_numpy(np.ascontiguousarray(r6).astype(np.float16))  # (R,R,6)
            out["rot_fg"] = torch.from_numpy(np.ascontiguousarray(rfg))                   # (R,R) bool
        if self.csc:
            # Canonical-surface-coordinate GT at a capped res. Downsample the FACE+BARY (nearest) and
            # recompute, rather than interpolating xyz/uv: interpolating would blend across occlusion
            # boundaries and across chart seams, inventing points that are on no surface. part/uv/xyz are
            # all derived from the SAME resampled face+bary, so they are mutually consistent by
            # construction -- the loss's validity check depends on that.
            R = min(cs, CSC_CAP)
            if R != cs:
                fwr = cv2.resize(fw.astype(np.float32), (R, R),
                                 interpolation=cv2.INTER_NEAREST).astype(np.int32)
                bwr = cv2.resize(bw, (R, R), interpolation=cv2.INTER_NEAREST)
            else:
                fwr, bwr = fw, bw
            cfg = fwr >= 0
            cxyz = np.zeros((R, R, 3), np.float32)
            cpart = np.zeros((R, R), np.int16)
            cuv = np.zeros((2, R, R), np.float32)
            if cfg.any():
                ff2 = fwr[cfg]
                cpart[cfg] = self.face2part[ff2]
                tv2 = self.faces[ff2]
                cxyz[cfg] = (bwr[cfg][:, :, None] * self.v_template[tv2]).sum(1)
                q = (bwr[cfg][:, :, None] * self.face_uv[ff2]).sum(1)
                cuv[0][cfg] = q[:, 0]; cuv[1][cfg] = q[:, 1]
            out["csc_xyz"] = torch.from_numpy(cxyz.astype(np.float16))     # (R,R,3) template metres
            out["csc_part"] = torch.from_numpy(cpart)                      # (R,R) int16, 0=bg
            out["csc_uv"] = torch.from_numpy(cuv.astype(np.float16))       # (2,R,R)
            out["csc_fg"] = torch.from_numpy(cfg)                          # (R,R) bool
        return out

    def _gt512(self, rgb, face, bary, meta, vert_rot6d, S0, rng):
        """GT lives in its own FIXED 512^2 canvas and is NEVER touched. Only the image is cropped.

        The prediction is warped INTO this canvas in the loss, via a pure scale+translate affine -- both
        crops are axis-aligned squares in SOURCE-FRAME coordinates, so we can write it in closed form.
        Warping the PREDICTION is legitimate (the network's output is an inherently smooth field, so this
        only EVALUATES it at the GT's pixel centres); warping the GT is what destroyed information.
        """
        G = face.shape[0]
        fg = face >= 0
        part = np.zeros((G, G), np.int64)
        uv = np.zeros((2, G, G), np.float32)
        xyz = np.zeros((G, G, 3), np.float32)
        rot6 = np.zeros((G, G, 6), np.float32)
        if fg.any():
            ff = face[fg]; bw = bary[fg]
            part[fg] = self.face2part[ff]
            q = (bw[:, :, None] * self.face_uv[ff]).sum(1)
            uv[0][fg] = q[:, 0]; uv[1][fg] = q[:, 1]
            tv = self.faces[ff]
            xyz[fg] = (bw[:, :, None] * self.v_template[tv]).sum(1)
            if vert_rot6d is not None:
                rot6[fg] = (bw[:, :, None] * vert_rot6d[tv]).sum(1)
        if self.hidden is not None and fg.any():
            # part stays filled (mask BCE uses part>0); only the fg gate for CE/UV/rot/CSC drops
            hid = np.zeros_like(fg)
            hid[fg] = self.hidden[face[fg]]
            fg = fg & ~hid

        # ---- image crop (the ONLY place augmentation lives) ----
        e = rng.uniform(*self.je) if self.train else self.ve
        cs = int(round((e / STORED_EXPAND) * S0)); cs = max(8, min(cs, S0))
        x0 = int(np.clip(S0 // 2 - cs // 2, 0, S0 - cs))
        y0 = int(np.clip(S0 // 2 - cs // 2, 0, S0 - cs))
        img = cv2.resize(np.ascontiguousarray(rgb[y0:y0 + cs, x0:x0 + cs]),
                         (self.z, self.z), interpolation=cv2.INTER_AREA)
        if self.train and self.color:
            img = self._color(img, rng)

        # ---- affine: GT-canvas normalised coords -> prediction normalised coords ----
        bx0, by0, side = meta["box"]                       # stored RGB canvas, in SOURCE px
        gx0, gy0, gside = meta["gt_box"]                   # fixed GT canvas,   in SOURCE px
        sps = side / S0                                    # source px per stored px
        ix0 = bx0 + x0 * sps; iy0 = by0 + y0 * sps; isz = cs * sps   # image crop, in SOURCE px
        sx = gside / isz                                   # <= 1: the GT box sits inside the image crop
        tx = 2.0 * (gx0 - ix0) / isz + sx - 1.0
        ty = 2.0 * (gy0 - iy0) / isz + sx - 1.0
        theta = np.array([[sx, 0.0, tx], [0.0, sx, ty]], np.float32)

        out = {
            "img": torch.from_numpy(np.ascontiguousarray(img)).permute(2, 0, 1).float() / 255,
            "part": torch.from_numpy(part),                        # (G,G) 0=bg 1..24
            "uv": torch.from_numpy(uv),                            # (2,G,G)
            "fg": torch.from_numpy(fg),                            # (G,G)
            "xyz": torch.from_numpy(xyz.astype(np.float16)),       # (G,G,3) template metres
            "theta": torch.from_numpy(theta),                      # (2,3)
        }
        if self.orient:
            out["rot6d"] = torch.from_numpy(rot6.astype(np.float16))   # (G,G,6) -- exact, never resampled
        return out

    @staticmethod
    def _color(rgb, rng):
        x = rgb.astype(np.float32)
        x *= rng.uniform(0.8, 1.2)
        m = x.mean((0, 1), keepdims=True)
        x = (x - m) * rng.uniform(0.8, 1.2) + m
        return np.clip(x, 0, 255).astype(np.uint8)


def collate_512(samples):
    """Everything is a fixed 512^2 -> real batched tensors (the old GT was variable-size, which is why
    dp_losses had to loop over the batch in python)."""
    b = {k: torch.stack([s[k] for s in samples]) for k in samples[0]}
    return b


def collate_dense(samples):
    """Stack imgs (fixed Z); keep variable-size dense GT as a per-sample list.
    GT tuple is (part, uv, fg) or (part, uv, fg, rot6d) when the orientation target is present."""
    imgs = torch.stack([s["img"] for s in samples])
    if "csc_xyz" in samples[0]:          # (part, uv, fg, rot6d, rot_fg, csc_xyz, csc_part, csc_uv, csc_fg)
        gt = [(s["part"], s["uv"], s["fg"], s["rot6d"], s["rot_fg"],
               s["csc_xyz"], s["csc_part"], s["csc_uv"], s["csc_fg"]) for s in samples]
    elif "rot6d" in samples[0]:
        gt = [(s["part"], s["uv"], s["fg"], s["rot6d"], s["rot_fg"]) for s in samples]
    else:
        gt = [(s["part"], s["uv"], s["fg"]) for s in samples]
    return {"img": imgs, "gt": gt}


def dense_gt(rgba, face2part=None, face_uv=None):
    """EXACT native-res dense GT for visualization only (no resampling).
    Returns (part HxW int, uv HxW2 float, fg HxW bool)."""
    if face2part is None:
        face2part = np.load(f"{ASSETS}/smplx_face_to_dp24.npy")
    if face_uv is None:
        face_uv = np.load(f"{ASSETS}/smplx_face_uv.npy")
    face, bary = unpack_rgba(rgba)
    fg = face >= 0
    part = np.zeros(face.shape, np.int64)
    uv = np.zeros((*face.shape, 2), np.float32)
    part[fg] = face2part[face[fg]]
    uv[fg] = (bary[fg][:, :, None] * face_uv[face[fg]]).sum(1)
    return part, uv, fg


def build_loader(split, batch=32, workers=8, input_size=384, n_points=1024,
                 train=None, shards=None, orient=False, csc=False, uv_asset="smplx_face_uv.npy",
                 gt512=False, hidden_mask=False, mamma=True, mamma_holdout=""):
    if train is None:
        train = (split == "train")
    if shards is None:
        root = (SHARDS_512 if gt512 else SHARDS_ROT) if orient else SHARDS
        mroot = SHARDS_512_MAMMA if gt512 else SHARDS_ROT_MAMMA
        shards = sorted(glob.glob(f"{root}/{split}/shard-*.tar"))
        if mamma and orient and split == "train" and os.path.isdir(f"{mroot}/train"):
            mshards = sorted(glob.glob(f"{mroot}/train/*.tar"))            # extend train with MAMMA-syn
            if mamma_holdout:
                # held-out MAMMA sequences (build_mamma_holdout.py): drop every shard of those seqs
                ho = {l.strip() for l in open(os.path.expanduser(mamma_holdout)) if l.strip()}
                n0 = len(mshards)
                mshards = [s for s in mshards
                           if re.sub(r"_\d+\.tar$", "", os.path.basename(s)) not in ho]
                print(f"[build_loader] mamma holdout: {n0 - len(mshards)} shards "
                      f"({len(ho)} seqs) excluded", flush=True)
            # ~50/50 mix: upweight the smaller BEDLAM2 set to parity by repeating its shard list
            rep = max(1, round(len(mshards) / max(len(shards), 1)))
            shards = shards * rep + mshards
            print(f"[build_loader] ~50/50 mix: {len(shards) - len(mshards)} (BEDLAM2 x{rep}) + "
                  f"{len(mshards)} MAMMA = {len(shards)} train shards", flush=True)
    assert shards, f"no shards for split={split} orient={orient}"
    dec = DPDecode(input_size=input_size, train=train, orient=orient, csc=csc, uv_asset=uv_asset,
                   gt512=gt512, hidden_mask=hidden_mask)
    ds = wds.WebDataset(shards, shardshuffle=1000 if train else False,
                        nodesplitter=wds.split_by_node,
                        workersplitter=wds.split_by_worker,
                        empty_check=False, handler=wds.warn_and_continue)
    if train:
        ds = ds.shuffle(2000).repeat()
    keys = ("rgb.jpg", "corr.png", "meta.json", "rot.npy") if orient else \
           ("rgb.jpg", "corr.png", "meta.json")
    ds = (ds.to_tuple(*keys, handler=wds.warn_and_continue)
            .map(dec, handler=wds.warn_and_continue)
            .batched(batch, collation_fn=(collate_512 if gt512 else collate_dense), partial=False))
    return wds.WebLoader(ds, batch_size=None, num_workers=workers, pin_memory=True,
                         persistent_workers=workers > 0,
                         prefetch_factor=4 if workers > 0 else None)


if __name__ == "__main__":
    import sys
    loader = build_loader(sys.argv[1] if len(sys.argv) > 1 else "train", batch=8, workers=2)
    b = next(iter(loader))
    print("img", tuple(b["img"].shape), b["img"].dtype, "| gt samples:", len(b["gt"]))
    part0, uv0, fg0 = b["gt"][0]
    print("gt[0] part", tuple(part0.shape), "uv", tuple(uv0.shape), "fg frac", fg0.float().mean().item())
    allp = torch.cat([g[0].flatten() for g in b["gt"]])
    print("part histogram (0=bg):", torch.bincount(allp, minlength=25).tolist())

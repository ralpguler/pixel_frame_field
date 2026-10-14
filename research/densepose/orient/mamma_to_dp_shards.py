"""Convert MAMMA-syn WebDataset shards -> our oriented-DensePose format (dp_shards_rot).

MAMMA-syn source (per .tar = one seq chunk): sample key <cam>/<frame> with members
  <k>.jpg (RGB), <k>.mask.jpg (person mask), <k>.data.pyd (pickle: {person_idx: dict}).
Each person dict carries `vertices3d` (10475,3) POSED SMPL-X verts already in CAMERA space
(OpenCV, +Z fwd) — verified: cam_int @ vertices3d reproduces vertices2d at ~0px, the cam_ext
route is off by thousands of px. So we use vertices3d directly for BOTH rasterization and rot6d.

Output (mirrors dp_shards_rot, one sample per (frame,person)):
  rgb.jpg  : EXPAND-context square RGB crop (person ~1/EXPAND of canvas, STORED_EXPAND=2.0)
  corr.png : RGBA pack of face index + barycentric (encode_correspondence.pack_rgba), same affine
  rot.npy  : per-vertex 6D rotation table (10475,6) f16 (rest neutral template -> camera frame)
  meta.json: uid=seq/cam/frame/person, gender, box, visibility

Per frame all people are rasterized together (z-buffered) for correct inter-person occlusion.
Parameterized as --shard <tar> --out <dir> so it drops into a Slurm array over all shards.
"""
import argparse
import io
import json
import os
import pickle
import sys
import tarfile

import cv2
import numpy as np
import torch
from PIL import Image

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE + "/..")                       # densepose/
sys.path.insert(0, os.path.expanduser("~/codes/wan/bedlam"))
import webdataset as wds                                                    # noqa: E402
from render_correspondence import _rasterize                               # noqa: E402
from encode_correspondence import pack_rgba                                # noqa: E402
from orient.rot6d import (get_template, rotation_from_deformation,          # noqa: E402
                          vertex_normals, _FLIP)

ASSETS = os.path.expanduser("~/data/bedlam2/dp_assets")
FACES = np.load(f"{ASSETS}/smplx_faces.npy").astype(np.int64)              # (20908,3)

EXPAND = 2.0            # == data.STORED_EXPAND: person fills ~1/EXPAND of the stored canvas
MAX_SIDE = 1280        # cap stored canvas long side; never upscale
MIN_FG_PX = 32         # min visible person pixels in the crop to keep a sample
MIN_VIS_RATE = 0.10    # skip a person whose visible-vertex rate is below this
BG_RGBA = (255, 255, 0, 0)     # FACE_BG sentinel -> unpacks to face=-1


# ----------------------------------------------------------------------------- rasterize
def rasterize_people(vlist, K, H, W, device, max_bbox_area=1 << 16):
    """Rasterize a frame's people together (shared K, per-person camera verts already applied).
    Returns face_idx (H,W int32,-1 bg, LOCAL face id), bary (H,W,3), inst (H,W int16,-1)."""
    Ft = torch.as_tensor(FACES, device=device)
    all_v, all_f, starts, voff, foff = [], [], [], 0, 0
    for v in vlist:
        vt = torch.as_tensor(v, dtype=torch.float32, device=device)
        all_v.append(vt); all_f.append(Ft + voff)
        starts.append(foff); voff += vt.shape[0]; foff += Ft.shape[0]
    verts = torch.cat(all_v, 0); faces = torch.cat(all_f, 0)
    gface, bary = _rasterize(verts, faces, K, H, W, max_bbox_area=max_bbox_area)
    face_idx = torch.full((H, W), -1, dtype=torch.int32, device=device)
    inst = torch.full((H, W), -1, dtype=torch.int16, device=device)
    fg = gface >= 0
    gfi = gface[fg].long()
    st = torch.tensor(starts, device=device)
    which = torch.bucketize(gfi, st, right=True) - 1
    face_idx[fg] = (gfi - st[which]).to(torch.int32)
    inst[fg] = which.to(torch.int16)
    return face_idx.cpu().numpy(), bary.cpu().numpy(), inst.cpu().numpy()


# ----------------------------------------------------------------------------- rot6d GT
def rot6d_from_camverts(vc):
    """Per-vertex 6D rotation table (10475,6) f16 from POSED camera-space verts (OpenCV).
    Same definition as orient.rot6d.vert_rot6d_for_body, minus the pose_instance step
    (MAMMA gives posed verts directly). Neutral rest template -> PyTorch3D camera frame."""
    v_template, faces, n_rest = get_template("cpu")            # neutral rest (numpy f64)
    vp = vc.astype(np.float64) * _FLIP                         # OpenCV cam -> PyTorch3D view
    n_cam = vertex_normals(vp, faces)
    R = rotation_from_deformation(v_template, vp, faces, n_rest, n_cam)     # (Nv,3,3)
    return np.concatenate([R[:, :, 0], R[:, :, 1]], axis=1).astype(np.float16)


GT_CANVAS = 512
GT_EXPAND = 1.2      # <= data.py's tightest train jitter (1.2), so the GT is inside every image crop


def crop_K(K, center, scale, canvas=GT_CANVAS, expand=GT_EXPAND):
    """Intrinsics of the fixed-size GT crop. MAMMA's tight body box side is scale*200."""
    cx, cy = float(center[0]), float(center[1])
    side = float(expand * scale * 200.0)
    gx0, gy0 = cx - side / 2.0, cy - side / 2.0
    s = canvas / side
    Kp = np.asarray(K, np.float32).copy()
    Kp[0, 0] *= s; Kp[1, 1] *= s
    Kp[0, 2] = (K[0, 2] - gx0) * s
    Kp[1, 2] = (K[1, 2] - gy0) * s
    return Kp, [gx0, gy0, side]


def render_gt512(vlist, K, idx, center, scale, device):
    """Rasterise EVERY person of the frame directly into this person's fixed 512^2 GT canvas.

    Replaces rasterise-at-frame-res + cv2.warpAffine. That old path cost MAMMA three separate
    corruptions, all stacking: (1) warpAffine works in pixel-CORNER coords while the rasteriser samples
    pixel CENTRES -> a half-pixel offset; (2) MAMMA's crops exceed MAX_SIDE=1280, so the correspondence
    was NEAREST-DOWNSCALED, point-sampling real label resolution away; (3) the loader then bilinearly
    resized the 6D rotations, AVERAGING two different surfaces across every occlusion boundary.
    Rendering straight into the target canvas has none of them, and the GT is never resampled again.

    All people are rasterised together, so inter-person occlusion -- the entire reason MAMMA exists --
    is correct IN THE CROP. Non-target people occlude, then become background.
    """
    Kp, box = crop_K(K, center, scale)
    face, bary, inst = rasterize_people(vlist, Kp, GT_CANVAS, GT_CANVAS, device)
    m = inst == idx
    if int(m.sum()) < MIN_FG_PX:
        return None
    fi = np.where(m, face, -1).astype(np.int32)
    b2 = bary[..., :2].astype(np.float16).copy(); b2[~m] = 0
    return pack_rgba(fi, b2), box


# ----------------------------------------------------------------------------- crop/warp
def make_crop(frame_rgb, face_full, bary_full, mask_i, center, scale):
    """Warp frame RGB + this person's correspondence into one EXPAND-context square canvas.
    Returns (rgb_crop uint8 HxWx3, rgba_crop uint8 HxWx4, box=[bx0,by0,side]) or None."""
    cx, cy = float(center[0]), float(center[1])
    side = EXPAND * float(scale) * 200.0
    bx0, by0 = cx - side / 2, cy - side / 2
    cz = int(round(min(side, MAX_SIDE)))
    if cz < 8:
        return None
    s = cz / side
    M = np.array([[s, 0, -bx0 * s], [0, s, -by0 * s]], np.float32)
    interp = cv2.INTER_AREA if s < 1 else cv2.INTER_LINEAR
    rgb = cv2.warpAffine(frame_rgb, M, (cz, cz), flags=interp,
                         borderMode=cv2.BORDER_REPLICATE)
    face_i = np.where(mask_i, face_full, -1).astype(np.int32)
    bary2 = bary_full[..., :2].astype(np.float16).copy()
    bary2[~mask_i] = 0
    rgba_full = pack_rgba(face_i, bary2)                       # (H,W,4) u8
    rgba = cv2.warpAffine(rgba_full, M, (cz, cz), flags=cv2.INTER_NEAREST,
                          borderMode=cv2.BORDER_CONSTANT, borderValue=BG_RGBA)
    if int((rgba[..., 0] != 255).sum()) < MIN_FG_PX:          # (255,255)=FACE_BG hi/lo
        return None
    return rgb, rgba, [bx0, by0, side]


# ----------------------------------------------------------------------------- driver
def load_frames(tar_path):
    """Yield (uidbase, cam, frame, rgb HxWx3, persons dict) per frame in a MAMMA-syn tar."""
    t = tarfile.open(tar_path)
    members = t.getmembers()
    keys = {}
    for m in members:
        if m.name.endswith(".data.pyd"):
            keys.setdefault(m.name[:-len(".data.pyd")], {})["pyd"] = m
        elif m.name.endswith(".mask.jpg"):
            keys.setdefault(m.name[:-len(".mask.jpg")], {})["mask"] = m
        elif m.name.endswith(".jpg"):
            keys.setdefault(m.name[:-len(".jpg")], {})["jpg"] = m
    for k in sorted(keys):
        e = keys[k]
        if "pyd" not in e or "jpg" not in e:
            continue
        parts = k.split("/")
        frame, cam, seq = parts[-1], parts[-2], parts[-3]
        rgb = cv2.imdecode(np.frombuffer(t.extractfile(e["jpg"]).read(), np.uint8),
                           cv2.IMREAD_COLOR)[..., ::-1]        # BGR->RGB
        persons = pickle.load(t.extractfile(e["pyd"]))
        yield seq, cam, frame, np.ascontiguousarray(rgb), persons
    t.close()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--shard", required=True, help="input MAMMA-syn .tar")
    ap.add_argument("--out", required=True, help="output dir (writes shard-<name>.tar)")
    ap.add_argument("--gt512", action="store_true", help="render corr into a FIXED 512^2 crop canvas")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--outname", default=None, help="output tar basename (default: shard-000000)")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    name = a.outname or "shard-000000"
    out_tar = f"{a.out}/{name}.tar"
    tmp = out_tar + ".tmp"

    n_in = n_out = n_skip_vis = n_skip_empty = 0
    with wds.TarWriter(tmp) as sink:
        for seq, cam, frame, rgb, persons in load_frames(a.shard):
            H, W = rgb.shape[:2]
            pids = sorted(persons.keys())
            vlist, K = [], None
            for pid in pids:
                p = persons[pid]
                vlist.append(np.asarray(p["vertices3d"], np.float64))
                if K is None:
                    K = np.asarray(p["cam_int"], np.float32)
            if K is None:
                continue
            face_full, bary_full, inst = rasterize_people(vlist, K, H, W, a.device)
            for idx, pid in enumerate(pids):
                p = persons[pid]
                n_in += 1
                vis = np.asarray(p["vertex_visibility"]).reshape(-1)
                vis_rate = float((vis > 0).mean())
                if vis_rate < MIN_VIS_RATE:
                    n_skip_vis += 1; continue
                mask_i = inst == idx
                if not mask_i.any():
                    n_skip_empty += 1; continue
                crop = make_crop(rgb, face_full, bary_full, mask_i,
                                 p["center"], p["scale"])
                if crop is None:
                    n_skip_empty += 1; continue
                rgb_c, rgba_c, box = crop
                if a.gt512:
                    g = render_gt512(vlist, K, idx, p["center"], p["scale"], a.device)
                    if g is None:
                        n_skip_empty += 1; continue
                    rgba_c, gt_box = g          # the 512^2 GT canvas replaces the warped correspondence
                rot = rot6d_from_camverts(np.asarray(p["vertices3d"], np.float64))
                uid = f"{seq}/{cam}/{frame}/{pid}"
                jpg = cv2.imencode(".jpg", rgb_c[..., ::-1],
                                   [cv2.IMWRITE_JPEG_QUALITY, 95])[1].tobytes()
                pbuf = io.BytesIO(); Image.fromarray(rgba_c, "RGBA").save(pbuf, "PNG")
                rbuf = io.BytesIO(); np.save(rbuf, rot)
                meta = {"uid": uid, "seq": seq, "cam": cam, "frame": frame,
                        "person": int(pid), "gender": str(p["gender"]),
                        "box": box, "vis_rate": vis_rate}
                if a.gt512:
                    meta["gt_box"] = gt_box; meta["canvas"] = GT_CANVAS; meta["gt_expand"] = GT_EXPAND
                sink.write({"__key__": uid.replace("/", "_"),
                            "rgb.jpg": jpg, "corr.png": pbuf.getvalue(),
                            "rot.npy": rbuf.getvalue(),
                            "meta.json": json.dumps(meta).encode()})
                n_out += 1
    os.rename(tmp, out_tar)
    print(f"[mamma->dp_rot] {os.path.basename(a.shard)}: {n_out} written "
          f"(seen {n_in}, skip_vis {n_skip_vis}, skip_empty {n_skip_empty}) -> {out_tar}",
          flush=True)


if __name__ == "__main__":
    main()

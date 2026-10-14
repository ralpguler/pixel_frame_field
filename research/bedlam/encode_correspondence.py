"""Pack per-instance dense correspondence into a compact RGBA PNG and process
a whole BEDLAM2 sequence.

RGBA layout (tight bbox per visible instance):
  R, G = face index (uint16) split hi/lo byte; sentinel 0xFFFF = background
  B, A = barycentric u, v in uint8 [0,255] (third = 1-u-v)
Lossless face index; barycentric quantized to 1/255 (sub-pixel within a tri).

Per sequence we write:
  <out>/<seq>/<sub_seq>/<frame>_<inst>.png   - the RGBA correspondence crops
  <out>/annotations/<seq>.jsonl              - COCO-style records (one per inst)
  <out>/posed3d/<seq>.npz                    - SMPL-X params + camera per uid
Posed 3D is stored as generative params (pose_world,betas,trans_world,cam_*),
not dense verts: ~1KB/instance and reconstructs the exact mesh on demand.
"""
import json
import os
import tarfile
import numpy as np
import torch
from PIL import Image

import imageio.v3 as iio
from render_correspondence import crops_from_frame, rasterize_frame

FACE_BG = 0xFFFF


def pack_rgba(face_crop, bary2_crop):
    """face_crop (h,w int32, -1 bg); bary2_crop (h,w,2 float16). -> (h,w,4) u8."""
    h, w = face_crop.shape
    rgba = np.zeros((h, w, 4), np.uint8)
    fg = face_crop >= 0
    fid = np.where(fg, face_crop, FACE_BG).astype(np.uint16)
    rgba[..., 0] = (fid >> 8).astype(np.uint8)
    rgba[..., 1] = (fid & 0xFF).astype(np.uint8)
    b = np.clip(bary2_crop.astype(np.float32), 0, 1)
    rgba[..., 2] = np.where(fg, np.round(b[..., 0] * 255), 0).astype(np.uint8)
    rgba[..., 3] = np.where(fg, np.round(b[..., 1] * 255), 0).astype(np.uint8)
    return rgba


def unpack_rgba(rgba):
    """Inverse, for verification. -> face (h,w int32,-1 bg), bary (h,w,3)."""
    fid = (rgba[..., 0].astype(np.uint16) << 8) | rgba[..., 1]
    fg = fid != FACE_BG
    face = np.where(fg, fid, -1).astype(np.int32)
    u = rgba[..., 2].astype(np.float32) / 255
    v = rgba[..., 3].astype(np.float32) / 255
    bary = np.stack([u, v, 1 - u - v], -1)
    bary[~fg] = 0
    return face, bary


def iter_mp4_frames(mp4_path, want_frames):
    """Yield (frame_idx, rgb) for frame indices in want_frames (sorted set)."""
    want = set(want_frames)
    hi = max(want)
    for i, fr in enumerate(iio.imiter(mp4_path)):
        if i in want:
            yield i, fr
        if i >= hi:
            break


def process_sequence(seq, labels_dir, movie_tar, out_dir, device="cuda",
                     limit_frames=None, limit_subs=None):
    # Load ONCE the arrays we need (NpzFile re-decompresses the whole array on
    # every access, so per-instance d[k][r] is both slow and a memory bomb; it
    # also avoids loading the huge unused gtkps/proj_verts arrays).
    with np.load(f"{labels_dir}/{seq}.npz", allow_pickle=True) as z:
        img = z["imgname"]
        d = {k: z[k] for k in ("pose_world", "shape", "trans_world",
                               "cam_int", "cam_ext", "gender")}
    # group rows by (sub_seq, frame)
    groups = {}
    for r, name in enumerate(img):
        sub, fn = name.split("/")[0], int(name.split("_")[-1].split(".")[0])
        groups.setdefault(sub, {}).setdefault(fn, []).append(r)

    seq_out = f"{out_dir}/{seq}"
    os.makedirs(f"{out_dir}/annotations", exist_ok=True)
    os.makedirs(f"{out_dir}/posed3d", exist_ok=True)
    ann = open(f"{out_dir}/annotations/{seq}.jsonl", "w")
    p3d = {k: [] for k in ("uid", "pose_world", "betas", "trans_world",
                           "cam_int", "cam_ext", "gender")}

    # index the movie tar; extract only the mp4 members we need, one at a time
    # (avoids extractall directory-permission issues on a shared/flaky scratch)
    import tempfile
    scratch = tempfile.mkdtemp(prefix=f"{seq}_", dir="/tmp")
    tar = tarfile.open(movie_tar)
    members = {os.path.basename(m.name): m for m in tar.getmembers()
               if m.name.endswith(".mp4")}

    n_inst = 0
    subs = sorted(groups)
    if limit_subs:
        subs = subs[:limit_subs]
    for sub in subs:
        m = members.get(f"{sub}.mp4")
        if m is None:
            continue
        mp4 = f"{scratch}/{sub}.mp4"
        try:
            with tar.extractfile(m) as src, open(mp4, "wb") as dst:
                dst.write(src.read())
        except Exception as e:
            print(f"    {seq}/{sub}: extract failed {e!r}", flush=True)
            continue
        frames = sorted(groups[sub])
        if limit_frames:
            frames = frames[:limit_frames]
        try:
            frame_iter = list(iter_mp4_frames(mp4, frames))
        except Exception as e:
            print(f"    {seq}/{sub}: decode failed {e!r}", flush=True)
            os.remove(mp4); continue
        for fn, rgb in frame_iter:
            rows = groups[sub][fn]
            H, W = rgb.shape[:2]
            K = d["cam_int"][rows[0]]            # K shared per frame
            instances = [dict(pose=d["pose_world"][r], betas=d["shape"][r],
                              trans=d["trans_world"][r], ext=d["cam_ext"][r],
                              gender=str(d["gender"][r])) for r in rows]
            face_idx, bary, inst_id, meta, _, _ = rasterize_frame(
                instances, K, H, W, device)
            crops = crops_from_frame(face_idx, bary, inst_id, len(instances))
            fdir = f"{seq_out}/{sub}"
            for i, c in enumerate(crops):
                if c is None:
                    continue
                os.makedirs(fdir, exist_ok=True)
                uid = f"{sub}_{fn:04d}_{i}"
                rgba = pack_rgba(c["face"], c["bary2"])
                Image.fromarray(rgba, "RGBA").save(f"{fdir}/{fn:04d}_{i}.png")
                ann.write(json.dumps({
                    "uid": uid, "seq": seq, "sub": sub, "frame": fn,
                    "instance": i, "bbox": c["bbox"], "area": c["area"],
                    "gender": instances[i]["gender"],
                    "png": f"{seq}/{sub}/{fn:04d}_{i}.png"}) + "\n")
                r = rows[i]
                p3d["uid"].append(uid)
                p3d["pose_world"].append(d["pose_world"][r])
                p3d["betas"].append(d["shape"][r])
                p3d["trans_world"].append(d["trans_world"][r])
                p3d["cam_int"].append(K); p3d["cam_ext"].append(d["cam_ext"][r])
                p3d["gender"].append(instances[i]["gender"])
                n_inst += 1
        os.remove(mp4)                       # free disk after each sub-seq
    tar.close()
    ann.close()
    np.savez_compressed(
        f"{out_dir}/posed3d/{seq}.npz",
        uid=np.array(p3d["uid"]),
        pose_world=np.array(p3d["pose_world"], np.float32),
        betas=np.array(p3d["betas"], np.float32),
        trans_world=np.array(p3d["trans_world"], np.float32),
        cam_int=np.array(p3d["cam_int"], np.float32),
        cam_ext=np.array(p3d["cam_ext"], np.float32),
        gender=np.array(p3d["gender"]))
    # cleanup scratch
    import shutil
    shutil.rmtree(scratch, ignore_errors=True)
    return n_inst

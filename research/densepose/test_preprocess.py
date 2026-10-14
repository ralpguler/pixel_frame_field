"""Geometry check for the (no-render) preprocess_shards: read movie frame + stored
corr PNG, warp into the expanded canvas, overlay official part LUT + UV on the RGB.
Writes dp_assets/preproc_check.png
"""
import json
import os
import sys
import tarfile
import tempfile

import numpy as np
from PIL import Image

sys.path.insert(0, "/home/u6ga/raguler.u6ga/codes/wan/bedlam")
sys.path.insert(0, "/home/u6ga/raguler.u6ga/codes/wan/densepose")
import cv2
from encode_correspondence import iter_mp4_frames
from preprocess_shards import CORR, MOVIES, expand_box, make_sample
from data import unpack_rgba, _PAL

ASSETS = os.path.expanduser("~/data/bedlam2/dp_assets")
SEQ = "20240806_1_250_ai1101_vcam"
N = 6


def main():
    face2part = np.load(f"{ASSETS}/smplx_face_to_dp24.npy")
    face_uv = np.load(f"{ASSETS}/smplx_face_uv.npy")
    recs = [json.loads(l) for l in open(f"{CORR}/annotations/{SEQ}.jsonl")]
    recs = [r for r in recs if max(r["bbox"][2] - r["bbox"][0],
                                   r["bbox"][3] - r["bbox"][1]) >= 28][:N]
    tar = tarfile.open(f"{MOVIES}/{SEQ}_mp4.tar")
    members = {os.path.basename(m.name): m for m in tar.getmembers()
               if m.name.endswith(".mp4")}
    tiles = []; sizes = []
    for r in recs:
        m = members[f"{r['sub']}.mp4"]
        tmp = tempfile.NamedTemporaryFile(suffix=".mp4", delete=False)
        tmp.write(tar.extractfile(m).read()); tmp.close()
        frame = dict(iter_mp4_frames(tmp.name, [r["frame"]]))[r["frame"]]
        os.unlink(tmp.name)
        corr = np.array(Image.open(f"{CORR}/{r['png']}"))
        rgb, rgba = make_sample(frame, corr, r["bbox"], *expand_box(*r["bbox"]))
        h, w = rgb.shape[:2]
        face, bary = unpack_rgba(rgba)
        fg = face >= 0
        part = np.zeros((h, w), np.int32); part[fg] = face2part[face[fg]]
        over = rgb.copy(); over[fg] = (0.4 * over[fg] + 0.6 * _PAL[part][fg]).astype(np.uint8)
        uv = np.zeros((h, w, 2), np.float32)
        uv[fg] = (bary[fg][:, :, None] * face_uv[face[fg]]).sum(1)
        uvimg = np.zeros((h, w, 3), np.uint8)
        uvimg[..., 0] = (uv[..., 0] * 255).astype(np.uint8)
        uvimg[..., 1] = (uv[..., 1] * 255).astype(np.uint8)
        uvimg[..., 2] = np.where(fg, 128, 0).astype(np.uint8)
        col = np.concatenate([rgb, over, uvimg], 0)                    # native size
        sizes.append(h)
        tiles.append(cv2.resize(col, (256, 768), interpolation=cv2.INTER_AREA))
    tar.close()
    print("native crop long sides (px):", sizes)
    Image.fromarray(np.concatenate(tiles, 1)).save(f"{ASSETS}/preproc_check.png")
    print(f"wrote {ASSETS}/preproc_check.png ({len(tiles)} instances; rows rgb/part/uv)")


if __name__ == "__main__":
    main()

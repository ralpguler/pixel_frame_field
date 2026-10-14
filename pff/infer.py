"""Inference on images: person crop from a box, one forward pass, and the maps pasted back into the image.

The crop is the one the model was evaluated with (research/densepose/dpeval/run_eval.py): a square of side
1.3 x max(w, h) centred on the person box, zero-padded where it leaves the image, resized to 512 x 512.

Per person and per pixel the result holds:
  part   0..24 (0 = background), the 24-way argmax where the body-mask logit is positive;
  U, V   the chart coordinates of the pixel's own part, in [0, 1] (v2 charts, assets/smplx_face_uv_v2.npy);
  R      (3,3) rotation from the body template to the camera (pff.template: PyTorch3D view space);
  body   body probability.
Pasting back: part and R nearest (R as the gathered 6D, re-orthonormalised after resampling), U, V and body
bilinear, as the evaluation does.
"""
import cv2
import numpy as np
import torch

from .geometry import gather_rot6d, part_pred
from .model import load_model
from .template import sixd_to_R_np

Z = 512
EXPAND = 1.3


def crop_box(rgb, box):
    """box = [x, y, w, h] (COCO convention). Returns the 512 x 512 crop and its window (ix0, iy0, S): the
    crop covers image pixels [ix0, ix0 + S) x [iy0, iy0 + S)."""
    H, W = rgb.shape[:2]
    bx, by, bw, bh = box
    cx, cy = bx + bw / 2, by + bh / 2
    side = EXPAND * max(bw, bh)
    sx0, sy0 = cx - side / 2, cy - side / 2
    S = int(round(side))
    canv = np.zeros((S, S, 3), np.uint8)
    ix0, iy0 = int(round(sx0)), int(round(sy0))
    ax0, ay0 = max(0, -ix0), max(0, -iy0)
    bx0, by0 = max(0, ix0), max(0, iy0)
    bx1, by1 = min(W, ix0 + S), min(H, iy0 + S)
    if bx1 > bx0 and by1 > by0:
        canv[ay0:ay0 + (by1 - by0), ax0:ax0 + (bx1 - bx0)] = rgb[by0:by1, bx0:bx1]
    return cv2.resize(canv, (Z, Z)), (ix0, iy0, S)


class Predictor:
    def __init__(self, checkpoint, device=None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = load_model(checkpoint, self.device)

    @torch.no_grad()
    def predict_crop(self, crop):
        """crop (512,512,3) uint8 RGB -> dict of (512,512) maps: part (int), U, V, body (float32), six (512,512,6),
        R (512,512,3,3)."""
        img = torch.from_numpy(np.ascontiguousarray(crop)).permute(2, 0, 1)[None].float().to(self.device) / 255
        if self.device.startswith("cuda"):
            with torch.autocast("cuda", dtype=torch.bfloat16):
                o = self.model(img)
        else:
            o = self.model(img)
        part = part_pred(o)[0]
        uv = o["uv"].float()[0].view(24, 2, Z, Z)
        sel = (part - 1).clamp(0)
        U = uv[:, 0].gather(0, sel[None])[0].clamp(0, 1)
        V = uv[:, 1].gather(0, sel[None])[0].clamp(0, 1)
        out = dict(part=part.cpu().numpy().astype(np.int64), U=U.cpu().numpy(), V=V.cpu().numpy(),
                   body=torch.sigmoid(o["mask"].float()[0, 0]).cpu().numpy())
        if "rot6d" in o:
            six = gather_rot6d(o["rot6d"].float(), part[None])[0].permute(1, 2, 0).cpu().numpy()
            out["six"] = six
            out["R"] = sixd_to_R_np(six.astype(np.float64)).astype(np.float32)
        return out

    def paste(self, pred, window, shape):
        """Resample one crop's maps to its window and clip to the image. Returns (y0, y1, x0, x1) of the image
        region and the maps over that region."""
        ix0, iy0, S = window
        H, W = shape[:2]
        x0, y0, x1, y1 = max(0, ix0), max(0, iy0), min(W, ix0 + S), min(H, iy0 + S)
        sx, sy = x0 - ix0, y0 - iy0
        cut = lambda a: a[sy:sy + (y1 - y0), sx:sx + (x1 - x0)]               # noqa: E731
        m = dict(part=cut(cv2.resize(pred["part"].astype(np.uint8), (S, S), interpolation=cv2.INTER_NEAREST)),
                 U=cut(cv2.resize(pred["U"], (S, S))), V=cut(cv2.resize(pred["V"], (S, S))),
                 body=cut(cv2.resize(pred["body"], (S, S))))
        if "six" in pred:
            six = np.stack([cv2.resize(pred["six"][..., k], (S, S), interpolation=cv2.INTER_NEAREST)
                            for k in range(6)], -1)
            m["R"] = sixd_to_R_np(cut(six).astype(np.float64)).astype(np.float32)
        return (y0, y1, x0, x1), m

    def predict_image(self, rgb, boxes):
        """rgb (H,W,3) uint8; boxes = [[x, y, w, h], ...] or [[x, y, w, h, score], ...].
        Returns {"persons": [per-person dicts with box, score, window, crop and its 512 maps],
                 "part", "U", "V", "body", "R", "owner"}: full-image maps where each person pixel shows the
        highest-scoring person covering it (owner = that person's index, -1 elsewhere)."""
        H, W = rgb.shape[:2]
        full = dict(part=np.zeros((H, W), np.int64), U=np.zeros((H, W), np.float32),
                    V=np.zeros((H, W), np.float32), body=np.zeros((H, W), np.float32),
                    R=np.tile(np.eye(3, dtype=np.float32), (H, W, 1, 1)), owner=np.full((H, W), -1, np.int64))
        persons = []
        for i, b in enumerate(boxes):
            crop, window = crop_box(rgb, b[:4])
            persons.append(dict(box=list(map(float, b[:4])), score=float(b[4]) if len(b) > 4 else 1.0,
                                window=window, crop=crop, pred=self.predict_crop(crop)))
        for i in sorted(range(len(persons)), key=lambda j: persons[j]["score"]):     # highest score last = wins
            (y0, y1, x0, x1), m = self.paste(persons[i]["pred"], persons[i]["window"], rgb.shape)
            fg = m["part"] > 0
            for k in ("part", "U", "V", "body") + (("R",) if "R" in m else ()):
                full[k][y0:y1, x0:x1][fg] = m[k][fg]
            full["owner"][y0:y1, x0:x1][fg] = i
        full["persons"] = persons
        return full

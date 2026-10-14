"""Crop padding written by field_dump (edge replication when the person crop leaves the video frame)."""
import numpy as np


def pad_white(im, thr=1.5, min_run=3):
    """field_dump pads crops that leave the video frame by EDGE REPLICATION (streaks) -> paint those borders white.
    A replicated strip is CONSTANT along the padding axis: every line in it matches the outermost line (no drift); smooth
    natural backgrounds drift past thr within a few lines."""
    im = im.copy(); f = im.astype(np.float32)
    def run(lines):                                            # lines ordered from the border inwards
        d_ = np.abs(lines - lines[0]).mean(tuple(range(1, lines.ndim)))
        n = int(np.argmax(d_ >= thr)) if (d_ >= thr).any() else len(d_)
        return n if n >= min_run else 0
    t, b = run(f), run(f[::-1]); l, r = run(np.transpose(f, (1, 0, 2))), run(np.transpose(f, (1, 0, 2))[::-1])
    if b: im[-b:] = 255
    if t: im[:t] = 255
    if r: im[:, -r:] = 255
    if l: im[:, :l] = 255
    return im, (t, b, l, r)


def valid_region(im):
    """bool (H, W): True where the crop holds real video pixels."""
    t, b, l, r = pad_white(im)[1]; H, W = im.shape[:2]; v = np.zeros((H, W), bool); v[t:H - b, l:W - r] = True
    return v

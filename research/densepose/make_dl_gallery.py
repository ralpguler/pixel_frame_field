"""Browsable HTML gallery of EXACT native-res DensePose GT across many sequences.
Each card: [input | part GT | UV GT] — official DensePose-24 + chart-consistent UV,
computed from the saved face+bary with NO label resampling.

Usage: python make_dl_gallery.py [--seqs 12] [--per-seq 6]
Then:  rsync -av u6ga.aip2.isambard:data/bedlam2/dp_assets/dl_gallery/ ./dl_gallery/
"""
import argparse
import glob
import json
import os

import numpy as np
from PIL import Image

from data import _PAL, ROOT
from viz_dataloader import gather, panel_exact

ASSETS = f"{ROOT}/dp_assets"
OUT = f"{ASSETS}/dl_gallery"
CORR = f"{ROOT}/correspondence"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seqs", type=int, default=12)
    ap.add_argument("--per-seq", type=int, default=6)
    args = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)
    names = json.load(open(f"{ASSETS}/dp24_names.json"))

    all_seqs = sorted(os.path.basename(p)[:-6]
                      for p in glob.glob(f"{CORR}/annotations/*.jsonl"))
    seqs = all_seqs[:: max(1, len(all_seqs) // args.seqs)][:args.seqs]

    cards = []
    for seq in seqs:
        try:
            samples = gather(seq, n=args.per_seq)
        except Exception as e:
            print(f"  {seq}: {e!r}"); continue
        files = []
        for i, (rgb, rgba) in enumerate(samples):
            fn = f"{seq}_{i}.png"
            Image.fromarray(panel_exact(rgb, rgba, h=360)).save(f"{OUT}/{fn}")
            files.append(fn)
        if files:
            cards.append((seq, files)); print(f"{seq}: {len(files)}", flush=True)

    legend = "".join(
        f"<span style='background:rgb({_PAL[i,0]},{_PAL[i,1]},{_PAL[i,2]});color:#fff;"
        f"padding:2px 6px;margin:1px;display:inline-block;border-radius:3px'>"
        f"{i} {names[str(i)]}</span>" for i in range(1, 25))
    html = ["<html><head><meta charset='utf-8'><title>DensePose GT gallery</title>",
            "<style>body{font:13px system-ui,monospace;background:#111;color:#ddd;margin:16px}"
            "img{max-width:100%;border:1px solid #333;margin:4px 0;background:#000}"
            "h2{background:#1c2330;padding:6px;border-radius:4px}.cap{color:#9ab}</style></head><body>",
            "<h1>BEDLAM2 → DensePose-24 ground truth</h1>",
            "<p class='cap'>Each row: <b>input crop</b> | <b>part GT</b> (official detectron2 "
            "DensePose-24, transferred to SMPL-X) | <b>UV GT</b> (R=U, G=V, chart-consistent). "
            "EXACT native-resolution GT from the saved face+bary — no label resampling.</p>",
            f"<div style='line-height:2'>{legend}</div>"]
    for seq, files in cards:
        html.append(f"<h2>{seq}</h2>")
        for f in files:
            html.append(f"<img src='{f}'>")
    html.append("</body></html>")
    open(f"{OUT}/index.html", "w").write("\n".join(html))
    n = sum(len(f) for _, f in cards)
    sz = sum(os.path.getsize(f"{OUT}/{x}") for x in os.listdir(OUT)) / 1e6
    print(f"\ngallery: {OUT}/index.html ({n} samples from {len(cards)} seqs, {sz:.0f} MB)")


if __name__ == "__main__":
    main()

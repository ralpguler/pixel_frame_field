"""Analysis plots for the paper (matplotlib, CVPR-appropriate: no chartjunk, serif-free, tight)."""
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT = os.path.expanduser("~/codes/wan/paper/figs")
os.makedirs(OUT, exist_ok=True)
plt.rcParams.update({
    "font.size": 8, "axes.labelsize": 8.5, "axes.titlesize": 9,
    "xtick.labelsize": 7.5, "ytick.labelsize": 7.5, "legend.fontsize": 7.5,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.linewidth": 0.7, "xtick.major.width": 0.7, "ytick.major.width": 0.7,
    "figure.dpi": 300, "savefig.dpi": 300, "savefig.bbox": "tight",
    "savefig.pad_inches": 0.02, "font.family": "DejaVu Sans",
})
OURS, BASE, ACC = "#6a3fc0", "#8a8f9c", "#c0392b"

# ---------------------------------------------------------------- frontier
# COCO AP vs held-out occlusion error for the data/recipe arms.
pts = [
    ("W-B  (BEDLAM only)",      0.613, 14.80, BASE, (10, 6),  "left"),
    ("W-M  (+MAMMA)",           0.621,  6.89, OURS, (10, 4),  "left"),
    ("W-J  (COCO in-training)", 0.699, 10.77, BASE, (0, -16), "center"),
    ("W-MF@1k",                 0.713,  7.75, OURS, (-6, 7),  "right"),
    ("W-MF@3k  (final)",        0.741,  8.24, OURS, (-2, -17),"right"),
]
fig, ax = plt.subplots(figsize=(3.35, 2.4))
for lbl, ap, occ, c, off, ha in pts:
    star = "final" in lbl
    ax.scatter(ap, occ, s=95 if star else 34, c=c, marker="*" if star else "o",
               zorder=3, edgecolors="white", linewidths=0.6)
    ax.annotate(lbl, (ap, occ), textcoords="offset points", xytext=off, ha=ha,
                fontsize=6.8, color="#333")
ax.annotate("", xy=(0.7365, 8.15), xytext=(0.6255, 6.95),
            arrowprops=dict(arrowstyle="->", color=OURS, lw=1.0, alpha=.5,
                            connectionstyle="arc3,rad=-0.22"), zorder=2)
ax.text(0.679, 5.9, "joint fine-tune", fontsize=6.6, color=OURS, ha="center", style="italic")
ax.set_xlabel("DensePose-COCO GPS AP  $\\uparrow$")
ax.set_ylabel("held-out occlusion err. (deg) $\\downarrow$")
ax.invert_yaxis()
ax.set_xlim(0.598, 0.775)
ax.set_ylim(16.2, 5.2)
ax.grid(alpha=.16, lw=.5)
fig.savefig(f"{OUT}/fig_frontier.pdf")
fig.savefig(f"{OUT}/fig_frontier.png")
print("wrote fig_frontier")

# ------------------------------------------------- downstream effect size
fig, ax = plt.subplots(figsize=(3.35, 1.9))
labels = ["MCLD$\\rightarrow$FPDM\n(published step)", "detectron2$\\rightarrow$ours\n(this work)"]
vals = [0.0037, 0.0116]
err = [0, 0.0116 - 0.0110]
bars = ax.barh(labels, vals, height=.5, color=[BASE, OURS],
               xerr=[[0, 0.0116 - 0.0110], [0, 0.0123 - 0.0116]],
               error_kw=dict(lw=.8, capsize=2.5, ecolor="#444"))
for y, v in enumerate(vals):
    ax.text(v + 0.0006, y, f"{v:+.4f}".replace("+", "−" if False else "−"),
            va="center", fontsize=7.2)
ax.set_xlabel("LPIPS improvement on DeepFashion @256  $\\rightarrow$")
ax.set_xlim(0, 0.0155)
ax.grid(axis="x", alpha=.16, lw=.5)
ax.tick_params(axis="y", length=0)
fig.savefig(f"{OUT}/fig_effect.pdf")
fig.savefig(f"{OUT}/fig_effect.png")
print("wrote fig_effect")

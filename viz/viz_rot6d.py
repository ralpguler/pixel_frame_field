"""Visualize the per-pixel 6D rotation GT, channel-by-channel, aligned with the RGB crop.

For sample instances from dp_shards_rot, reconstruct per-pixel 6D (barycentric interp of the triangle's
3 vertex vert_rot6d) and R (Gram-Schmidt). Render, fg-masked & aligned with RGB:
  RGB | a0x a0y a0z | a1x a1y a1z   (6 channels, diverging cmap [-1,1])
      | a0->RGB | a1->RGB | R*N_template (normal map)
where a0=R[:,0], a1=R[:,1] are the two camera-space basis columns the 6D stores. -> rot6d_viz/index.html
"""
import os  # noqa: E402
import sys  # noqa: E402
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import _env  # noqa: E402  (repository paths: viz/_env.py)
_env.take_out_arg()
import os, io, glob, json, html, numpy as np, cv2
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation
import webdataset as wds
import sys; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)) + "/..")
from data import unpack_rgba
from orient.rot6d import get_template, sixd_to_R_np

ROOT = _env.data("bedlam2")
OUT = _env.out("rot6d_gt_viz")
os.makedirs(OUT + "/imgs", exist_ok=True)
N = 10
_, faces, n_rest = get_template("cpu")
CH = ["a0x", "a0y", "a0z", "a1x", "a1y", "a1z"]


def recon(face, bary, vert_rot6d):
    mask = face >= 0
    tv = faces[np.where(mask, face, 0)]                       # (H,W,3) vert ids
    six = (bary[..., None] * vert_rot6d[tv]).sum(2)           # (H,W,6) GT 6D
    R = sixd_to_R_np(six.astype(np.float32))                  # (H,W,3,3)
    nt = (bary[..., None] * n_rest[tv]).sum(2)
    nt = nt / np.clip(np.linalg.norm(nt, axis=-1, keepdims=True), 1e-9, None)
    ncam = np.einsum("hwij,hwj->hwi", R, nt)                  # R * N_template
    twist = twist_about_normal(R, nt, mask)                   # (H,W) deg, signed, NaN on bg
    return mask, six, ncam, twist


def twist_about_normal(R, nt, mask):
    """Swing-twist split (gauge-free, doc §2): signed twist angle (deg) of R about the TEMPLATE normal.
    quaternion q=(w,xyz); twist about axis n -> tau = 2*atan2(xyz·n, w), wrapped to (-180,180]."""
    out = np.full(R.shape[:2], np.nan, np.float32)
    if not mask.any():
        return out
    q = Rotation.from_matrix(R[mask].reshape(-1, 3, 3)).as_quat()   # (n,4) xyzw
    n = nt[mask]
    proj = (q[:, :3] * n).sum(1)
    tau = 2.0 * np.arctan2(proj, q[:, 3])                            # radians, (-2pi,2pi]
    tau = (tau + np.pi) % (2 * np.pi) - np.pi                        # wrap to (-pi,pi]
    out[mask] = np.degrees(tau)
    return out


def render(rgb, mask, six, ncam, twist, path):
    fig, ax = plt.subplots(1, 11, figsize=(33, 3.4))
    ax[0].imshow(rgb); ax[0].set_title("RGB", fontsize=9)
    for c in range(6):
        ch = np.where(mask, six[..., c], np.nan)
        im = ax[c + 1].imshow(ch, cmap="coolwarm", vmin=-1, vmax=1)
        ax[c + 1].set_title(CH[c], fontsize=9)
    a0 = np.clip(six[..., 0:3] * 0.5 + 0.5, 0, 1); a0[~mask] = 1
    a1 = np.clip(six[..., 3:6] * 0.5 + 0.5, 0, 1); a1[~mask] = 1
    nm = np.clip(ncam * 0.5 + 0.5, 0, 1); nm[~mask] = 1
    ax[7].imshow(a0); ax[7].set_title("a0->RGB", fontsize=9)
    ax[8].imshow(a1); ax[8].set_title("a1->RGB", fontsize=9)
    ax[9].imshow(nm); ax[9].set_title("R·N (normal)", fontsize=9)
    tw = ax[10].imshow(twist, cmap="twilight", vmin=-180, vmax=180)    # signed twist about normal (deg)
    ax[10].set_title("twist about N (deg)", fontsize=9)
    for a in ax: a.axis("off")
    fig.colorbar(im, ax=ax[1:10].tolist(), fraction=0.010, pad=0.01)   # [-1,1] for 6D channels
    fig.colorbar(tw, ax=[ax[10]], fraction=0.046, pad=0.02)            # cyclic [-180,180] for twist
    fig.savefig(path, dpi=70, bbox_inches="tight"); plt.close(fig)


shards = sorted(glob.glob(f"{ROOT}/dp_shards_rot/val/*.tar"))
ds = wds.WebDataset(shards, shardshuffle=False, empty_check=False, handler=wds.warn_and_continue)
rows = []
for s in ds:
    if len(rows) >= N:
        break
    try:
        rgb = cv2.imdecode(np.frombuffer(s["rgb.jpg"], np.uint8), cv2.IMREAD_COLOR)[..., ::-1]
        face, bary = unpack_rgba(np.array(__import__("PIL.Image", fromlist=["Image"]).open(io.BytesIO(s["corr.png"]))))
        vr = np.load(io.BytesIO(s["rot.npy"])).astype(np.float32)
        if (face >= 0).mean() < 0.02:
            continue
        mask, six, ncam, twist = recon(face, bary, vr)
        fn = f"imgs/{len(rows):02d}.png"
        render(rgb, mask, six, ncam, twist, OUT + "/" + fn)
        rows.append((fn, json.loads(s["meta.json"]).get("uid", s["__key__"])))
        print("rendered", fn, flush=True)
    except Exception as e:
        print("skip", s.get("__key__", "?"), repr(e), flush=True)

H = ['<html><head><meta charset=utf-8><style>body{font-family:sans-serif;background:#fff;color:#111}'
     'img{width:100%;max-width:1900px}.it{border-bottom:1px solid #ccc;padding:6px 0}</style></head><body>']
H.append("<h2>BEDLAM2 oriented-DensePose — per-pixel 6D rotation GT, by channel</h2>")
H.append("<p>Each row: <b>RGB</b> | 6 channels of the 6D = the two camera-space basis columns "
         "<b>a0=R[:,0]</b> (a0x,a0y,a0z), <b>a1=R[:,1]</b> (a1x,a1y,a1z), diverging colormap in "
         "[-1,1] | <b>a0,a1 as RGB</b> | <b>R·N_template</b> (surface normal map) | <b>twist about N</b> "
         "(signed spin about the template normal, deg, cyclic colormap) — the gauge-free 3rd DoF "
         "(swing-twist split, doc §2). Smooth, seam-free variation = correct orientation GT.</p>")
for fn, uid in rows:
    H.append(f"<div class=it>{html.escape(str(uid))}<br><img src='{fn}'></div>")
H.append("</body></html>")
open(OUT + "/index.html", "w").write("\n".join(H))
print(f"wrote {OUT}/index.html with {len(rows)} instances", flush=True)

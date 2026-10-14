"""Framefield studio v2. Pose an SMPL-X body, compute the per-vertex SO(3) framefield (rest->posed),
build a proper SURFACE frame (primary axis = outward posed normal; the two tangent axes carry the TWIST
our rotation head predicts), sample EQUIDISTANT points via farthest-point sampling (the template vertex
density is very non-uniform), and emit:
  - a SHADED preview PNG (fake-lit arrow tripods, 3 views)
  - tripods.ply         : combined shaded tripod geometry with vertex colors (import straight into Blender)
  - points_frames.csv    : equidistant points + frame quaternion (wxyz) + outward normal, for GN instancing
  - blender_framefield.py: a ready-to-run Blender script (instances a shaded arrow on each point, lit)

Usage: framefield_studio.py --n 700 [--pose-idx -1]
"""
import os  # noqa: E402
import sys  # noqa: E402
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import _env  # noqa: E402  (repository paths: viz/_env.py)
import os, sys, argparse, numpy as np, torch
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from orient.rot6d import get_template, vertex_normals, rotation_from_deformation
from render_correspondence import get_smplx

ap = argparse.ArgumentParser()
ap.add_argument("--n", type=int, default=700, help="number of equidistant tripods")
ap.add_argument("--pose-idx", type=int, default=-1)
ap.add_argument("--twist-color", action="store_true", help="color the normal arrow by twist instead of blue")
ap.add_argument("--arrow", type=float, default=0.07, help="arrow length scale (smaller = shorter)")
ap.add_argument("--out", default=_env.out("framefield_studio"))
a = ap.parse_args()
dev = "cuda"; os.makedirs(a.out, exist_ok=True); torch.set_grad_enabled(False)

# ---------- pose a body + framefield ----------
v_template, faces, n_rest = get_template(dev)
gt = np.load(_env.camerahmr("cameramhr_3dpw_preds.npz"), allow_pickle=True)["gt_pose"]
idx = int(np.argsort(np.linalg.norm(gt[:, 3:66], axis=1))[int(0.85*len(gt))]) if a.pose_idx < 0 else a.pose_idx
pose = gt[idx].astype(np.float32)
m = get_smplx("neutral", 16, dev)
bp = torch.zeros(1, 63, device=dev); bp[0, :min(63, pose.shape[0]-3)] = torch.tensor(pose[3:66], device=dev)
V = m(global_orient=torch.zeros(1, 3, device=dev), body_pose=bp,
      betas=torch.zeros(1, 16, device=dev)).vertices[0].cpu().numpy().astype(np.float64)
n_pose = vertex_normals(V, faces)
R = rotation_from_deformation(v_template, V, faces, n_rest, n_pose)   # (Nv,3,3) rest->posed

# ---------- proper SURFACE frame: primary = outward normal; tangents carry the twist ----------
# template tangent frame (smooth gauge): project world-up onto tangent plane (fallback world-x near poles)
ref = np.tile([0., 1., 0.], (len(n_rest), 1))
deg = np.abs((ref * n_rest).sum(1)) > 0.95
ref[deg] = [1., 0., 0.]
t_rest = ref - (ref * n_rest).sum(1, keepdims=True) * n_rest
t_rest /= np.clip(np.linalg.norm(t_rest, axis=1, keepdims=True), 1e-9, None)
b_rest = np.cross(n_rest, t_rest)
Tf = np.stack([t_rest, b_rest, n_rest], axis=2)                       # (Nv,3,3) columns = template surface frame
D = R @ Tf                                                            # display frame: col2 = R@n_rest = posed normal (OUT)
# Z-up for viewing/Blender (SMPL is Y-up): 90deg about X, applied to points AND frames
Pm = np.array([[1., 0., 0.], [0., 0., -1.], [0., 1., 0.]])
V = V @ Pm.T; D = Pm[None] @ D
c = V.mean(0); V = V - c; sc = 1.0/np.abs(V-V.mean(0)).max(); V = V*sc

# ---------- equidistant farthest-point sampling ----------
def fps(P, k, seed=0):
    n = len(P); sel = np.empty(k, np.int64); sel[0] = seed
    d = np.sum((P - P[seed])**2, 1)
    for i in range(1, k):
        j = int(np.argmax(d)); sel[i] = j
        d = np.minimum(d, np.sum((P - P[j])**2, 1))
    return sel
si = fps(V, min(a.n, len(V)))
Ps, Ds = V[si], D[si]
twist = (np.arctan2(R[si, 1, 0], R[si, 0, 0]) + np.pi) / (2*np.pi)    # gauge twist in [0,1]
print(f"[studio] pose {idx}, {len(si)} equidistant tripods, primary=outward normal", flush=True)

# ---------- shaded arrow-tripod geometry ----------
def arrow(length, rs=0.06, rc=0.14, hc=0.34, nseg=10):
    """unit arrow along +Z: cylinder shaft + cone head. returns (verts, tris)."""
    ang = np.linspace(0, 2*np.pi, nseg, endpoint=False)
    cs, sn = np.cos(ang), np.sin(ang); hz = length*(1-hc)
    ring0 = np.stack([rs*cs, rs*sn, np.zeros(nseg)], 1)
    ring1 = np.stack([rs*cs, rs*sn, np.full(nseg, hz)], 1)
    ring2 = np.stack([rc*cs, rc*sn, np.full(nseg, hz)], 1)
    tip = np.array([[0, 0, length]]); base = np.array([[0, 0, 0.]])
    v = np.concatenate([ring0, ring1, ring2, tip, base]); o1, o2, o3, T, B = 0, nseg, 2*nseg, 3*nseg, 3*nseg+1
    tris = []
    for i in range(nseg):
        j = (i+1) % nseg
        tris += [[o1+i, o1+j, o2+j], [o1+i, o2+j, o2+i]]     # shaft side
        tris += [[o3+i, o3+j, T]]                             # cone
        tris += [[B, o1+j, o1+i]]                             # base cap
    return v, np.array(tris)

# canonical tripod: 3 arrows (x=red short, y=green short, z=blue LONG = the outward normal)
AV, AF = arrow(1.0)
axes = [(0, [0.95,0.25,0.25], 0.55), (1, [0.3,0.9,0.35], 0.55), (2, [0.35,0.55,1.0], 1.0)]
def rot_z_to(axis):                                         # rotation mapping +Z to local axis e_k
    return {0: np.array([[0,0,1],[0,1,0],[-1,0,0.]]), 1: np.array([[1,0,0],[0,0,-1],[0,1,0.]]),
            2: np.eye(3)}[axis]
tri_v, tri_f, tri_c = [], [], []; off = 0
for ax, col, ln in axes:
    v = (AV * [1,1,ln]) @ rot_z_to(ax).T
    tri_v.append(v); tri_f.append(AF + off); tri_c.append(np.tile(col, (len(v), 1))); off += len(v)
TRI_V = np.concatenate(tri_v); TRI_F = np.concatenate(tri_f); TRI_C = np.concatenate(tri_c)  # canonical tripod

# instance the tripod at every equidistant point, oriented by its display frame D
S = a.arrow
allV, allF, allC = [], [], []; voff = 0
for i in range(len(si)):
    vw = (Ds[i] @ (TRI_V * S).T).T + Ps[i]
    col = TRI_C.copy()
    if a.twist_color:                                       # tint the normal(blue) arrow by twist
        znv = len(TRI_V) - len(arrow(1.0)[0])              # last arrow = z
        col[znv:] = plt.cm.twilight(twist[i])[:3]
    allV.append(vw); allF.append(TRI_F + voff); allC.append(col); voff += len(vw)
GV = np.concatenate(allV); GF = np.concatenate(allF); GC = np.concatenate(allC)
print(f"[studio] geometry {GV.shape[0]} verts / {GF.shape[0]} tris", flush=True)

# ---------- exports ----------
def R_to_quat(Rm):
    q = np.zeros((len(Rm), 4)); t = np.trace(Rm, 1, 2)
    for k in range(len(Rm)):
        M, tr = Rm[k], t[k]
        if tr > 0:
            s = np.sqrt(tr+1)*2; q[k] = [.25*s, (M[2,1]-M[1,2])/s, (M[0,2]-M[2,0])/s, (M[1,0]-M[0,1])/s]
        else:
            i = int(np.argmax([M[0,0], M[1,1], M[2,2]]))
            if i == 0: s = np.sqrt(1+M[0,0]-M[1,1]-M[2,2])*2; q[k] = [(M[2,1]-M[1,2])/s,.25*s,(M[0,1]+M[1,0])/s,(M[0,2]+M[2,0])/s]
            elif i == 1: s = np.sqrt(1+M[1,1]-M[0,0]-M[2,2])*2; q[k] = [(M[0,2]-M[2,0])/s,(M[0,1]+M[1,0])/s,.25*s,(M[1,2]+M[2,1])/s]
            else: s = np.sqrt(1+M[2,2]-M[0,0]-M[1,1])*2; q[k] = [(M[1,0]-M[0,1])/s,(M[0,2]+M[2,0])/s,(M[1,2]+M[2,1])/s,.25*s]
    return q/np.clip(np.linalg.norm(q, axis=1, keepdims=True), 1e-9, None)

with open(f"{a.out}/tripods.ply", "w") as f:                # shaded geometry w/ vertex colors
    f.write("ply\nformat ascii 1.0\n")
    f.write(f"element vertex {len(GV)}\nproperty float x\nproperty float y\nproperty float z\n")
    f.write("property uchar red\nproperty uchar green\nproperty uchar blue\n")
    f.write(f"element face {len(GF)}\nproperty list uchar int vertex_indices\nend_header\n")
    C8 = np.clip(GC*255, 0, 255).astype(int)
    for v, cc in zip(GV, C8): f.write(f"{v[0]:.5f} {v[1]:.5f} {v[2]:.5f} {cc[0]} {cc[1]} {cc[2]}\n")
    for t in GF: f.write(f"3 {t[0]} {t[1]} {t[2]}\n")
np.savetxt(f"{a.out}/points_frames.csv", np.concatenate([Ps, R_to_quat(Ds), Ds[:, :, 2], twist[:, None]], 1),
           delimiter=",", header="x,y,z,qw,qx,qy,qz,nx,ny,nz,twist", comments="")
with open(f"{a.out}/body.obj", "w") as f:                    # ghost body for context in Blender
    for v in V: f.write(f"v {v[0]:.5f} {v[1]:.5f} {v[2]:.5f}\n")
    for t in faces + 1: f.write(f"f {t[0]} {t[1]} {t[2]}\n")
print(f"[studio] wrote tripods.ply + points_frames.csv + body.obj", flush=True)

# ---------- shaded matplotlib preview ----------
def shade(V3, F3, C3, light=(0.4, 0.5, 0.8)):
    tri = V3[F3]; fn = np.cross(tri[:, 1]-tri[:, 0], tri[:, 2]-tri[:, 0])
    fn /= np.clip(np.linalg.norm(fn, axis=1, keepdims=True), 1e-9, None)
    L = np.array(light)/np.linalg.norm(light)
    lam = np.clip((fn*L).sum(1), 0, 1); shade = 0.35 + 0.65*lam                  # ambient + diffuse
    fc = C3[F3[:, 0]] * shade[:, None]
    z = tri[:, :, 1].mean(1)                                                     # depth sort (y is view-depth-ish)
    return tri, np.clip(fc, 0, 1), z
lo, hi = V.min(0), V.max(0); ext = hi-lo; ctr = (hi+lo)/2
gb = V[np.random.RandomState(1).choice(len(V), 2600, replace=False)]
fig = plt.figure(figsize=(15, 7.6), facecolor="#0b0b10")
for j, az in enumerate([-88, -35, 30]):
    ax = fig.add_subplot(1, 3, j+1, projection="3d"); ax.set_facecolor("#0b0b10")
    for pane in (ax.xaxis, ax.yaxis, ax.zaxis):
        pane.set_pane_color((0,0,0,0)); pane.line.set_color((0,0,0,0)); pane.set_ticks([])
    ax.grid(False)
    for sl, cc, ee in [(ax.set_xlim, ctr[0], ext[0]), (ax.set_ylim, ctr[1], ext[1]), (ax.set_zlim, ctr[2], ext[2])]:
        sl(cc-ee/2*1.06, cc+ee/2*1.06)
    try: ax.set_box_aspect(ext)
    except Exception: pass
    ax.scatter(gb[:,0], gb[:,1], gb[:,2], s=1.0, c="#20202a", alpha=.4, depthshade=False)
    tri, fc, z = shade(GV, GF, GC)
    pc = Poly3DCollection(tri[np.argsort(z)], facecolors=fc[np.argsort(z)], edgecolors="none")
    ax.add_collection3d(pc); ax.view_init(elev=6, azim=az)
fig.suptitle("framefield · shaded tripods (primary = outward normal, equidistant)", color="w", fontsize=15, y=.99)
fig.subplots_adjust(left=0, right=1, top=.93, bottom=0, wspace=0)
fig.savefig(f"{a.out}/tripods_shaded.png", dpi=95, facecolor="#0b0b10"); plt.close(fig)
print(f"wrote {a.out}/tripods_shaded.png\nDONE", flush=True)

"""Self-contained page: canonical TEMPLATE (left) beside the REPROJECTED image sequence (right).

Both panels always draw the SAME set of tracked points -- left with the canonical template surface frame,
right with the predicted per-frame SO(3) frame reprojected onto the real image.

Two point modes:
  FIXED  the first NFIX farthest-point samples -- tripods stay on the same body points and are tracked.
  SWEEP  a band travels over the body and lights up whatever DENSE points fall inside it (size+brightness
         fade at the band edges). Nothing is fixed: the sample locations are a continuum, which is the
         point -- the frame field is predicted densely, not at a chosen set of anchors.
Greedy FPS has a prefix property, so the first NFIX of the dense set ARE the fixed set: one export, both modes.

Two temporal filters (blinking comes from points dithering across the visibility threshold):
  min-run K : drop visibility runs shorter than K frames    (grazing-angle pop-ins)
  bridge  G : close dropouts of <= G frames by interpolating between the bracketing OBSERVED frames
              (linear u,v + nlerp quat). Never extrapolates, never invents past a gap edge.

Usage: framefield_pair_html.py [--dir SEQ_DIR] [--tmpl TMPL_DIR] [--nfix 340] [--out framefield.html]
"""
import os  # noqa: E402
import sys  # noqa: E402
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import _env  # noqa: E402  (repository paths: viz/_env.py)
import tempfile  # noqa: E402
import os, json, base64, glob, argparse, numpy as np, cv2, imageio.v2 as iio

H = os.path.expanduser("~")
ap = argparse.ArgumentParser()
ap.add_argument("--dir", default=_env.data("dp_viz_compare/ff_dense"))
ap.add_argument("--tmpl", default=_env.data("dp_viz_compare/ff_dense_tmpl"))
ap.add_argument("--out", default=_env.out("framefield.html"))
ap.add_argument("--nfix", type=int, default=340, help="FPS prefix used as the FIXED tracked point set")
ap.add_argument("--vid-px", type=int, default=512)
ap.add_argument("--fps", type=float, default=16.0)
ap.add_argument("--gmax", type=int, default=6, help="largest gap we ever precompute a bridge for")
ap.add_argument("--nflow", type=int, default=640, help="FLOW mode: number of particles drifting over the surface")
a = ap.parse_args()

frames = sorted(int(os.path.basename(f)[4:8]) for f in glob.glob(f"{a.dir}/pts_*.csv"))
assert frames, f"no pts_*.csv in {a.dir}"
F = len(frames)

# ---------- template: body mesh + canonical frame per point ----------
V, FA = [], []
for ln in open(f"{a.tmpl}/tmpl_body.obj"):
    if ln.startswith("v "): V.append([float(x) for x in ln.split()[1:4]])
    elif ln.startswith("f "): FA.append([int(t.split("/")[0]) - 1 for t in ln.split()[1:4]])
V = np.array(V, np.float32); FA = np.array(FA, np.int32)
assert V.shape[0] < 65536 and FA.max() < 65536
T = np.loadtxt(f"{a.tmpl}/tmpl_pts.csv", delimiter=",", skiprows=1, ndmin=2)   # x,y,z,qw,qx,qy,qz,twist
N = len(T)

ctr = (V.max(0) + V.min(0)) / 2                                # bbox center: mesh is only MEAN-centered,
rng = float((V.max(0) - V.min(0)).max())                       # so ctr != 0 (~.44m in z). Points take the
vq = np.clip(np.round((V - ctr) / rng * 32000), -32767, 32767).astype(np.int16)   # SAME shift or they float.
tpos = (T[:, :3] - ctr).astype(np.float32)
tq = T[:, [4, 5, 6, 3]].astype(np.float32)                     # wxyz -> xyzw


# ---------- FLOW mode: surface flow paths over the template ----------
# Particles follow the gradient of GEODESIC DISTANCE FROM THE CROWN, walking the mesh graph away from the
# head: down the torso and legs, and OUTWARD ALONG THE ARMS to the fingertips.
# (Gravity is the wrong field here: on the zero-pose template the arms are horizontal, so -Z projected onto
# the tangent plane runs around the arm's circumference, not along it -- arm particles stalled on the
# underside after ~8 vertices and got dropped. A body-intrinsic field is well defined on every limb.)
from scipy.spatial import cKDTree
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import dijkstra

adj = [set() for _ in range(len(V))]
E = set()
for t in FA:
    for i in range(3):
        u_, v_ = int(t[i]), int(t[(i + 1) % 3]); adj[u_].add(v_); adj[v_].add(u_)
        E.add((min(u_, v_), max(u_, v_)))
adj = [np.fromiter(s, int) for s in adj]
VN = np.zeros_like(V)
fn = np.cross(V[FA[:, 1]] - V[FA[:, 0]], V[FA[:, 2]] - V[FA[:, 0]])
for k in range(3): np.add.at(VN, FA[:, k], fn)
VN /= np.clip(np.linalg.norm(VN, axis=1, keepdims=True), 1e-9, None)
dtree = cKDTree(T[:, :3])                                      # dense sample positions
dvert = cKDTree(V).query(T[:, :3])[1]                          # dense sample -> mesh vertex
DN = VN[dvert]                                                 # dense sample normals

ei = np.array(sorted(E))
w = np.linalg.norm(V[ei[:, 0]] - V[ei[:, 1]], axis=1)
G = coo_matrix((np.r_[w, w], (np.r_[ei[:, 0], ei[:, 1]], np.r_[ei[:, 1], ei[:, 0]])), shape=(len(V),) * 2)
crown = int(np.argmax(V[:, 2]))                                # top of the head
geo = dijkstra(G.tocsr(), indices=crown)                       # geodesic distance from the crown
geo[~np.isfinite(geo)] = geo[np.isfinite(geo)].max()


# CONTINUOUS advection in the geodesic gradient. Greedy steepest-ascent on the mesh GRAPH collapses
# trajectories onto a tree -- at 17 cm of travel, 21% of particles had snapped onto a shared vertex, which
# is what produced the streaky sparsity. Sub-vertex stepping keeps neighbouring trajectories distinct.
p0, p1, p2 = V[FA[:, 0]], V[FA[:, 1]], V[FA[:, 2]]
fnv = np.cross(p1 - p0, p2 - p0)
A2 = np.linalg.norm(fnv, axis=1, keepdims=True)                # = 2 * area
un = fnv / np.clip(A2, 1e-12, None)
g0, g1, g2 = geo[FA[:, 0]], geo[FA[:, 1]], geo[FA[:, 2]]
fgrad = (g0[:, None] * np.cross(un, p2 - p1) + g1[:, None] * np.cross(un, p0 - p2)
         + g2[:, None] * np.cross(un, p1 - p0)) / np.clip(A2, 1e-12, None)
print(f"[flow] |grad geo| per face: median {np.median(np.linalg.norm(fgrad,axis=1)):.2f} (a distance field -> ~1)",
      flush=True)

def _pt_tri(P, A, B, C):
    """Exact closest point on a triangle (Ericson), vectorised."""
    ab, ac, ap = B - A, C - A, P - A
    d1 = np.einsum("ij,ij->i", ab, ap); d2 = np.einsum("ij,ij->i", ac, ap)
    bp = P - B; d3 = np.einsum("ij,ij->i", ab, bp); d4 = np.einsum("ij,ij->i", ac, bp)
    cp_ = P - C; d5 = np.einsum("ij,ij->i", ab, cp_); d6 = np.einsum("ij,ij->i", ac, cp_)
    va, vb, vc = d3 * d6 - d5 * d4, d5 * d2 - d1 * d6, d1 * d4 - d3 * d2
    out = np.empty_like(P)
    m1 = (d1 <= 0) & (d2 <= 0)
    m2 = ~m1 & (d3 >= 0) & (d4 <= d3)
    m3 = ~m1 & ~m2 & (vc <= 0) & (d1 >= 0) & (d3 <= 0)
    m4 = ~m1 & ~m2 & ~m3 & (d6 >= 0) & (d5 <= d6)
    m5 = ~m1 & ~m2 & ~m3 & ~m4 & (vb <= 0) & (d2 >= 0) & (d6 <= 0)
    m6 = ~m1 & ~m2 & ~m3 & ~m4 & ~m5 & (va <= 0) & ((d4 - d3) >= 0) & ((d5 - d6) >= 0)
    m7 = ~(m1 | m2 | m3 | m4 | m5 | m6)
    sf = lambda a, b: a / np.where(np.abs(b) < 1e-20, 1e-20, b)
    out[m1] = A[m1]; out[m2] = B[m2]; out[m4] = C[m4]
    out[m3] = A[m3] + sf(d1, d1 - d3)[m3, None] * ab[m3]
    out[m5] = A[m5] + sf(d2, d2 - d6)[m5, None] * ac[m5]
    out[m6] = B[m6] + sf(d4 - d3, (d4 - d3) + (d5 - d6))[m6, None] * (C[m6] - B[m6])
    den = va + vb + vc
    out[m7] = A[m7] + ab[m7] * sf(vb, den)[m7, None] + ac[m7] * sf(vc, den)[m7, None]
    return out


ftree = cKDTree(V[FA].mean(1))                                 # KD over face centroids


def closest_on_mesh(X, k=16):
    _, cf = ftree.query(X, k=k)
    M = len(X)
    A_, B_, C_ = (V[FA[cf, j]].reshape(-1, 3) for j in range(3))
    cp = _pt_tri(np.repeat(X[:, None, :], k, 1).reshape(-1, 3), A_, B_, C_).reshape(M, k, 3)
    j = np.argmin(np.linalg.norm(cp - X[:, None, :], axis=2), axis=1)
    r = np.arange(M)
    return cp[r, j], cf[r, j]


STEP, TRAVEL_MAX = 0.008, 0.52                                 # 8 mm steps, up to the slider's max travel
NSTEP = int(np.ceil(TRAVEL_MAX / STEP))
NPART = min(a.nflow, N)
X = T[:NPART, :3].copy()                                       # seeds = the FPS prefix: blue-noise, ALL kept
track = np.zeros((NPART, NSTEP + 1, 3), np.float32)
for i in range(NSTEP + 1):
    cp, fid = closest_on_mesh(X)                               # snap back onto the surface
    track[:, i] = cp
    g = fgrad[fid]; ln = np.linalg.norm(g, axis=1, keepdims=True)
    X = cp + np.where(ln > 1e-6, g / np.clip(ln, 1e-9, None), 0.0) * STEP
paths = None
pathP = (track.reshape(-1, 3) - ctr).astype(np.float32)        # same shift as the body mesh
pathOff = (np.arange(NPART + 1) * (NSTEP + 1)).astype(np.int32)
pathVpos = track.reshape(-1, 3)                                # world (unshifted) for the KD lookups below
mv = np.linalg.norm(np.diff(track, axis=1), axis=2).sum(1)
print(f"[flow] {NPART} particles | {NSTEP+1} samples each | advected arclength: median {np.median(mv):.2f} m",
      flush=True)

# per path sample: blend the 3 nearest DENSE samples (normal-agreeing -- never blend across a limb)
pn = VN[cKDTree(V).query(pathVpos)[1]]                          # surface normal at each path sample
d8, i8 = dtree.query(pathVpos, k=8)
nn = np.zeros((len(pathVpos), 3), np.uint16); ww = np.zeros((len(pathVpos), 3), np.uint8)
for r in range(len(pathVpos)):
    ok = (d8[r] < 0.08) & (DN[i8[r]] @ pn[r] > 0.3)
    cand = i8[r][ok][:3]; dist = d8[r][ok][:3]
    if len(cand) == 0: cand, dist = i8[r][:1], d8[r][:1]        # fall back to the plain nearest
    w = 1.0 / (dist + 1e-3); w /= w.sum()
    nn[r, :len(cand)] = cand; ww[r, :len(cand)] = np.round(w * 255)


def qpack(q):                                                  # q,-q are the same rotation: force w>=0 and
    q = np.asarray(q, np.float64)                              # ship only xyz (w = sqrt(1-|xyz|^2) in JS)
    q = np.where((q[..., 3:] < 0), -q, q)
    return np.clip(np.round(q[..., :3] * 32767), -32767, 32767).astype(np.int16)


# ---------- observed tracks ----------
obs = {}
for j, fi in enumerate(frames):
    d = np.loadtxt(f"{a.dir}/pts_{fi:04d}.csv", delimiter=",", skiprows=1, ndmin=2)
    for r in d:                                                # idx,u,v,qw,qx,qy,qz,twist
        obs[(int(r[0]), j)] = np.array([r[1], r[2], r[4], r[5], r[6], r[3], r[7]])   # u,v,quat(xyzw),twist

vis = np.zeros((N, F), bool)
for (i, j) in obs: vis[i, j] = True

# ---------- bridge interior gaps (<= gmax) ----------
recs = [(j, i, p, 0) for (i, j), p in obs.items()]             # gap 0 == observed
for i in range(N):
    fj = np.flatnonzero(vis[i])
    for k in range(len(fj) - 1):
        f0, f1 = int(fj[k]), int(fj[k + 1]); g = f1 - f0 - 1
        if not (1 <= g <= a.gmax): continue
        p0, p1 = obs[(i, f0)], obs[(i, f1)]
        q0, q1 = p0[2:6], p1[2:6].copy()
        if float(q0 @ q1) < 0: q1 = -q1                        # shortest arc
        for f in range(f0 + 1, f1):
            w = (f - f0) / (f1 - f0)
            q = (1 - w) * q0 + w * q1; q /= max(float(np.linalg.norm(q)), 1e-9)      # nlerp
            dt = float(p1[6] - p0[6]); dt -= round(dt)         # twist is cyclic in [0,1)
            recs.append((f, i, np.concatenate([[(1-w)*p0[0] + w*p1[0], (1-w)*p0[1] + w*p1[1]],
                                               q, [(p0[6] + w*dt) % 1.0]]), g))
nbridge = len(recs) - len(obs)
recs.sort(key=lambda r: (r[0], r[1]))

counts = [0] * F
for r in recs: counts[r[0]] += 1
P = np.stack([r[2] for r in recs])
idx = np.array([r[1] for r in recs], np.uint16)
gap = np.array([r[3] for r in recs], np.uint8)
uv = np.clip(np.round(P[:, :2] * 65535), 0, 65535).astype(np.uint16)
qb = qpack(P[:, 2:6])
tw = np.clip(np.round(P[:, 6] * 255), 0, 255).astype(np.uint8)

# ---------- background video ----------
tmp = os.path.join(tempfile.gettempdir(), f"ff_bg_{os.getpid()}.mp4")
w = iio.get_writer(tmp, fps=a.fps, codec="libx264", quality=6, pixelformat="yuv420p", macro_block_size=8)
for fi in frames:
    w.append_data(cv2.resize(cv2.imread(f"{a.dir}/bg_{fi:04d}.png")[..., ::-1], (a.vid_px, a.vid_px)))
w.close()

b64 = lambda x: base64.b64encode(x.tobytes()).decode()
data = dict(F=F, fps=a.fps, N=N, NFIX=a.nfix, gmax=a.gmax, counts=counts, rng=rng, maxc=int(max(counts)),
            idxB=b64(idx), uvB=b64(uv), qB=b64(qb), twB=b64(tw), gapB=b64(gap),
            vB=b64(vq), fB=b64(FA.astype(np.uint16)), tpB=b64(tpos), tqB=b64(qpack(tq)),
            NP=NPART, ppB=b64(pathP), poB=b64(pathOff), nnB=b64(nn), wwB=b64(ww),
            vid=base64.b64encode(open(tmp, "rb").read()).decode())
print(f"[pair] {F} frames | {N} dense pts (fixed prefix {a.nfix}) | {len(obs)} observed + {nbridge} bridged"
      f" | max {max(counts)}/frame | body {len(V)}v/{len(FA)}f", flush=True)

HTML = r"""<!-- FRAMEFIELD_PAIR_SWEEP -->
<title>Frame Field · template vs. reprojected sequence</title>
<style>
:root{ color-scheme:dark; --ink:#dfe6f2; --dim:#8a93a6; --line:rgba(140,160,200,.16);
  --xr:#ff5470; --yg:#3ddc84; --zb:#5b8cff; --accent:#5b8cff; --panel:rgba(14,18,28,.72);
  --mono:ui-monospace,"SFMono-Regular",Menlo,Consolas,monospace; --b:.26 }
*{box-sizing:border-box} html,body{margin:0;height:100%}
body{background:radial-gradient(120% 90% at 50% -10%,#141924 0%,#0a0c12 55%,#06070b 100%);
  color:var(--ink);font-family:var(--mono);overflow:hidden;-webkit-font-smoothing:antialiased;
  display:flex;flex-direction:column;align-items:center;justify-content:center;gap:13px}
#wrap{position:relative;display:flex;gap:14px}
.pane{position:relative;width:min(40vh,40vw);aspect-ratio:1/1;border:1px solid var(--line);border-radius:13px;
  overflow:hidden;box-shadow:0 24px 70px rgba(0,0,0,.55)}
#pL{background:linear-gradient(180deg,#10141d,#0a0d13);cursor:grab}#pL:active{cursor:grabbing}
#vid{position:absolute;inset:0;width:100%;height:100%;object-fit:cover;filter:brightness(var(--b))}
#gl{position:absolute;inset:0;display:block;pointer-events:none}
.tag{position:absolute;top:9px;left:11px;z-index:2;font-size:10px;letter-spacing:.17em;text-transform:uppercase;
  color:var(--dim);background:rgba(8,10,16,.6);padding:4px 7px;border-radius:6px;backdrop-filter:blur(6px)}
.hd{text-align:center;max-width:78ch}
.ey{font-size:10.5px;letter-spacing:.32em;color:var(--accent);text-transform:uppercase;margin:0 0 5px}
.ti{font-family:system-ui,sans-serif;font-weight:650;font-size:clamp(16px,1.9vw,23px);margin:0;letter-spacing:-.01em}
.sub{margin:6px 0 0;color:var(--dim);font-size:11.5px;line-height:1.5}.sub b{color:var(--zb);font-weight:600}
.bar{display:flex;align-items:center;gap:11px;width:min(84vh,84vw);flex-wrap:wrap;justify-content:center}
button{font-family:var(--mono);font-size:10.5px;letter-spacing:.06em;text-transform:uppercase;color:var(--ink);
  background:var(--panel);border:1px solid var(--line);border-radius:8px;padding:7px 10px;cursor:pointer}
button:hover{border-color:rgba(120,150,220,.5)}
button[aria-pressed="true"]{color:#fff;border-color:var(--accent);background:rgba(91,140,255,.16)}
button:focus-visible{outline:2px solid var(--accent);outline-offset:2px}
.play{width:34px;height:34px;border-radius:50%;display:grid;place-items:center;padding:0;flex:0 0 auto}
.lbl{font-size:9.5px;letter-spacing:.11em;color:var(--dim);text-transform:uppercase;white-space:nowrap}
.val{font-size:10px;color:var(--ink);min-width:2.2em;font-variant-numeric:tabular-nums}
input[type=range]{appearance:none;height:4px;border-radius:3px;background:rgba(140,160,200,.25);outline:none;cursor:pointer}
input[type=range]::-webkit-slider-thumb{appearance:none;width:12px;height:12px;border-radius:50%;background:#fff;box-shadow:0 0 8px var(--accent)}
input[type=range]::-moz-range-thumb{width:12px;height:12px;border:none;border-radius:50%;background:#fff}
input[type=range]:disabled{opacity:.35;cursor:default}
#scrub{flex:1;min-width:140px}
.tc{font-size:10.5px;color:var(--dim);letter-spacing:.08em;font-variant-numeric:tabular-nums;min-width:64px;text-align:right}
.lg{display:flex;gap:12px;font-size:11px;color:var(--dim);align-items:center}
.sw{width:17px;height:3px;border-radius:2px;display:inline-block;margin-right:5px;vertical-align:middle;box-shadow:0 0 9px currentColor}
.grp{display:flex;align-items:center;gap:7px;padding:5px 9px;border:1px solid var(--line);border-radius:9px;background:rgba(12,15,23,.5)}
</style>
<div class="hd">
  <p class="ey">Oriented DensePose</p>
  <h1 class="ti">The same frames, on the template and on the image</h1>
  <p class="sub">Left: the canonical <b>template</b> frame at every visible point. Right: the predicted <b>SO(3) frame</b> reprojected onto the real clip. Blue = outward normal. Drag the left panel to orbit.<br>
  <b>Fixed</b> tracks chosen anchors. <b>Band</b> sweeps a plane over a dense point set. <b>Flow</b> lets sparse frames drift continuously down the surface &mdash; no anchors at all, because the field is predicted densely.</p>
</div>
<div id="wrap">
  <div class="pane" id="pL"><span class="tag">template · canonical</span></div>
  <div class="pane" id="pR"><span class="tag">sequence · reprojected</span><video id="vid" muted loop playsinline></video></div>
  <canvas id="gl"></canvas>
</div>
<div class="bar">
  <button id="bPlay" class="play" aria-pressed="true" aria-label="Play or pause">&#10074;&#10074;</button>
  <input id="scrub" type="range" min="0" max="1000" value="0" aria-label="Timeline">
  <span class="tc" id="tc">000 / 000</span>
  <span class="lg"><span><i class="sw" style="background:var(--zb);color:var(--zb)"></i>normal</span>
  <span><i class="sw" style="background:var(--xr);color:var(--xr)"></i>tan X</span>
  <span><i class="sw" style="background:var(--yg);color:var(--yg)"></i>tan Y</span></span>
</div>
<div class="bar">
  <div class="grp"><button id="mFix" aria-pressed="true">Fixed</button><button id="mBand">Band</button><button id="mFlow">Flow</button>
    <span class="lbl">band/travel</span><input id="band" type="range" min="2" max="30" value="10" style="width:64px" disabled>
    <span class="lbl">speed</span><input id="spd" type="range" min="1" max="40" value="14" style="width:56px" disabled></div>
  <div class="grp"><span class="lbl">bridge</span><input id="bridge" type="range" min="0" max="6" value="3" style="width:66px"><span class="val" id="vG">3</span>
    <span class="lbl">min run</span><input id="minrun" type="range" min="1" max="20" value="3" style="width:66px"><span class="val" id="vK">3</span></div>
  <div class="grp"><span class="lbl">dim</span><input id="dim" type="range" min="5" max="100" value="26" style="width:62px">
    <span class="lbl">size</span><input id="sz" type="range" min="10" max="70" value="30" style="width:62px"></div>
  <button id="bTan" aria-pressed="true">Tangents</button><button id="bTwist" aria-pressed="false">Twist</button>
  <span class="lbl" id="stat" style="min-width:9em"></span>
</div>
<script>const DATA=__DATA__;</script>
<script>
const vid=document.getElementById("vid"),cv=document.getElementById("gl"),wrap=document.getElementById("wrap"),
      pL=document.getElementById("pL"),pR=document.getElementById("pR"),
      gl=cv.getContext("webgl2",{antialias:true,alpha:true});
vid.src="data:video/mp4;base64,"+DATA.vid;
if(!gl){document.body.insertAdjacentHTML("beforeend","<p style='color:#8a93a6'>WebGL2 not available.</p>");}
else{
const $=id=>document.getElementById(id);
function unb64(s){const b=atob(s),u=new Uint8Array(b.length);for(let i=0;i<b.length;i++)u[i]=b.charCodeAt(i);return u;}
const IDX=new Uint16Array(unb64(DATA.idxB).buffer),UV=new Uint16Array(unb64(DATA.uvB).buffer),
      QI=new Int16Array(unb64(DATA.qB).buffer),TWI=unb64(DATA.twB),GAP=unb64(DATA.gapB),
      BV=new Int16Array(unb64(DATA.vB).buffer),BF=new Uint16Array(unb64(DATA.fB).buffer),
      TP=new Float32Array(unb64(DATA.tpB).buffer),TQI=new Int16Array(unb64(DATA.tqB).buffer),
      PP=new Float32Array(unb64(DATA.ppB).buffer),PO=new Int32Array(unb64(DATA.poB).buffer),
      FNN=new Uint16Array(unb64(DATA.nnB).buffer),FWW=unb64(DATA.wwB);
const F=DATA.F,N=DATA.N,NFIX=DATA.NFIX,counts=DATA.counts,off=[0];
for(let i=0;i<F;i++)off.push(off[i]+counts[i]);
// quats ship as xyz with w>=0 implied
function q4(A,i){const x=A[i*3]/32767,y=A[i*3+1]/32767,z=A[i*3+2]/32767;
 return [x,y,z,Math.sqrt(Math.max(0,1-x*x-y*y-z*z))];}
// sweep coordinate = height on the template (z is up); precompute range
let zlo=1e9,zhi=-1e9;for(let p=0;p<N;p++){const z=TP[p*3+2];if(z<zlo)zlo=z;if(z>zhi)zhi=z;}
// ---- temporal filter: bridge gaps <= G, then drop runs < K ----
const keep=new Uint8Array(N*F),pres=new Uint8Array(N*F);
function refilter(K,G){
 pres.fill(0);
 for(let f=0;f<F;f++)for(let i=off[f];i<off[f+1];i++) if(GAP[i]<=G) pres[IDX[i]*F+f]=1;
 keep.fill(0);
 for(let p=0;p<N;p++){let c=0;
  for(let f=0;f<=F;f++){const on=f<F&&pres[p*F+f];
   if(on)c++;else{if(c>=K)for(let g=f-c;g<f;g++)keep[p*F+g]=1;c=0;}}}
}
// ---- arrow along +Z ----
function arrow(seg=10){const P=[],Nn=[],I=[],rs=.055,sh=.62,rc=.16,h=1.;
 for(let i=0;i<seg;i++){const a=i/seg*6.2831853,c=Math.cos(a),s=Math.sin(a);P.push(rs*c,rs*s,0,rs*c,rs*s,sh);Nn.push(c,s,0,c,s,0);}
 for(let i=0;i<seg;i++){const j=(i+1)%seg;I.push(i*2,j*2,j*2+1,i*2,j*2+1,i*2+1);}
 const b=P.length/3;for(let i=0;i<seg;i++){const a=i/seg*6.2831853,c=Math.cos(a),s=Math.sin(a),l=Math.hypot(c,s,.5);P.push(rc*c,rc*s,sh);Nn.push(c/l,s/l,.5/l);}
 const t=P.length/3;P.push(0,0,h);Nn.push(0,0,1);for(let i=0;i<seg;i++){const j=(i+1)%seg;I.push(b+i,b+j,t);}
 return{P,N:Nn,I};}
const AR=arrow(10);
function prog(vs,fs){const p=gl.createProgram();for(const[t,s]of[[gl.VERTEX_SHADER,vs],[gl.FRAGMENT_SHADER,fs]]){
 const sh=gl.createShader(t);gl.shaderSource(sh,s);gl.compileShader(sh);
 if(!gl.getShaderParameter(sh,gl.COMPILE_STATUS))console.error(gl.getShaderInfoLog(sh));gl.attachShader(p,sh);}
 gl.linkProgram(p);return p;}
const QR=`vec3 qr(vec4 q,vec3 v){return v+2.0*cross(q.xyz,cross(q.xyz,v)+q.w*v);}
vec3 turbo(float t){return clamp(vec3(.5+.6*sin(6.28*t),.5+.6*sin(6.28*(t+.33)),.5+.6*sin(6.28*(t-.33))),0.,1.);}`;
const LIT=`#version 300 es
precision highp float;in vec3 vN;in vec3 vC;out vec4 o;
void main(){vec3 L=normalize(vec3(.45,.6,.85));float d=max(dot(normalize(vN),L),0.0);
 o=vec4(vC*(0.5+0.62*d),1.0);}`;
// right pane: image-space orthographic
const prI=prog(`#version 300 es
layout(location=0) in vec3 aPos;layout(location=1) in vec3 aNorm;
layout(location=2) in vec4 iQ;layout(location=3) in vec2 iUV;layout(location=4) in vec3 iC;
layout(location=5) in float iT;layout(location=6) in float iW;
uniform float uScale;uniform int uTwist;out vec3 vN;out vec3 vC;${QR}
void main(){vec3 p=qr(iQ,aPos*uScale*iW);vN=normalize(qr(iQ,aNorm));
 vC=(uTwist==1?turbo(iT):iC)*(0.35+0.65*iW);
 vec2 c=vec2(iUV.x*2.0-1.0,iUV.y*2.0-1.0)+p.xy*2.0;gl_Position=vec4(c,-p.z*0.15,1.0);}`,LIT);
// left pane: perspective on the template
const prT=prog(`#version 300 es
layout(location=0) in vec3 aPos;layout(location=1) in vec3 aNorm;
layout(location=2) in vec4 iQ;layout(location=3) in vec3 iP;layout(location=4) in vec3 iC;
layout(location=5) in float iT;layout(location=6) in float iW;
uniform mat4 uMVP;uniform float uScale;uniform int uTwist;out vec3 vN;out vec3 vC;${QR}
void main(){vec3 p=iP+qr(iQ,aPos*uScale*iW);vN=normalize(qr(iQ,aNorm));
 vC=(uTwist==1?turbo(iT):iC)*(0.35+0.65*iW);gl_Position=uMVP*vec4(p,1.0);}`,LIT);
const prB=prog(`#version 300 es
layout(location=0) in vec3 aPos;layout(location=1) in vec3 aNorm;
uniform mat4 uMVP;out vec3 vN;void main(){vN=aNorm;gl_Position=uMVP*vec4(aPos,1.0);}`,
`#version 300 es
precision highp float;in vec3 vN;out vec4 o;
void main(){vec3 L=normalize(vec3(.45,.6,.85));vec3 n=normalize(vN);float d=max(dot(n,L),0.0);
 float rim=pow(1.0-abs(n.z),2.0);
 o=vec4(vec3(.16,.18,.23)*(0.35+0.75*d)+rim*vec3(.09,.12,.2),1.0);}`);
// ---- body mesh: dequantize + per-vertex normals ----
const nV=BV.length/3,BP=new Float32Array(nV*3),BN=new Float32Array(nV*3),sc=DATA.rng/32000;
for(let i=0;i<nV*3;i++)BP[i]=BV[i]*sc;
for(let t=0;t<BF.length;t+=3){const a=BF[t]*3,b=BF[t+1]*3,c=BF[t+2]*3;
 const e1=[BP[b]-BP[a],BP[b+1]-BP[a+1],BP[b+2]-BP[a+2]],e2=[BP[c]-BP[a],BP[c+1]-BP[a+1],BP[c+2]-BP[a+2]];
 const n=[e1[1]*e2[2]-e1[2]*e2[1],e1[2]*e2[0]-e1[0]*e2[2],e1[0]*e2[1]-e1[1]*e2[0]];
 for(const o of [a,b,c])for(let k=0;k<3;k++)BN[o+k]+=n[k];}
for(let i=0;i<nV;i++){const l=Math.hypot(BN[i*3],BN[i*3+1],BN[i*3+2])||1;for(let k=0;k<3;k++)BN[i*3+k]/=l;}
// ---- FLOW: arclength along each descent path + a per-particle phase so they don't march in lockstep ----
const NP=DATA.NP,CUM=new Float32Array(PP.length/3),PLEN=new Float32Array(NP),PH=new Float32Array(NP);
for(let p=0;p<NP;p++){let s=0;CUM[PO[p]]=0;
 for(let k=PO[p]+1;k<PO[p+1];k++){const a=(k-1)*3,b=k*3;
  s+=Math.hypot(PP[b]-PP[a],PP[b+1]-PP[a+1],PP[b+2]-PP[a+2]);CUM[k]=s;}
 PLEN[p]=s;PH[p]=(p*0.6180339887)%1;}      // staggered ages -> constant on-screen count
const fU=new Float32Array(NP*2),fQ=new Float32Array(NP*4),fT=new Float32Array(NP*4),
      fP=new Float32Array(NP*3),fW=new Float32Array(NP),fTw=new Float32Array(NP);
function buf(tg,arr){const b=gl.createBuffer();gl.bindBuffer(tg,b);gl.bufferData(tg,arr,gl.STATIC_DRAW);return b;}
const arP=buf(gl.ARRAY_BUFFER,new Float32Array(AR.P)),arN=buf(gl.ARRAY_BUFFER,new Float32Array(AR.N)),
      arI=buf(gl.ELEMENT_ARRAY_BUFFER,new Uint16Array(AR.I)),
      bdP=buf(gl.ARRAY_BUFFER,BP),bdN=buf(gl.ARRAY_BUFFER,BN),bdI=buf(gl.ELEMENT_ARRAY_BUFFER,BF);
function attr(loc,b,n,div){gl.bindBuffer(gl.ARRAY_BUFFER,b);gl.enableVertexAttribArray(loc);
 gl.vertexAttribPointer(loc,n,gl.FLOAT,false,0,0);if(div)gl.vertexAttribDivisor(loc,div);}
const M=DATA.maxc*3;                                    // 3 axes per visible point, worst-case frame
const iQb=gl.createBuffer(),iUVb=gl.createBuffer(),iCb=gl.createBuffer(),iTb=gl.createBuffer(),
      iWb=gl.createBuffer(),tQb=gl.createBuffer(),tPb=gl.createBuffer();
const aQ=new Float32Array(M*4),aUV=new Float32Array(M*2),aC=new Float32Array(M*3),aT=new Float32Array(M),
      aW=new Float32Array(M),bQ=new Float32Array(M*4),bP=new Float32Array(M*3);
const BUFS=[[iQb,aQ],[iUVb,aUV],[iCb,aC],[iTb,aT],[iWb,aW],[tQb,bQ],[tPb,bP]];
for(const [b,arr] of BUFS){gl.bindBuffer(gl.ARRAY_BUFFER,b);gl.bufferData(gl.ARRAY_BUFFER,arr.byteLength,gl.DYNAMIC_DRAW);}
const vaoI=gl.createVertexArray();gl.bindVertexArray(vaoI);
 attr(0,arP,3);attr(1,arN,3);attr(2,iQb,4,1);attr(3,iUVb,2,1);attr(4,iCb,3,1);attr(5,iTb,1,1);attr(6,iWb,1,1);
 gl.bindBuffer(gl.ELEMENT_ARRAY_BUFFER,arI);
const vaoT=gl.createVertexArray();gl.bindVertexArray(vaoT);
 attr(0,arP,3);attr(1,arN,3);attr(2,tQb,4,1);attr(3,tPb,3,1);attr(4,iCb,3,1);attr(5,iTb,1,1);attr(6,iWb,1,1);
 gl.bindBuffer(gl.ELEMENT_ARRAY_BUFFER,arI);
const vaoB=gl.createVertexArray();gl.bindVertexArray(vaoB);
 attr(0,bdP,3);attr(1,bdN,3);gl.bindBuffer(gl.ELEMENT_ARRAY_BUFFER,bdI);
gl.bindVertexArray(null);
// ---- per-frame instance build (both panes get the identical point set) ----
const AXC=[[.36,.55,1],[1,.33,.44],[.24,.86,.52]],s2=Math.SQRT1_2,
      AXQ=[[0,0,0,1],[0,s2,0,s2],[-s2,0,0,s2]];       // +Z -> normal, +X, +Y  (xyzw)
function qmul(a,b){return[a[3]*b[0]+a[0]*b[3]+a[1]*b[2]-a[2]*b[1],a[3]*b[1]-a[0]*b[2]+a[1]*b[3]+a[2]*b[0],
 a[3]*b[2]+a[0]*b[1]-a[1]*b[0]+a[2]*b[3],a[3]*b[3]-a[0]*b[0]-a[1]*b[1]-a[2]*b[2]];}
let nDraw=0,mapF=-1,mapDirty=true;
const selI=new Int32Array(DATA.maxc),selW=new Float32Array(DATA.maxc);
const recOf=new Int32Array(N);
function recMap(f){recOf.fill(-1);
 for(let i=off[f];i<off[f+1];i++){const p=IDX[i];if(keep[p*F+f])recOf[p]=i;}
 mapF=f;mapDirty=false;}
function upload(n){nDraw=n;
 for(const [b,arr] of BUFS){gl.bindBuffer(gl.ARRAY_BUFFER,b);gl.bufferSubData(gl.ARRAY_BUFFER,0,arr);}}
// FIXED (mode 0) / BAND (mode 1): draw the exported samples themselves
function buildSampled(f,band,zc,hw){
 let n=0;
 for(let i=off[f];i<off[f+1];i++){
  const p=IDX[i];
  if(!keep[p*F+f])continue;
  let w=1;
  if(band){const t=Math.abs(TP[p*3+2]-zc)/hw;if(t>=1)continue;w=1-t*t*(3-2*t);}
  else if(p>=NFIX)continue;
  selI[n]=i;selW[n]=w;n++;
 }
 for(let ax=0;ax<3;ax++)for(let k=0;k<n;k++){
  const i=selI[k],p=IDX[i],j=ax*n+k;
  const cq=qmul(q4(QI,i),AXQ[ax]);
  aQ[j*4]=cq[0];aQ[j*4+1]=cq[1];aQ[j*4+2]=cq[2];aQ[j*4+3]=cq[3];
  aUV[j*2]=UV[i*2]/65535;aUV[j*2+1]=UV[i*2+1]/65535;
  const tq2=qmul(q4(TQI,p),AXQ[ax]);
  bQ[j*4]=tq2[0];bQ[j*4+1]=tq2[1];bQ[j*4+2]=tq2[2];bQ[j*4+3]=tq2[3];
  bP[j*3]=TP[p*3];bP[j*3+1]=TP[p*3+1];bP[j*3+2]=TP[p*3+2];
  aC[j*3]=AXC[ax][0];aC[j*3+1]=AXC[ax][1];aC[j*3+2]=AXC[ax][2];
  aT[j]=TWI[i]/255;aW[j]=selW[k];
 }
 upload(n);$("stat").textContent=n+(band?" in band":" pts");
}
// FLOW (mode 2): read the field at a CONTINUOUS surface position by blending the 3 nearest dense
// samples (normal-agreeing, so it never blends across a limb). Position is not an anchor any more.
function blend(k){
 let ws=0,u=0,v=0,tw=0,bw=-1,ref=null,tref=null;
 const q=[0,0,0,0],tq=[0,0,0,0];
 for(let j=0;j<3;j++){
  const w=FWW[k*3+j]/255;if(w<=0)continue;
  const p=FNN[k*3+j],r=recOf[p];if(r<0)continue;         // that sample isn't visible this frame
  ws+=w;u+=w*UV[r*2]/65535;v+=w*UV[r*2+1]/65535;
  if(w>bw){bw=w;tw=TWI[r]/255;}                          // twist is cyclic: take it from the dominant one
  let a=q4(QI,r);
  if(ref===null)ref=a;else if(a[0]*ref[0]+a[1]*ref[1]+a[2]*ref[2]+a[3]*ref[3]<0)a=[-a[0],-a[1],-a[2],-a[3]];
  for(let c=0;c<4;c++)q[c]+=w*a[c];
  let b=q4(TQI,p);
  if(tref===null)tref=b;else if(b[0]*tref[0]+b[1]*tref[1]+b[2]*tref[2]+b[3]*tref[3]<0)b=[-b[0],-b[1],-b[2],-b[3]];
  for(let c=0;c<4;c++)tq[c]+=w*b[c];
 }
 if(ws<1e-4)return null;
 const nq=Math.hypot(q[0],q[1],q[2],q[3])||1,nt=Math.hypot(tq[0],tq[1],tq[2],tq[3])||1;
 for(let c=0;c<4;c++){q[c]/=nq;tq[c]/=nt;}
 return {w:ws,u:u/ws,v:v/ws,tw:tw,q:q,tq:tq};
}
function buildFlow(f,t,spd,travel){
 // Each particle has a short LIFETIME and respawns at its OWN blue-noise (FPS) seed. So at any instant the
 // live set IS the uniform seed set, each member displaced only `travel` metres along the flow. Letting
 // particles free-run instead lets the flow's convergence funnel them into shared channels -> streaks.
 if(mapF!==f||mapDirty)recMap(f);
 const T=Math.max(travel/Math.max(spd,1e-3),.05);              // lifetime = travel / speed
 let n=0;
 for(let p=0;p<NP;p++){
  const L=PLEN[p];if(L<1e-4)continue;
  const age=(((t/T+PH[p])%1)+1)%1;                             // 0 = born at the seed, 1 = die
  const s=Math.min(age*travel,L);                              // short path -> stall at its end
  let lo=PO[p],hi=PO[p+1]-1;
  while(hi-lo>1){const m=(lo+hi)>>1;if(CUM[m]<=s)lo=m;else hi=m;}
  const fr=Math.min(1,Math.max(0,(s-CUM[lo])/Math.max(CUM[hi]-CUM[lo],1e-6)));
  const A=blend(lo),B=blend(hi);
  if(!A||!B)continue;
  let al=(Math.min(A.w,B.w)-0.25)/0.45;al=Math.min(1,Math.max(0,al));al=al*al*(3-2*al);
  const fi=Math.min(1,age/0.12),fo=Math.min(1,(1-age)/0.12);   // quick fade in/out, long plateau
  const wt=al*Math.min(fi,fo)*(3-2*Math.min(fi,fo))*Math.min(fi,fo);
  if(wt<0.02)continue;
  const sg=(A.q[0]*B.q[0]+A.q[1]*B.q[1]+A.q[2]*B.q[2]+A.q[3]*B.q[3])<0?-1:1;
  const st=(A.tq[0]*B.tq[0]+A.tq[1]*B.tq[1]+A.tq[2]*B.tq[2]+A.tq[3]*B.tq[3])<0?-1:1;
  const q=[0,0,0,0],tq=[0,0,0,0];
  for(let c=0;c<4;c++){q[c]=A.q[c]*(1-fr)+sg*B.q[c]*fr;tq[c]=A.tq[c]*(1-fr)+st*B.tq[c]*fr;}
  const nq=Math.hypot(q[0],q[1],q[2],q[3])||1,nt=Math.hypot(tq[0],tq[1],tq[2],tq[3])||1;
  for(let c=0;c<4;c++){fQ[n*4+c]=q[c]/nq;fT[n*4+c]=tq[c]/nt;}
  const a3=lo*3,b3=hi*3;
  for(let c=0;c<3;c++)fP[n*3+c]=PP[a3+c]+(PP[b3+c]-PP[a3+c])*fr;
  fU[n*2]=A.u+(B.u-A.u)*fr;fU[n*2+1]=A.v+(B.v-A.v)*fr;
  fW[n]=wt;fTw[n]=fr<0.5?A.tw:B.tw;n++;
 }
 for(let ax=0;ax<3;ax++)for(let k=0;k<n;k++){
  const j=ax*n+k;
  const cq=qmul([fQ[k*4],fQ[k*4+1],fQ[k*4+2],fQ[k*4+3]],AXQ[ax]);
  aQ[j*4]=cq[0];aQ[j*4+1]=cq[1];aQ[j*4+2]=cq[2];aQ[j*4+3]=cq[3];
  aUV[j*2]=fU[k*2];aUV[j*2+1]=fU[k*2+1];
  const tq2=qmul([fT[k*4],fT[k*4+1],fT[k*4+2],fT[k*4+3]],AXQ[ax]);
  bQ[j*4]=tq2[0];bQ[j*4+1]=tq2[1];bQ[j*4+2]=tq2[2];bQ[j*4+3]=tq2[3];
  for(let c=0;c<3;c++)bP[j*3+c]=fP[k*3+c];
  aC[j*3]=AXC[ax][0];aC[j*3+1]=AXC[ax][1];aC[j*3+2]=AXC[ax][2];
  aT[j]=fTw[k];aW[j]=fW[k];
 }
 upload(n);$("stat").textContent=n+" flowing";
}
// ---- camera (left pane) ----
let az=0,el=.06,R=2.6;
function mul(a,b){const o=new Float32Array(16);for(let i=0;i<4;i++)for(let j=0;j<4;j++){let s=0;
 for(let k=0;k<4;k++)s+=a[k*4+j]*b[i*4+k];o[i*4+j]=s;}return o;}
function persp(fy,as,n,f){const t=1/Math.tan(fy/2);return new Float32Array([t/as,0,0,0, 0,t,0,0, 0,0,(f+n)/(n-f),-1, 0,0,2*f*n/(n-f),0]);}
function look(e,c,u){const z=[e[0]-c[0],e[1]-c[1],e[2]-c[2]];let l=Math.hypot(z[0],z[1],z[2]);z.forEach((v,i)=>z[i]=v/l);
 const x=[u[1]*z[2]-u[2]*z[1],u[2]*z[0]-u[0]*z[2],u[0]*z[1]-u[1]*z[0]];l=Math.hypot(x[0],x[1],x[2])||1;x.forEach((v,i)=>x[i]=v/l);
 const y=[z[1]*x[2]-z[2]*x[1],z[2]*x[0]-z[0]*x[2],z[0]*x[1]-z[1]*x[0]];
 return new Float32Array([x[0],y[0],z[0],0, x[1],y[1],z[1],0, x[2],y[2],z[2],0,
  -(x[0]*e[0]+x[1]*e[1]+x[2]*e[2]),-(y[0]*e[0]+y[1]*e[1]+y[2]*e[2]),-(z[0]*e[0]+z[1]*e[1]+z[2]*e[2]),1]);}
function mvp(as){const e=[R*Math.cos(el)*Math.sin(az),-R*Math.cos(el)*Math.cos(az),R*Math.sin(el)];
 return mul(persp(.62,as,.05,20),look(e,[0,0,0],[0,0,1]));}
let drag=null;
pL.addEventListener("pointerdown",e=>{drag=[e.clientX,e.clientY];pL.setPointerCapture(e.pointerId);});
pL.addEventListener("pointermove",e=>{if(!drag)return;az-=(e.clientX-drag[0])*.008;
 el=Math.max(-1.2,Math.min(1.2,el+(e.clientY-drag[1])*.006));drag=[e.clientX,e.clientY];});
addEventListener("pointerup",()=>drag=null);
pL.addEventListener("wheel",e=>{e.preventDefault();R=Math.max(1.2,Math.min(6,R*(1+Math.sign(e.deltaY)*.09)));},{passive:false});
// ---- UI ----
let showTan=true,twist=false,playing=true,mode=0,shown=-1,K=3,G=3,t0=null;
refilter(K,G);
$("bTan").onclick=()=>{showTan=!showTan;$("bTan").setAttribute("aria-pressed",showTan);};
$("bTwist").onclick=()=>{twist=!twist;$("bTwist").setAttribute("aria-pressed",twist);};
function setMode(m){mode=m;
 $("mFix").setAttribute("aria-pressed",m===0);$("mBand").setAttribute("aria-pressed",m===1);
 $("mFlow").setAttribute("aria-pressed",m===2);
 $("band").disabled=m===0;$("spd").disabled=m===0;shown=-1;}
$("mFix").onclick=()=>setMode(0);$("mBand").onclick=()=>setMode(1);$("mFlow").onclick=()=>setMode(2);
$("bPlay").onclick=()=>{playing=!playing;playing?vid.play():vid.pause();
 $("bPlay").innerHTML=playing?"&#10074;&#10074;":"&#9654;";$("bPlay").setAttribute("aria-pressed",playing);};
$("dim").oninput=e=>document.documentElement.style.setProperty("--b",e.target.value/100);
$("minrun").oninput=e=>{K=+e.target.value;$("vK").textContent=K;refilter(K,G);shown=-1;mapDirty=true;};
$("bridge").oninput=e=>{G=+e.target.value;$("vG").textContent=G;refilter(K,G);shown=-1;mapDirty=true;};
const scrub=$("scrub"),tc=$("tc");
scrub.addEventListener("input",()=>{playing=false;vid.pause();$("bPlay").innerHTML="&#9654;";
 $("bPlay").setAttribute("aria-pressed",false);vid.currentTime=(scrub.value/1000)*(F/DATA.fps);});
let DPR=1,vpL=[0,0,1,1],vpR=[0,0,1,1];
function resize(){const w=wrap.getBoundingClientRect();DPR=Math.min(devicePixelRatio||1,2);
 cv.style.width=w.width+"px";cv.style.height=w.height+"px";cv.style.position="absolute";cv.style.left="0";cv.style.top="0";
 cv.width=Math.round(w.width*DPR);cv.height=Math.round(w.height*DPR);
 const f=r=>[Math.round((r.left-w.left)*DPR),Math.round((w.bottom-r.bottom)*DPR),Math.round(r.width*DPR),Math.round(r.height*DPR)];
 vpL=f(pL.getBoundingClientRect());vpR=f(pR.getBoundingClientRect());}
addEventListener("resize",resize);new ResizeObserver(resize).observe(wrap);
gl.enable(gl.DEPTH_TEST);
if(matchMedia("(prefers-reduced-motion:reduce)").matches){playing=false;$("bPlay").innerHTML="&#9654;";}
vid.addEventListener("loadeddata",()=>{resize();if(playing)vid.play().catch(()=>{});});
function loop(ts){
 if(t0===null)t0=ts;
 const f=Math.min(F-1,Math.max(0,Math.round(vid.currentTime*DATA.fps)));
 const tsec=(ts-t0)/1000;
 if(mode===2){ buildFlow(f,tsec,(+$("spd").value)/100*0.45,(+$("band").value)/200); shown=f; }  // m/s, m
 else if(mode===1){ const span=zhi-zlo,hw=span*(+$("band").value)/200;
  const ph=(tsec*(+$("spd").value)/100)%1;                                      // travels top -> bottom, loops
  buildSampled(f,true,(zhi+hw)-ph*(span+2*hw),hw); shown=f; }                   // starts/ends outside: no pop
 else if(f!==shown){ buildSampled(f,false,0,0); shown=f; }
 if(f!==+scrub.dataset.f){scrub.value=Math.round(f/(F-1)*1000);scrub.dataset.f=f;
  tc.textContent=String(f+1).padStart(3,"0")+" / "+String(F).padStart(3,"0");}
 gl.disable(gl.SCISSOR_TEST);gl.clearColor(0,0,0,0);gl.clear(gl.COLOR_BUFFER_BIT|gl.DEPTH_BUFFER_BIT);
 gl.enable(gl.SCISSOR_TEST);
 const S=(+$("sz").value)/1000,inst=(showTan&&!twist)?nDraw*3:nDraw;
 gl.viewport(...vpL);gl.scissor(...vpL);
 const Pm=mvp(vpL[2]/vpL[3]);
 gl.useProgram(prB);gl.bindVertexArray(vaoB);gl.uniformMatrix4fv(gl.getUniformLocation(prB,"uMVP"),false,Pm);
 gl.drawElements(gl.TRIANGLES,BF.length,gl.UNSIGNED_SHORT,0);
 gl.useProgram(prT);gl.bindVertexArray(vaoT);gl.uniformMatrix4fv(gl.getUniformLocation(prT,"uMVP"),false,Pm);
 gl.uniform1f(gl.getUniformLocation(prT,"uScale"),S*2.2);gl.uniform1i(gl.getUniformLocation(prT,"uTwist"),twist?1:0);
 gl.drawElementsInstanced(gl.TRIANGLES,AR.I.length,gl.UNSIGNED_SHORT,0,inst);
 gl.viewport(...vpR);gl.scissor(...vpR);
 gl.useProgram(prI);gl.bindVertexArray(vaoI);
 gl.uniform1f(gl.getUniformLocation(prI,"uScale"),S);gl.uniform1i(gl.getUniformLocation(prI,"uTwist"),twist?1:0);
 gl.drawElementsInstanced(gl.TRIANGLES,AR.I.length,gl.UNSIGNED_SHORT,0,inst);
 requestAnimationFrame(loop);}
resize();requestAnimationFrame(loop);
}
</script>
"""
open(a.out, "w").write(HTML.replace("__DATA__", json.dumps(data, separators=(",", ":"))))
print(f"[pair] wrote {a.out} ({os.path.getsize(a.out)/1e6:.2f} MB)", flush=True)

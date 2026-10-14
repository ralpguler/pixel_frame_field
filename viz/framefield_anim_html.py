"""Self-contained WebGL2 page that ANIMATES the surface frame field over a body motion loop (baked).
Per frame: posed mesh + each tripod tracking its surface point. Orbitable, play/pause/scrub. No external assets.
Positions are quantized to int16 + base64 to keep the file small; normals are computed in-browser."""
import os  # noqa: E402
import sys  # noqa: E402
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import _env  # noqa: E402  (repository paths: viz/_env.py)
_env.take_out_arg()
import os, sys, json, base64, numpy as np, torch
from scipy.spatial.transform import Rotation, Slerp
from orient.rot6d import get_template, vertex_normals, rotation_from_deformation
from render_correspondence import get_smplx

OUT = _env.out("framefield.html")
NKEY, INTERP, NTRIP = 5, 9, 340          # keyframes, frames between them -> ~45 frames; tripods
import torch as _t; dev = "cuda" if _t.cuda.is_available() else "cpu"; torch.set_grad_enabled(False)

v_template, faces, n_rest = get_template(dev)
faces = np.asarray(faces); n_rest = np.asarray(n_rest); v_template = np.asarray(v_template)
gt = np.load(_env.camerahmr("cameramhr_3dpw_preds.npz"), allow_pickle=True)["gt_pose"]
mag = np.linalg.norm(gt[:, 3:66], axis=1)
picks = np.argsort(mag)[(np.linspace(0.60, 0.92, NKEY - 2) * (len(mag) - 1)).astype(int)]  # varied dance poses
# loop: ZERO-POSE SMPL TEMPLATE -> dance poses -> back to template (frame 0 = the canonical template)
key_poses = [np.zeros(63)] + [gt[p, 3:66] for p in picks] + [np.zeros(63)]
m = get_smplx("neutral", 16, dev)

# smooth motion: per-joint slerp between keyframe body poses
key_rot = [Rotation.from_rotvec(kp.reshape(21, 3)) for kp in key_poses]
frames_pose = []
for a in range(len(key_poses) - 1):
    sl = [Slerp([0, 1], Rotation.concatenate([key_rot[a][j], key_rot[a+1][j]])) for j in range(21)]
    for t in np.linspace(0, 1, INTERP, endpoint=False):
        frames_pose.append(np.concatenate([sl[j](t).as_rotvec() for j in range(21)]))
F = len(frames_pose)

# fixed FPS points (equidistant on the template) -> tracked consistently across frames
def fps(P, k):
    d = ((P - P[0]) ** 2).sum(1); sel = [0]
    for _ in range(k - 1):
        j = int(np.argmax(d)); sel.append(j); d = np.minimum(d, ((P - P[j]) ** 2).sum(1))
    return np.array(sel)
S = fps(v_template, NTRIP)
# template surface frame at the fixed points
ref = np.tile([0., 1., 0.], (len(n_rest), 1)); dg = np.abs((ref * n_rest).sum(1)) > 0.95; ref[dg] = [1, 0, 0]
tT = ref - (ref * n_rest).sum(1, keepdims=True) * n_rest; tT /= np.clip(np.linalg.norm(tT, axis=1, keepdims=True), 1e-9, None)
Tf = np.stack([tT, np.cross(n_rest, tT), n_rest], 2)


def R2q(M):                                                    # xyzw
    t = np.trace(M)
    if t > 0:
        s = np.sqrt(t+1)*2; q = [(M[2,1]-M[1,2])/s, (M[0,2]-M[2,0])/s, (M[1,0]-M[0,1])/s, .25*s]
    else:
        i = int(np.argmax([M[0,0], M[1,1], M[2,2]]))
        if i == 0: s = np.sqrt(1+M[0,0]-M[1,1]-M[2,2])*2; q = [.25*s, (M[0,1]+M[1,0])/s, (M[0,2]+M[2,0])/s, (M[2,1]-M[1,2])/s]
        elif i == 1: s = np.sqrt(1+M[1,1]-M[0,0]-M[2,2])*2; q = [(M[0,1]+M[1,0])/s, .25*s, (M[1,2]+M[2,1])/s, (M[0,2]-M[2,0])/s]
        else: s = np.sqrt(1+M[2,2]-M[0,0]-M[1,1])*2; q = [(M[0,2]+M[2,0])/s, (M[1,2]+M[2,1])/s, .25*s, (M[1,0]-M[0,1])/s]
    q = np.array(q); return q / max(np.linalg.norm(q), 1e-9)


# forward every frame -> verts, framefield; collect
allV = np.zeros((F, len(v_template), 3), np.float32)
tripQ = np.zeros((F, NTRIP, 4), np.float32)
tripT = np.zeros((F, NTRIP), np.float32)
for f in range(F):
    bp = torch.tensor(frames_pose[f], device=dev, dtype=torch.float32)[None]
    V = m(global_orient=torch.zeros(1, 3, device=dev), body_pose=bp,
          betas=torch.zeros(1, 16, device=dev)).vertices[0].cpu().numpy().astype(np.float64)
    np_ = vertex_normals(V, faces)
    R = rotation_from_deformation(v_template, V, faces, n_rest, np_)
    D = R[S] @ Tf[S]
    allV[f] = V.astype(np.float32)
    for i in range(NTRIP): tripQ[f, i] = R2q(D[i])
    tripT[f] = (np.arctan2(R[S, 1, 0], R[S, 0, 0]) + np.pi) / (2*np.pi)
    if f % 10 == 0: print(f"[anim] frame {f}/{F}", flush=True)

# center + scale using the whole motion's bounds
ctr = allV.reshape(-1, 3).mean(0); allV -= ctr
scl = 1.8 / (allV.reshape(-1, 3).max(0) - allV.reshape(-1, 3).min(0)).max(); allV *= scl

# ---- pack: positions int16 (per-axis min/scale), faces uint16, tripod quats float32, twist float32 ----
mn = allV.reshape(-1, 3).min(0); mx = allV.reshape(-1, 3).max(0); rng = np.maximum(mx - mn, 1e-6)
q16 = np.clip(np.round((allV - mn) / rng * 65535.0), 0, 65535).astype(np.uint16)   # (F,Nv,3)
def b64(a): return base64.b64encode(a.tobytes()).decode()
data = dict(
    F=F, Nv=int(len(v_template)), nTrip=NTRIP,
    mn=[float(x) for x in mn], rng=[float(x) for x in rng],
    posB=b64(q16),                                    # uint16 F*Nv*3
    facesB=b64(faces.astype(np.uint16).reshape(-1)),  # uint16 Nf*3
    fps=[int(x) for x in S],                          # tripod vertex indices
    quatB=b64(tripQ.astype(np.float32).reshape(-1)),  # float32 F*nTrip*4
    twB=b64(tripT.astype(np.float32).reshape(-1)),    # float32 F*nTrip
)
print(f"[anim] {F} frames, {NTRIP} tripods, body {len(v_template)}v/{len(faces)}f", flush=True)

HTML = r"""<!-- FRAMEFIELD_ANIM -->
<title>Surface Frame Field · Oriented DensePose</title>
<style>
:root{ color-scheme:dark;
  --ink:#dfe6f2; --dim:#8a93a6; --line:rgba(140,160,200,.16);
  --xr:#ff5470; --yg:#3ddc84; --zb:#5b8cff; --accent:#5b8cff; --panel:rgba(14,18,28,.66);
  --mono:ui-monospace,"SFMono-Regular",Menlo,Consolas,monospace; }
*{box-sizing:border-box} html,body{margin:0;height:100%}
body{background:radial-gradient(120% 90% at 50% -10%,#161b28 0%,#0a0c12 55%,#06070b 100%);
  color:var(--ink);font-family:var(--mono);overflow:hidden;-webkit-font-smoothing:antialiased}
#gl{position:fixed;inset:0;width:100%;height:100%;display:block;touch-action:none}
.hud{position:fixed;z-index:2;pointer-events:none}
.tl{top:22px;left:26px}
.ey{font-size:11px;letter-spacing:.32em;color:var(--accent);text-transform:uppercase;margin:0 0 6px}
.ti{font-family:system-ui,sans-serif;font-weight:650;font-size:clamp(20px,2.6vw,30px);letter-spacing:-.01em;margin:0}
.sub{max-width:44ch;margin:10px 0 0;color:var(--dim);font-size:12.5px;line-height:1.55}
.sub b{color:var(--zb);font-weight:600}
.legend{bottom:78px;left:26px;display:flex;flex-direction:column;gap:7px}
.lg{display:flex;align-items:center;gap:9px;font-size:11.5px;color:var(--dim)}
.sw{width:22px;height:3px;border-radius:2px;box-shadow:0 0 10px currentColor}
.panel{bottom:78px;right:26px;pointer-events:auto;display:flex;flex-direction:column;gap:10px;align-items:flex-end}
.row{display:flex;gap:8px}
button{pointer-events:auto;font-family:var(--mono);font-size:11px;letter-spacing:.06em;text-transform:uppercase;
  color:var(--ink);background:var(--panel);border:1px solid var(--line);border-radius:9px;padding:8px 11px;cursor:pointer;backdrop-filter:blur(8px);transition:.15s}
button:hover{border-color:rgba(120,150,220,.5)}
button[aria-pressed="true"]{color:#fff;border-color:var(--accent);background:rgba(91,140,255,.16)}
button:focus-visible{outline:2px solid var(--accent);outline-offset:2px}
.transport{position:fixed;left:26px;right:26px;bottom:26px;z-index:2;display:flex;align-items:center;gap:14px;pointer-events:auto}
.play{width:42px;height:42px;border-radius:50%;display:grid;place-items:center;flex:0 0 auto}
#scrub{flex:1;appearance:none;height:4px;border-radius:3px;background:linear-gradient(90deg,var(--accent) 0%,var(--accent) var(--p,0%),rgba(140,160,200,.22) var(--p,0%));outline:none;cursor:pointer}
#scrub::-webkit-slider-thumb{appearance:none;width:14px;height:14px;border-radius:50%;background:#fff;box-shadow:0 0 8px var(--accent);cursor:pointer}
#scrub::-moz-range-thumb{width:14px;height:14px;border:none;border-radius:50%;background:#fff}
.tc{font-size:11px;color:var(--dim);letter-spacing:.1em;font-variant-numeric:tabular-nums;min-width:64px;text-align:right}
.badge{position:fixed;top:24px;right:26px;z-index:2;font-size:10.5px;letter-spacing:.14em;color:var(--dim);text-transform:uppercase;border:1px solid var(--line);border-radius:999px;padding:5px 12px;background:var(--panel);backdrop-filter:blur(8px)}
@media (max-width:640px){ .sub{display:none} .tl{top:16px;left:16px} .legend{left:16px} .panel{right:16px} .badge{display:none} .transport{left:16px;right:16px} }
</style>
<canvas id="gl"></canvas>
<div class="hud tl">
  <p class="ey">Oriented DensePose</p>
  <h1 class="ti">Surface Frame Field</h1>
  <p class="sub">A predicted <b>SO(3) frame</b> at every surface point, animated over a motion loop — each tripod tracks its point on the body. Blue = outward <b>normal</b>; red/green are the orthonormal tangent axes carrying the <b>twist</b>. Drag to orbit.</p>
</div>
<div class="badge">SMPL-X &middot; <span id="cnt"></span> frames &middot; live</div>
<div class="hud legend">
  <div class="lg"><span class="sw" style="background:var(--zb);color:var(--zb)"></span>normal</div>
  <div class="lg"><span class="sw" style="background:var(--xr);color:var(--xr)"></span>tangent&nbsp;X</div>
  <div class="lg"><span class="sw" style="background:var(--yg);color:var(--yg)"></span>tangent&nbsp;Y</div>
</div>
<div class="hud panel">
  <div class="row"><button id="bBody" aria-pressed="true">Body</button><button id="bTan" aria-pressed="true">Tangents</button></div>
  <div class="row"><button id="bTwist" aria-pressed="false">Twist color</button><button id="bSpin" aria-pressed="true">Spin</button></div>
</div>
<div class="transport">
  <button id="bPlay" class="play" aria-pressed="true" aria-label="Play / pause">❚❚</button>
  <input id="scrub" type="range" min="0" max="1000" value="0" aria-label="Timeline">
  <span class="tc" id="tc">00 / 00</span>
</div>
<script>const DATA=__DATA__;</script>
<script>
const cv=document.getElementById("gl"),gl=cv.getContext("webgl2",{antialias:true});
document.getElementById("cnt").textContent=DATA.F;
if(!gl){document.body.insertAdjacentHTML("beforeend","<p style='position:fixed;inset:0;display:grid;place-items:center;color:#8a93a6'>WebGL2 not available.</p>");}
else{
const sub=(a,b)=>[a[0]-b[0],a[1]-b[1],a[2]-b[2]],dot=(a,b)=>a[0]*b[0]+a[1]*b[1]+a[2]*b[2];
const cross=(a,b)=>[a[1]*b[2]-a[2]*b[1],a[2]*b[0]-a[0]*b[2],a[0]*b[1]-a[1]*b[0]];
const nrm=a=>{let l=Math.hypot(a[0],a[1],a[2])||1;return[a[0]/l,a[1]/l,a[2]/l];};
const M4={persp:(f,a,n,fa)=>[1/Math.tan(f/2)/a,0,0,0,0,1/Math.tan(f/2),0,0,0,0,(fa+n)/(n-fa),-1,0,0,2*fa*n/(n-fa),0],
 look:(e,c,u)=>{let z=nrm(sub(e,c)),x=nrm(cross(u,z)),y=cross(z,x);return[x[0],y[0],z[0],0,x[1],y[1],z[1],0,x[2],y[2],z[2],0,-dot(x,e),-dot(y,e),-dot(z,e),1];},
 mul:(a,b)=>{let o=new Array(16);for(let i=0;i<4;i++)for(let j=0;j<4;j++){let s=0;for(let k=0;k<4;k++)s+=a[k*4+j]*b[i*4+k];o[i*4+j]=s;}return o;}};
function prog(vs,fs){const p=gl.createProgram();for(const[t,s]of[[gl.VERTEX_SHADER,vs],[gl.FRAGMENT_SHADER,fs]]){const sh=gl.createShader(t);gl.shaderSource(sh,s);gl.compileShader(sh);if(!gl.getShaderParameter(sh,gl.COMPILE_STATUS))console.error(gl.getShaderInfoLog(sh));gl.attachShader(p,sh);}gl.linkProgram(p);return p;}
function unb64(s){const bin=atob(s),u=new Uint8Array(bin.length);for(let i=0;i<bin.length;i++)u[i]=bin.charCodeAt(i);return u;}

// ---- decode animation ----
const F=DATA.F,Nv=DATA.Nv,nT=DATA.nTrip;
const posU16=new Uint16Array(unb64(DATA.posB).buffer);         // F*Nv*3
const faces=new Uint16Array(unb64(DATA.facesB).buffer);        // Nf*3
const quat=new Float32Array(unb64(DATA.quatB).buffer);         // F*nT*4
const tw=new Float32Array(unb64(DATA.twB).buffer);             // F*nT
const mn=DATA.mn,rg=DATA.rng,fpsIdx=DATA.fps;
// dequantize all frames + compute normals once
const framePos=[],frameNorm=[];
for(let f=0;f<F;f++){const P=new Float32Array(Nv*3),o=f*Nv*3;
 for(let i=0;i<Nv*3;i++){P[i]=mn[i%3]+posU16[o+i]/65535*rg[i%3];}
 const N=new Float32Array(Nv*3);
 for(let t=0;t<faces.length;t+=3){const a=faces[t]*3,b=faces[t+1]*3,c=faces[t+2]*3;
  const ux=P[b]-P[a],uy=P[b+1]-P[a+1],uz=P[b+2]-P[a+2],vx=P[c]-P[a],vy=P[c+1]-P[a+1],vz=P[c+2]-P[a+2];
  const nx=uy*vz-uz*vy,ny=uz*vx-ux*vz,nz=ux*vy-uy*vx;
  N[a]+=nx;N[a+1]+=ny;N[a+2]+=nz;N[b]+=nx;N[b+1]+=ny;N[b+2]+=nz;N[c]+=nx;N[c+1]+=ny;N[c+2]+=nz;}
 for(let i=0;i<Nv*3;i+=3){const l=Math.hypot(N[i],N[i+1],N[i+2])||1;N[i]/=l;N[i+1]/=l;N[i+2]/=l;}
 framePos.push(P);frameNorm.push(N);}

// ---- body program (dynamic pos+norm) ----
const pBody=prog(`#version 300 es
layout(location=0) in vec3 aPos;layout(location=1) in vec3 aNorm;uniform mat4 uVP;out vec3 vN;
void main(){vN=aNorm;gl_Position=uVP*vec4(aPos,1.0);}`,
`#version 300 es
precision highp float;in vec3 vN;out vec4 o;uniform vec3 uL;
void main(){vec3 n=normalize(vN);float d=max(dot(n,normalize(uL)),0.0);float rim=pow(1.0-abs(n.z),2.0);
 o=vec4(vec3(0.278,0.302,0.349)*(0.30+0.72*d)+vec3(0.10,0.13,0.20)*rim,1.0);}`);
const bodyVAO=gl.createVertexArray();gl.bindVertexArray(bodyVAO);
const bPos=gl.createBuffer(),bNor=gl.createBuffer();
gl.bindBuffer(gl.ARRAY_BUFFER,bPos);gl.bufferData(gl.ARRAY_BUFFER,framePos[0],gl.DYNAMIC_DRAW);gl.enableVertexAttribArray(0);gl.vertexAttribPointer(0,3,gl.FLOAT,false,0,0);
gl.bindBuffer(gl.ARRAY_BUFFER,bNor);gl.bufferData(gl.ARRAY_BUFFER,frameNorm[0],gl.DYNAMIC_DRAW);gl.enableVertexAttribArray(1);gl.vertexAttribPointer(1,3,gl.FLOAT,false,0,0);
const bIdx=gl.createBuffer();gl.bindBuffer(gl.ELEMENT_ARRAY_BUFFER,bIdx);gl.bufferData(gl.ELEMENT_ARRAY_BUFFER,faces,gl.STATIC_DRAW);

// ---- arrow geometry + program (instanced) ----
function arrow(seg=10){const P=[],N=[],I=[],rs=.055,sh=.62,rc=.16,h=1.0;
 for(let i=0;i<seg;i++){const a=i/seg*6.2831853,c=Math.cos(a),s=Math.sin(a);P.push(rs*c,rs*s,0,rs*c,rs*s,sh);N.push(c,s,0,c,s,0);}
 for(let i=0;i<seg;i++){const j=(i+1)%seg;I.push(i*2,j*2,j*2+1,i*2,j*2+1,i*2+1);}
 const base=P.length/3;for(let i=0;i<seg;i++){const a=i/seg*6.2831853,c=Math.cos(a),s=Math.sin(a),l=Math.hypot(c,s,.5);P.push(rc*c,rc*s,sh);N.push(c/l,s/l,.5/l);}
 const tip=P.length/3;P.push(0,0,h);N.push(0,0,1);for(let i=0;i<seg;i++){const j=(i+1)%seg;I.push(base+i,base+j,tip);}
 return{P,N,I};}
const AR=arrow(10);
const pArr=prog(`#version 300 es
layout(location=0) in vec3 aPos;layout(location=1) in vec3 aNorm;
layout(location=2) in vec4 iQ;layout(location=3) in vec3 iP;layout(location=4) in vec3 iC;layout(location=5) in float iT;
uniform mat4 uVP;uniform float uScale;uniform int uTwist;out vec3 vN;out vec3 vC;
vec3 qr(vec4 q,vec3 v){return v+2.0*cross(q.xyz,cross(q.xyz,v)+q.w*v);}
vec3 turbo(float t){return clamp(vec3(.5+.6*sin(6.28*t),.5+.6*sin(6.28*(t+.33)),.5+.6*sin(6.28*(t-.33))),0.0,1.0);}
void main(){vec3 p=qr(iQ,aPos*uScale)+iP;vN=normalize(qr(iQ,aNorm));vC=uTwist==1?turbo(iT):iC;gl_Position=uVP*vec4(p,1.0);}`,
`#version 300 es
precision highp float;in vec3 vN;in vec3 vC;out vec4 o;uniform vec3 uL;
void main(){float d=max(dot(normalize(vN),normalize(uL)),0.0);o=vec4(vC*(0.5+0.62*d),1.0);}`);
const arrVAO=gl.createVertexArray();gl.bindVertexArray(arrVAO);
function sb(loc,arr,n){const b=gl.createBuffer();gl.bindBuffer(gl.ARRAY_BUFFER,b);gl.bufferData(gl.ARRAY_BUFFER,new Float32Array(arr),gl.STATIC_DRAW);gl.enableVertexAttribArray(loc);gl.vertexAttribPointer(loc,n,gl.FLOAT,false,0,0);return b;}
sb(0,AR.P,3);sb(1,AR.N,3);
const aIdx=gl.createBuffer();gl.bindBuffer(gl.ELEMENT_ARRAY_BUFFER,aIdx);gl.bufferData(gl.ELEMENT_ARRAY_BUFFER,new Uint16Array(AR.I),gl.STATIC_DRAW);
// per-instance dynamic buffers: quat, pos, color(static per axis), twist. 3 axes -> 3*nT instances.
const AXR=[[0,0,1,0,1,0,-1,0,0],[1,0,0,0,0,-1,0,1,0],[1,0,0,0,1,0,0,0,1]];   // +Z-> +X,+Y,+Z (row-major 3x3)
const AXC=[[1,.33,.44],[.24,.86,.52],[.36,.55,1]];
function qmul(a,b){return[a[3]*b[0]+a[0]*b[3]+a[1]*b[2]-a[2]*b[1], a[3]*b[1]-a[0]*b[2]+a[1]*b[3]+a[2]*b[0], a[3]*b[2]+a[0]*b[1]-a[1]*b[0]+a[2]*b[3], a[3]*b[3]-a[0]*b[0]-a[1]*b[1]-a[2]*b[2]];}
function m2q(m){const t=m[0]+m[4]+m[8];let q;if(t>0){const s=Math.sqrt(t+1)*2;q=[(m[7]-m[5])/s,(m[2]-m[6])/s,(m[3]-m[1])/s,.25*s];}else if(m[0]>m[4]&&m[0]>m[8]){const s=Math.sqrt(1+m[0]-m[4]-m[8])*2;q=[.25*s,(m[1]+m[3])/s,(m[2]+m[6])/s,(m[7]-m[5])/s];}else if(m[4]>m[8]){const s=Math.sqrt(1+m[4]-m[0]-m[8])*2;q=[(m[1]+m[3])/s,.25*s,(m[5]+m[7])/s,(m[2]-m[6])/s];}else{const s=Math.sqrt(1+m[8]-m[0]-m[4])*2;q=[(m[2]+m[6])/s,(m[5]+m[7])/s,.25*s,(m[3]-m[1])/s];}const l=Math.hypot(q[0],q[1],q[2],q[3]);return[q[0]/l,q[1]/l,q[2]/l,q[3]/l];}
const axQ=AXR.map(m2q);                                        // base-axis quaternions (xyzw)
const NI=nT*3;
const iQ=new Float32Array(NI*4),iP=new Float32Array(NI*3),iC=new Float32Array(NI*3),iT=new Float32Array(NI);
for(let ax=0;ax<3;ax++)for(let i=0;i<nT;i++){const k=ax*nT+i;iC[k*3]=AXC[ax][0];iC[k*3+1]=AXC[ax][1];iC[k*3+2]=AXC[ax][2];}
const bQ=gl.createBuffer(),bP=gl.createBuffer(),bC=gl.createBuffer(),bT=gl.createBuffer();
function idyn(loc,b,n){gl.bindBuffer(gl.ARRAY_BUFFER,b);gl.enableVertexAttribArray(loc);gl.vertexAttribPointer(loc,n,gl.FLOAT,false,0,0);gl.vertexAttribDivisor(loc,1);}
gl.bindBuffer(gl.ARRAY_BUFFER,bQ);gl.bufferData(gl.ARRAY_BUFFER,iQ.byteLength,gl.DYNAMIC_DRAW);idyn(2,bQ,4);
gl.bindBuffer(gl.ARRAY_BUFFER,bP);gl.bufferData(gl.ARRAY_BUFFER,iP.byteLength,gl.DYNAMIC_DRAW);idyn(3,bP,3);
gl.bindBuffer(gl.ARRAY_BUFFER,bC);gl.bufferData(gl.ARRAY_BUFFER,iC,gl.STATIC_DRAW);idyn(4,bC,3);
gl.bindBuffer(gl.ARRAY_BUFFER,bT);gl.bufferData(gl.ARRAY_BUFFER,iT.byteLength,gl.DYNAMIC_DRAW);idyn(5,bT,1);

function setFrame(f){
 gl.bindBuffer(gl.ARRAY_BUFFER,bPos);gl.bufferSubData(gl.ARRAY_BUFFER,0,framePos[f]);
 gl.bindBuffer(gl.ARRAY_BUFFER,bNor);gl.bufferSubData(gl.ARRAY_BUFFER,0,frameNorm[f]);
 const P=framePos[f];
 for(let ax=0;ax<3;ax++)for(let i=0;i<nT;i++){const k=ax*nT+i,qo=(f*nT+i)*4;
  const q=[quat[qo],quat[qo+1],quat[qo+2],quat[qo+3]];const cq=qmul(q,axQ[ax]);
  iQ[k*4]=cq[0];iQ[k*4+1]=cq[1];iQ[k*4+2]=cq[2];iQ[k*4+3]=cq[3];
  const v=fpsIdx[i]*3;iP[k*3]=P[v];iP[k*3+1]=P[v+1];iP[k*3+2]=P[v+2];iT[k]=tw[f*nT+i];}
 gl.bindBuffer(gl.ARRAY_BUFFER,bQ);gl.bufferSubData(gl.ARRAY_BUFFER,0,iQ);
 gl.bindBuffer(gl.ARRAY_BUFFER,bP);gl.bufferSubData(gl.ARRAY_BUFFER,0,iP);
 gl.bindBuffer(gl.ARRAY_BUFFER,bT);gl.bufferSubData(gl.ARRAY_BUFFER,0,iT);
}

// ---- state / camera / transport ----
let az=-0.5,el=0.1,rad=3.4,showBody=true,showTan=true,twmode=false,spin=true,playing=true;
let curF=0,acc=0;const FPS=DATA.F>1?18:1;
const el_=id=>document.getElementById(id);
const btn=(id,on,f)=>{el_(id).setAttribute("aria-pressed",on);el_(id).onclick=()=>f(el_(id));};
btn("bBody",true,b=>{showBody=!showBody;b.setAttribute("aria-pressed",showBody);});
btn("bTan",true,b=>{showTan=!showTan;b.setAttribute("aria-pressed",showTan);});
btn("bTwist",false,b=>{twmode=!twmode;b.setAttribute("aria-pressed",twmode);});
btn("bSpin",true,b=>{spin=!spin;b.setAttribute("aria-pressed",spin);});
const play=el_("bPlay");play.onclick=()=>{playing=!playing;play.textContent=playing?"❚❚":"▶";play.setAttribute("aria-pressed",playing);};
const scrub=el_("scrub"),tc=el_("tc");
scrub.addEventListener("input",()=>{playing=false;play.textContent="▶";play.setAttribute("aria-pressed",false);curF=Math.round(scrub.value/1000*(F-1));});
let drag=false,px=0,py=0;
cv.addEventListener("pointerdown",e=>{drag=true;px=e.clientX;py=e.clientY;cv.setPointerCapture(e.pointerId);});
cv.addEventListener("pointerup",()=>drag=false);
cv.addEventListener("pointermove",e=>{if(!drag)return;az-=(e.clientX-px)*.008;el+=(e.clientY-py)*.008;el=Math.max(-1.45,Math.min(1.45,el));px=e.clientX;py=e.clientY;});
cv.addEventListener("wheel",e=>{e.preventDefault();rad*=(1+Math.sign(e.deltaY)*.08);rad=Math.max(1.6,Math.min(7,rad));},{passive:false});
function resize(){const d=Math.min(devicePixelRatio||1,2);cv.width=innerWidth*d;cv.height=innerHeight*d;gl.viewport(0,0,cv.width,cv.height);}
addEventListener("resize",resize);resize();
gl.enable(gl.DEPTH_TEST);
const reduce=matchMedia("(prefers-reduced-motion:reduce)").matches;if(reduce){spin=false;playing=false;play.textContent="▶";}
const L=[.5,.7,.9];let last=performance.now(),shownF=-1;
function loop(now){const dt=Math.min((now-last)/1000,.1);last=now;
 if(spin)az+=.0032;
 if(playing&&F>1){acc+=dt*FPS;while(acc>=1){acc-=1;curF=(curF+1)%F;}}
 if(curF!==shownF){setFrame(curF);shownF=curF;scrub.value=Math.round(curF/(F-1)*1000);scrub.style.setProperty("--p",(curF/(F-1)*100)+"%");tc.textContent=String(curF+1).padStart(2,"0")+" / "+String(F).padStart(2,"0");}
 const eye=[rad*Math.cos(el)*Math.sin(az),rad*Math.sin(el),rad*Math.cos(el)*Math.cos(az)];
 const VP=M4.mul(M4.persp(.86,cv.width/cv.height,.05,40),M4.look(eye,[0,0,0],[0,1,0]));
 gl.clearColor(.039,.047,.071,1);gl.clear(gl.COLOR_BUFFER_BIT|gl.DEPTH_BUFFER_BIT);
 if(showBody){gl.useProgram(pBody);gl.bindVertexArray(bodyVAO);gl.bindBuffer(gl.ELEMENT_ARRAY_BUFFER,bIdx);
  gl.uniformMatrix4fv(gl.getUniformLocation(pBody,"uVP"),false,VP);gl.uniform3fv(gl.getUniformLocation(pBody,"uL"),L);
  gl.drawElements(gl.TRIANGLES,faces.length,gl.UNSIGNED_SHORT,0);}
 gl.useProgram(pArr);gl.bindVertexArray(arrVAO);gl.bindBuffer(gl.ELEMENT_ARRAY_BUFFER,aIdx);
 gl.uniformMatrix4fv(gl.getUniformLocation(pArr,"uVP"),false,VP);gl.uniform3fv(gl.getUniformLocation(pArr,"uL"),L);
 gl.uniform1f(gl.getUniformLocation(pArr,"uScale"),.085);gl.uniform1i(gl.getUniformLocation(pArr,"uTwist"),twmode?1:0);
 const count=(showTan&&!twmode)?nT*3:nT;
 gl.drawElementsInstanced(gl.TRIANGLES,AR.I.length,gl.UNSIGNED_SHORT,0,count);
 requestAnimationFrame(loop);}
setFrame(0);requestAnimationFrame(loop);
}
</script>
"""
html = HTML.replace("__DATA__", json.dumps(data, separators=(",", ":")))
open(OUT, "w").write(html)
print(f"[anim] wrote {OUT} ({os.path.getsize(OUT)/1e6:.2f} MB)", flush=True)

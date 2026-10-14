"""Generate a self-contained interactive WebGL2 page: posed body mesh + the predicted per-surface-point
SO(3) frame field as instanced RGB tripods you can orbit. Writes framefield.html (no external assets)."""
import os, sys, json, numpy as np, torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)) + "/..")
sys.path.insert(0, os.path.expanduser("~/codes/wan/bedlam"))
from orient.rot6d import get_template, vertex_normals, rotation_from_deformation
from render_correspondence import get_smplx

OUT = os.path.expanduser("~/data/dp_viz_compare/framefield.html")
N = 620                                              # number of tripods (equidistant)
dev = "cuda"; torch.set_grad_enabled(False)

# ---- pose a body + compute the framefield (rest -> posed SO(3) per vertex) ----
v_template, faces, n_rest = get_template(dev)
gt = np.load(os.path.expanduser("~/codes/CameraHMR/cameramhr_3dpw_preds.npz"), allow_pickle=True)["gt_pose"]
idx = int(np.argsort(np.linalg.norm(gt[:, 3:66], axis=1))[int(0.85 * len(gt))])
pose = gt[idx].astype(np.float32)
m = get_smplx("neutral", 16, dev)
bp = torch.zeros(1, 63, device=dev); bp[0, :min(63, pose.shape[0]-3)] = torch.tensor(pose[3:66], device=dev)
V = m(global_orient=torch.zeros(1, 3, device=dev), body_pose=bp,
      betas=torch.zeros(1, 16, device=dev)).vertices[0].cpu().numpy().astype(np.float64)
faces = np.asarray(faces); n_rest = np.asarray(n_rest); v_template = np.asarray(v_template)
n_pose = vertex_normals(V, faces)
R = rotation_from_deformation(v_template, V, faces, n_rest, n_pose)     # (Nv,3,3) rest->posed

# surface frame per vertex: primary = posed normal; tangents carry the twist (world-up gauge)
ref = np.tile([0., 1., 0.], (len(n_rest), 1)); deg = np.abs((ref * n_rest).sum(1)) > 0.95; ref[deg] = [1, 0, 0]
tT = ref - (ref * n_rest).sum(1, keepdims=True) * n_rest; tT /= np.clip(np.linalg.norm(tT, axis=1, keepdims=True), 1e-9, None)
Tf = np.stack([tT, np.cross(n_rest, tT), n_rest], 2)                     # template surface frame
D = R @ Tf                                                              # posed display frame (col2 = normal)

# center + scale to ~2 units, keep Y-up (WebGL up = +Y)
c = V.mean(0); V = V - c; sc = 1.8 / (V.max(0) - V.min(0)).max(); V = V * sc
n_pose = vertex_normals(V, faces)

# ---- equidistant farthest-point sampling ----
def fps(P, k):
    d = ((P - P[0]) ** 2).sum(1); sel = [0]
    for _ in range(k - 1):
        j = int(np.argmax(d)); sel.append(j); d = np.minimum(d, ((P - P[j]) ** 2).sum(1))
    return np.array(sel)
S = fps(V, min(N, len(V)))
Ps, Ds = V[S], D[S]
twist = (np.arctan2(R[S, 1, 0], R[S, 0, 0]) + np.pi) / (2 * np.pi)


def R_to_quat_xyzw(M):
    t = np.trace(M)
    if t > 0:
        s = np.sqrt(t + 1) * 2; q = [(M[2,1]-M[1,2])/s, (M[0,2]-M[2,0])/s, (M[1,0]-M[0,1])/s, 0.25*s]
    else:
        i = int(np.argmax([M[0,0], M[1,1], M[2,2]]))
        if i == 0: s = np.sqrt(1+M[0,0]-M[1,1]-M[2,2])*2; q = [0.25*s, (M[0,1]+M[1,0])/s, (M[0,2]+M[2,0])/s, (M[2,1]-M[1,2])/s]
        elif i == 1: s = np.sqrt(1+M[1,1]-M[0,0]-M[2,2])*2; q = [(M[0,1]+M[1,0])/s, 0.25*s, (M[1,2]+M[2,1])/s, (M[0,2]-M[2,0])/s]
        else: s = np.sqrt(1+M[2,2]-M[0,0]-M[1,1])*2; q = [(M[0,2]+M[2,0])/s, (M[1,2]+M[2,1])/s, 0.25*s, (M[1,0]-M[0,1])/s]
    q = np.array(q); return q / max(np.linalg.norm(q), 1e-9)


# base-axis rotations: map +Z (arrow axis) -> +X, +Y, +Z
A = {0: np.array([[0,0,1],[0,1,0],[-1,0,0.]]), 1: np.array([[1,0,0],[0,0,-1],[0,1,0.]]), 2: np.eye(3)}
AXCOL = {0: [1.0, 0.33, 0.44], 1: [0.24, 0.86, 0.52], 2: [0.36, 0.55, 1.0]}   # X red, Y green, Z blue(normal)
# instance order: [all Z normals (N)], [all X (N)], [all Y (N)] -> lets us draw normals-only or full tripod
inst_q, inst_p, inst_c, inst_t = [], [], [], []
for ax in (2, 0, 1):
    for i in range(len(S)):
        inst_q.append(R_to_quat_xyzw(Ds[i] @ A[ax]))
        inst_p.append(Ps[i]); inst_c.append(AXCOL[ax]); inst_t.append(float(twist[i]))

data = dict(
    bodyPos=[round(float(x), 4) for x in V.reshape(-1)],
    bodyNorm=[round(float(x), 3) for x in n_pose.reshape(-1)],
    bodyIdx=[int(x) for x in faces.reshape(-1)],
    iQuat=[round(float(x), 4) for x in np.array(inst_q).reshape(-1)],
    iPos=[round(float(x), 4) for x in np.array(inst_p).reshape(-1)],
    iCol=[round(float(x), 3) for x in np.array(inst_c).reshape(-1)],
    iTwist=[round(float(x), 3) for x in inst_t],
    nTrip=int(len(S)),
)
print(f"[html] pose {idx}, {len(S)} tripods, body {len(V)} v / {len(faces)} f", flush=True)

HTML = r"""<!-- FRAMEFIELD_HTML -->
<style>
:root{ color-scheme: dark;
  --bg:#0a0c12; --bg2:#12151f; --ink:#dfe6f2; --dim:#8a93a6; --line:rgba(140,160,200,.16);
  --xr:#ff5470; --yg:#3ddc84; --zb:#5b8cff; --accent:#5b8cff; --panel:rgba(14,18,28,.66);
  --mono:ui-monospace,"SFMono-Regular",Menlo,Consolas,monospace; }
*{box-sizing:border-box} html,body{margin:0;height:100%}
body{background:radial-gradient(120% 90% at 50% -10%,#161b28 0%,#0a0c12 55%,#06070b 100%);
  color:var(--ink); font-family:var(--mono); overflow:hidden; -webkit-font-smoothing:antialiased}
#gl{position:fixed;inset:0;width:100%;height:100%;display:block;touch-action:none}
.hud{position:fixed;z-index:2;pointer-events:none}
.tl{top:22px;left:26px}
.ey{font-size:11px;letter-spacing:.32em;color:var(--accent);text-transform:uppercase;margin:0 0 6px}
.ti{font-family:system-ui,sans-serif;font-weight:650;font-size:clamp(20px,2.6vw,30px);letter-spacing:-.01em;margin:0;line-height:1.05}
.sub{max-width:46ch;margin:10px 0 0;color:var(--dim);font-size:12.5px;line-height:1.55}
.sub b{color:var(--zb);font-weight:600}
.legend{bottom:22px;left:26px;display:flex;flex-direction:column;gap:7px}
.lg{display:flex;align-items:center;gap:9px;font-size:11.5px;color:var(--dim);letter-spacing:.02em}
.sw{width:22px;height:3px;border-radius:2px;box-shadow:0 0 10px currentColor}
.panel{bottom:22px;right:26px;pointer-events:auto;display:flex;flex-direction:column;gap:10px;align-items:flex-end}
.row{display:flex;gap:8px}
button{pointer-events:auto;font-family:var(--mono);font-size:11px;letter-spacing:.06em;text-transform:uppercase;
  color:var(--ink);background:var(--panel);border:1px solid var(--line);border-radius:9px;padding:8px 11px;
  cursor:pointer;backdrop-filter:blur(8px);transition:border-color .15s,color .15s,background .15s}
button:hover{border-color:rgba(120,150,220,.5)}
button[aria-pressed="true"]{color:#fff;border-color:var(--accent);background:rgba(91,140,255,.16);box-shadow:0 0 0 1px rgba(91,140,255,.25) inset}
button:focus-visible{outline:2px solid var(--accent);outline-offset:2px}
.hint{font-size:10.5px;color:var(--dim);letter-spacing:.08em}
.badge{position:fixed;top:24px;right:26px;z-index:2;font-size:10.5px;letter-spacing:.14em;color:var(--dim);
  text-transform:uppercase;border:1px solid var(--line);border-radius:999px;padding:5px 12px;background:var(--panel);backdrop-filter:blur(8px)}
.vidcard{position:fixed;top:58px;right:26px;z-index:3;width:min(30vw,288px);pointer-events:auto;border:1px solid var(--line);border-radius:12px;overflow:hidden;background:#05070b;box-shadow:0 12px 44px rgba(0,0,0,.55)}
.vidcard video{display:block;width:100%;height:auto}
.vidcard .cap{font-size:9.5px;letter-spacing:.13em;text-transform:uppercase;color:var(--dim);padding:7px 10px;border-top:1px solid var(--line);background:var(--panel)}
.vidcard.hidden{display:none}
@media (max-width:640px){ .sub{display:none} .tl{top:16px;left:16px} .legend{left:16px;bottom:16px} .panel{right:16px;bottom:16px} .badge{display:none} .vidcard{width:44vw;top:auto;bottom:150px;right:16px} }
</style>

<canvas id="gl"></canvas>
<div class="vidcard" id="vidcard">
  <video autoplay muted loop playsinline src="data:video/mp4;base64,__VIDB64__"></video>
  <div class="cap">Predicted field &middot; tracked on the dance clip</div>
</div>
<div class="hud tl">
  <p class="ey">Oriented DensePose</p>
  <h1 class="ti">Surface Frame Field</h1>
  <p class="sub">A predicted <b>SO(3) frame</b> at every point on the body surface — the extra orientation our rotation head learns beyond dense correspondence. Blue points along the outward <b>normal</b>; red and green span the tangent plane and carry the <b>twist</b>. Drag to orbit.</p>
</div>
<div class="badge">SMPL-X · <span id="cnt"></span> frames</div>
<div class="hud legend">
  <div class="lg"><span class="sw" style="background:var(--zb);color:var(--zb)"></span>normal &nbsp;<span style="opacity:.6">R·n</span></div>
  <div class="lg"><span class="sw" style="background:var(--xr);color:var(--xr)"></span>tangent&nbsp;X</div>
  <div class="lg"><span class="sw" style="background:var(--yg);color:var(--yg)"></span>tangent&nbsp;Y</div>
</div>
<div class="hud panel">
  <div class="row">
    <button id="bBody" aria-pressed="true">Body</button>
    <button id="bTan" aria-pressed="true">Tangents</button>
  </div>
  <div class="row">
    <button id="bTwist" aria-pressed="false">Twist color</button>
    <button id="bSpin" aria-pressed="true">Spin</button>
    <button id="bVid" aria-pressed="true">Video</button>
  </div>
  <div class="hint">drag&nbsp;·&nbsp;orbit &nbsp; scroll&nbsp;·&nbsp;zoom</div>
</div>

<script>const DATA=__DATA__;</script>
<script>
const cv=document.getElementById("gl");const gl=cv.getContext("webgl2",{antialias:true});
document.getElementById("cnt").textContent=DATA.nTrip;
if(!gl){document.body.insertAdjacentHTML("beforeend","<p style='position:fixed;inset:0;display:grid;place-items:center;color:#8a93a6'>WebGL2 not available in this browser.</p>");}
else{
// ---- tiny mat4 ----
const M4={
 persp:(f,a,n,fa)=>{const t=1/Math.tan(f/2);return[t/a,0,0,0, 0,t,0,0, 0,0,(fa+n)/(n-fa),-1, 0,0,2*fa*n/(n-fa),0];},
 look:(e,c,u)=>{let z=nrm(sub(e,c)),x=nrm(cross(u,z)),y=cross(z,x);
   return[x[0],y[0],z[0],0, x[1],y[1],z[1],0, x[2],y[2],z[2],0, -dot(x,e),-dot(y,e),-dot(z,e),1];},
 mul:(a,b)=>{let o=new Array(16);for(let i=0;i<4;i++)for(let j=0;j<4;j++){let s=0;for(let k=0;k<4;k++)s+=a[k*4+j]*b[i*4+k];o[i*4+j]=s;}return o;}
};
const sub=(a,b)=>[a[0]-b[0],a[1]-b[1],a[2]-b[2]];const dot=(a,b)=>a[0]*b[0]+a[1]*b[1]+a[2]*b[2];
const cross=(a,b)=>[a[1]*b[2]-a[2]*b[1],a[2]*b[0]-a[0]*b[2],a[0]*b[1]-a[1]*b[0]];
const nrm=a=>{let l=Math.hypot(a[0],a[1],a[2])||1;return[a[0]/l,a[1]/l,a[2]/l];};

function prog(vs,fs){const p=gl.createProgram();for(const[t,s]of[[gl.VERTEX_SHADER,vs],[gl.FRAGMENT_SHADER,fs]]){const sh=gl.createShader(t);gl.shaderSource(sh,s);gl.compileShader(sh);if(!gl.getShaderParameter(sh,gl.COMPILE_STATUS))console.error(gl.getShaderInfoLog(sh));gl.attachShader(p,sh);}gl.linkProgram(p);return p;}

// ---- body ----
const pBody=prog(`#version 300 es
layout(location=0) in vec3 aPos; layout(location=1) in vec3 aNorm;
uniform mat4 uVP; out vec3 vN;
void main(){ vN=aNorm; gl_Position=uVP*vec4(aPos,1.0); }`,
`#version 300 es
precision highp float; in vec3 vN; out vec4 o; uniform vec3 uL;
void main(){ vec3 n=normalize(vN); float d=max(dot(n,normalize(uL)),0.0);
 float rim=pow(1.0-abs(n.z),2.0);
 vec3 c=vec3(0.278,0.302,0.349)*(0.32+0.72*d)+vec3(0.10,0.13,0.20)*rim; o=vec4(c,1.0);} `);
const bodyVAO=gl.createVertexArray();gl.bindVertexArray(bodyVAO);
function buf(loc,arr,n){const b=gl.createBuffer();gl.bindBuffer(gl.ARRAY_BUFFER,b);gl.bufferData(gl.ARRAY_BUFFER,new Float32Array(arr),gl.STATIC_DRAW);gl.enableVertexAttribArray(loc);gl.vertexAttribPointer(loc,n,gl.FLOAT,false,0,0);return b;}
buf(0,DATA.bodyPos,3);buf(1,DATA.bodyNorm,3);
const bIdx=gl.createBuffer();gl.bindBuffer(gl.ELEMENT_ARRAY_BUFFER,bIdx);gl.bufferData(gl.ELEMENT_ARRAY_BUFFER,new Uint32Array(DATA.bodyIdx),gl.STATIC_DRAW);
const bodyCount=DATA.bodyIdx.length;

// ---- arrow base geometry (procedural, +Z) ----
function arrow(seg=10){const P=[],Nn=[],I=[];const rs=0.055,sh=0.62,rc=0.16,h=1.0;
 for(let i=0;i<seg;i++){const a=i/seg*6.2831853,cx=Math.cos(a),sy=Math.sin(a);
  P.push(rs*cx,rs*sy,0, rs*cx,rs*sy,sh); Nn.push(cx,sy,0, cx,sy,0);}
 for(let i=0;i<seg;i++){const j=(i+1)%seg;const a=i*2,b=i*2+1,c=j*2,d=j*2+1;I.push(a,c,d, a,d,b);}
 const base=P.length/3;
 for(let i=0;i<seg;i++){const a=i/seg*6.2831853,cx=Math.cos(a),sy=Math.sin(a);
  P.push(rc*cx,rc*sy,sh); const nl=Math.hypot(cx,sy,0.5);Nn.push(cx/nl,sy/nl,0.5/nl);}
 const tip=P.length/3;P.push(0,0,h);Nn.push(0,0,1);
 for(let i=0;i<seg;i++){const j=(i+1)%seg;I.push(base+i,base+j,tip);}
 return{P,Nn,I};}
const AR=arrow(10);
const pArr=prog(`#version 300 es
layout(location=0) in vec3 aPos; layout(location=1) in vec3 aNorm;
layout(location=2) in vec4 iQ; layout(location=3) in vec3 iP; layout(location=4) in vec3 iC; layout(location=5) in float iT;
uniform mat4 uVP; uniform float uScale; uniform int uTwist;
out vec3 vN; out vec3 vC;
vec3 qr(vec4 q,vec3 v){return v+2.0*cross(q.xyz,cross(q.xyz,v)+q.w*v);}
vec3 turbo(float t){ return clamp(vec3(0.5+0.6*sin(6.28*(t+0.0)),0.5+0.6*sin(6.28*(t+0.33)),0.5+0.6*sin(6.28*(t-0.33))),0.0,1.0);}
void main(){ vec3 p=qr(iQ,aPos*uScale)+iP; vN=normalize(qr(iQ,aNorm));
 vC = uTwist==1 ? turbo(iT) : iC; gl_Position=uVP*vec4(p,1.0);} `,
`#version 300 es
precision highp float; in vec3 vN; in vec3 vC; out vec4 o; uniform vec3 uL;
void main(){ float d=max(dot(normalize(vN),normalize(uL)),0.0); vec3 c=vC*(0.5+0.62*d); o=vec4(c,1.0);} `);
const arrVAO=gl.createVertexArray();gl.bindVertexArray(arrVAO);
buf(0,AR.P,3);buf(1,AR.Nn,3);
const aIdx=gl.createBuffer();gl.bindBuffer(gl.ELEMENT_ARRAY_BUFFER,aIdx);gl.bufferData(gl.ELEMENT_ARRAY_BUFFER,new Uint16Array(AR.I),gl.STATIC_DRAW);
function ibuf(loc,arr,n){const b=gl.createBuffer();gl.bindBuffer(gl.ARRAY_BUFFER,b);gl.bufferData(gl.ARRAY_BUFFER,new Float32Array(arr),gl.STATIC_DRAW);gl.enableVertexAttribArray(loc);gl.vertexAttribPointer(loc,n,gl.FLOAT,false,0,0);gl.vertexAttribDivisor(loc,1);}
ibuf(2,DATA.iQuat,4);ibuf(3,DATA.iPos,3);ibuf(4,DATA.iCol,3);ibuf(5,DATA.iTwist,1);
const arrCount=AR.I.length, nT=DATA.nTrip;

// ---- state + camera ----
let az=-0.5,el=0.12,rad=3.4,tgt=[0,0,0];
let showBody=true,showTan=true,twist=false,spin=true;
const btn=(id,on,f)=>{const b=document.getElementById(id);b.setAttribute("aria-pressed",on);b.onclick=()=>{f(b);};};
btn("bBody",true,b=>{showBody=!showBody;b.setAttribute("aria-pressed",showBody);});
btn("bTan",true,b=>{showTan=!showTan;b.setAttribute("aria-pressed",showTan);});
btn("bTwist",false,b=>{twist=!twist;b.setAttribute("aria-pressed",twist);});
btn("bSpin",true,b=>{spin=!spin;b.setAttribute("aria-pressed",spin);});
btn("bVid",true,b=>{const v=document.getElementById("vidcard");const h=v.classList.toggle("hidden");b.setAttribute("aria-pressed",!h);});
let drag=false,px=0,py=0;
cv.addEventListener("pointerdown",e=>{drag=true;px=e.clientX;py=e.clientY;spin=false;document.getElementById("bSpin").setAttribute("aria-pressed",false);cv.setPointerCapture(e.pointerId);});
cv.addEventListener("pointerup",()=>drag=false);
cv.addEventListener("pointermove",e=>{if(!drag)return;az-=(e.clientX-px)*0.008;el+=(e.clientY-py)*0.008;el=Math.max(-1.45,Math.min(1.45,el));px=e.clientX;py=e.clientY;});
cv.addEventListener("wheel",e=>{e.preventDefault();rad*=(1+Math.sign(e.deltaY)*0.08);rad=Math.max(1.6,Math.min(7,rad));},{passive:false});

function resize(){const d=Math.min(devicePixelRatio||1,2);cv.width=innerWidth*d;cv.height=innerHeight*d;gl.viewport(0,0,cv.width,cv.height);}
addEventListener("resize",resize);resize();
gl.enable(gl.DEPTH_TEST);   // no back-face cull: procedural arrows may not have consistent winding
const reduce=matchMedia("(prefers-reduced-motion:reduce)").matches;if(reduce)spin=false;
const L=[0.5,0.7,0.9];
function frame(){
 if(spin)az+=0.0032;
 const eye=[tgt[0]+rad*Math.cos(el)*Math.sin(az),tgt[1]+rad*Math.sin(el),tgt[2]+rad*Math.cos(el)*Math.cos(az)];
 const P=M4.persp(0.86,cv.width/cv.height,0.05,40),Vm=M4.look(eye,tgt,[0,1,0]),VP=M4.mul(P,Vm);
 gl.clearColor(0.039,0.047,0.071,1);gl.clear(gl.COLOR_BUFFER_BIT|gl.DEPTH_BUFFER_BIT);
 if(showBody){gl.useProgram(pBody);gl.bindVertexArray(bodyVAO);
   gl.uniformMatrix4fv(gl.getUniformLocation(pBody,"uVP"),false,VP);gl.uniform3fv(gl.getUniformLocation(pBody,"uL"),L);
   gl.bindBuffer(gl.ELEMENT_ARRAY_BUFFER,bIdx);gl.drawElements(gl.TRIANGLES,bodyCount,gl.UNSIGNED_INT,0);}
 gl.useProgram(pArr);gl.bindVertexArray(arrVAO);gl.bindBuffer(gl.ELEMENT_ARRAY_BUFFER,aIdx);
 gl.uniformMatrix4fv(gl.getUniformLocation(pArr,"uVP"),false,VP);gl.uniform3fv(gl.getUniformLocation(pArr,"uL"),L);
 gl.uniform1f(gl.getUniformLocation(pArr,"uScale"),0.085);gl.uniform1i(gl.getUniformLocation(pArr,"uTwist"),twist?1:0);
 const count=(showTan && !twist)? nT*3 : nT;                 // normals-only when twist or tangents off
 gl.drawElementsInstanced(gl.TRIANGLES,arrCount,gl.UNSIGNED_SHORT,0,count);
 requestAnimationFrame(frame);
}
frame();
}
</script>
"""

vidb64 = open("/lus/lfs1aip2/projects/u6ga/rguler/data/dp_viz_compare/ff_vid_b64.txt").read().strip()
html = HTML.replace("__DATA__", json.dumps(data, separators=(",", ":"))).replace("__VIDB64__", vidb64)
open(OUT, "w").write(html)
print(f"[html] wrote {OUT}  ({os.path.getsize(OUT)/1e6:.2f} MB)", flush=True)

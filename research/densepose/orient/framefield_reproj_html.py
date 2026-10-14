"""Self-contained page: the predicted frame field REPROJECTED onto the real image sequence.
The dance clip plays as the background; on top, each tracked surface point draws its SO(3) tripod,
oriented by the per-frame predicted quaternion, through the same orthographic camera the image uses.
Everything is inlined (video as a data URI, per-frame points as base64 float32). No external assets.

Usage: framefield_reproj_html.py [--dir EXPORT_DIR] [--out framefield.html]
"""
import os, sys, json, base64, glob, argparse, numpy as np, cv2, imageio.v2 as iio

ap = argparse.ArgumentParser()
ap.add_argument("--dir", default="/lus/lfs1aip2/projects/u6ga/rguler/data/dp_viz_compare/framefield_video_fixed")
ap.add_argument("--out", default=os.path.expanduser("~/data/dp_viz_compare/framefield.html"))
ap.add_argument("--vid-px", type=int, default=520, help="background video size (smaller = smaller file)")
ap.add_argument("--fps", type=float, default=16.0)
a = ap.parse_args()

frames = sorted(int(os.path.basename(f)[4:8]) for f in glob.glob(f"{a.dir}/pts_*.csv"))
assert frames, f"no pts_*.csv in {a.dir}"
F = len(frames)

# ---- background video from the frame crops (clean; darkened in the page via CSS) ----
tmp = "/tmp/ff_reproj_bg.mp4"
w = iio.get_writer(tmp, fps=a.fps, codec="libx264", quality=6, pixelformat="yuv420p", macro_block_size=8)
for fi in frames:
    im = cv2.imread(f"{a.dir}/bg_{fi:04d}.png")[..., ::-1]
    w.append_data(cv2.resize(im, (a.vid_px, a.vid_px)))
w.close()
vid_b64 = base64.b64encode(open(tmp, "rb").read()).decode()
print(f"[reproj] bg video {os.path.getsize(tmp)/1e3:.0f} KB -> b64 {len(vid_b64)/1e3:.0f} KB", flush=True)

# ---- per-frame tracked points: u,v, quat(xyzw), twist ----
counts, flat = [], []
for fi in frames:
    d = np.loadtxt(f"{a.dir}/pts_{fi:04d}.csv", delimiter=",", skiprows=1, ndmin=2)
    # csv: u,v,qw,qx,qy,qz,twist  ->  pack u,v,qx,qy,qz,qw,twist  (shader wants xyzw)
    if d.size == 0:
        counts.append(0); continue
    p = np.stack([d[:, 0], d[:, 1], d[:, 3], d[:, 4], d[:, 5], d[:, 2], d[:, 6]], 1).astype(np.float32)
    counts.append(len(p)); flat.append(p)
flat = np.concatenate(flat).reshape(-1) if flat else np.zeros(0, np.float32)
data = dict(F=F, fps=a.fps, counts=[int(c) for c in counts],
            ptsB=base64.b64encode(flat.astype(np.float32).tobytes()).decode(),
            vid=vid_b64)
print(f"[reproj] {F} frames, {sum(counts)} tripod-instances total (avg {sum(counts)//F}/frame)", flush=True)

HTML = r"""<!-- FRAMEFIELD_REPROJ -->
<title>Frame Field · reprojected on the sequence</title>
<style>
:root{ color-scheme:dark;
  --ink:#dfe6f2; --dim:#8a93a6; --line:rgba(140,160,200,.16);
  --xr:#ff5470; --yg:#3ddc84; --zb:#5b8cff; --accent:#5b8cff; --panel:rgba(14,18,28,.7);
  --mono:ui-monospace,"SFMono-Regular",Menlo,Consolas,monospace; }
*{box-sizing:border-box} html,body{margin:0;height:100%}
body{background:radial-gradient(120% 90% at 50% -10%,#141924 0%,#0a0c12 55%,#06070b 100%);
  color:var(--ink);font-family:var(--mono);overflow:hidden;-webkit-font-smoothing:antialiased;
  display:flex;align-items:center;justify-content:center}
#stage{position:relative;width:min(78vh,78vw);aspect-ratio:1/1;border:1px solid var(--line);border-radius:14px;
  overflow:hidden;box-shadow:0 30px 90px rgba(0,0,0,.6)}
#vid{position:absolute;inset:0;width:100%;height:100%;object-fit:cover;filter:brightness(var(--b,.26));}
#gl{position:absolute;inset:0;width:100%;height:100%;display:block}
.tl{position:fixed;top:24px;left:28px;z-index:3}
.ey{font-size:11px;letter-spacing:.32em;color:var(--accent);text-transform:uppercase;margin:0 0 6px}
.ti{font-family:system-ui,sans-serif;font-weight:650;font-size:clamp(19px,2.3vw,27px);margin:0;letter-spacing:-.01em}
.sub{max-width:38ch;margin:9px 0 0;color:var(--dim);font-size:12.5px;line-height:1.55}
.sub b{color:var(--zb);font-weight:600}
.legend{position:fixed;bottom:26px;left:28px;z-index:3;display:flex;flex-direction:column;gap:7px}
.lg{display:flex;align-items:center;gap:9px;font-size:11.5px;color:var(--dim)}
.sw{width:22px;height:3px;border-radius:2px;box-shadow:0 0 10px currentColor}
.panel{position:fixed;bottom:26px;right:28px;z-index:3;display:flex;flex-direction:column;gap:10px;align-items:flex-end}
.row{display:flex;gap:8px;align-items:center}
button{font-family:var(--mono);font-size:11px;letter-spacing:.06em;text-transform:uppercase;color:var(--ink);
  background:var(--panel);border:1px solid var(--line);border-radius:9px;padding:8px 11px;cursor:pointer;backdrop-filter:blur(8px)}
button:hover{border-color:rgba(120,150,220,.5)}
button[aria-pressed="true"]{color:#fff;border-color:var(--accent);background:rgba(91,140,255,.16)}
button:focus-visible{outline:2px solid var(--accent);outline-offset:2px}
.lbl{font-size:10px;letter-spacing:.12em;color:var(--dim);text-transform:uppercase}
input[type=range]{appearance:none;height:4px;border-radius:3px;background:rgba(140,160,200,.25);outline:none;cursor:pointer}
input[type=range]::-webkit-slider-thumb{appearance:none;width:13px;height:13px;border-radius:50%;background:#fff;box-shadow:0 0 8px var(--accent)}
input[type=range]::-moz-range-thumb{width:13px;height:13px;border:none;border-radius:50%;background:#fff}
.transport{position:fixed;left:28px;right:28px;bottom:86px;z-index:3;display:flex;align-items:center;gap:14px}
.play{width:40px;height:40px;border-radius:50%;display:grid;place-items:center;flex:0 0 auto}
#scrub{flex:1}
.tc{font-size:11px;color:var(--dim);letter-spacing:.1em;font-variant-numeric:tabular-nums;min-width:70px;text-align:right}
@media (max-width:760px){ .sub{display:none} .transport{bottom:78px;left:16px;right:16px} .legend{left:16px} .panel{right:16px} .tl{left:16px;top:16px} }
</style>
<div id="stage"><video id="vid" muted loop playsinline></video><canvas id="gl"></canvas></div>
<div class="tl">
  <p class="ey">Oriented DensePose</p>
  <h1 class="ti">Frame field, reprojected</h1>
  <p class="sub">The predicted <b>SO(3) frame</b> at tracked surface points, drawn back onto the frames of the real clip. Each tripod follows its own body point and disappears when it turns away. Blue = outward <b>normal</b>.</p>
</div>
<div class="legend">
  <div class="lg"><span class="sw" style="background:var(--zb);color:var(--zb)"></span>normal</div>
  <div class="lg"><span class="sw" style="background:var(--xr);color:var(--xr)"></span>tangent&nbsp;X</div>
  <div class="lg"><span class="sw" style="background:var(--yg);color:var(--yg)"></span>tangent&nbsp;Y</div>
</div>
<div class="transport">
  <button id="bPlay" class="play" aria-pressed="true" aria-label="Play or pause">&#10074;&#10074;</button>
  <input id="scrub" type="range" min="0" max="1000" value="0" aria-label="Timeline">
  <span class="tc" id="tc">000 / 000</span>
</div>
<div class="panel">
  <div class="row"><button id="bTan" aria-pressed="true">Tangents</button><button id="bTwist" aria-pressed="false">Twist color</button></div>
  <div class="row"><span class="lbl">dim</span><input id="dim" type="range" min="5" max="100" value="26" style="width:110px"><span class="lbl">size</span><input id="sz" type="range" min="10" max="70" value="30" style="width:80px"></div>
</div>
<script>const DATA=__DATA__;</script>
<script>
const vid=document.getElementById("vid"),cv=document.getElementById("gl"),gl=cv.getContext("webgl2",{antialias:true,alpha:true});
vid.src="data:video/mp4;base64,"+DATA.vid;
if(!gl){document.body.insertAdjacentHTML("beforeend","<p style='position:fixed;inset:0;display:grid;place-items:center;color:#8a93a6'>WebGL2 not available.</p>");}
else{
function unb64(s){const b=atob(s),u=new Uint8Array(b.length);for(let i=0;i<b.length;i++)u[i]=b.charCodeAt(i);return u;}
const PTS=new Float32Array(unb64(DATA.ptsB).buffer);      // flat: u,v,qx,qy,qz,qw,tw
const F=DATA.F,counts=DATA.counts,off=[0];for(let i=0;i<F;i++)off.push(off[i]+counts[i]);
const MAXC=Math.max(...counts);
function prog(vs,fs){const p=gl.createProgram();for(const[t,s]of[[gl.VERTEX_SHADER,vs],[gl.FRAGMENT_SHADER,fs]]){const sh=gl.createShader(t);gl.shaderSource(sh,s);gl.compileShader(sh);if(!gl.getShaderParameter(sh,gl.COMPILE_STATUS))console.error(gl.getShaderInfoLog(sh));gl.attachShader(p,sh);}gl.linkProgram(p);return p;}
// arrow along +Z
function arrow(seg=10){const P=[],N=[],I=[],rs=.055,sh=.62,rc=.16,h=1.;
 for(let i=0;i<seg;i++){const a=i/seg*6.2831853,c=Math.cos(a),s=Math.sin(a);P.push(rs*c,rs*s,0,rs*c,rs*s,sh);N.push(c,s,0,c,s,0);}
 for(let i=0;i<seg;i++){const j=(i+1)%seg;I.push(i*2,j*2,j*2+1,i*2,j*2+1,i*2+1);}
 const b=P.length/3;for(let i=0;i<seg;i++){const a=i/seg*6.2831853,c=Math.cos(a),s=Math.sin(a),l=Math.hypot(c,s,.5);P.push(rc*c,rc*s,sh);N.push(c/l,s/l,.5/l);}
 const t=P.length/3;P.push(0,0,h);N.push(0,0,1);for(let i=0;i<seg;i++){const j=(i+1)%seg;I.push(b+i,b+j,t);}
 return{P,N,I};}
const AR=arrow(10);
const pr=prog(`#version 300 es
layout(location=0) in vec3 aPos;layout(location=1) in vec3 aNorm;
layout(location=2) in vec4 iQ;layout(location=3) in vec2 iUV;layout(location=4) in vec3 iC;layout(location=5) in float iT;
uniform float uScale;uniform int uTwist;out vec3 vN;out vec3 vC;
vec3 qr(vec4 q,vec3 v){return v+2.0*cross(q.xyz,cross(q.xyz,v)+q.w*v);}
vec3 turbo(float t){return clamp(vec3(.5+.6*sin(6.28*t),.5+.6*sin(6.28*(t+.33)),.5+.6*sin(6.28*(t-.33))),0.,1.);}
void main(){ vec3 p=qr(iQ,aPos*uScale);
 vN=normalize(qr(iQ,aNorm)); vC=uTwist==1?turbo(iT):iC;
 // orthographic, image space: u,v in [0,1] -> clip [-1,1]; y already up
 vec2 c=vec2(iUV.x*2.0-1.0, iUV.y*2.0-1.0)+p.xy*2.0;
 gl_Position=vec4(c, -p.z*0.15, 1.0); }`,
`#version 300 es
precision highp float;in vec3 vN;in vec3 vC;out vec4 o;
void main(){ vec3 L=normalize(vec3(.45,.6,.85)); float d=max(dot(normalize(vN),L),0.0);
 o=vec4(vC*(0.5+0.62*d),1.0); }`);
const vao=gl.createVertexArray();gl.bindVertexArray(vao);
function sbuf(loc,arr,n){const b=gl.createBuffer();gl.bindBuffer(gl.ARRAY_BUFFER,b);gl.bufferData(gl.ARRAY_BUFFER,new Float32Array(arr),gl.STATIC_DRAW);gl.enableVertexAttribArray(loc);gl.vertexAttribPointer(loc,n,gl.FLOAT,false,0,0);return b;}
sbuf(0,AR.P,3);sbuf(1,AR.N,3);
const ib=gl.createBuffer();gl.bindBuffer(gl.ELEMENT_ARRAY_BUFFER,ib);gl.bufferData(gl.ELEMENT_ARRAY_BUFFER,new Uint16Array(AR.I),gl.STATIC_DRAW);
const NI=MAXC*3;
const aQ=new Float32Array(NI*4),aUV=new Float32Array(NI*2),aC=new Float32Array(NI*3),aT=new Float32Array(NI);
const AXC=[[.36,.55,1],[1,.33,.44],[.24,.86,.52]];         // z(normal)=blue, x=red, y=green
const s2=Math.SQRT1_2;
const AXQ=[[0,0,0,1],[0,s2,0,s2],[-s2,0,0,s2]];            // +Z -> +Z, +X, +Y   (xyzw)
function qmul(a,b){return[a[3]*b[0]+a[0]*b[3]+a[1]*b[2]-a[2]*b[1],a[3]*b[1]-a[0]*b[2]+a[1]*b[3]+a[2]*b[0],
 a[3]*b[2]+a[0]*b[1]-a[1]*b[0]+a[2]*b[3],a[3]*b[3]-a[0]*b[0]-a[1]*b[1]-a[2]*b[2]];}
const bQ=gl.createBuffer(),bUV=gl.createBuffer(),bC=gl.createBuffer(),bT=gl.createBuffer();
function dyn(loc,b,n,bytes){gl.bindBuffer(gl.ARRAY_BUFFER,b);gl.bufferData(gl.ARRAY_BUFFER,bytes,gl.DYNAMIC_DRAW);gl.enableVertexAttribArray(loc);gl.vertexAttribPointer(loc,n,gl.FLOAT,false,0,0);gl.vertexAttribDivisor(loc,1);}
dyn(2,bQ,4,aQ.byteLength);dyn(3,bUV,2,aUV.byteLength);dyn(4,bC,3,aC.byteLength);dyn(5,bT,1,aT.byteLength);
let curCount=0;
function setFrame(f){
 const n=counts[f],o=off[f];curCount=n;
 for(let ax=0;ax<3;ax++)for(let i=0;i<n;i++){const k=ax*n+i,s=(o+i)*7;
  const q=[PTS[s+2],PTS[s+3],PTS[s+4],PTS[s+5]];const cq=qmul(q,AXQ[ax]);
  aQ[k*4]=cq[0];aQ[k*4+1]=cq[1];aQ[k*4+2]=cq[2];aQ[k*4+3]=cq[3];
  aUV[k*2]=PTS[s];aUV[k*2+1]=PTS[s+1];
  aC[k*3]=AXC[ax][0];aC[k*3+1]=AXC[ax][1];aC[k*3+2]=AXC[ax][2];aT[k]=PTS[s+6];}
 gl.bindBuffer(gl.ARRAY_BUFFER,bQ);gl.bufferSubData(gl.ARRAY_BUFFER,0,aQ);
 gl.bindBuffer(gl.ARRAY_BUFFER,bUV);gl.bufferSubData(gl.ARRAY_BUFFER,0,aUV);
 gl.bindBuffer(gl.ARRAY_BUFFER,bC);gl.bufferSubData(gl.ARRAY_BUFFER,0,aC);
 gl.bindBuffer(gl.ARRAY_BUFFER,bT);gl.bufferSubData(gl.ARRAY_BUFFER,0,aT);
}
// ---- UI ----
let showTan=true,twist=false,playing=true,shown=-1;
const $=id=>document.getElementById(id);
$("bTan").onclick=e=>{showTan=!showTan;$("bTan").setAttribute("aria-pressed",showTan);};
$("bTwist").onclick=e=>{twist=!twist;$("bTwist").setAttribute("aria-pressed",twist);};
$("bPlay").onclick=()=>{playing=!playing;playing?vid.play():vid.pause();$("bPlay").innerHTML=playing?"&#10074;&#10074;":"&#9654;";$("bPlay").setAttribute("aria-pressed",playing);};
$("dim").oninput=e=>document.documentElement.style.setProperty("--b",e.target.value/100);
const scrub=$("scrub"),tc=$("tc");
scrub.addEventListener("input",()=>{playing=false;vid.pause();$("bPlay").innerHTML="&#9654;";$("bPlay").setAttribute("aria-pressed",false);
 vid.currentTime=(scrub.value/1000)*(F/DATA.fps);});
function resize(){const r=cv.getBoundingClientRect(),d=Math.min(devicePixelRatio||1,2);
 cv.width=r.width*d;cv.height=r.height*d;gl.viewport(0,0,cv.width,cv.height);}
addEventListener("resize",resize);
const ro=new ResizeObserver(resize);ro.observe(cv);
gl.enable(gl.DEPTH_TEST);
if(matchMedia("(prefers-reduced-motion:reduce)").matches){playing=false;$("bPlay").innerHTML="&#9654;";}
vid.addEventListener("loadeddata",()=>{resize();if(playing)vid.play().catch(()=>{});});
function loop(){
 const f=Math.min(F-1,Math.max(0,Math.round(vid.currentTime*DATA.fps)));
 if(f!==shown){setFrame(f);shown=f;scrub.value=Math.round(f/(F-1)*1000);
  tc.textContent=String(f+1).padStart(3,"0")+" / "+String(F).padStart(3,"0");}
 gl.clearColor(0,0,0,0);gl.clear(gl.COLOR_BUFFER_BIT|gl.DEPTH_BUFFER_BIT);
 gl.useProgram(pr);gl.bindVertexArray(vao);gl.bindBuffer(gl.ELEMENT_ARRAY_BUFFER,ib);
 gl.uniform1f(gl.getUniformLocation(pr,"uScale"),(+$("sz").value)/1000);
 gl.uniform1i(gl.getUniformLocation(pr,"uTwist"),twist?1:0);
 const inst=(showTan&&!twist)?curCount*3:curCount;
 gl.drawElementsInstanced(gl.TRIANGLES,AR.I.length,gl.UNSIGNED_SHORT,0,inst);
 requestAnimationFrame(loop);}
resize();requestAnimationFrame(loop);
}
</script>
"""
html = HTML.replace("__DATA__", json.dumps(data, separators=(",", ":")))
open(a.out, "w").write(html)
print(f"[reproj] wrote {a.out} ({os.path.getsize(a.out)/1e6:.2f} MB)", flush=True)

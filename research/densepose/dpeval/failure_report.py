"""Failure analysis: rank COCO-minival instances by per-instance GPS (official metric),
take the bottom 5%, render GT point-targets (colored by per-point GPS) + our predicted IUV,
and emit an HTML report. GT-box eval (isolates dense-head quality)."""
import os, sys, json, html, numpy as np, cv2, torch
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
sys.path.insert(0,".."); sys.path.insert(0,".")
from model import DINOv3DPT
from data import _PAL
from pycocotools.coco import COCO
from densepose_cocoeval import denseposeCOCOeval
dev="cuda"; Z=512
CKPT="/tmp/vitl_fail.pt"; BACKBONE="vit_large_patch16_dinov3"; LAYERS=(4,11,17,23)
IMG=os.path.expanduser("~/data/coco_val2014_minival"); GT="data/densepose_coco_2014_minival.json"
OUT=os.path.expanduser("~/codes/wan/densepose/dpeval/failure_report"); os.makedirs(OUT+"/imgs",exist_ok=True)
NAMES=json.load(open(os.path.expanduser("~/data/bedlam2/dp_assets/dp24_names.json")))

m=DINOv3DPT(backbone=BACKBONE,layers=LAYERS,pretrained=False).to(dev).eval()
m.load_state_dict(torch.load(CKPT,map_location="cpu")["model"]); print("loaded",CKPT,flush=True)
coco=COCO(GT)
def crop_for(rgb,bbox,exp=1.3):
    bx,by,bw,bh=bbox; cx,cy=bx+bw/2,by+bh/2; s=int(round(exp*max(bw,bh)))
    H,W=rgb.shape[:2]; ix0,iy0=int(round(cx-s/2)),int(round(cy-s/2)); canv=np.zeros((s,s,3),np.uint8)
    ax0,ay0=max(0,-ix0),max(0,-iy0); bx0,by0=max(0,ix0),max(0,iy0); bx1,by1=min(W,ix0+s),min(H,iy0+s)
    canv[ay0:ay0+(by1-by0),ax0:ax0+(bx1-bx0)]=rgb[by0:by1,bx0:bx1]; return canv,ix0,iy0,s
@torch.no_grad()
def predict(rgb,bbox):
    canv,sx0,sy0,S=crop_for(rgb,bbox)
    img=torch.from_numpy(cv2.resize(canv,(Z,Z))).permute(2,0,1)[None].float().to(dev)/255
    with torch.autocast("cuda",dtype=torch.bfloat16): o=m(img)
    part=o["part"].float().argmax(1)[0]; uv=o["uv"].float()[0].view(24,2,Z,Z)
    sel=(part-1).clamp(0); U=uv[:,0].gather(0,sel[None])[0].clamp(0,1); V=uv[:,1].gather(0,sel[None])[0].clamp(0,1)
    part=part.cpu().numpy().astype(np.uint8); U=U.cpu().numpy(); V=V.cpu().numpy()
    bx,by,bw,bh=bbox; Wt,Ht=int(bw)+1,int(bh)+1
    pS=cv2.resize(part,(S,S),interpolation=cv2.INTER_NEAREST); US=cv2.resize(U,(S,S)); VS=cv2.resize(V,(S,S))
    ox,oy=int(round(bx-sx0)),int(round(by-sy0)); ox=min(max(ox,0),S-Wt); oy=min(max(oy,0),S-Ht)
    I=pS[oy:oy+Ht,ox:ox+Wt]; uvarr=np.stack([I,(US[oy:oy+Ht,ox:ox+Wt]*255).astype(np.uint8),(VS[oy:oy+Ht,ox:ox+Wt]*255).astype(np.uint8)],0)
    return uvarr,(sx0,sy0,S)

# build predictions per dp-annotated person instance
items=[]
for ii in coco.getImgIds():
    info=coco.loadImgs(ii)[0]; p=os.path.join(IMG,info["file_name"])
    if not os.path.exists(p): continue
    anns=[a for a in coco.loadAnns(coco.getAnnIds(imgIds=ii)) if "dp_x" in a]
    if not anns: continue
    rgb=cv2.imread(p)[...,::-1].copy()
    for a in anns:
        uvarr,crop=predict(rgb,a["bbox"])
        items.append({"image_id":ii,"file":info["file_name"],"bbox":a["bbox"],"uv":uvarr,"crop":crop,"gt":a})
print("predicted",len(items),"instances",flush=True)
results=[{"image_id":it["image_id"],"category_id":1,"bbox":it["bbox"],"score":1.0,"uv":it["uv"]} for it in items]
ev=denseposeCOCOeval(coco,coco.loadRes(results),iouType='uv')
ev._loadGEval()   # sets Part_UVs/Part_ClosestVertInds/Pdist_matrix/Mean_Distances/CoarseParts (normally done in _prepare)

def per_point_gps(gt,uv,bbox):
    g_=bbox; dt_bbox=bbox
    dp_x=np.array(gt['dp_x'])*g_[2]/255.; dp_y=np.array(gt['dp_y'])*g_[3]/255.
    px=(dp_y+g_[1]-dt_bbox[1]).astype(int); py=(dp_x+g_[0]-dt_bbox[0]).astype(int)
    dx,dy=uv.shape[1],uv.shape[2]
    valid=(px>=0)&(py>=0)&(px<dx)&(py<dy)
    pxc=np.clip(px,0,dx-1); pyc=np.clip(py,0,dy-1)
    ip=uv[0,pxc,pyc].astype(float); up=uv[1,pxc,pyc]/255.; vp=uv[2,pxc,pyc]/255.; ip[~valid]=0
    cV,cVGT=ev.findAllClosestVerts(gt,up,vp,ip)
    dist=ev.getDistances(cVGT,cV)              # over cVGT>0, in order
    gtmask=cVGT>0
    MeanD=ev.Mean_Distances[ev.CoarseParts[ev.Part_ids[cVGT[gtmask].astype(int)-1]]]
    og=np.exp(-(dist**2)/(2*(MeanD**2))) if len(dist) else np.array([])
    gps=float(np.sum(og)/len(dist)) if len(dist) else 0.0
    # map ogps back to original points (gtmask order)
    pp=np.full(len(dp_x),np.nan); pp[np.where(gtmask)[0]]=og
    return gps,pp,(dp_x,dp_y,np.array(gt['dp_I']))

for it in items:
    it["gps"],it["pp"],it["pts"]=per_point_gps(it["gt"],it["uv"],it["bbox"])
items.sort(key=lambda x:x["gps"])
k=max(1,int(0.05*len(items))); worst=items[:k]
print(f"bottom 5% = {k} instances, GPS range {worst[0]['gps']:.3f}..{worst[-1]['gps']:.3f}",flush=True)

# aggregate: mean per-point GPS by GT part across the worst set (systematic failure parts)
part_acc={}
for it in worst:
    dx,dy,I=it["pts"]
    for i,og in zip(I,it["pp"]):
        if not np.isnan(og): part_acc.setdefault(int(i),[]).append(og)
part_summary=sorted(((p,np.mean(v),len(v)) for p,v in part_acc.items()),key=lambda x:x[1])

def render(it,idx):
    p=os.path.join(IMG,it["file"]); rgb=cv2.imread(p)[...,::-1]
    sx0,sy0,S=it["crop"]; H,W=rgb.shape[:2]
    x0,y0=max(0,sx0),max(0,sy0); x1,y1=min(W,sx0+S),min(H,sy0+S)
    crop=rgb[y0:y1,x0:x1]
    dx,dy,I=it["pts"]; bx,by,bw,bh=it["bbox"]
    ix=bx+dx-x0; iy=by+dy-y0                      # gt point coords in crop
    fig,ax=plt.subplots(1,2,figsize=(11,5.5))
    ax[0].imshow(crop); ax[0].set_title(f"GT points (color=per-point GPS)  mean GPS={it['gps']:.3f}",fontsize=9)
    sc=ax[0].scatter(ix,iy,c=np.nan_to_num(it["pp"]),cmap="RdYlGn",vmin=0,vmax=1,s=22,edgecolors='k',linewidths=0.3)
    ax[0].axis("off"); fig.colorbar(sc,ax=ax[0],fraction=0.046)
    # our predicted part overlay
    pal=_PAL; uvarr=it["uv"]; ov=crop.copy()
    pbx,pby=int(round(bx-x0)),int(round(by-y0)); Ht,Wt=uvarr.shape[1],uvarr.shape[2]
    sub=ov[pby:pby+Ht,pbx:pbx+Wt]
    if sub.shape[:2]==(Ht,Wt):
        Ipart=uvarr[0]; fg=Ipart>0; col=pal[Ipart]
        sub[fg]=(0.5*sub[fg]+0.5*col[fg]).astype(np.uint8); ov[pby:pby+Ht,pbx:pbx+Wt]=sub
    ax[1].imshow(ov); ax[1].set_title("our predicted parts (overlay)",fontsize=9); ax[1].axis("off")
    fig.tight_layout(); fn=f"imgs/w{idx:03d}.png"; fig.savefig(OUT+"/"+fn,dpi=80); plt.close(fig)
    return fn

rows=[]
for idx,it in enumerate(worst):
    fn=render(it,idx)
    nbad=int(np.sum(np.nan_to_num(it["pp"])<0.5))
    rows.append((it,fn,nbad))
# HTML
H=['<html><head><meta charset=utf-8><style>body{font-family:sans-serif;background:#111;color:#ddd}img{max-width:900px}'
   '.it{border-bottom:1px solid #333;padding:10px}.s{color:#f88}</style></head><body>']
H.append(f"<h2>DensePose COCO-minival — worst {k} instances ({100*k/len(items):.0f}%) by GPS</h2>")
H.append(f"<p>model: ViT-L @512 (step 65k), GT-box. {len(items)} instances total. worst GPS {worst[0]['gps']:.3f}.</p>")
H.append("<h3>Systematic failure: mean per-point GPS by GT part (worst-set)</h3><table border=1 cellpadding=4><tr><th>part</th><th>mean GPS</th><th>#pts</th></tr>")
for pid,mg,n in part_summary[:12]:
    H.append(f"<tr><td>{pid}: {html.escape(NAMES.get(str(pid),'?'))}</td><td>{mg:.3f}</td><td>{n}</td></tr>")
H.append("</table><hr>")
for it,fn,nbad in rows:
    H.append(f"<div class=it><span class=s>GPS={it['gps']:.3f}</span> &nbsp; img {it['image_id']} ({html.escape(it['file'])}) "
             f"&nbsp; pts={len(it['pts'][0])} bad(&lt;0.5)={nbad}<br><img src='{fn}'></div>")
H.append("</body></html>")
open(OUT+"/index.html","w").write("\n".join(H))
print("wrote",OUT+"/index.html","with",len(rows),"instances",flush=True)

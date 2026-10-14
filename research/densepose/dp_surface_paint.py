import os, sys, glob, numpy as np, cv2, torch
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from scipy.spatial import cKDTree
from scipy.interpolate import LinearNDInterpolator
sys.path.insert(0,".")
from model import DINOv3DPT
import smplx
from torchvision.models.detection import fasterrcnn_resnet50_fpn, FasterRCNN_ResNet50_FPN_Weights
dev="cuda"; Z=384; A=os.path.expanduser("~/data/bedlam2/dp_assets")
# SMPL-X template + per-vertex part/UV
sm=smplx.create(os.path.expanduser("~/data/smplx/models_lockedhead"),model_type="smplx",gender="neutral",num_betas=16,use_pca=False); V=sm.v_template.detach().cpu().numpy().astype(np.float32)  # (10475,3)
vpart=np.load(f"{A}/smplx_vert_to_dp24.npy").astype(int); vuv=np.load(f"{A}/smplx_vert_uv.npy")
trees={p: (np.where(vpart==p)[0], cKDTree(vuv[vpart==p])) for p in range(1,25) if (vpart==p).any()}
# per-part barycentric interpolator: UV -> continuous template 3D (Delaunay over part vertices)
interp={p: LinearNDInterpolator(vuv[vpart==p], V[vpart==p]) for p in trees if (vpart==p).sum()>=4}
# models
det=fasterrcnn_resnet50_fpn(weights=FasterRCNN_ResNet50_FPN_Weights.DEFAULT).eval().to(dev)
m=DINOv3DPT(pretrained=False).to(dev).eval()
m.load_state_dict(torch.load(os.path.expanduser("~/data/bedlam2/dp_runs/dp_v1/final.pt"),map_location="cpu")["model"])
def largest_person(rgb):
    t=torch.from_numpy(rgb).permute(2,0,1).float().to(dev)/255
    with torch.no_grad(): o=det([t])[0]
    bs=[b.cpu().numpy() for b,l,s in zip(o["boxes"],o["labels"],o["scores"]) if int(l)==1 and float(s)>0.9]
    return max(bs,key=lambda b:(b[2]-b[0])*(b[3]-b[1])) if bs else None
def crop_square(rgb,box,exp=1.3):
    x0,y0,x1,y1=box; cx,cy=(x0+x1)/2,(y0+y1)/2; s=int(exp*max(x1-x0,y1-y0))
    H,W=rgb.shape[:2]; sx0,sy0=int(cx-s/2),int(cy-s/2); canv=np.zeros((s,s,3),np.uint8)
    ax0,ay0=max(0,-sx0),max(0,-sy0); bx0,by0=max(0,sx0),max(0,sy0); bx1,by1=min(W,sx0+s),min(H,sy0+s)
    canv[ay0:ay0+(by1-by0),ax0:ax0+(bx1-bx0)]=rgb[by0:by1,bx0:bx1]; return cv2.resize(canv,(Z,Z))
def infer(crop):
    img=torch.from_numpy(crop).permute(2,0,1)[None].float().to(dev)/255
    with torch.no_grad(), torch.autocast("cuda",dtype=torch.bfloat16): o=m(img)
    return o["part"].float().argmax(1)[0].cpu().numpy(), o["uv"].float()[0].cpu().numpy().reshape(24,2,Z,Z)
def pixel_cloud(crop,part,uv):
    pts=[]; cols=[]
    for p,(vidx,tree) in trees.items():
        sel=part==p
        if not sel.any(): continue
        q=np.stack([uv[p-1,0][sel], uv[p-1,1][sel]],1)
        rgb=crop[sel].astype(float)/255
        xyz=interp[p](q) if p in interp else np.full((len(q),3),np.nan)   # continuous barycentric 3D
        bad=np.isnan(xyz[:,0])
        if bad.any(): _,nn=tree.query(q[bad]); xyz[bad]=V[vidx[nn]]        # fallback outside UV hull
        pts.append(xyz); cols.append(rgb)
    return np.concatenate(pts), np.concatenate(cols)
def render(ax,pts,cols,azim,title):
    ax.scatter(V[:,0],V[:,2],V[:,1],s=0.5,c='0.85',alpha=0.06,linewidths=0)   # faint template for shape
    ax.scatter(pts[:,0],pts[:,2],pts[:,1],s=2,c=np.clip(cols,0,1),linewidths=0,alpha=0.45)
    ax.set_box_aspect((V[:,0].ptp(),V[:,2].ptp(),V[:,1].ptp())); ax.view_init(elev=8,azim=azim)
    ax.set_title(title); ax.set_axis_off()
# rank by coverage, paint top 2
cands=[]
for p in sorted(glob.glob("/tmp/coco/*.jpg")):
    rgb=cv2.imread(p)[...,::-1].copy(); b=largest_person(rgb)
    if b is None: continue
    crop=crop_square(rgb,b); part,uv=infer(crop); cands.append((np.unique(part[part>0]).size,p,crop,part,uv))
cands.sort(key=lambda c:-c[0])
for rank,(nc,p,crop,part,uv) in enumerate(cands[:2]):
    pts,cols=pixel_cloud(crop,part,uv)
    fig=plt.figure(figsize=(14,6))
    axi=fig.add_subplot(1,3,1); axi.imshow(crop); axi.axis("off"); axi.set_title(f"input ({os.path.basename(p)})")
    render(fig.add_subplot(1,3,2,projection='3d'),pts,cols,-90,"template — front")
    render(fig.add_subplot(1,3,3,projection='3d'),pts,cols, 90,"template — back")
    fig.suptitle(f"every fg pixel on SMPL-X template via BARYCENTRIC (part,U,V) | {len(pts)} pixels (continuous)",fontsize=12)
    out=f"/home/u6ga/raguler.u6ga/codes/wan/vfi_bench/dp_barycloud_{rank}.png"
    fig.tight_layout(); fig.savefig(out,dpi=95); plt.close(fig); print("wrote",out)

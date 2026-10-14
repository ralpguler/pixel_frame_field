import os, sys, glob, json, numpy as np, cv2, torch
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
sys.path.insert(0,".")
from model import DINOv3DPT
from torchvision.models.detection import fasterrcnn_resnet50_fpn, FasterRCNN_ResNet50_FPN_Weights
dev="cuda"; Z=384
NAMES=json.load(open(os.path.expanduser("~/data/bedlam2/dp_assets/dp24_names.json")))
det=fasterrcnn_resnet50_fpn(weights=FasterRCNN_ResNet50_FPN_Weights.DEFAULT).eval().to(dev)
m=DINOv3DPT(pretrained=False).to(dev).eval()
m.load_state_dict(torch.load(os.path.expanduser("~/data/bedlam2/dp_runs/dp_v1/final.pt"),map_location="cpu")["model"])
yy,xx=np.mgrid[0:Z,0:Z]
poscol=np.stack([xx/Z, yy/Z, 0.18*np.ones((Z,Z))],-1)   # vivid: R=x, G=y, low B
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
cands=[]
for p in sorted(glob.glob("/tmp/coco/*.jpg")):
    rgb=cv2.imread(p)[...,::-1].copy(); b=largest_person(rgb)
    if b is None: continue
    crop=crop_square(rgb,b); part,uv=infer(crop)
    cands.append((np.unique(part[part>0]).size, p, crop, part, uv))
cands.sort(key=lambda c:-c[0])
for rank,(ncov,p,crop,part,uv) in enumerate(cands[:3]):
    # well-covered parts only, up to 12, ordered by pixel count
    counts=[(pid,(part==pid).sum()) for pid in range(1,25)]
    parts=[pid for pid,c in sorted(counts,key=lambda x:-x[1]) if c>150][:12]
    ncol=4; nrow=int(np.ceil(len(parts)/ncol))
    fig=plt.figure(figsize=(4*ncol+5, 3.2*max(nrow,2)))
    gs=fig.add_gridspec(max(nrow,2), ncol+2)
    ax=fig.add_subplot(gs[0,0:2]); ax.imshow(crop); ax.set_title("input crop"); ax.axis("off")
    fg=part>0; key=crop.copy()/255.0; key[fg]=poscol[fg]
    ax2=fig.add_subplot(gs[1,0:2]); ax2.imshow(key); ax2.set_title("image-space colour key  (R=left→right, G=top→bottom)"); ax2.axis("off")
    for j,pid in enumerate(parts):
        r,c=divmod(j,ncol); axp=fig.add_subplot(gs[r,2+c])
        sel=part==pid; U=uv[pid-1,0][sel]; V=uv[pid-1,1][sel]; col=np.clip(poscol[sel],0,1)
        n=sel.sum(); idx=np.random.choice(n,min(n,5000),replace=False)
        axp.scatter(U[idx],V[idx],s=14,c=col[idx],marker='.',linewidths=0,alpha=0.55)
        axp.set_xlim(-0.02,1.02); axp.set_ylim(-0.02,1.02); axp.invert_yaxis()
        axp.set_xticks([0,1]); axp.set_yticks([0,1]); axp.tick_params(labelsize=6)
        axp.set_title(f"{pid}: {NAMES[str(pid)]}  (n={int(n//1000)}k)",fontsize=9)
    fig.suptitle(f"image→UV scatter per part, coloured by image position | {os.path.basename(p)} | smooth gradient = good mapping",fontsize=12)
    out=f"/home/u6ga/raguler.u6ga/codes/wan/vfi_bench/dp_uvscatter_{rank}.png"
    fig.tight_layout(); fig.savefig(out,dpi=95); plt.close(fig); print("wrote",out,"parts",len(parts))

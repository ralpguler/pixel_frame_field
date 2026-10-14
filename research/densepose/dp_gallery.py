"""Qualitative DensePose gallery on real images: detect persons -> crop (val-style 1.3x square)
-> run trained model -> [input | part(I) | UV] rows."""
import sys, glob, numpy as np, cv2, torch
import torch.nn.functional as F
sys.path.insert(0, ".")
from model import DINOv3DPT, N_CLASSES, N_PARTS
from data import _PAL
from torchvision.models.detection import fasterrcnn_resnet50_fpn, FasterRCNN_ResNet50_FPN_Weights
dev="cuda"; Z=384
det=fasterrcnn_resnet50_fpn(weights=FasterRCNN_ResNet50_FPN_Weights.DEFAULT).eval().to(dev)
m=DINOv3DPT(pretrained=False).to(dev).eval()
sd=torch.load("/home/u6ga/raguler.u6ga/data/bedlam2/dp_runs/dp_v1/final.pt",map_location="cpu")
m.load_state_dict(sd["model"]); print("loaded final.pt step",sd.get("step"))
pal=torch.from_numpy(_PAL)
def person_boxes(rgb):
    t=torch.from_numpy(rgb).permute(2,0,1).float().to(dev)/255
    with torch.no_grad(): o=det([t])[0]
    out=[]
    for b,l,s in zip(o["boxes"],o["labels"],o["scores"]):
        if int(l)==1 and float(s)>0.90: out.append(b.cpu().numpy())
    return out
def crop_square(rgb,box,exp=1.3):
    x0,y0,x1,y1=box; cx,cy=(x0+x1)/2,(y0+y1)/2; side=exp*max(x1-x0,y1-y0)
    H,W=rgb.shape[:2]; sx0,sy0=int(cx-side/2),int(cy-side/2); s=int(side)
    canvas=np.zeros((s,s,3),np.uint8)
    ax0,ay0=max(0,-sx0),max(0,-sy0); bx0,by0=max(0,sx0),max(0,sy0)
    bx1,by1=min(W,sx0+s),min(H,sy0+s); 
    canvas[ay0:ay0+(by1-by0),ax0:ax0+(bx1-bx0)]=rgb[by0:by1,bx0:bx1]
    return cv2.resize(canvas,(Z,Z))
def infer(crop):
    img=torch.from_numpy(crop).permute(2,0,1)[None].float().to(dev)/255
    with torch.no_grad(), torch.autocast("cuda",dtype=torch.bfloat16):
        o=m(img)
    part=o["part"].float().argmax(1)[0].cpu()           # (Z,Z)
    uv=o["uv"].float()[0].cpu()                          # (48,Z,Z)
    rgb=torch.from_numpy(crop)
    fg=part>0
    pov=torch.where(fg[...,None],pal[part],rgb)
    sel=(part-1).clamp(0); u=uv.view(24,2,Z,Z)
    gu=u.gather(0,sel.view(1,1,Z,Z).expand(1,2,Z,Z)).squeeze(0)
    uvimg=torch.zeros_like(rgb)
    uvimg[...,0]=(gu[0].clamp(0,1)*255).byte(); uvimg[...,1]=(gu[1].clamp(0,1)*255).byte()
    uvimg[...,2]=torch.where(fg,torch.tensor(128,dtype=torch.uint8),torch.tensor(0,dtype=torch.uint8))
    uvimg=torch.where(fg[...,None],uvimg,rgb)
    return torch.cat([rgb,pov,uvimg],1).numpy()          # (Z,3Z,3)
rows=[]
for p in sorted(glob.glob("/tmp/coco/*.jpg")):
    rgb=cv2.imread(p)[...,::-1].copy()
    bs=person_boxes(rgb)
    if not bs: continue
    bs=sorted(bs,key=lambda b:-(b[2]-b[0])*(b[3]-b[1]))   # largest person
    rows.append(infer(crop_square(rgb,bs[0])))
    if len(rows)>=8: break
gal=np.concatenate(rows,0)
cv2.imwrite("/home/u6ga/raguler.u6ga/codes/wan/vfi_bench/dp_gallery_coco.png", gal[...,::-1])
print("gallery rows:",len(rows),"-> dp_gallery_coco.png", gal.shape)

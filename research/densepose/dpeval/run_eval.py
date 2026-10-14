"""Evaluate trained DensePose model on COCO minival using the official GPS eval.
GT-box eval: each densepose-annotated person GT box is a score-1 detection; model
predicts dense IUV on the val-style crop -> encoded as dt['uv']=(3,h,w) [I, U*255, V*255]."""
import os, sys, argparse, numpy as np, cv2, torch
sys.path.insert(0,"..")                      # for model.py, data.py
sys.path.insert(0,".")                       # for densepose_cocoeval
from model import DINOv3DPT, part_pred, backbone_from_state_dict, default_layers
from pycocotools.coco import COCO
from densepose_cocoeval import denseposeCOCOeval
ap=argparse.ArgumentParser(); ap.add_argument("--limit",type=int,default=0)
ap.add_argument("--ckpt",default=os.path.expanduser("~/data/bedlam2/dp_runs/dp_v1/final.pt"))
ap.add_argument("--backbone",default="vit_base_patch16_dinov3")
ap.add_argument("--layers",default="2,5,8,11")
ap.add_argument("--input",type=int,default=384)
ap.add_argument("--detector",action="store_true")   # use fasterrcnn person boxes (fair/leaderboard); default = GT boxes
ap.add_argument("--det-thr",type=float,default=0.5)
ap.add_argument("--boxes",default="")               # precomputed detections json {image_id:[[x,y,w,h,score],...]} (e.g. detectron2)
ap.add_argument("--uv-remap",default="")            # (25,K,K,2) v2->v1 chart remap for face_uv_v2-trained ckpts
a=ap.parse_args()
REMAP=None
if a.uv_remap:
    REMAP=torch.from_numpy(np.load(os.path.expanduser(a.uv_remap)).astype(np.float32))
    print(f"uv remap active: {a.uv_remap} {tuple(REMAP.shape)} (COCO GT is v1-chart; remapping parts 3-6)",flush=True)
BOXES=None
if a.boxes:
    import json as _json
    BOXES={int(k):v for k,v in _json.load(open(a.boxes)).items()}
    print(f"using precomputed boxes from {a.boxes} ({len(BOXES)} images)",flush=True)
dev="cuda"; Z=a.input
IMG=os.path.expanduser("~/data/coco_val2014_minival")
GT="data/densepose_coco_2014_minival.json"
_sd=torch.load(a.ckpt,map_location="cpu")["model"]
_dec=any(k.startswith("mask_head") for k in _sd)        # auto-detect decoupled vs legacy 25-way head
_bb=backbone_from_state_dict(_sd)                      # V-JEPA 2.1 ckpts carry bb.img_mod_embed
if _bb.startswith("vjepa2_1"): a.backbone, a.layers = _bb, ",".join(map(str, default_layers(_bb)))
m=DINOv3DPT(backbone=a.backbone, layers=tuple(int(x) for x in a.layers.split(",")), pretrained=False, decoupled=_dec).to(dev).eval()
m.load_state_dict(_sd, strict=False)   # strict=False: tolerate an orient ckpt's extra rot_head (unused for IUV/GPS)
print(f"eval ckpt={a.ckpt} backbone={a.backbone} Z={Z} decoupled={_dec}",flush=True)
coco=COCO(GT)
img_ids=coco.getImgIds()
if a.limit: img_ids=img_ids[:a.limit]
@torch.no_grad()
def infer_maps(crop):
    img=torch.from_numpy(crop).permute(2,0,1)[None].float().to(dev)/255
    with torch.autocast("cuda",dtype=torch.bfloat16): o=m(img)
    part=part_pred(o)[0]                                      # (Z,Z) 0..24 (handles both heads)
    uv=o["uv"].float()[0].view(24,2,Z,Z)
    sel=(part-1).clamp(0)
    U=uv[:,0].gather(0,sel[None])[0].clamp(0,1); V=uv[:,1].gather(0,sel[None])[0].clamp(0,1)  # (Z,Z) in [0,1]
    if REMAP is not None:
        pc,Uc,Vc=part.cpu().numpy(),U.cpu().numpy(),V.cpu().numpy()
        K=REMAP.shape[1]
        for pp in (3,4,5,6):
            mm=pc==pp
            if mm.any():
                iu=np.clip((Uc[mm]*K).astype(int),0,K-1); iv=np.clip((Vc[mm]*K).astype(int),0,K-1)
                r=REMAP[pp].numpy()[iv,iu]
                Uc[mm]=r[:,0]; Vc[mm]=r[:,1]
        return pc.astype(np.uint8), np.clip(Uc,0,1), np.clip(Vc,0,1)
    return part.cpu().numpy().astype(np.uint8), U.cpu().numpy(), V.cpu().numpy()
DET=None
if a.detector:
    from torchvision.models.detection import fasterrcnn_resnet50_fpn_v2, FasterRCNN_ResNet50_FPN_V2_Weights
    DET=fasterrcnn_resnet50_fpn_v2(weights=FasterRCNN_ResNet50_FPN_V2_Weights.DEFAULT).eval().to(dev)
    print("DETECTOR mode: fasterrcnn_resnet50_fpn_V2 person boxes",flush=True)
@torch.no_grad()
def boxes_for(ii,rgb):
    if BOXES is not None:   # precomputed detections (e.g. detectron2): [[x,y,w,h,score],...]
        return [([d[0],d[1],d[2],d[3]],float(d[4])) for d in BOXES.get(ii,[])]
    if DET is None:   # GT-box: each densepose GT person box, score 1
        return [(an["bbox"],1.0) for an in coco.loadAnns(coco.getAnnIds(imgIds=ii)) if "dp_x" in an]
    t=torch.from_numpy(rgb).permute(2,0,1).float().to(dev)/255
    o=DET([t])[0]; out=[]
    for b,l,s in zip(o["boxes"],o["labels"],o["scores"]):
        if int(l)==1 and float(s)>a.det_thr:
            x0,y0,x1,y1=b.tolist(); out.append(([x0,y0,x1-x0,y1-y0],float(s)))
    return out
results=[]; n_inst=0
for ii in img_ids:
    info=coco.loadImgs(ii)[0]; p=os.path.join(IMG,info["file_name"])
    if not os.path.exists(p): continue
    rgb=cv2.imread(p)[...,::-1].copy(); H,W=rgb.shape[:2]
    bxs=boxes_for(ii,rgb)
    if not bxs: continue
    for bbox,score in bxs:
        bx,by,bw,bh=bbox
        cx,cy=bx+bw/2,by+bh/2; side=1.3*max(bw,bh)
        sx0,sy0=cx-side/2,cy-side/2; S=int(round(side))
        canv=np.zeros((S,S,3),np.uint8)
        ix0,iy0=int(round(sx0)),int(round(sy0))
        ax0,ay0=max(0,-ix0),max(0,-iy0); bx0,by0=max(0,ix0),max(0,iy0)
        bx1,by1=min(W,ix0+S),min(H,iy0+S)
        canv[ay0:ay0+(by1-by0),ax0:ax0+(bx1-bx0)]=rgb[by0:by1,bx0:bx1]
        crop=cv2.resize(canv,(Z,Z))
        part,U,V=infer_maps(crop)
        # square-grid maps at image px, then crop the tight bbox window
        partS=cv2.resize(part,(S,S),interpolation=cv2.INTER_NEAREST)
        US=cv2.resize(U,(S,S)); VS=cv2.resize(V,(S,S))
        ox,oy=int(round(bx-sx0)),int(round(by-sy0)); Wt,Ht=int(bw)+1,int(bh)+1
        ox=min(max(ox,0),S-Wt); oy=min(max(oy,0),S-Ht)
        I=partS[oy:oy+Ht,ox:ox+Wt]; Uu=(US[oy:oy+Ht,ox:ox+Wt]*255); Vv=(VS[oy:oy+Ht,ox:ox+Wt]*255)
        uv=np.stack([I.astype(np.uint8),Uu.astype(np.uint8),Vv.astype(np.uint8)],0)  # (3,Ht,Wt)
        results.append({"image_id":ii,"category_id":1,"bbox":[bx,by,bw,bh],"score":score,"uv":uv})
        n_inst+=1
print(f"built {n_inst} predictions over {len(set(r['image_id'] for r in results))} images",flush=True)
cocoDt=coco.loadRes(results)
ev=denseposeCOCOeval(coco,cocoDt,iouType='uv'); ev.params.imgIds=img_ids
ev.evaluate(); ev.accumulate(); ev.summarize()

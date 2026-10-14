"""Rebuild the two evidence figures as WIDE strips: more legible per unit of page height,
so they can run as full-width figure* without costing the vertical space a tall block does."""
import os, cv2, numpy as np
B = os.path.expanduser("~/data/dp_viz_compare/beauty"); RA = f"{B}/report_assets"
OUT = os.path.expanduser("~/codes/wan/paper/figs"); CLIP = "15731089"
NPZ = f"{B}/{CLIP}_cscC/f_0075.npz"
m = np.load(NPZ)["mask"].astype(bool)

def crop(im, pad=0.03):
    ys, xs = np.where(m); H, W = im.shape[:2]
    sy, sx = H/m.shape[0], W/m.shape[1]
    y0,y1,x0,x1 = int(ys.min()*sy),int(ys.max()*sy),int(xs.min()*sx),int(xs.max()*sx)
    py,px = int((y1-y0)*pad), int((x1-x0)*pad)
    return im[max(0,y0-py):min(H,y1+py), max(0,x0-px):min(W,x1+px)]

def strip(ims, h, gap=8):
    out=[]
    for im in ims:
        s=h/im.shape[0]; out.append(cv2.resize(im,(int(im.shape[1]*s),h),interpolation=cv2.INTER_AREA))
    parts=[]
    for i,im in enumerate(out):
        parts.append(im)
        if i<len(out)-1: parts.append(np.full((h,gap,3),255,np.uint8))
    return np.concatenate(parts,1)

# ---- framefield: 1x6 horizontal
tags=["satin_0075","sequin_f0075","chrome_0075","diffract_f0075","fur_0075","objects_0075"]
ims=[]
for t in tags:
    p=f"{RA}/{CLIP}_{t}.jpg"
    im=cv2.imread(p if os.path.exists(p) else f"{RA}/15153663_{t}.jpg")
    ims.append(crop(im))
s=strip(ims, 620)
cv2.imwrite(f"{OUT}/fig_framefield.png", s, [cv2.IMWRITE_PNG_COMPRESSION,6])
print("framefield", s.shape, f"ar={s.shape[1]/s.shape[0]:.2f}")

# ---- garment: 3 rows -> transpose to a wide 3-row block is still tall; instead keep 3 rows
src=cv2.imread(os.path.expanduser("~/data/mcld/report_figs/qualitative.jpg"))
rh=src.shape[0]//5
keep=[0,1,4]                     # two largest gains + the worst regression
g=np.concatenate([src[i*rh:(i+1)*rh] for i in keep],0)
cv2.imwrite(f"{OUT}/fig_garment.png", g, [cv2.IMWRITE_PNG_COMPRESSION,6])
print("garment", g.shape, f"ar={g.shape[1]/g.shape[0]:.2f}")

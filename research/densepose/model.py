"""Top-down DensePose predictor: DINOv3 ViT-B/16 backbone + compact DPT decoder
+ two heads (part I: 25 classes incl. bg; UV: 24x2 per-part regressors).

forward(img[B,3,H,W] in [0,1]) -> {"part": [B,25,H,W], "uv": [B,48,H,W] in [0,1]}
At a pixel, the UV loss/readout uses the 2 channels of the predicted/GT part.
"""
import math

import os, sys
import timm
import torch
import torch.nn as nn
import torch.nn.functional as F

N_PARTS = 24
N_CLASSES = N_PARTS + 1            # + background
ROT_BAND_W = 0.5   # weight of the dilated-band rotation term relative to the own-part term


class ResConv(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.c1 = nn.Conv2d(dim, dim, 3, 1, 1)
        self.c2 = nn.Conv2d(dim, dim, 3, 1, 1)

    def forward(self, x):
        y = self.c2(F.relu(self.c1(F.relu(x))))
        return x + y


class Reassemble(nn.Module):
    """Project a token grid to `dim` and resample to `scale`x the grid size."""
    def __init__(self, in_dim, dim, scale):
        super().__init__()
        self.proj = nn.Conv2d(in_dim, dim, 1)
        self.scale = scale

    def forward(self, x):
        x = self.proj(x)
        if self.scale != 1:
            x = F.interpolate(x, scale_factor=self.scale, mode="bilinear",
                              align_corners=False)
        return x


class Fusion(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.skip = ResConv(dim)
        self.out = ResConv(dim)

    def forward(self, x, skip=None):
        if skip is not None:
            x = x + self.skip(skip)
        x = self.out(x)
        return F.interpolate(x, scale_factor=2, mode="bilinear", align_corners=False)


VJEPA_REPO = os.path.expanduser("~/codes/vjepa2")               # github.com/facebookresearch/vjepa2 (MIT)
VJEPA_CKPTS = {"vjepa2_1_vit_large": ("vit_large", "~/data/ckpts/vjepa2/vjepa2_1_vitl_dist_vitG_384.pt"),
               "vjepa2_1_vit_base": ("vit_base", "~/data/ckpts/vjepa2/vjepa2_1_vitb_dist_vitG_384.pt")}


class VJEPA21Backbone(nn.Module):
    """V-JEPA 2.1 encoder behind the slice of timm's ViT API that DINOv3DPT uses (.blocks, .patch_embed,
    .embed_dim, forward_intermediates). Images go through 2.1's IMAGE patchifier (16x16, T=1) plus its learned
    image-modality token: the input must carry an explicit singleton time axis (B,3,1,H,W), otherwise the encoder
    silently takes the video path. RoPE with interpolate_rope maps any token grid onto the pretrained range and
    handle_nonsquare_inputs (default True) passes the real grid -- verified at 512 px (32x32). Intermediate taps
    must be the encoder's hierarchical layers (ViT-L: 5, 11, 17, 23); each has its own trained norm (norms_block).
    Weights: the EMA encoder, as in the official hub (checkpoint_key="ema_encoder"). The encoder's modules are
    re-exposed as OUR attributes so parameter names read bb.blocks.N / bb.patch_embed... (LLRD, freezing);
    the unused video patchifier and video-modality token are frozen."""
    def __init__(self, name="vjepa2_1_vit_large", pretrained=True, layers=(5, 11, 17, 23)):
        super().__init__()
        if VJEPA_REPO not in sys.path:
            sys.path.insert(0, VJEPA_REPO)
        from app.vjepa_2_1.models import vision_transformer as V
        arch, ck = VJEPA_CKPTS[name]
        enc = V.__dict__[arch](patch_size=16, img_size=(384, 384), num_frames=64, tubelet_size=2, use_sdpa=True,
                               use_SiLU=False, wide_SiLU=True, uniform_power=False, use_rope=True,
                               img_temporal_dim_size=1, interpolate_rope=True, out_layers=list(layers))
        assert all(l in enc.hierarchical_layers for l in layers), f"V-JEPA taps must be in {enc.hierarchical_layers}"
        if pretrained:
            sd = torch.load(os.path.expanduser(ck), map_location="cpu", mmap=True)["ema_encoder"]
            enc.load_state_dict({k.replace("module.", "").replace("backbone.", ""): v for k, v in sd.items()}, strict=True)
        self.embed_dim = enc.embed_dim
        self.blocks = enc.blocks
        self.patch_embed = enc.patch_embed_img            # the image patchifier we actually use
        self.img_mod_embed = enc.img_mod_embed
        self.norms_block = enc.norms_block
        self.patch_embed_video = enc.patch_embed          # unused (video path): frozen
        self.video_mod_embed = enc.video_mod_embed
        for p in list(self.patch_embed_video.parameters()) + [self.video_mod_embed]:
            p.requires_grad_(False)
        self._enc = [enc]                                 # unregistered handle: same module objects as above

    def forward_intermediates(self, x, indices=None, norm=True, output_fmt="NCHW", intermediates_only=True):
        assert list(indices) == list(self._enc[0].out_layers) and output_fmt == "NCHW" and intermediates_only
        B, _, H, W = x.shape
        outs = self._enc[0](x.unsqueeze(2))               # (B,3,T=1,H,W): image patchifier + image-modality token
        return [o.transpose(1, 2).reshape(B, self.embed_dim, H // 16, W // 16) for o in outs]


def backbone_from_state_dict(sd):
    """Backbone name of a saved DINOv3DPT checkpoint: V-JEPA 2.1 checkpoints carry bb.img_mod_embed."""
    return "vjepa2_1_vit_large" if any(k.startswith("bb.img_mod_embed") for k in sd) else "vit_large_patch16_dinov3"


def default_layers(backbone):
    return (5, 11, 17, 23) if backbone.startswith("vjepa2_1") else (4, 11, 17, 23)


class DINOv3DPT(nn.Module):
    def __init__(self, backbone="vit_base_patch16_dinov3", dim=256,
                 layers=(2, 5, 8, 11), pretrained=True, freeze_patch=True, decoupled=True,
                 orient=False, rot_per_part=True):
        super().__init__()
        self.decoupled = decoupled
        self.orient = orient
        self.rot_per_part = rot_per_part
        self.is_vjepa = backbone.startswith("vjepa2_1")
        if self.is_vjepa:                                  # V-JEPA 2.1 encoder, image patchifier (VJEPA21Backbone)
            self.bb = VJEPA21Backbone(backbone, pretrained=pretrained, layers=layers)
        else:
            self.bb = timm.create_model(backbone, pretrained=pretrained, num_classes=0)
        self.layers = list(layers)
        ed = self.bb.embed_dim
        self.patch = 16 if self.is_vjepa else self.bb.patch_embed.patch_size[0]
        # shallow->deep reassemble scales (DPT): 4x,2x,1x,0.5x of token grid
        scales = [4, 2, 1, 0.5]
        self.reasm = nn.ModuleList([Reassemble(ed, dim, s) for s in scales])
        self.fuse = nn.ModuleList([Fusion(dim) for _ in range(4)])
        if decoupled:
            self.mask_head = nn.Sequential(nn.Conv2d(dim, dim, 3, 1, 1), nn.ReLU(True),
                                           nn.Conv2d(dim, 1, 1))        # body vs background
            self.part_head = nn.Sequential(nn.Conv2d(dim, dim, 3, 1, 1), nn.ReLU(True),
                                           nn.Conv2d(dim, N_PARTS, 1))  # 24 parts, NO bg class
        else:                                                          # legacy: flat 25-way
            self.part_head = nn.Sequential(nn.Conv2d(dim, dim, 3, 1, 1), nn.ReLU(True),
                                           nn.Conv2d(dim, N_CLASSES, 1))
        self.uv_head = nn.Sequential(nn.Conv2d(dim, dim, 3, 1, 1), nn.ReLU(True),
                                     nn.Conv2d(dim, 2 * N_PARTS, 1))
        if orient:    # 6D rotation: per-part (24x6, gathered by part like UV) or single shared (6)
            rc = 6 * N_PARTS if rot_per_part else 6
            self.rot_head = nn.Sequential(nn.Conv2d(dim, dim, 3, 1, 1), nn.ReLU(True),
                                          nn.Conv2d(dim, rc, 1))
        if self.is_vjepa:                                  # V-JEPA 2.1 app/vjepa_2_1/transforms.py
            mean, std = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)
        else:
            cfg = timm.data.resolve_model_data_config(self.bb); mean, std = cfg["mean"], cfg["std"]
        self.register_buffer("mean", torch.tensor(mean).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor(std).view(1, 3, 1, 1))
        if freeze_patch:
            for p in self.bb.patch_embed.parameters():
                p.requires_grad_(False)

    def forward(self, img):
        H, W = img.shape[-2:]
        x = (img - self.mean) / self.std
        feats = self.bb.forward_intermediates(
            x, indices=self.layers, norm=True, output_fmt="NCHW",
            intermediates_only=True)                         # list of (B,ed,g,g)
        r = [self.reasm[i](feats[i]) for i in range(4)]      # [4x,2x,1x,0.5x]
        p = self.fuse[3](r[3])                               # 0.5x -> 1x
        p = self.fuse[2](r[2], p)                            # -> 2x
        p = self.fuse[1](r[1], p)                            # -> 4x
        p = self.fuse[0](r[0], p)                            # -> 8x
        part = self.part_head(p)
        uv = torch.sigmoid(self.uv_head(p))
        part = F.interpolate(part, (H, W), mode="bilinear", align_corners=False)
        uv = F.interpolate(uv, (H, W), mode="bilinear", align_corners=False)
        out = {"part": part, "uv": uv}
        if self.decoupled:
            mask = F.interpolate(self.mask_head(p), (H, W), mode="bilinear", align_corners=False)
            out["mask"] = mask
        if self.orient:
            rot6d = F.interpolate(self.rot_head(p), (H, W), mode="bilinear", align_corners=False)
            out["rot6d"] = rot6d              # (B, 6 or 6*N_PARTS, H, W); gather_rot6d -> 6 -> sixd_to_R
        return out


def gather_rot6d(rot6d, part):
    """Select the active part's 6D from a rotation prediction, handling BOTH modes.
    rot6d: (B,C,H,W) with C=6 (single shared) or C=6*N_PARTS (per-part). part: (B,H,W) long, 0=bg 1..24.
    Returns (B,6,H,W). Single-head -> returns rot6d unchanged (part ignored)."""
    if rot6d.shape[1] == 6:
        return rot6d
    B, _, H, W = rot6d.shape
    r = rot6d.view(B, N_PARTS, 6, H, W)
    idx = (part - 1).clamp(min=0)                                      # (B,H,W) -> 0-based part
    return r.gather(1, idx[:, None, None].expand(B, 1, 6, H, W)).squeeze(1)


def sixd_to_R(six):
    """(...,6) -> (...,3,3) rotation via Gram-Schmidt. `six` channels = first two columns [a0|a1].
    Use on GT 6D and on the net's rot6d output (move the 6 to the last dim first)."""
    a0, a1 = six[..., :3], six[..., 3:]
    e0 = F.normalize(a0, dim=-1)
    a1 = a1 - (a1 * e0).sum(-1, keepdim=True) * e0
    e1 = F.normalize(a1, dim=-1)
    e2 = torch.cross(e0, e1, dim=-1)
    return torch.stack([e0, e1, e2], dim=-1)              # columns e0,e1,e2


def part_pred(out):
    """0..24 part-id map from model output, handling both head types (mask>0 + 24-way, or flat 25-way)."""
    if "mask" in out:
        pid = out["part"].argmax(1) + 1
        pid[out["mask"][:, 0] <= 0] = 0
        return pid
    return out["part"].argmax(1)


def dp_losses(out, batch, w_uv=1.0, w_mask=1.0, w_rot=0.0,
              w_xyz=0.0, w_xg=0.0, csc=None, csc_gather="gt", csc_tau=0.005, csc_clip=0.05):
    """Decoupled head, NO GT resampling: (1) BCE body-mask over ALL pixels; (2) 24-way part CE
    over GT-FOREGROUND pixels only (part head has no bg class -> can't 'pick bg' at seams);
    (3) UV smooth-L1 over fg (read GT-part channel). Preds interpolated UP to native GT res.
    (4) ORIENTATION (if w_rot>0 and out has 'rot6d' and gt tuple has a 4th elem rot6d_gt (H,W,6)):
    chordal loss ||R_pred-R_gt||_F^2 over fg, both via sixd_to_R; reports geodesic angle (deg)."""
    decoupled = "mask" in out
    pl, uvp = out["part"], out["uv"]
    ml = out["mask"] if decoupled else None
    rotp = out.get("rot6d")
    dev = pl.device
    B = pl.shape[0]
    m_l = pl.new_zeros(())
    ce = pl.new_zeros(())
    uv_l = pl.new_zeros(())
    rot_l = pl.new_zeros(())
    geo_deg = pl.new_zeros(())
    xyz_l = pl.new_zeros(())
    xg_l = pl.new_zeros(())
    surf_mm = pl.new_zeros(())
    n_fg = 0; n_ce = 0; n_rot = 0; n_csc = 0
    for b, g in enumerate(batch["gt"]):
        part, uv, fg = g[0], g[1], g[2]
        rot_gt = g[3] if len(g) > 3 else None        # (Rh,Rw,6) capped-res GT 6D
        rot_fg = g[4] if len(g) > 4 else None        # (Rh,Rw) bool, matches rot_gt res
        part = part.to(dev, non_blocking=True)       # (H,W) long, 0=bg 1..24=parts
        H, W = part.shape
        if decoupled:
            body = (part > 0).float()
            m_up = F.interpolate(ml[b:b + 1], (H, W), mode="bilinear", align_corners=False)[0, 0]
            m_l = m_l + F.binary_cross_entropy_with_logits(m_up, body)
        else:                                         # legacy: 25-way CE over ALL pixels
            p_all = F.interpolate(pl[b:b + 1], (H, W), mode="bilinear", align_corners=False)
            ce = ce + F.cross_entropy(p_all, part[None]); n_ce += 1
        if fg.any():
            fg = fg.to(dev, non_blocking=True); uv = uv.to(dev, non_blocking=True)  # (2,H,W)
            p_up = F.interpolate(pl[b:b + 1], (H, W), mode="bilinear", align_corners=False)[0]
            if decoupled:
                ce = ce + F.cross_entropy(p_up.permute(1, 2, 0)[fg], part[fg] - 1); n_ce += 1  # fg, 24-way
            u_up = F.interpolate(uvp[b:b + 1], (H, W), mode="bilinear",
                                 align_corners=False)[0].view(N_PARTS, 2, H, W)
            idx = (part - 1).clamp(min=0)            # (H,W)
            sel = u_up.gather(0, idx[None, None].expand(1, 2, H, W)).squeeze(0)  # (2,H,W)
            uv_l = uv_l + F.smooth_l1_loss(sel.permute(1, 2, 0)[fg], uv.permute(1, 2, 0)[fg])
            n_fg += 1
        # ORIENTATION at its OWN capped res (decoupled from native part-fg; bounds memory)
        if w_rot > 0 and rotp is not None and rot_gt is not None and rot_fg.any():
            rot_fg = rot_fg.to(dev, non_blocking=True)
            Rh, Rw = rot_fg.shape
            rot_gt = rot_gt.to(dev, non_blocking=True).float()                  # (Rh,Rw,6) GT (single 6D/pixel)
            pr = F.interpolate(part[None, None].float(), (Rh, Rw), mode="nearest")[0, 0].long()  # GT part @ rot res
            r_pp = F.interpolate(rotp[b:b + 1], (Rh, Rw), mode="bilinear", align_corners=False)  # (1,C,Rh,Rw)
            r_up = gather_rot6d(r_pp, pr[None])[0].permute(1, 2, 0)             # (Rh,Rw,6) GT-part's pred 6D
            Rp = sixd_to_R(r_up[rot_fg]); Rg = sixd_to_R(rot_gt[rot_fg])        # (n,3,3)
            rot_l = rot_l + ((Rp - Rg) ** 2).sum(dim=(1, 2)).mean()            # chordal^2
            with torch.no_grad():
                tr = torch.matmul(Rp.transpose(1, 2), Rg).diagonal(dim1=1, dim2=2).sum(1)
                geo_deg = geo_deg + torch.arccos(((tr - 1) / 2).clamp(-1, 1)).mean() * (180 / math.pi)
            n_rot += 1
        # CANONICAL SURFACE COORDINATES: map the predicted (part,uv) to template XYZ and supervise it in
        # METRES, plus its image-space GRADIENT (MiDaS-style). Fixes two things the uv smooth-L1 cannot
        # see: (a) every chart is renormalised by its own span, so the same uv error costs 8x more surface
        # error on the torso than on the smallest chart; (b) uv is DISCONTINUOUS across chart seams while
        # XYZ is not, so the gradient term forces neighbouring charts to agree in 3D at their boundary.
        if (w_xyz > 0 or w_xg > 0) and csc is not None and len(g) > 8 and g[8].any():
            from orient.csc import xyz_grad_loss
            cfg = g[8].to(dev, non_blocking=True)
            R = cfg.shape[0]
            Xg = g[5].to(dev, non_blocking=True).float().permute(2, 0, 1)       # (3,R,R) template metres
            cpart = g[6].to(dev, non_blocking=True).long().clamp(min=1)          # (R,R) 1..24
            cuv = g[7].to(dev, non_blocking=True).float()                        # (2,R,R)
            with torch.autocast(device_type=dev.type, enabled=False):            # bf16 uv ~= 1 texel: fatal
                # Validity, exactly: the chart must REPRODUCE the GT point. Where it cannot, (part,uv) does
                # not determine a surface point (the hand/foot fold, the head's hidden eyeball + interior
                # geometry) and training there would fit an arbitrary branch. No heuristic needed -- we
                # have the true point, so we just ask.
                with torch.no_grad():
                    ok = (csc(cuv, cpart) - Xg).norm(dim=0) < csc_tau
                m = cfg & ok
                if m.any():
                    u_r = F.interpolate(uvp[b:b + 1].float(), (R, R), mode="bilinear",
                                        align_corners=False)[0].view(N_PARTS, 2, R, R)
                    if csc_gather == "soft":
                        # run E. MEASURED: given the GT part the uv regression already reproduces the
                        # 301 mm occlusion step to 1.3 mm and is SMOOTHER than GT on the interior -- the
                        # blur and the local noise live in the PART LOGITS, not in uv (argmax is 96%
                        # accurate but its boundary is smeared, so the canonical map ramps instead of
                        # stepping). Only a SOFT mixture sends metric 3D gradient into those logits: a
                        # hedged posterior at an occlusion edge lands in mid-air and is charged 300 mm,
                        # where cross-entropy charges it nothing extra.
                        p_r = F.interpolate(pl[b:b + 1].float(), (R, R), mode="bilinear",
                                            align_corners=False)[0]                # (24,R,R)
                        wk, ik = p_r.softmax(0).topk(3, dim=0)                     # (3,R,R)
                        wk = wk / wk.sum(0, keepdim=True).clamp(min=1e-6)
                        Xp = 0
                        for kk in range(3):
                            pk = ik[kk] + 1
                            sk = u_r.gather(0, (pk - 1)[None, None].expand(1, 2, R, R)).squeeze(0)
                            Xp = Xp + wk[kk] * csc(sk, pk)
                        pid = ik[0] + 1
                    else:
                        if csc_gather == "pred":  # run D: the channel the model would pick at TEST time
                            p_r = F.interpolate(pl[b:b + 1].float(), (R, R), mode="bilinear",
                                                align_corners=False)[0]
                            pid = (p_r.argmax(0) + 1)
                        else:                     # run C: gather by the GT part (same as the uv loss)
                            pid = cpart
                        sel = u_r.gather(0, (pid - 1).clamp(min=0)[None, None].expand(1, 2, R, R)).squeeze(0)
                        Xp = csc(sel, pid)                                         # (3,R,R), diff'able in uv
                    xyz_l = xyz_l + (Xp - Xg).abs().permute(1, 2, 0)[m].mean()
                    xg_l = xg_l + xyz_grad_loss(Xp, Xg, m, clip=csc_clip)
                    with torch.no_grad():
                        surf_mm = surf_mm + (Xp - Xg).norm(dim=0)[m].median() * 1000
                    n_csc += 1
    m_l = m_l / B
    ce = ce / max(n_ce, 1)
    uv_l = uv_l / max(n_fg, 1)
    rot_l = rot_l / max(n_rot, 1)
    geo_deg = geo_deg / max(n_rot, 1)
    xyz_l = xyz_l / max(n_csc, 1)
    xg_l = xg_l / max(n_csc, 1)
    surf_mm = surf_mm / max(n_csc, 1)
    return {"loss": w_mask * m_l + ce + w_uv * uv_l + w_rot * rot_l + w_xyz * xyz_l + w_xg * xg_l,
            "mask": m_l.detach(), "ce": ce.detach(), "uv": uv_l.detach(),
            "rot": rot_l.detach(), "geo": geo_deg.detach(),
            "xyz": xyz_l.detach(), "xg": xg_l.detach(), "mm": surf_mm.detach()}


def dp_losses_sparse(out, batch, w_uv=1.0, w_mask=1.0, uv_skip=()):
    """SPARSE supervision (DensePose-COCO): point-wise 24-way part CE + per-point UV smooth-L1
    (both via grid_sample at the annotated surface points) + body-mask BCE (decoupled head).
    batch['gt_sparse'] = list of {pts[N,2] in [-1,1], I[N] 0..23, uv[N,2], mask[Z,Z]}.
    Mirrors dp_losses' decomposition so COCO and BEDLAM optimize the same heads identically.
    uv_skip: 1-indexed parts whose UV loss is dropped (part CE kept). Needed when the model
    predicts v2 charts: COCO UV are v1-chart coordinates, which on the re-flattened hands/feet
    (3-6) would supervise the wrong chart -- and v1 folds there, so the target is ambiguous anyway."""
    assert "mask" in out, "sparse loss assumes the decoupled head"
    pl, uvp, ml = out["part"], out["uv"], out["mask"]
    dev = pl.device
    B = pl.shape[0]
    m_l = pl.new_zeros(()); ce = pl.new_zeros(()); uv_l = pl.new_zeros(())
    n_inst = 0
    for b, g in enumerate(batch["gt_sparse"]):
        mask = g["mask"].to(dev, non_blocking=True)                     # (Z,Z) float
        Z = mask.shape[-1]
        m_up = F.interpolate(ml[b:b + 1], (Z, Z), mode="bilinear", align_corners=False)[0, 0]
        m_l = m_l + F.binary_cross_entropy_with_logits(m_up.float(), mask)
        pts = g["pts"].to(dev, non_blocking=True)                       # (N,2)
        if pts.shape[0] == 0:
            continue
        I = g["I"].to(dev, non_blocking=True)                           # (N,) 0..23
        uvg = g["uv"].to(dev, non_blocking=True)                        # (N,2)
        grid = pts.view(1, -1, 1, 2)
        pl_pt = F.grid_sample(pl[b:b + 1].float(), grid, align_corners=False,
                              mode="bilinear")[0, :, :, 0].t()          # (N,24)
        ce = ce + F.cross_entropy(pl_pt, I)
        uv_pt = F.grid_sample(uvp[b:b + 1].float(), grid, align_corners=False,
                              mode="bilinear")[0, :, :, 0].t().view(-1, N_PARTS, 2)  # (N,24,2)
        N = uv_pt.shape[0]
        sel = uv_pt[torch.arange(N, device=dev), I]                     # (N,2)
        if uv_skip:
            keep = ~torch.isin(I + 1, torch.as_tensor(uv_skip, device=dev))
            if keep.any():
                uv_l = uv_l + F.smooth_l1_loss(sel[keep], uvg[keep])
        else:
            uv_l = uv_l + F.smooth_l1_loss(sel, uvg)
        n_inst += 1
    m_l = m_l / B
    ce = ce / max(n_inst, 1)
    uv_l = uv_l / max(n_inst, 1)
    return {"loss": w_mask * m_l + ce + w_uv * uv_l,
            "mask": m_l.detach(), "ce": ce.detach(), "uv": uv_l.detach()}


if __name__ == "__main__":
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    m = DINOv3DPT(pretrained=True).to(dev)
    n = sum(p.numel() for p in m.parameters()) / 1e6
    nt = sum(p.numel() for p in m.parameters() if p.requires_grad) / 1e6
    print(f"params {n:.1f}M (trainable {nt:.1f}M)")
    img = torch.rand(2, 3, 384, 384, device=dev)
    out = m(img)
    print("part", tuple(out["part"].shape), "uv", tuple(out["uv"].shape))
    # dense GT list with variable native sizes
    gt = []
    for hw in (512, 832):
        gt.append((torch.randint(0, 25, (hw, hw), device=dev),
                   torch.rand(2, hw, hw, device=dev),
                   torch.rand(hw, hw, device=dev) > 0.5))
    L = dp_losses(out, {"gt": gt})
    print({k: float(v) for k, v in L.items()})
    L["loss"].backward()
    print("backward ok")


def dp_losses_512(out, batch, w_uv=1.0, w_mask=1.0, w_rot=0.0,
                  w_xyz=0.0, w_xg=0.0, csc=None, csc_gather="gt", csc_tau=0.005, csc_clip=0.20,
                  rot_dilate=0):
    """All losses in the FIXED 512^2 GT canvas.

    The GT is NEVER resampled -- it was rasterised straight into this canvas. Instead the PREDICTION is
    warped INTO it by a pure scale+translate affine (data.py builds `theta`). That direction is the safe
    one: the network's output is an inherently smooth field, so warping only EVALUATES it at the GT's
    pixel centres, whereas warping the GT -- a field with true discontinuities at occlusion boundaries --
    is what destroyed information (bilinear on 6D rotations averaged an arm with the torso behind it:
    3.2% of labels wrong by >30 deg, up to a 180 deg flip).

    Because the GT is fixed-size it finally STACKS, so this is fully batched -- no python loop over the
    batch, and no ROT_CAP / CSC_CAP: every head is supervised at the network's own output resolution.
    """
    dev = out["part"].device
    part = batch["part"].to(dev, non_blocking=True)                 # (B,G,G) long, 0=bg
    uv = batch["uv"].to(dev, non_blocking=True)                     # (B,2,G,G)
    fg = batch["fg"].to(dev, non_blocking=True)                     # (B,G,G) bool
    theta = batch["theta"].to(dev, non_blocking=True).float()       # (B,2,3)
    B, G, _ = part.shape

    grid = F.affine_grid(theta, (B, 1, G, G), align_corners=False)  # ONE grid, reused by every head

    def warp(x, fp32=True):
        """Warp a prediction into GT space. fp32=False keeps bf16: the 144-channel rotation field at
        512^2 is the memory hog (4.8 GB of fp32 activations at B=32), and it does not need fp32 -- the
        6 channels we actually gather are upcast right after. UV DOES need fp32 (the CSC operator
        resolves a [0,1] uv to ~1 texel in bf16)."""
        g = grid if (fp32 or x.dtype == grid.dtype) else grid.to(x.dtype)
        return F.grid_sample(x.float() if fp32 else x, g, mode="bilinear",
                             padding_mode="border", align_corners=False)

    pl = warp(out["part"])                                          # (B,24,G,G)
    body = (part > 0).float()
    m_l = F.binary_cross_entropy_with_logits(warp(out["mask"])[:, 0], body)
    ce = F.cross_entropy(pl.permute(0, 2, 3, 1)[fg], part[fg] - 1) if fg.any() else pl.new_zeros(())

    uvp = warp(out["uv"]).view(B, N_PARTS, 2, G, G)
    idx = (part - 1).clamp(min=0)
    sel = uvp.gather(1, idx[:, None, None].expand(B, 1, 2, G, G)).squeeze(1)      # (B,2,G,G) GT-part chan
    uv_l = F.smooth_l1_loss(sel.permute(0, 2, 3, 1)[fg], uv.permute(0, 2, 3, 1)[fg]) \
        if fg.any() else pl.new_zeros(())

    rot_l = pl.new_zeros(()); geo_deg = pl.new_zeros(())
    if w_rot > 0 and "rot6d" in out and "rot6d" in batch:
        rg = batch["rot6d"].to(dev, non_blocking=True).float()                    # (B,G,G,6) EXACT
        rp = gather_rot6d(warp(out["rot6d"], fp32=False),
                          part.clamp(min=1)).permute(0, 2, 3, 1).float()   # (B,G,G,6) upcast AFTER gather
        if fg.any():
            Rp = sixd_to_R(rp[fg]); Rg = sixd_to_R(rg[fg])
            rot_l = ((Rp - Rg) ** 2).sum(dim=(1, 2)).mean()
            with torch.no_grad():
                tr = torch.matmul(Rp.transpose(1, 2), Rg).diagonal(dim1=1, dim2=2).sum(1)
                geo_deg = torch.arccos(((tr - 1) / 2).clamp(-1, 1)).mean() * (180 / math.pi)
        if rot_dilate > 0 and fg.any():
            # DILATED per-part rotation supervision. A rotation channel is only ever trained INSIDE its
            # own part, so one pixel outside it the channel has drifted (measured: 7.75 deg -> 16.99 deg
            # against the same true R). When argmax flips at a boundary we read that drifted channel and
            # the frame field JUMPS 13 deg where the truth is flat -- on ~2% of edges, and invisible to
            # geo_e2e (it moves 0.04 deg). The rotation field is a per-vertex barycentric field, i.e.
            # CONTINUOUS across a part boundary, so the correct target for channel p just outside part p
            # is EXACTLY the same R we already have. Supervising a dilated band therefore costs no new GT
            # and forces the 24 channels to agree at their boundaries -- which makes an argmax flip
            # HARMLESS instead of catastrophic.
            oh = F.one_hot(part, N_PARTS + 1).permute(0, 3, 1, 2).float()          # (B,25,G,G)
            k = 2 * rot_dilate + 1
            dil = F.max_pool2d(oh, k, stride=1, padding=rot_dilate) > 0.5
            band = dil[:, 1:] & fg[:, None] & ~(oh[:, 1:] > 0.5)                   # near part p, not IN it
            rw = warp(out["rot6d"], fp32=False).view(B, N_PARTS, 6, G, G)
            bsum = rot_l.new_zeros(()); nb = 0
            for p in range(N_PARTS):
                msk = band[:, p]
                if not msk.any():
                    continue
                Rb = sixd_to_R(rw[:, p].permute(0, 2, 3, 1)[msk].float())
                bsum = bsum + ((Rb - sixd_to_R(rg[msk])) ** 2).sum(dim=(1, 2)).mean()
                nb += 1
            if nb:
                # MEAN over the parts that have a band, not the SUM: summing 24 band terms against one
                # own-part term hands the band ~98% of the rotation loss and swamps the primary signal.
                rot_band = bsum / nb
                rot_l = rot_l + ROT_BAND_W * rot_band

    xyz_l = pl.new_zeros(()); xg_l = pl.new_zeros(()); surf_mm = pl.new_zeros(())
    if (w_xyz > 0 or w_xg > 0) and csc is not None and "xyz" in batch:
        from orient.csc import xyz_grad_loss
        Xg = batch["xyz"].to(dev, non_blocking=True).float().permute(0, 3, 1, 2)  # (B,3,G,G)
        with torch.autocast(device_type=dev.type, enabled=False):                 # bf16 uv ~= 1 texel
            pid_gt = part.clamp(min=1)
            with torch.no_grad():   # the chart must REPRODUCE the GT point, else (part,uv) is degenerate
                ok = torch.stack([(csc(uv[b], pid_gt[b]) - Xg[b]).norm(dim=0) < csc_tau
                                  for b in range(B)])
            m = fg & ok
            if m.any():
                if csc_gather == "soft":
                    wk, ik = pl.softmax(1).topk(3, dim=1)
                    wk = wk / wk.sum(1, keepdim=True).clamp(min=1e-6)
                    Xp = 0
                    for kk in range(3):
                        pk = ik[:, kk] + 1
                        sk = uvp.gather(1, (pk - 1)[:, None, None].expand(B, 1, 2, G, G)).squeeze(1)
                        Xp = Xp + wk[:, kk:kk + 1] * torch.stack(
                            [csc(sk[b], pk[b]) for b in range(B)])
                else:
                    pid = (pl.argmax(1) + 1) if csc_gather == "pred" else pid_gt
                    sk = uvp.gather(1, (pid - 1)[:, None, None].expand(B, 1, 2, G, G)).squeeze(1)
                    Xp = torch.stack([csc(sk[b], pid[b]) for b in range(B)])      # (B,3,G,G)
                xyz_l = (Xp - Xg).abs().permute(0, 2, 3, 1)[m].mean()
                xg_l = torch.stack([xyz_grad_loss(Xp[b], Xg[b], m[b], clip=csc_clip)
                                    for b in range(B)]).mean()
                with torch.no_grad():
                    surf_mm = (Xp - Xg).norm(dim=1)[m].median() * 1000

    return {"loss": w_mask * m_l + ce + w_uv * uv_l + w_rot * rot_l + w_xyz * xyz_l + w_xg * xg_l,
            "mask": m_l.detach(), "ce": ce.detach(), "uv": uv_l.detach(),
            "rot": rot_l.detach(), "geo": geo_deg.detach(),
            "xyz": xyz_l.detach(), "xg": xg_l.detach(), "mm": surf_mm.detach()}

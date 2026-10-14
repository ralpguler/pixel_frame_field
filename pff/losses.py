"""Training losses (research/densepose/model.py: dp_losses_512 and dp_losses_sparse, with the options of the
released recipe).

dense_losses   synthetic data (BEDLAM2, MAMMA): every head supervised at every pixel of a FIXED 512 x 512 ground-truth
               canvas. The ground truth is never resampled; the prediction is warped into the canvas by the
               scale-and-translate affine `theta` the loader computes, because the prediction is smooth while the
               ground truth has real discontinuities at occlusion boundaries.
sparse_losses  DensePose-COCO: part cross-entropy and UV smooth-L1 at the annotated points, body-mask BCE.
"""
import math

import torch
import torch.nn.functional as F

from .csc import xyz_grad_loss
from .geometry import N_PARTS, gather_rot6d, sixd_to_R


def dense_losses(out, batch, w_uv=1.0, w_mask=1.0, w_rot=1.0, w_xyz=2.0, w_xg=5.0, csc=None, csc_tau=0.005,
                 csc_clip=0.20):
    """Mask BCE + part CE + UV smooth-L1 + rotation (squared chordal) + canonical-surface-coordinate L1 and
    multi-scale gradient terms (metres on the body template, pff.csc). Defaults = the released recipe."""
    dev = out["part"].device
    part = batch["part"].to(dev, non_blocking=True)                 # (B,G,G) long, 0 = background
    uv = batch["uv"].to(dev, non_blocking=True)                     # (B,2,G,G)
    fg = batch["fg"].to(dev, non_blocking=True)                     # (B,G,G) supervised pixels
    theta = batch["theta"].to(dev, non_blocking=True).float()       # (B,2,3) canvas -> prediction coordinates
    B, G, _ = part.shape
    grid = F.affine_grid(theta, (B, 1, G, G), align_corners=False)

    def warp(x, fp32=True):
        """The prediction evaluated at the canvas pixels. The 144-channel rotation field stays in bf16 (memory);
        UV needs fp32 (the CSC operator resolves a [0,1] uv to about one texel in bf16)."""
        g = grid if (fp32 or x.dtype == grid.dtype) else grid.to(x.dtype)
        return F.grid_sample(x.float() if fp32 else x, g, mode="bilinear", padding_mode="border",
                             align_corners=False)

    pl = warp(out["part"])
    m_l = F.binary_cross_entropy_with_logits(warp(out["mask"])[:, 0], (part > 0).float())
    ce = F.cross_entropy(pl.permute(0, 2, 3, 1)[fg], part[fg] - 1) if fg.any() else pl.new_zeros(())

    uvp = warp(out["uv"]).view(B, N_PARTS, 2, G, G)
    idx = (part - 1).clamp(min=0)
    sel = uvp.gather(1, idx[:, None, None].expand(B, 1, 2, G, G)).squeeze(1)
    uv_l = F.smooth_l1_loss(sel.permute(0, 2, 3, 1)[fg], uv.permute(0, 2, 3, 1)[fg]) if fg.any() else pl.new_zeros(())

    rot_l = pl.new_zeros(()); geo_deg = pl.new_zeros(())
    if w_rot > 0 and "rot6d" in out and "rot6d" in batch and fg.any():
        rg = batch["rot6d"].to(dev, non_blocking=True).float()                        # (B,G,G,6), exact
        rp = gather_rot6d(warp(out["rot6d"], fp32=False), part.clamp(min=1)).permute(0, 2, 3, 1).float()
        Rp, Rg = sixd_to_R(rp[fg]), sixd_to_R(rg[fg])
        rot_l = ((Rp - Rg) ** 2).sum(dim=(1, 2)).mean()
        with torch.no_grad():
            tr = torch.matmul(Rp.transpose(1, 2), Rg).diagonal(dim1=1, dim2=2).sum(1)
            geo_deg = torch.arccos(((tr - 1) / 2).clamp(-1, 1)).mean() * (180 / math.pi)

    xyz_l = pl.new_zeros(()); xg_l = pl.new_zeros(()); surf_mm = pl.new_zeros(())
    if (w_xyz > 0 or w_xg > 0) and csc is not None and "xyz" in batch:
        Xg = batch["xyz"].to(dev, non_blocking=True).float().permute(0, 3, 1, 2)      # (B,3,G,G) template metres
        with torch.autocast(device_type=dev.type, enabled=False):
            pid = part.clamp(min=1)
            with torch.no_grad():       # only pixels where the chart reproduces the ground-truth point
                ok = torch.stack([(csc(uv[b], pid[b]) - Xg[b]).norm(dim=0) < csc_tau for b in range(B)])
            m = fg & ok
            if m.any():
                sk = uvp.gather(1, (pid - 1)[:, None, None].expand(B, 1, 2, G, G)).squeeze(1)
                Xp = torch.stack([csc(sk[b], pid[b]) for b in range(B)])
                xyz_l = (Xp - Xg).abs().permute(0, 2, 3, 1)[m].mean()
                xg_l = torch.stack([xyz_grad_loss(Xp[b], Xg[b], m[b], clip=csc_clip) for b in range(B)]).mean()
                with torch.no_grad():
                    surf_mm = (Xp - Xg).norm(dim=1)[m].median() * 1000

    loss = w_mask * m_l + ce + w_uv * uv_l + w_rot * rot_l + w_xyz * xyz_l + w_xg * xg_l
    return {"loss": loss, "mask": m_l.detach(), "ce": ce.detach(), "uv": uv_l.detach(), "rot": rot_l.detach(),
            "geo": geo_deg.detach(), "xyz": xyz_l.detach(), "xg": xg_l.detach(), "mm": surf_mm.detach()}


def sparse_losses(out, batch, w_uv=1.0, w_mask=1.0, uv_skip=(3, 4, 5, 6)):
    """batch["gt_sparse"]: per instance {pts (N,2) in [-1,1], I (N,) 0..23, uv (N,2), mask (Z,Z)}.
    uv_skip: parts (1..24) without a UV term. The model predicts the fold-free v2 charts, while COCO's UV are on
    the official v1 charts, which differ (and fold) on the hands and feet (parts 3-6); their part labels still
    train."""
    pl, uvp, ml = out["part"], out["uv"], out["mask"]
    dev = pl.device
    m_l = pl.new_zeros(()); ce = pl.new_zeros(()); uv_l = pl.new_zeros(())
    n_inst = 0
    for b, g in enumerate(batch["gt_sparse"]):
        mask = g["mask"].to(dev, non_blocking=True)
        Z = mask.shape[-1]
        m_up = F.interpolate(ml[b:b + 1], (Z, Z), mode="bilinear", align_corners=False)[0, 0]
        m_l = m_l + F.binary_cross_entropy_with_logits(m_up.float(), mask)
        pts = g["pts"].to(dev, non_blocking=True)
        if pts.shape[0] == 0:
            continue
        I = g["I"].to(dev, non_blocking=True)
        uvg = g["uv"].to(dev, non_blocking=True)
        grid = pts.view(1, -1, 1, 2)
        pl_pt = F.grid_sample(pl[b:b + 1].float(), grid, align_corners=False, mode="bilinear")[0, :, :, 0].t()
        ce = ce + F.cross_entropy(pl_pt, I)
        uv_pt = F.grid_sample(uvp[b:b + 1].float(), grid, align_corners=False,
                              mode="bilinear")[0, :, :, 0].t().view(-1, N_PARTS, 2)
        sel = uv_pt[torch.arange(uv_pt.shape[0], device=dev), I]
        if uv_skip:
            keep = ~torch.isin(I + 1, torch.as_tensor(uv_skip, device=dev))
            if keep.any():
                uv_l = uv_l + F.smooth_l1_loss(sel[keep], uvg[keep])
        else:
            uv_l = uv_l + F.smooth_l1_loss(sel, uvg)
        n_inst += 1
    m_l = m_l / pl.shape[0]
    ce = ce / max(n_inst, 1)
    uv_l = uv_l / max(n_inst, 1)
    return {"loss": w_mask * m_l + ce + w_uv * uv_l, "mask": m_l.detach(), "ce": ce.detach(), "uv": uv_l.detach()}

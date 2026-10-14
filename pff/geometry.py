"""Part ids, 6D rotations and the per-pixel read-out of a prediction.

Conventions: 24 DensePose parts, 1..24, 0 = background. A rotation is stored as 6D = its first two COLUMNS and
turned into a matrix by Gram-Schmidt; matrices act on column vectors. The predicted rotation at a pixel takes the
body template's surface directions (rest pose, template axes) to the camera.
"""
import torch
import torch.nn.functional as F

N_PARTS = 24


def sixd_to_R(six):
    """(...,6) -> (...,3,3) rotation by Gram-Schmidt; the 6 channels are the first two columns [a0 | a1]."""
    a0, a1 = six[..., :3], six[..., 3:]
    e0 = F.normalize(a0, dim=-1)
    a1 = a1 - (a1 * e0).sum(-1, keepdim=True) * e0
    e1 = F.normalize(a1, dim=-1)
    return torch.stack([e0, e1, torch.cross(e0, e1, dim=-1)], dim=-1)


def gather_rot6d(rot6d, part):
    """The 6D of the pixel's own part. rot6d (B,6*24,H,W) per-part (or (B,6,H,W) shared); part (B,H,W) 0..24.
    Returns (B,6,H,W)."""
    if rot6d.shape[1] == 6:
        return rot6d
    B, _, H, W = rot6d.shape
    r = rot6d.view(B, N_PARTS, 6, H, W)
    idx = (part - 1).clamp(min=0)
    return r.gather(1, idx[:, None, None].expand(B, 1, 6, H, W)).squeeze(1)


def part_pred(out):
    """(B,H,W) part id 0..24: the 24-way argmax where the body-mask logit is positive, else 0."""
    pid = out["part"].argmax(1) + 1
    pid[out["mask"][:, 0] <= 0] = 0
    return pid


def read_out(out):
    """Network output -> per-pixel maps: part (B,H,W) 0..24, uv (B,2,H,W) in [0,1] of the pixel's own part
    (0 on background), R (B,H,W,3,3) or None, body probability (B,H,W)."""
    part = part_pred(out)
    B, _, H, W = out["uv"].shape
    uv = out["uv"].float().view(B, N_PARTS, 2, H, W)
    idx = (part - 1).clamp(min=0)
    uv = uv.gather(1, idx[:, None, None].expand(B, 1, 2, H, W)).squeeze(1) * (part > 0)[:, None]
    R = None
    if "rot6d" in out:
        R = sixd_to_R(gather_rot6d(out["rot6d"].float(), part).permute(0, 2, 3, 1))
    return dict(part=part, uv=uv, R=R, body=torch.sigmoid(out["mask"][:, 0].float()))

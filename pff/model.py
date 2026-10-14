"""The Pixel Frame Field network: a DINOv3 ViT-L/16 backbone, a compact DPT decoder and four dense heads.

For a person crop (input 512 x 512, RGB in [0,1]) it predicts, at every pixel:
  * mask    one logit: body against background;
  * part    24 logits: the DensePose body part (argmax where the mask logit is positive);
  * uv      24 x 2 values in [0,1]: the chart coordinates (U,V) for each part, read at the pixel's own part;
  * rot6d   24 x 6 values: for each part, the 6D rotation (first two columns) taking the body template's
            surface directions to the camera, read at the pixel's own part (pff.geometry.read_out).
Parameter names are those of the research code (research/densepose/model.py, DINOv3DPT), so its checkpoints load
strictly.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import timm

from .geometry import N_PARTS

BACKBONE = "vit_large_patch16_dinov3"
LAYERS = (4, 11, 17, 23)        # backbone blocks tapped by the decoder (shallow to deep)


class ResConv(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.c1 = nn.Conv2d(dim, dim, 3, 1, 1)
        self.c2 = nn.Conv2d(dim, dim, 3, 1, 1)

    def forward(self, x):
        return x + self.c2(F.relu(self.c1(F.relu(x))))


class Reassemble(nn.Module):
    """Project a token grid to `dim` channels and resample it by `scale`."""

    def __init__(self, in_dim, dim, scale):
        super().__init__()
        self.proj = nn.Conv2d(in_dim, dim, 1)
        self.scale = scale

    def forward(self, x):
        x = self.proj(x)
        if self.scale != 1:
            x = F.interpolate(x, scale_factor=self.scale, mode="bilinear", align_corners=False)
        return x


class Fusion(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.skip = ResConv(dim)
        self.out = ResConv(dim)

    def forward(self, x, skip=None):
        if skip is not None:
            x = x + self.skip(skip)
        return F.interpolate(self.out(x), scale_factor=2, mode="bilinear", align_corners=False)


def head(dim, n_out):
    return nn.Sequential(nn.Conv2d(dim, dim, 3, 1, 1), nn.ReLU(True), nn.Conv2d(dim, n_out, 1))


class PixelFrameField(nn.Module):
    def __init__(self, backbone=BACKBONE, layers=LAYERS, dim=256, pretrained=False, orient=True):
        """pretrained: load the DINOv3 backbone weights through timm (training from scratch only)."""
        super().__init__()
        self.orient = orient
        self.bb = timm.create_model(backbone, pretrained=pretrained, num_classes=0)
        self.layers = list(layers)
        ed = self.bb.embed_dim
        self.reasm = nn.ModuleList([Reassemble(ed, dim, s) for s in (4, 2, 1, 0.5)])   # DPT: 4x .. 0.5x the grid
        self.fuse = nn.ModuleList([Fusion(dim) for _ in range(4)])
        self.mask_head = head(dim, 1)
        self.part_head = head(dim, N_PARTS)
        self.uv_head = head(dim, 2 * N_PARTS)
        if orient:
            self.rot_head = head(dim, 6 * N_PARTS)
        cfg = timm.data.resolve_model_data_config(self.bb)
        self.register_buffer("mean", torch.tensor(cfg["mean"]).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor(cfg["std"]).view(1, 3, 1, 1))
        for p in self.bb.patch_embed.parameters():          # the patch embedding stays frozen, as in training
            p.requires_grad_(False)

    def forward(self, img):
        """img (B,3,H,W) RGB in [0,1] -> dict of (B,C,H,W) maps: mask (1), part (24), uv (48, sigmoid),
        rot6d (144)."""
        H, W = img.shape[-2:]
        feats = self.bb.forward_intermediates((img - self.mean) / self.std, indices=self.layers, norm=True,
                                             output_fmt="NCHW", intermediates_only=True)
        r = [self.reasm[i](feats[i]) for i in range(4)]
        p = self.fuse[3](r[3])
        p = self.fuse[2](r[2], p)
        p = self.fuse[1](r[1], p)
        p = self.fuse[0](r[0], p)

        def up(x):
            return F.interpolate(x, (H, W), mode="bilinear", align_corners=False)

        out = dict(part=up(self.part_head(p)), uv=up(torch.sigmoid(self.uv_head(p))), mask=up(self.mask_head(p)))
        if self.orient:
            out["rot6d"] = up(self.rot_head(p))
        return out


def load_model(checkpoint, device="cuda"):
    """A trained model (a research checkpoint {"model": state_dict, ...} or a weights-only export), strict."""
    ck = torch.load(checkpoint, map_location="cpu", weights_only=False)
    sd = ck.get("model", ck)
    model = PixelFrameField(orient=any(k.startswith("rot_head.") for k in sd))
    model.load_state_dict(sd, strict=True)
    return model.to(device).eval()

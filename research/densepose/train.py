"""DDP training for DensePose-from-bbox (DINOv3+DPT, IUV).

Single full GH200 node: torchrun --standalone --nproc_per_node=4 train.py [args]
Also runs single-process if launched without torchrun (debug).
"""
import argparse
import math
import os
import time

import numpy as np
import torch
import torch.distributed as dist
import torch.nn.functional as F
from PIL import Image
from torch.nn.parallel import DistributedDataParallel as DDP

from data import build_loader, _PAL
from model import DINOv3DPT, N_CLASSES, dp_losses

RUNS = os.path.expanduser("~/data/bedlam2/dp_runs")


def ddp_setup():
    if "RANK" in os.environ:
        rank = int(os.environ["RANK"]); world = int(os.environ["WORLD_SIZE"])
        local = int(os.environ["LOCAL_RANK"])
        torch.cuda.set_device(local)
        dist.init_process_group("nccl")
        return rank, world, local
    return 0, 1, 0


def is_main():
    return (not dist.is_initialized()) or dist.get_rank() == 0


def lr_lambda(step, warmup, total):
    if step < warmup:
        return step / max(1, warmup)
    t = (step - warmup) / max(1, total - warmup)
    return 0.5 * (1 + math.cos(math.pi * min(t, 1.0)))


def param_groups(model, lr, bb_mult):
    bb, head = [], []
    for n, p in model.named_parameters():
        if not p.requires_grad:
            continue
        (bb if n.startswith("bb.") else head).append(p)
    return [{"params": bb, "lr": lr * bb_mult}, {"params": head, "lr": lr}]


@torch.no_grad()
def evaluate(model, loader, device, max_batches=40):
    model.eval()
    conf = torch.zeros(N_CLASSES, N_CLASSES, dtype=torch.float64, device=device)
    uv_err = torch.zeros(2, device=device)   # sum, count
    for i, b in enumerate(loader):
        if i >= max_batches:
            break
        img = b["img"].to(device, non_blocking=True)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            out = model(img)
        # per-sample: interpolate preds to native GT res (same as the loss); no GT resize
        for bi, (part, uv, fg) in enumerate(b["gt"]):
            part = part.to(device); H, W = part.shape
            p_up = F.interpolate(out["part"][bi:bi + 1].float(), (H, W),
                                 mode="bilinear", align_corners=False)
            m_up = F.interpolate(out["mask"][bi:bi + 1].float(), (H, W),
                                 mode="bilinear", align_corners=False)[0, 0]
            pred = p_up.argmax(1)[0] + 1                      # (H,W) 1..24
            pred[m_up <= 0] = 0                               # bg from mask head
            k = (part * N_CLASSES + pred).flatten()
            conf += torch.bincount(k, minlength=N_CLASSES ** 2).reshape(
                N_CLASSES, N_CLASSES).to(conf)
            if fg.any():
                fg = fg.to(device); uv = uv.to(device)
                u_up = F.interpolate(out["uv"][bi:bi + 1].float(), (H, W),
                                     mode="bilinear", align_corners=False)[0].view(24, 2, H, W)
                idx = (part - 1).clamp(min=0)
                sel = u_up.gather(0, idx[None, None].expand(1, 2, H, W)).squeeze(0)
                uv_err[0] += (sel.permute(1, 2, 0)[fg] - uv.permute(1, 2, 0)[fg]).abs().sum()
                uv_err[1] += fg.sum() * 2
    if dist.is_initialized():
        dist.all_reduce(conf); dist.all_reduce(uv_err)
    inter = conf.diag()
    union = conf.sum(0) + conf.sum(1) - inter
    iou = (inter / union.clamp(min=1))
    valid = union > 0
    miou = iou[valid].mean().item()
    fg = valid.clone(); fg[0] = False
    miou_fg = iou[fg].mean().item() if fg.any() else 0.0
    acc = (inter.sum() / conf.sum().clamp(min=1)).item()
    uvmae = (uv_err[0] / uv_err[1].clamp(min=1)).item()
    model.train()
    return {"mIoU": miou, "mIoU_fg": miou_fg, "pixAcc": acc, "uv_mae": uvmae}


def save_overlays(model, loader, device, path, n=6):
    """input | predicted part | predicted UV (dense argmax at input grid)."""
    model.eval()
    b = next(iter(loader))
    img = b["img"][:n].to(device)
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        out = model(img)
    part = (out["part"].float().argmax(1) + 1)                 # (n,H,W) 1..24
    part[out["mask"][:, 0] <= 0] = 0                            # bg from mask head
    part = part.cpu()
    uv = out["uv"].float().cpu()                                # (n,48,H,W)
    pal = torch.from_numpy(_PAL)
    rows = []
    for i in range(min(n, img.shape[0])):
        rgb = (b["img"][i].permute(1, 2, 0) * 255).byte()
        p = part[i]
        fg = p > 0
        pov = torch.where(fg[..., None], pal[p], rgb)
        sel = (p - 1).clamp(0)
        u = uv[i].view(24, 2, *p.shape)
        uimg = torch.zeros_like(rgb)
        gu = u.gather(0, sel.view(1, 1, *p.shape).expand(1, 2, *p.shape)).squeeze(0)
        uimg[..., 0] = (gu[0] * 255).clamp(0, 255).byte()
        uimg[..., 1] = (gu[1] * 255).clamp(0, 255).byte()
        uimg[..., 2] = torch.where(fg, torch.tensor(128, dtype=torch.uint8),
                                   torch.tensor(0, dtype=torch.uint8))
        uimg = torch.where(fg[..., None], uimg, rgb)
        rows.append(torch.cat([rgb, pov, uimg], 0))
    Image.fromarray(torch.cat(rows, 1).numpy()).save(path)
    model.train()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", default="dp_v1")   # NOT --run: torchrun abbrev-collides with --run-path
    ap.add_argument("--steps", type=int, default=40000)
    ap.add_argument("--warmup", type=int, default=1000)
    ap.add_argument("--batch", type=int, default=32)        # per GPU
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--bb-mult", type=float, default=0.1)
    ap.add_argument("--workers", type=int, default=12)
    ap.add_argument("--input-size", type=int, default=384)
    ap.add_argument("--backbone", default="vit_base_patch16_dinov3")
    ap.add_argument("--layers", default="2,5,8,11")
    ap.add_argument("--w-uv", type=float, default=1.0)
    ap.add_argument("--val-every", type=int, default=2000)
    ap.add_argument("--save-every", type=int, default=2000)
    ap.add_argument("--log-every", type=int, default=50)
    args = ap.parse_args()

    rank, world, local = ddp_setup()
    device = f"cuda:{local}"
    rundir = f"{RUNS}/{args.name}"
    if is_main():
        os.makedirs(rundir, exist_ok=True)
        from torch.utils.tensorboard import SummaryWriter
        tb = SummaryWriter(rundir)
        print(f"world={world} batch/gpu={args.batch} global_batch={args.batch*world}",
              flush=True)

    model = DINOv3DPT(backbone=args.backbone, layers=tuple(int(x) for x in args.layers.split(",")), pretrained=True).to(device)
    if dist.is_initialized():
        # backbone's post-block norm head is untapped by forward_intermediates(intermediates_only=True)
        # -> some params get no grad -> DDP needs find_unused_parameters=True
        model = DDP(model, device_ids=[local], find_unused_parameters=True)
    raw = model.module if isinstance(model, DDP) else model

    opt = torch.optim.AdamW(param_groups(raw, args.lr, args.bb_mult),
                            weight_decay=0.05, betas=(0.9, 0.999))
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: lr_lambda(s, args.warmup, args.steps))

    train_loader = build_loader("train", batch=args.batch, workers=args.workers,
                                input_size=args.input_size, train=True)
    val_loader = build_loader("val", batch=args.batch, workers=4,
                              input_size=args.input_size, train=False)

    model.train()
    step = 0
    it = iter(train_loader)
    t_log = time.time()
    while step < args.steps:
        try:
            b = next(it)
        except StopIteration:
            it = iter(train_loader); b = next(it)
        img = b["img"].to(device, non_blocking=True)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            out = model(img)
            L = dp_losses(out, b, w_uv=args.w_uv)
        opt.zero_grad(set_to_none=True)
        L["loss"].backward()
        torch.nn.utils.clip_grad_norm_(raw.parameters(), 1.0)
        opt.step(); sched.step(); step += 1

        if is_main() and step % args.log_every == 0:
            lr = sched.get_last_lr()[-1]
            torch.cuda.synchronize(device)
            dt = time.time() - t_log
            ips = args.batch * world * args.log_every / max(dt, 1e-6)
            mem = torch.cuda.max_memory_allocated(device) / 1e9
            res = torch.cuda.max_memory_reserved(device) / 1e9
            print(f"step {step}/{args.steps} loss {L['loss'].item():.3f} "
                  f"mask {L['mask'].item():.3f} ce {L['ce'].item():.3f} uv {L['uv'].item():.4f} lr {lr:.2e} "
                  f"| {ips:.1f} img/s | mem {mem:.1f}/{res:.1f}G", flush=True)
            tb.add_scalar("train/loss", L["loss"].item(), step)
            tb.add_scalar("train/mask", L["mask"].item(), step)
            tb.add_scalar("train/ce", L["ce"].item(), step)
            tb.add_scalar("train/uv", L["uv"].item(), step)
            tb.add_scalar("perf/img_s", ips, step)
            tb.add_scalar("perf/mem_GB", mem, step)
            tb.add_scalar("lr", lr, step)
            t_log = time.time()

        if step % args.val_every == 0:
            metrics = evaluate(model, val_loader, device)
            if is_main():
                print(f"  [val step {step}] " +
                      " ".join(f"{k}={v:.4f}" for k, v in metrics.items()), flush=True)
                for k, v in metrics.items():
                    tb.add_scalar(f"val/{k}", v, step)
                save_overlays(model, val_loader, device, f"{rundir}/val_{step:06d}.png")
        if is_main() and step % args.save_every == 0:
            torch.save({"step": step, "model": raw.state_dict(),
                        "opt": opt.state_dict()}, f"{rundir}/ckpt.pt")

    if is_main():
        torch.save({"step": step, "model": raw.state_dict()}, f"{rundir}/final.pt")
        print("done", flush=True)
    if dist.is_initialized():
        dist.destroy_process_group()


if __name__ == "__main__":
    main()

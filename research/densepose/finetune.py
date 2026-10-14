"""Fine-tune the BEDLAM-pretrained DensePose model on real DensePose-COCO.

Two modes (Riza's plan):
  --data coco   : pure real fine-tune on COCO (sparse point CE + UV + body-mask BCE).
  --data joint  : alternate steps BEDLAM (dense) <-> COCO (sparse), same heads/losses.

Init from a BEDLAM checkpoint (e.g. dp_v3_L512_dec/final.pt). The model is the decoupled head
(mask + 24-part); both data sources optimize it identically via dp_losses / dp_losses_sparse.
Authoritative metric is run_eval.py (COCO GPS) on the saved checkpoints; here we log a fast sparse
minival proxy (point part-acc + UV MAE) for the trend.

Single full GH200 node: torchrun --standalone --nproc_per_node=4 finetune.py [args]
"""
import argparse
import math
import os
import time

import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel as DDP

from model import DINOv3DPT, dp_losses, dp_losses_512, dp_losses_sparse
from coco_dp import build_coco_loader
from data import build_loader

RUNS = os.path.expanduser("~/data/bedlam2/dp_runs")
HOME = os.path.expanduser("~")
COCO_TRAIN_ANN = f"{HOME}/data/densepose_coco/densepose_coco_2014_train.json"
COCO_TRAIN_IMG = f"{HOME}/data/densepose_coco/train2014"
COCO_VAL_ANN = f"{HOME}/codes/wan/densepose/dpeval/data/densepose_coco_2014_minival.json"
COCO_VAL_IMG = f"{HOME}/data/coco_val2014_minival"


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
    """Flat: backbone at lr*bb_mult, heads at lr (the old CNN-detection heuristic)."""
    bb, head = [], []
    for n, p in model.named_parameters():
        if not p.requires_grad:
            continue
        (bb if n.startswith("bb.") else head).append(p)
    return [{"params": bb, "lr": lr * bb_mult}, {"params": head, "lr": lr}]


def param_groups_llrd(model, lr, decay):
    """Layer-wise LR decay (MAE/BEiT recipe): heads + top backbone block at full `lr`, each deeper
    block scaled by `decay`^depth (top->bottom), embeddings smallest. The literature-standard way to
    fine-tune a pretrained ViT on a dense task — top blocks adapt at ~lr, early blocks stay near-frozen,
    instead of a single flat backbone multiplier. decay 0.75 (ViT-L MAE) / 0.8 (ViT-L BEiTv2)."""
    n_blocks = len(model.bb.blocks)
    num_layers = n_blocks + 1                                          # blocks + 1 (embeddings=0)

    def layer_id(name):
        if not name.startswith("bb."):
            return num_layers                                         # heads/decoder: full lr
        nm = name[3:]
        if nm.startswith("blocks."):
            return int(nm.split(".")[1]) + 1                          # block i -> i+1
        if any(nm.startswith(k) for k in ("patch_embed", "cls_token", "pos_embed", "reg_token", "mask_token",
                                          "storage_tokens", "img_mod_embed", "video_mod_embed")):   # + V-JEPA 2.1 modality tokens
            return 0                                                  # embeddings: smallest lr
        return num_layers                                            # bb.norm etc: top

    groups = {}
    for n, p in model.named_parameters():
        if not p.requires_grad:
            continue
        lid = layer_id(n)
        scale = decay ** (num_layers - lid)
        g = groups.setdefault(lid, {"params": [], "lr": lr * scale})
        g["params"].append(p)
    return [groups[k] for k in sorted(groups)]


@torch.no_grad()
def sparse_val(model, loader, device, max_batches=30):
    """Fast proxy on COCO minival: point part-accuracy + per-point UV MAE (NOT the GPS metric)."""
    model.eval()
    correct = torch.zeros(2, device=device)   # correct, total
    uverr = torch.zeros(2, device=device)      # abs sum, count
    for i, b in enumerate(loader):
        if i >= max_batches:
            break
        img = b["img"].to(device, non_blocking=True)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            out = model(img)
        for bi, g in enumerate(b["gt_sparse"]):
            pts = g["pts"].to(device)
            if pts.shape[0] == 0:
                continue
            I = g["I"].to(device); uvg = g["uv"].to(device)
            grid = pts.view(1, -1, 1, 2)
            pl = F.grid_sample(out["part"][bi:bi + 1].float(), grid, align_corners=False)[0, :, :, 0].t()
            pred = pl.argmax(1)
            correct[0] += (pred == I).sum(); correct[1] += I.numel()
            uvpt = F.grid_sample(out["uv"][bi:bi + 1].float(), grid, align_corners=False)[0, :, :, 0].t().view(-1, 24, 2)
            sel = uvpt[torch.arange(uvpt.shape[0], device=device), I]
            uverr[0] += (sel - uvg).abs().sum(); uverr[1] += I.numel() * 2
    if dist.is_initialized():
        dist.all_reduce(correct); dist.all_reduce(uverr)
    model.train()
    return {"pt_acc": (correct[0] / correct[1].clamp(min=1)).item(),
            "uv_mae": (uverr[0] / uverr[1].clamp(min=1)).item()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", default="dp_ft_coco")
    ap.add_argument("--data", choices=["coco", "joint", "bedlam"], default="coco")
    # joint mode: 1 COCO step every N steps (2 = the historical 50/50 alternation)
    ap.add_argument("--coco-period", type=int, default=2)
    ap.add_argument("--orient", action="store_true")        # add rotation head + loss (BEDLAM)
    ap.add_argument("--w-rot", type=float, default=1.0)
    ap.add_argument("--rot-per-part", action=argparse.BooleanOptionalAction, default=True)  # 24x6 vs single 6
    ap.add_argument("--init", default=f"{RUNS}/dp_v3_L512_dec/final.pt")
    ap.add_argument("--resume", default="")                  # resume model+opt+sched+step (overrides --init)
    ap.add_argument("--steps", type=int, default=10000)
    ap.add_argument("--warmup", type=int, default=500)
    ap.add_argument("--batch", type=int, default=32)        # per GPU
    ap.add_argument("--lr", type=float, default=5e-5)       # low: fine-tune
    ap.add_argument("--bb-mult", type=float, default=0.1)            # flat backbone LR multiplier
    ap.add_argument("--llrd", action="store_true")                   # layer-wise LR decay (overrides bb-mult)
    ap.add_argument("--llrd-decay", type=float, default=0.75)        # 0.75 ViT-L MAE / 0.8 BEiTv2
    ap.add_argument("--workers", type=int, default=12)
    ap.add_argument("--input-size", type=int, default=512)
    ap.add_argument("--backbone", default="vit_large_patch16_dinov3")
    ap.add_argument("--layers", default="4,11,17,23")
    ap.add_argument("--w-uv", type=float, default=1.0)
    # --- canonical surface coordinates (orient/csc.py) ---
    ap.add_argument("--w-xyz", type=float, default=0.0)          # L1 on template XYZ, in metres
    ap.add_argument("--w-xyzgrad", type=float, default=0.0)      # multi-scale image-space gradient of it
    ap.add_argument("--csc-gather", choices=["gt", "pred", "soft"], default="gt")  # C=gt D=argmax E=soft
    ap.add_argument("--csc-tau", type=float, default=0.005)      # chart must reproduce GT to 5 mm
    ap.add_argument("--csc-clip", type=float, default=0.20)      # clip |grad| (m/px). NOT 0.05: GT steps at
    # occlusion boundaries are ~300 mm/px, so a 50 mm ceiling clamps BOTH sides and the loss goes blind to
    # exactly the blur it is meant to fix (measured: model predicts 185 mm where GT is 301 mm).
    ap.add_argument("--csc-K", type=int, default=512)
    ap.add_argument("--uv-asset", default="smplx_face_uv.npy")   # smplx_face_uv_v2.npy = fold-free
    # Fixed 512^2 GT canvas: the GT is rasterised straight into it and NEVER resampled. Kills ROT_CAP,
    # CSC_CAP, and the bilinear-rotation bug; supervises every head at the network's own resolution.
    ap.add_argument("--gt512", action="store_true")
    # SMPL-X hidden geometry (eyeballs + mouth/socket interior, build_hidden_faces.py): drop those
    # pixels' unlearnable labels from part/UV/rot/CSC supervision (mask BCE keeps the full person).
    ap.add_argument("--hidden-mask", action="store_true")
    ap.add_argument("--no-mamma", action="store_true")           # BEDLAM2-only synthetic stream
    ap.add_argument("--mamma-holdout", default="")               # seq list -> excluded from training
    # Supervise each rotation channel in a DILATED band around its own part, so the 24 channels agree at
    # their boundaries and an argmax flip stops producing a 13 deg jump. 0 = off (current behaviour).
    ap.add_argument("--rot-dilate", type=int, default=0)
    ap.add_argument("--val-every", type=int, default=1000)
    ap.add_argument("--save-every", type=int, default=1000)
    ap.add_argument("--log-every", type=int, default=50)
    args = ap.parse_args()

    rank, world, local = ddp_setup()
    device = f"cuda:{local}"
    rundir = f"{RUNS}/{args.name}"
    if is_main():
        os.makedirs(rundir, exist_ok=True)
        from torch.utils.tensorboard import SummaryWriter
        tb = SummaryWriter(rundir)
        print(f"world={world} data={args.data} init={args.init} "
              f"batch/gpu={args.batch} steps={args.steps} lr={args.lr}", flush=True)

    scratch = args.init.lower() in ("none", "scratch", "")
    # scratch: DINOv3 pretrained backbone + fresh decoder/heads. else: load our IUV ckpt (backbone incl).
    # resume: full model+opt+sched+step from a prior run (no backbone download needed).
    model = DINOv3DPT(backbone=args.backbone,
                      layers=tuple(int(x) for x in args.layers.split(",")),
                      pretrained=(scratch and not args.resume), decoupled=True, orient=args.orient,
                      rot_per_part=args.rot_per_part).to(device)
    resume_ck = None
    if args.resume:
        resume_ck = torch.load(args.resume, map_location="cpu")
        model.load_state_dict(resume_ck["model"])           # full architecture match
        if is_main():
            print(f"RESUME weights from {args.resume} @ step {resume_ck.get('step', 0)}", flush=True)
    elif scratch:
        if is_main():
            print(f"from-scratch: pretrained {args.backbone} backbone, fresh decoder+heads", flush=True)
    else:
        sd = torch.load(args.init, map_location="cpu")["model"]
        # non-strict: a fresh rot_head is absent from the IUV init checkpoint
        missing, unexpected = model.load_state_dict(sd, strict=False)
        if is_main():
            print(f"loaded init {args.init} | missing={[m.split('.')[0] for m in missing][:4]} "
                  f"unexpected={len(unexpected)}", flush=True)
    if dist.is_initialized():
        model = DDP(model, device_ids=[local], find_unused_parameters=True)
    raw = model.module if isinstance(model, DDP) else model

    pg = (param_groups_llrd(raw, args.lr, args.llrd_decay) if args.llrd
          else param_groups(raw, args.lr, args.bb_mult))
    if is_main():
        lrs = sorted({round(g["lr"], 8) for g in pg})
        print(f"optim: {'LLRD decay=%.2f' % args.llrd_decay if args.llrd else 'flat bb_mult=%.2f' % args.bb_mult}"
              f" | {len(pg)} groups | lr range [{min(lrs):.2e}, {max(lrs):.2e}]", flush=True)
    opt = torch.optim.AdamW(pg, weight_decay=0.05, betas=(0.9, 0.999))
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: lr_lambda(s, args.warmup, args.steps))

    start_step = 0
    if resume_ck is not None:
        start_step = resume_ck.get("step", 0)
        if resume_ck.get("opt") is not None and resume_ck.get("sched") is not None:
            opt.load_state_dict(resume_ck["opt"]); sched.load_state_dict(resume_ck["sched"])  # true resume
        else:                                                # legacy model-only ckpt: seed weights only,
            for _ in range(start_step):                      # fast-forward LR onto the (new) --steps cosine
                sched.step()
        if is_main():
            print(f"resumed at step {start_step}/{args.steps}, lr now {sched.get_last_lr()[-1]:.2e}", flush=True)

    coco_loader = build_coco_loader(COCO_TRAIN_ANN, COCO_TRAIN_IMG, batch=args.batch,
                                    workers=args.workers, input_size=args.input_size,
                                    train=True, seed=rank, distributed=dist.is_initialized())
    coco_val = build_coco_loader(COCO_VAL_ANN, COCO_VAL_IMG, batch=args.batch,
                                 workers=4, input_size=args.input_size, train=False,
                                 distributed=dist.is_initialized())
    use_csc = args.w_xyz > 0 or args.w_xyzgrad > 0
    csc = None
    if use_csc:
        from orient.csc import CSC
        csc = CSC(args.csc_K).to(device)          # buffers only, no params: stays outside DDP
        if is_main():
            print(f"[csc] K={args.csc_K} gather={args.csc_gather} tau={args.csc_tau*1000:.0f}mm "
                  f"w_xyz={args.w_xyz} w_grad={args.w_xyzgrad} uv_asset={args.uv_asset}", flush=True)
    bedlam_it = None
    if args.data in ("joint", "bedlam"):
        bedlam_loader = build_loader("train", batch=args.batch, workers=args.workers,
                                     input_size=args.input_size, train=True, orient=args.orient,
                                     csc=use_csc, uv_asset=args.uv_asset, gt512=args.gt512,
                                     hidden_mask=args.hidden_mask, mamma=not args.no_mamma,
                                     mamma_holdout=args.mamma_holdout)
        bedlam_it = iter(bedlam_loader)

    def coco_epochs():
        ep = 0
        while True:
            if coco_loader.sampler_ref is not None:
                coco_loader.sampler_ref.set_epoch(ep)
            for b in coco_loader:
                yield b
            ep += 1
    coco_it = coco_epochs()

    model.train()
    step = start_step
    t_log = time.time()
    while step < args.steps:
        # bedlam: always BEDLAM dense; joint: 1 COCO step per --coco-period; coco: always COCO sparse
        use_bedlam = (args.data == "bedlam") or \
            (args.data == "joint" and step % args.coco_period != args.coco_period - 1)
        if use_bedlam:
            try:
                b = next(bedlam_it)
            except StopIteration:
                bedlam_it = iter(bedlam_loader); b = next(bedlam_it)
            img = b["img"].to(device, non_blocking=True)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                out = model(img)
                lossfn = dp_losses_512 if args.gt512 else dp_losses
                kw = dict(w_uv=args.w_uv, w_rot=args.w_rot, w_xyz=args.w_xyz, w_xg=args.w_xyzgrad,
                          csc=csc, csc_gather=args.csc_gather, csc_tau=args.csc_tau,
                          csc_clip=args.csc_clip)
                if args.gt512:
                    kw["rot_dilate"] = args.rot_dilate
                L = lossfn(out, b, **kw)
        else:
            b = next(coco_it)
            img = b["img"].to(device, non_blocking=True)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                # v2-chart models: COCO UV is v1-chart -> skip UV loss on the re-flattened
                # hands/feet (folded in v1 = ambiguous target anyway); part CE keeps all points
                skip = (3, 4, 5, 6) if "v2" in args.uv_asset else ()
                out = model(img); L = dp_losses_sparse(out, b, w_uv=args.w_uv, uv_skip=skip)
        opt.zero_grad(set_to_none=True)
        L["loss"].backward()
        torch.nn.utils.clip_grad_norm_(raw.parameters(), 1.0)
        opt.step(); sched.step(); step += 1

        if is_main() and step % args.log_every == 0:
            lr = sched.get_last_lr()[-1]
            torch.cuda.synchronize(device)
            dt = time.time() - t_log
            ips = args.batch * world * args.log_every / max(dt, 1e-6)
            src = "BED" if use_bedlam else "COCO"
            rstr = f" rot {L['rot'].item():.3f} geo {L['geo'].item():.1f}" if "rot" in L and use_bedlam else ""
            if use_csc and use_bedlam:
                rstr += f" xyz {L['xyz'].item()*1000:.1f}mm grad {L['xg'].item()*1000:.2f} surf {L['mm'].item():.1f}mm"
            print(f"step {step}/{args.steps} [{src}] loss {L['loss'].item():.3f} "
                  f"mask {L['mask'].item():.3f} ce {L['ce'].item():.3f} uv {L['uv'].item():.4f}{rstr} "
                  f"lr {lr:.2e} | {ips:.1f} img/s", flush=True)
            tb.add_scalar("train/loss", L["loss"].item(), step)
            tb.add_scalar("train/ce", L["ce"].item(), step)
            tb.add_scalar("train/uv", L["uv"].item(), step)
            tb.add_scalar("train/mask", L["mask"].item(), step)
            if "rot" in L and use_bedlam:
                tb.add_scalar("train/rot", L["rot"].item(), step)
                tb.add_scalar("train/geo_deg", L["geo"].item(), step)
            if use_csc and use_bedlam:
                tb.add_scalar("train/xyz_mm", L["xyz"].item() * 1000, step)
                tb.add_scalar("train/xyz_grad", L["xg"].item() * 1000, step)
                tb.add_scalar("train/surf_err_mm", L["mm"].item(), step)
            t_log = time.time()

        if step % args.val_every == 0:
            mv = sparse_val(model, coco_val, device)
            if is_main():
                print(f"  [val step {step}] pt_acc={mv['pt_acc']:.4f} uv_mae={mv['uv_mae']:.4f}",
                      flush=True)
                for k, v in mv.items():
                    tb.add_scalar(f"val/{k}", v, step)
        if is_main() and step % args.save_every == 0:
            torch.save({"step": step, "model": raw.state_dict(),
                        "opt": opt.state_dict(), "sched": sched.state_dict()},  # full state -> true resume
                       f"{rundir}/ckpt_{step:06d}.pt")

    if is_main():
        torch.save({"step": step, "model": raw.state_dict(),
                    "opt": opt.state_dict(), "sched": sched.state_dict()}, f"{rundir}/final.pt")
        print("done", flush=True)
    if dist.is_initialized():
        dist.destroy_process_group()


if __name__ == "__main__":
    main()

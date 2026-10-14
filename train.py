"""Train a Pixel Frame Field model (research/densepose/finetune.py, the script both released stages ran).

  torchrun --standalone --nproc_per_node=4 train.py --config configs/stage1_synthetic.yaml
  torchrun --standalone --nproc_per_node=4 train.py --config configs/stage2_joint_coco.yaml

Stage 1 trains from the DINOv3 backbone on synthetic data only (BEDLAM2 + MAMMA, about 50/50); stage 2 fine-tunes
it jointly with DensePose-COCO (one COCO step in every `coco_period`). Rerunning the same command resumes from the
newest checkpoint of the run directory (model, optimiser, schedule and step).
"""
import argparse
import glob
import math
import os
import time

import torch
import torch.distributed as dist
import torch.nn.functional as F
import yaml
from torch.nn.parallel import DistributedDataParallel as DDP

from pff.csc import CSC
from pff.data.coco import build_coco_loader
from pff.data.synthetic import build_loader, shard_list
from pff.losses import dense_losses, sparse_losses
from pff.model import PixelFrameField


def lr_lambda(step, warmup, total):
    """Linear warm-up, then one cosine to zero."""
    if step < warmup:
        return step / max(1, warmup)
    t = (step - warmup) / max(1, total - warmup)
    return 0.5 * (1 + math.cos(math.pi * min(t, 1.0)))


def param_groups_flat(model, lr, bb_mult):
    """Backbone at lr * bb_mult, decoder and heads at lr."""
    bb, rest = [], []
    for n, p in model.named_parameters():
        if p.requires_grad:
            (bb if n.startswith("bb.") else rest).append(p)
    return [{"params": bb, "lr": lr * bb_mult}, {"params": rest, "lr": lr}]


def param_groups_llrd(model, lr, decay):
    """Layer-wise learning-rate decay: decoder, heads and the top block at lr, each deeper block times `decay`,
    the embeddings smallest."""
    n_layers = len(model.bb.blocks) + 1

    def layer_id(name):
        if not name.startswith("bb."):
            return n_layers
        nm = name[3:]
        if nm.startswith("blocks."):
            return int(nm.split(".")[1]) + 1
        if any(nm.startswith(k) for k in ("patch_embed", "cls_token", "pos_embed", "reg_token", "mask_token")):
            return 0
        return n_layers

    groups = {}
    for n, p in model.named_parameters():
        if p.requires_grad:
            lid = layer_id(n)
            groups.setdefault(lid, {"params": [], "lr": lr * decay ** (n_layers - lid)})["params"].append(p)
    return [groups[k] for k in sorted(groups)]


@torch.no_grad()
def sparse_val(model, loader, device, max_batches=30):
    """Quick COCO minival trend (point part accuracy, per-point UV error); the real metric is eval/coco_gps.py."""
    model.eval()
    correct = torch.zeros(2, device=device); uverr = torch.zeros(2, device=device)
    for i, b in enumerate(loader):
        if i >= max_batches:
            break
        with torch.autocast("cuda", dtype=torch.bfloat16):
            out = model(b["img"].to(device, non_blocking=True))
        for bi, g in enumerate(b["gt_sparse"]):
            pts = g["pts"].to(device)
            if pts.shape[0] == 0:
                continue
            I = g["I"].to(device); uvg = g["uv"].to(device)
            grid = pts.view(1, -1, 1, 2)
            pl = F.grid_sample(out["part"][bi:bi + 1].float(), grid, align_corners=False)[0, :, :, 0].t()
            correct[0] += (pl.argmax(1) == I).sum(); correct[1] += I.numel()
            uvpt = F.grid_sample(out["uv"][bi:bi + 1].float(), grid, align_corners=False)[0, :, :, 0].t().view(-1, 24, 2)
            uverr[0] += (uvpt[torch.arange(uvpt.shape[0], device=device), I] - uvg).abs().sum()
            uverr[1] += I.numel() * 2
    if dist.is_initialized():
        dist.all_reduce(correct); dist.all_reduce(uverr)
    model.train()
    return {"pt_acc": (correct[0] / correct[1].clamp(min=1)).item(), "uv_mae": (uverr[0] / uverr[1].clamp(min=1)).item()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    args = ap.parse_args()
    cfg = yaml.safe_load(open(args.config))
    E = os.path.expanduser
    tr, dc = cfg["train"], cfg["data"]

    if "RANK" in os.environ:
        rank, world, local = int(os.environ["RANK"]), int(os.environ["WORLD_SIZE"]), int(os.environ["LOCAL_RANK"])
        torch.cuda.set_device(local); dist.init_process_group("nccl")
    else:
        rank, world, local = 0, 1, 0
    main_rank = rank == 0
    device = f"cuda:{local}"
    rundir = E(cfg["run_dir"])
    if main_rank:
        os.makedirs(rundir, exist_ok=True)
        from torch.utils.tensorboard import SummaryWriter
        tb = SummaryWriter(rundir)

    # init: "pretrained" = the DINOv3 backbone through timm and fresh decoder and heads (stage 1); a path = a
    # trained checkpoint (stage 2); the newest checkpoint of run_dir, if any, always wins (resume).
    ckpts = sorted(glob.glob(f"{rundir}/ckpt_*.pt"))
    resume = ckpts[-1] if ckpts else None
    model = PixelFrameField(pretrained=(cfg["init"] == "pretrained" and resume is None)).to(device)
    resume_ck = None
    if resume:
        resume_ck = torch.load(resume, map_location="cpu", weights_only=False)
        model.load_state_dict(resume_ck["model"])
    elif cfg["init"] != "pretrained":
        sd = torch.load(E(cfg["init"]), map_location="cpu", weights_only=False)
        model.load_state_dict(sd.get("model", sd))
    if main_rank:
        print(f"world {world} | init {cfg['init']} | resume {resume}", flush=True)
    if dist.is_initialized():
        model = DDP(model, device_ids=[local], find_unused_parameters=True)
    raw = model.module if isinstance(model, DDP) else model

    pg = (param_groups_llrd(raw, tr["lr"], tr["llrd_decay"]) if tr.get("llrd_decay")
          else param_groups_flat(raw, tr["lr"], tr["bb_mult"]))
    opt = torch.optim.AdamW(pg, weight_decay=0.05, betas=(0.9, 0.999))
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: lr_lambda(s, tr["warmup"], tr["steps"]))
    step = 0
    if resume_ck is not None:
        step = resume_ck["step"]
        opt.load_state_dict(resume_ck["opt"]); sched.load_state_dict(resume_ck["sched"])

    L = cfg["loss"]
    csc = CSC(512).to(device) if (L["w_xyz"] > 0 or L["w_xg"] > 0) else None
    synth = build_loader(shard_list(E(dc["bedlam_shards"]), E(dc["mamma_shards"]) if dc.get("mamma_shards") else None,
                                    "train", E(dc["mamma_holdout"]) if dc.get("mamma_holdout") else None),
                         batch=tr["batch"], workers=tr["workers"], train=True, uv_asset=dc["uv_asset"],
                         hidden_mask=dc["hidden_mask"])
    synth_it = iter(synth)
    coco_period = tr.get("coco_period", 0)                 # 0 = synthetic only
    coco_it = None
    if coco_period:
        coco = build_coco_loader(E(dc["coco_train_ann"]), E(dc["coco_train_img"]), batch=tr["batch"],
                                 workers=tr["workers"], train=True, seed=rank, distributed=dist.is_initialized())

        def coco_epochs():
            ep = 0
            while True:
                if coco.sampler_ref is not None:
                    coco.sampler_ref.set_epoch(ep)
                yield from coco
                ep += 1
        coco_it = coco_epochs()
    val = None
    if dc.get("coco_val_ann"):
        val = build_coco_loader(E(dc["coco_val_ann"]), E(dc["coco_val_img"]), batch=tr["batch"], workers=4,
                                train=False, distributed=dist.is_initialized())

    model.train()
    t_log = time.time()
    while step < tr["steps"]:
        use_synth = not coco_period or step % coco_period != coco_period - 1
        if use_synth:
            b = next(synth_it)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                out = model(b["img"].to(device, non_blocking=True))
                losses = dense_losses(out, b, w_uv=L["w_uv"], w_rot=L["w_rot"], w_xyz=L["w_xyz"], w_xg=L["w_xg"],
                                      csc=csc, csc_tau=L["csc_tau"], csc_clip=L["csc_clip"])
        else:
            b = next(coco_it)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                out = model(b["img"].to(device, non_blocking=True))
                losses = sparse_losses(out, b, w_uv=L["w_uv"])
        opt.zero_grad(set_to_none=True)
        losses["loss"].backward()
        torch.nn.utils.clip_grad_norm_(raw.parameters(), 1.0)
        opt.step(); sched.step(); step += 1

        if main_rank and step % tr["log_every"] == 0:
            dt = time.time() - t_log
            src = "SYN" if use_synth else "COCO"
            msg = " ".join(f"{k} {v.item():.4f}" for k, v in losses.items())
            print(f"step {step}/{tr['steps']} [{src}] {msg} lr {sched.get_last_lr()[-1]:.2e} | "
                  f"{tr['batch'] * world * tr['log_every'] / max(dt, 1e-6):.1f} img/s", flush=True)
            for k, v in losses.items():
                tb.add_scalar(f"train/{k}", v.item(), step)
            t_log = time.time()
        if val is not None and step % tr["val_every"] == 0:
            mv = sparse_val(model, val, device)
            if main_rank:
                print(f"  [val {step}] pt_acc {mv['pt_acc']:.4f} uv_mae {mv['uv_mae']:.4f}", flush=True)
                for k, v in mv.items():
                    tb.add_scalar(f"val/{k}", v, step)
        if main_rank and (step % tr["save_every"] == 0 or step == tr["steps"]):
            torch.save({"step": step, "model": raw.state_dict(), "opt": opt.state_dict(), "sched": sched.state_dict()},
                       f"{rundir}/ckpt_{step:06d}.pt")
    if dist.is_initialized():
        dist.destroy_process_group()


if __name__ == "__main__":
    main()

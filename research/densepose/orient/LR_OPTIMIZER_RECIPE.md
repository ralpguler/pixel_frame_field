# Optimizer & learning-rate recipe for fine-tuning DINOv3 (ViT-L/16) on a dense task

This is the setup we use to train a DINOv3 ViT-L/16 backbone + DPT decoder + task heads
(DensePose IUV + per-pixel orientation) on BEDLAM. It is written so another agent can reproduce
the LR/optimizer strategy on a similar dense-prediction task. Code lives in
`densepose/finetune.py` (`param_groups`, `param_groups_llrd`, `main`) and `densepose/model.py`.

---

## TL;DR recipe (what to actually set)

- **Optimizer:** `AdamW`, `betas=(0.9, 0.999)`, `weight_decay=0.05`.
- **Backbone LR strategy: layer-wise LR decay (LLRD), NOT a flat backbone multiplier.**
  - Heads/decoder and the **top** transformer block train at the full base LR.
  - Each deeper block is scaled by `decay**depth` toward the input; embeddings get the smallest LR.
  - `decay = 0.75` for ViT-L (MAE convention); `0.8` for ViT-L/16 (BEiT v2). We use **0.75**.
- **Base LR:** `3e-4` is a safe strong default for from-scratch-head training on a large dataset;
  `1e-3` (the MAE ViT-L value) also works and was marginally better on our dense rotation metric.
  For fine-tuning *from an existing task checkpoint* (heads already good), drop to `~5e-5`.
- **Schedule:** linear **warmup** (we use 500 steps) → **cosine decay to 0** over the run.
- **Grad clipping:** global-norm clip at **1.0** (important with the higher backbone LR).
- **Precision:** `bf16` autocast. **DDP** with `find_unused_parameters=True` (heads not always all used).
- **Freeze:** only `patch_embed` is frozen (`requires_grad=False`); everything else (302M/303M params)
  trains. The whole backbone is updated, just at layer-scaled LRs.
- **Effective batch:** 32/GPU × 4 GPUs = 128. Input 512×512.

---

## The optimizer / param-group code

```python
# AdamW over LLRD param groups
pg = param_groups_llrd(model, lr=3e-4, decay=0.75)   # or param_groups(model, lr, bb_mult) for flat
opt = torch.optim.AdamW(pg, weight_decay=0.05, betas=(0.9, 0.999))
# one cosine-with-warmup lambda applied to ALL groups (scales each group's own base lr)
sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: lr_lambda(s, warmup=500, total=75000))

def lr_lambda(step, warmup, total):
    if step < warmup:
        return step / max(1, warmup)                 # linear warmup
    t = (step - warmup) / max(1, total - warmup)
    return 0.5 * (1 + math.cos(math.pi * min(t, 1.0)))   # cosine -> 0

def param_groups_llrd(model, lr, decay):
    """Heads + top backbone block at full `lr`; each deeper block scaled by decay**depth."""
    n_blocks = len(model.bb.blocks)
    num_layers = n_blocks + 1                         # embeddings = layer 0

    def layer_id(name):
        if not name.startswith("bb."):
            return num_layers                         # heads / decoder -> full lr
        nm = name[3:]
        if nm.startswith("blocks."):
            return int(nm.split(".")[1]) + 1          # block i -> i+1
        if any(nm.startswith(k) for k in ("patch_embed", "cls_token", "pos_embed",
                                          "reg_token", "mask_token", "storage_tokens")):
            return 0                                   # embeddings -> smallest lr
        return num_layers                             # final norm etc -> top

    groups = {}
    for n, p in model.named_parameters():
        if not p.requires_grad:
            continue
        lid = layer_id(n)
        scale = decay ** (num_layers - lid)           # top=decay^0=1, embeddings=decay^num_layers
        groups.setdefault(lid, {"params": [], "lr": lr * scale})["params"].append(p)
    return [groups[k] for k in sorted(groups)]
```

For ViT-L/24 with `lr=3e-4, decay=0.75` this produces **26 param groups**, LR range
`[2.26e-7 (embeddings) … 3.0e-4 (heads + top block)]`. Top backbone block ≈ `2.25e-4` — i.e. ~7.5×
the LR a flat `0.1×` multiplier would give the whole backbone (`3e-5`).

The flat alternative (what we moved away from) for reference:
```python
def param_groups(model, lr, bb_mult):    # bb_mult=0.1 == the old CNN-detection heuristic
    bb   = [p for n,p in model.named_parameters() if p.requires_grad and     n.startswith("bb.")]
    head = [p for n,p in model.named_parameters() if p.requires_grad and not n.startswith("bb.")]
    return [{"params": bb, "lr": lr*bb_mult}, {"params": head, "lr": lr}]
```

---

## Reasoning

1. **Why not a flat `1/10` backbone LR?** The "backbone at 0.1× the head LR" rule comes from
   CNN object detection (Detectron2 / Mask R-CNN) with ImageNet-*classification* backbones. It is
   not the ViT fine-tuning convention. A single flat multiplier is wrong in both directions: too low
   for the **top** blocks (which should adapt most to the new task) and needlessly high for the
   **early** blocks (which encode general low-level features you want to preserve).

2. **Why LLRD?** A pretrained ViT is a hierarchy: early blocks = generic edges/texture, late blocks =
   semantic/task-relevant. LLRD lets late blocks move at ~full LR while early blocks stay near-frozen
   (`decay**depth` shrinks LR toward the input). This is the standard, well-tuned ViT fine-tuning
   recipe from MAE/BEiT. For **DINOv3 specifically** it matters more: DINOv3's dense features are
   Gram-anchored and degrade if perturbed too hard, so preserving the early layers (tiny LR) while
   adapting the top is exactly the right shape — many DINOv3 dense results even *freeze* the backbone
   entirely (the conservative extreme of this same axis).

3. **Why AdamW, wd=0.05, betas (0.9, 0.999)?** Standard for transformer fine-tuning; decoupled weight
   decay is the norm for ViTs. (A refinement we did *not* do but is common: exclude bias/norm/1-D
   params and positional embeddings from weight decay. Worth adding for a from-scratch run.)

4. **Why warmup + cosine?** A pretrained backbone is fragile in the first steps at a high LR; linear
   warmup (a few hundred steps) prevents early feature destruction. Cosine-to-0 is the default ViT
   schedule and gives a clean, monotone late-training improvement. With the *higher* backbone LR,
   warmup + grad-clip(1.0) are what keep the aggressive (1e-3) run stable — we verified no divergence.

5. **Base LR choice.** MAE uses base `1e-3` (ViT-L, decay 0.75). We tested `3e-4` and `1e-3`; both are
   stable. On a large synthetic dataset training fresh heads, `3e-4` is a conservative-strong default;
   `1e-3` fit the dense target slightly faster with no generalization penalty once warmed up. Fine-tuning
   from an already-good task checkpoint should use a much smaller base (`~5e-5`) to avoid forgetting.

---

## Empirical evidence (our task: per-pixel orientation on BEDLAM, ViT-L/16, ~37k steps, held-out val geodesic error)

| backbone LR strategy        | rotation geo° (lower=better) |
|-----------------------------|------------------------------|
| flat `0.1×`, base 3e-4      | 9.25 |
| **LLRD 0.75, base 3e-4**    | 8.98 |
| **LLRD 0.75, base 1e-3**    | **8.74** |

Rotation error improves **monotonically with backbone LR** (≈5.5% relative from flat→LLRD-1e-3), and the
higher-LR run had **no** drop on the real-image (COCO) generalization proxy after warmup. Conclusion:
the flat `0.1×` was under-training the backbone; LLRD with the top blocks near the head LR is better.

---

## Resources

- **MAE** — He et al., *Masked Autoencoders Are Scalable Vision Learners* (CVPR 2022): origin of the
  ViT fine-tuning recipe (AdamW, layer-wise LR decay, base LR ~1e-3 / decay 0.75 for ViT-L, warmup+cosine).
  Recipe details also in the follow-up *MAE Are Secretly Efficient Learners* (CVPRW 2024):
  https://openaccess.thecvf.com/content/CVPR2024W/ECV24/papers/Wei_Masked_Autoencoders_are_Secretly_Efficient_Learners__CVPRW_2024_paper.pdf
- **BEiT / BEiT v2** — layer-wise LR decay 0.65 (ViT-B) / **0.8 (ViT-L/16)**:
  https://arxiv.org/pdf/2208.06366
- **DINOv3** — Meta AI; Gram anchoring & dense-feature preservation, backbone-LR multiplier for
  fine-tuning stability, frozen-backbone dense results: https://ai.meta.com/blog/dinov3-self-supervised-vision-model/
- **Reference implementation** of `param_groups_lrd` (the canonical LLRD layer-id mapping this code mirrors):
  MAE repo `util/lr_decay.py` (github.com/facebookresearch/mae).
- Our code: `densepose/finetune.py` (`param_groups_llrd`, `lr_lambda`, `param_groups`, `main`),
  `densepose/model.py` (`DINOv3DPT`, `freeze_patch`).

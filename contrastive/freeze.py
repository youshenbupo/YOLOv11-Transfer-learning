"""
Freezing strategies and layer-wise learning rate utilities for contrastive learning.
Adapted for YOLO11 architecture.

YOLO11-nano block layout:
  Backbone: blocks 0-10 (Conv, C3k2, SPPF, C2PSA)
  Neck:     blocks 11-22 (Upsample, Concat, C3k2)
  Detect:   block 23
"""

import re
import torch
import torch.nn as nn

# YOLO11 default block boundaries
YOLO11_BACKBONE_END = 10  # Last backbone block
YOLO11_NECK_START = 11    # First neck block
YOLO11_DETECT_IDX = 23    # Detect head block


def _set_bn_eval(m: nn.Module):
    """Set BatchNorm to eval mode and freeze affine parameters."""
    if isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d, nn.SyncBatchNorm)):
        m.eval()
        if m.affine:
            if m.weight is not None:
                m.weight.requires_grad = False
            if m.bias is not None:
                m.bias.requires_grad = False


def freeze_backbone_upto_index(online_model: nn.Module,
                               upto: int,
                               freeze_bn_stats: bool = False,
                               verbose: bool = True):
    """
    Freeze backbone blocks from 0 up to 'upto' (inclusive).
    - upto >= 0: freeze 0..upto
    - upto < 0: unfreeze all backbone blocks
    Projection head is always kept trainable.
    """
    net = online_model.module if hasattr(online_model, "module") else online_model
    yolo = net.backbone_neck
    assert isinstance(yolo, nn.Module), "unexpected backbone_neck type"

    # Unfreeze all first
    for p in yolo.parameters():
        p.requires_grad = True

    frozen_blocks = set()
    if upto >= 0:
        for name, m in yolo.named_modules():
            mobj = re.match(r"^model\.(\d+)(?:\.|$)", name)
            if not mobj:
                continue
            idx = int(mobj.group(1))
            if idx <= upto:
                for p in m.parameters(recurse=True):
                    p.requires_grad = False
                if freeze_bn_stats:
                    _set_bn_eval(m)
                frozen_blocks.add(idx)

    # Projection heads always trainable
    proj_module = getattr(net, "projection_heads", None)
    if proj_module is None:
        proj_module = getattr(net, "projection_head")
    for p in proj_module.parameters():
        p.requires_grad = True

    if verbose:
        trn = sum(p.numel() for p in net.parameters() if p.requires_grad)
        frz = sum(p.numel() for p in net.parameters() if not p.requires_grad)
        print(f"[Freeze<=#{upto}] frozen_blocks={len(frozen_blocks)} "
              f"({sorted(list(frozen_blocks))[:8]}{' ...' if len(frozen_blocks) > 8 else ''}), "
              f"trainable_params={trn:,}, frozen_params={frz:,}")


def freeze_by_name_regex(online_model: nn.Module,
                         name_patterns,
                         freeze_bn_stats: bool = False,
                         verbose: bool = True,
                         ensure_other_trainable: bool = False):
    """Freeze modules matching given regex patterns."""
    net = online_model.module if hasattr(online_model, "module") else online_model
    assert hasattr(net, "backbone_neck"), "ContrastiveYOLO missing backbone_neck"
    yolo = net.backbone_neck
    assert isinstance(yolo, nn.Module), "unexpected backbone_neck type"

    regs = [re.compile(p) for p in name_patterns]
    matched_names = set()

    for name, m in yolo.named_modules():
        if name == "":
            continue
        if any(r.search(name) for r in regs):
            matched_names.add(name)
            for p in m.parameters(recurse=True):
                p.requires_grad = False
            if freeze_bn_stats:
                m.apply(_set_bn_eval)

    if ensure_other_trainable:
        for name, m in yolo.named_modules():
            if name == "" or name in matched_names:
                continue
            for p in m.parameters(recurse=True):
                p.requires_grad = True

    trainable = sum(p.numel() for p in net.parameters() if p.requires_grad)
    frozen = sum(p.numel() for p in net.parameters() if not p.requires_grad)
    if verbose:
        print(f"[Freeze-Regex] patterns={name_patterns}, "
              f"matched_modules={len(matched_names)}, "
              f"trainable_params={trainable:,}, frozen_params={frozen:,}")
        if matched_names:
            show = sorted(matched_names)[:10]
            print("[Freeze-Regex] first 10 matched:", show)


def build_optimizer_layerwise(model_or_ddp: nn.Module,
                              base_lr: float,
                              weight_decay: float,
                              layerwise_decay: float):
    """
    Build AdamW optimizer with layer-wise learning rate decay.
    Later blocks (closer to head/neck) get larger LR.
    """
    net = model_or_ddp.module if hasattr(model_or_ddp, "module") else model_or_ddp

    params_info = []
    for n, p in net.named_parameters():
        if not p.requires_grad:
            continue
        m = re.search(r"\bmodel\.(\d+)\b", n) if ("backbone_neck" in n or "model." in n) else None
        blk = int(m.group(1)) if m else None
        params_info.append((n, p, blk))

    trainable_blocks = sorted({blk for _, _, blk in params_info if blk is not None})
    blk2rank = {blk: rank for rank, blk in enumerate(reversed(trainable_blocks))}

    groups = {}
    for n, p, blk in params_info:
        if blk is None:
            lr = base_lr
        else:
            rank = blk2rank[blk]
            lr = base_lr * (layerwise_decay ** rank)
        key = round(lr, 12)
        groups.setdefault(key, {"params": [], "lr": lr, "weight_decay": weight_decay})
        groups[key]["params"].append(p)

    print(f"[LW-LR] base_lr={base_lr} decay={layerwise_decay} groups={len(groups)}")
    sample = sorted((v["lr"], len(v["params"])) for v in groups.values())
    print(f"[LW-LR] lr groups (lr, #params): {sample[:6]}{' ...' if len(sample) > 6 else ''}")

    return torch.optim.AdamW(list(groups.values()), lr=base_lr, weight_decay=0.0)


def print_trainable_report(model: nn.Module, max_names: int = 10):
    """Print a summary of trainable vs frozen parameters."""
    b_trn = sum(p.numel() for _, p in model.backbone_neck.named_parameters() if p.requires_grad)
    b_frz = sum(p.numel() for _, p in model.backbone_neck.named_parameters() if not p.requires_grad)
    proj_module = getattr(model, "projection_heads", None)
    if proj_module is None:
        proj_module = getattr(model, "projection_head")
    h_trn = sum(p.numel() for _, p in proj_module.named_parameters() if p.requires_grad)
    h_frz = sum(p.numel() for _, p in proj_module.named_parameters() if not p.requires_grad)
    trn_names = [n for n, p in model.named_parameters() if p.requires_grad]
    print("[Trainable Report] ==============================")
    print(f"[Trainable Report] Backbone_neck -> trainable={b_trn:,}, frozen={b_frz:,}")
    print(f"[Trainable Report] Proj-Head -> trainable={h_trn:,}, frozen={h_frz:,}")
    print(f"[Trainable Report] Trainable param entries: {len(trn_names)}")
    print(f"[Trainable Report] First {max_names} trainable names: {trn_names[:max_names]}")
    print("[Trainable Report] ==============================")

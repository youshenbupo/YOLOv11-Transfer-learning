#!/usr/bin/env python
"""
Unified Contrastive Pretraining for YOLO11-UDA

Supports both SimCLR and MoCo methods with configurable freezing strategies.
Adapted for YOLO11 (Ultralytics) architecture.

Usage:
    # SimCLR with staged unfreezing (recommended)
    python train_contrastive.py --method simclr --staged-unfreeze

    # MoCo with DDP
    python train_contrastive.py --method moco --device 0,1,2,3

    # Freeze backbone only
    python train_contrastive.py --method simclr --freeze-target backbone

    # No freezing
    python train_contrastive.py --method simclr --freeze-target none
"""

import os
import sys
import copy
import argparse
import warnings
from typing import List

import numpy as np
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.multiprocessing as mp
from torch.utils.data import DataLoader
from torchvision.transforms import v2 as transforms
from modules.losses import domain_aware_contrastive_loss

from contrastive import (
    setup_environment, get_yolo_backbone_neck, ContrastiveYOLO,
    CombinedDataset, RatioBatchSampler,
    freeze_backbone_upto_index, freeze_by_name_regex,
    build_optimizer_layerwise, print_trainable_report,
    info_nce_loss, moco_loss, dequeue_and_enqueue,
    save_loss_curve, _csv_ints, _csv_list,
    init_distributed_mode, concat_all_gather,
    batch_shuffle_ddp, batch_unshuffle_ddp,
    YOLO11_BACKBONE_END, YOLO11_NECK_START, YOLO11_DETECT_IDX,
)

warnings.filterwarnings("ignore", message="torch.meshgrid:.*indexing")


# ----------------------------
# Debug helpers
# ----------------------------
def print_model_blocks(model: nn.Module, max_children: int = 8):
    """Print top-level model blocks for debugging."""
    if hasattr(model, "model") and isinstance(model.model, nn.Module):
        container = model.model
    else:
        container = model
    items = list(container._modules.items())
    n = len(items)
    print(f"[Arch] top blocks = {n} (container type: {type(container).__name__})")
    for i, (name, m) in enumerate(items):
        child_names = [n for n, _ in m.named_children()]
        shown = child_names[:max_children]
        extra = "" if len(child_names) <= max_children else f" ...(+{len(child_names)-max_children})"
        print(f"  model.{i:<2} | key='{name}' | {m.__class__.__name__:<22} | children: {shown}{extra}")


# ----------------------------
# Training engine
# ----------------------------
def build_gpu_augmenter(img_size: int, device):
    """Build GPU-based data augmentation pipeline for contrastive learning."""
    return nn.Sequential(
        transforms.RandomResizedCrop(size=(img_size, img_size), scale=(0.2, 1.0), antialias=True),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomApply([transforms.ColorJitter(0.4, 0.4, 0.2, 0.1)], p=0.8),
        transforms.RandomGrayscale(p=0.2),
        transforms.GaussianBlur(kernel_size=23, sigma=(0.1, 2.0)),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ).to(device)


def build_phases(opt) -> List[dict]:
    """Build training phases based on freeze strategy."""
    if opt.use_layerlock:
        freeze_steps = opt.layerlock_freeze_steps
        feature_levels = [level.lower() for level in opt.layerlock_feature_levels]
        if len(freeze_steps) != len(feature_levels):
            raise ValueError("layerlock freeze steps and feature levels must have the same length")

        phase_epochs = opt.layerlock_phase_epochs
        if phase_epochs:
            if len(phase_epochs) != len(freeze_steps):
                raise ValueError("layerlock phase epochs must match layerlock freeze steps")
        else:
            phase_epochs = _distribute_epochs(opt.epochs, len(freeze_steps))

        phases = []
        max_index = max(1, len(freeze_steps) - 1)
        for idx, (freeze_upto, feature_level, epochs) in enumerate(zip(freeze_steps, feature_levels, phase_epochs)):
            if epochs <= 0:
                continue
            ratio = idx / max_index
            loss_scale = opt.layerlock_min_loss_scale + ratio * (1.0 - opt.layerlock_min_loss_scale)
            lr_scale = 1.0 - ratio * (1.0 - opt.lr_drop)
            phases.append({
                "name": f"layerlock_{feature_level}_freeze<=#{freeze_upto}",
                "freeze_upto": freeze_upto,
                "epochs": epochs,
                "base_lr": opt.lr * lr_scale,
                "feature_level": feature_level,
                "domain_weight_scale": loss_scale,
                "adv_weight_scale": loss_scale,
            })
        return phases

    phases = []

    if opt.freeze_target == "none":
        phases.append({
            "name": "full_train",
            "freeze_upto": -1,
            "epochs": opt.epochs,
            "base_lr": opt.lr,
        })
    elif opt.freeze_target == "backbone":
        phases.append({
            "name": "stage1_freeze_backbone",
            "freeze_upto": opt.backbone_freeze_upto,
            "epochs": opt.stage1_epochs,
            "base_lr": opt.lr,
        })
        if opt.staged_unfreeze:
            for upto in opt.unfreeze_steps:
                phases.append({
                    "name": f"stage2_unfreeze<=#{upto}",
                    "freeze_upto": upto,
                    "epochs": opt.stage2_epochs,
                    "base_lr": opt.lr * opt.lr_drop,
                })
        else:
            phases.append({
                "name": "stage2_full_unfreeze",
                "freeze_upto": -1,
                "epochs": opt.epochs - opt.stage1_epochs,
                "base_lr": opt.lr * opt.lr_drop,
            })
    elif opt.freeze_target == "neck":
        neck_end = opt.detect_idx - 1
        patterns = [f"^model\\.(?:{i})(?:\\.|$)" for i in range(opt.neck_start_idx, neck_end + 1)]
        phases.append({
            "name": "stage1_freeze_neck",
            "freeze_upto": -1,
            "freeze_patterns": patterns,
            "epochs": opt.stage1_epochs,
            "base_lr": opt.lr,
        })
        if opt.staged_unfreeze:
            phases.append({
                "name": "stage2_unfreeze_all",
                "freeze_upto": -1,
                "epochs": opt.epochs - opt.stage1_epochs,
                "base_lr": opt.lr * opt.lr_drop,
            })
        else:
            phases.append({
                "name": "stage2_full_unfreeze",
                "freeze_upto": -1,
                "epochs": opt.epochs - opt.stage1_epochs,
                "base_lr": opt.lr * opt.lr_drop,
            })
    elif opt.freeze_target == "both":
        phases.append({
            "name": "stage1_freeze_backbone",
            "freeze_upto": opt.backbone_freeze_upto,
            "epochs": opt.stage1_epochs,
            "base_lr": opt.lr,
        })
        if opt.staged_unfreeze:
            for upto in opt.unfreeze_steps:
                phases.append({
                    "name": f"stage2_unfreeze<=#{upto}",
                    "freeze_upto": upto,
                    "epochs": opt.stage2_epochs,
                    "base_lr": opt.lr * opt.lr_drop,
                })
        else:
            phases.append({
                "name": "stage2_full_unfreeze",
                "freeze_upto": -1,
                "epochs": opt.epochs - opt.stage1_epochs,
                "base_lr": opt.lr * opt.lr_drop,
            })

    return phases


def _distribute_epochs(total_epochs: int, num_phases: int) -> List[int]:
    base = total_epochs // num_phases
    remainder = total_epochs % num_phases
    return [base + (1 if idx < remainder else 0) for idx in range(num_phases)]


def main_worker(local_rank_cli: int, opt):
    """Main training worker (supports single-GPU and multi-GPU DDP)."""
    if "WORLD_SIZE" not in os.environ:
        os.environ["WORLD_SIZE"] = os.environ.get("WORLD_SIZE", "1")
    is_dist, rank, world_size, local_rank = init_distributed_mode(local_rank_cli)

    # Device setup
    if is_dist and torch.cuda.is_available():
        torch.cuda.set_device(local_rank)
        device = torch.device(f"cuda:{local_rank}")
    else:
        if opt.device == "cpu" or not torch.cuda.is_available():
            device = torch.device("cpu")
        else:
            dev_id = opt.device.split(",")[0].strip()
            device = torch.device(f"cuda:{dev_id}")

    if rank == 0:
        print(f"[rank={rank}] device={device} world_size={world_size}")

    # Environment setup
    setup_environment(rank, opt.yolo_weights_path, opt.pos_dir, opt.neg_dir)

    # Data transforms (CPU-side)
    cpu_transform = transforms.Compose([
        transforms.ToImage(),
        transforms.ToDtype(torch.float32, scale=True),
        transforms.Resize((opt.img_size, opt.img_size), antialias=True)
    ])

    # Load YOLO11 model
    backbone_neck, p5_dim, config = get_yolo_backbone_neck(opt.yolo_weights_path, device, rank)
    feature_dims = config.get("feature_dims", {"p5": p5_dim})

    if rank == 0:
        print_model_blocks(backbone_neck, max_children=8)

    # Build contrastive model
    if opt.method == "moco":
        online = ContrastiveYOLO(
            copy.deepcopy(backbone_neck),
            feature_dims,
            proj_dim=opt.proj_dim,
            use_cbam=opt.use_cbam,
            cbam_reduction=opt.cbam_reduction,
            use_domain_adversarial=opt.use_domain_adversarial,
            domain_hidden_dim=opt.domain_hidden_dim,
            grl_alpha=opt.grl_alpha,
        ).to(device)
        momentum = ContrastiveYOLO(
            copy.deepcopy(backbone_neck),
            feature_dims,
            proj_dim=opt.proj_dim,
            use_cbam=opt.use_cbam,
            cbam_reduction=opt.cbam_reduction,
            use_domain_adversarial=False,
        ).to(device)
        for pq, pk in zip(online.parameters(), momentum.parameters()):
            pk.data.copy_(pq.data)
            pk.requires_grad = False
    else:
        online = ContrastiveYOLO(
            copy.deepcopy(backbone_neck),
            feature_dims,
            proj_dim=opt.proj_dim,
            use_cbam=opt.use_cbam,
            cbam_reduction=opt.cbam_reduction,
            use_domain_adversarial=opt.use_domain_adversarial,
            domain_hidden_dim=opt.domain_hidden_dim,
            grl_alpha=opt.grl_alpha,
        ).to(device)
        momentum = None

    # Convert BN to SyncBN for DDP
    if is_dist:
        online = nn.SyncBatchNorm.convert_sync_batchnorm(online)

    # Wrap with DDP
    if is_dist:
        online = nn.parallel.DistributedDataParallel(
            online,
            device_ids=[local_rank] if torch.cuda.is_available() else None,
            output_device=local_rank if torch.cuda.is_available() else None,
            broadcast_buffers=False,
            find_unused_parameters=(opt.method == "moco"),
        )

    # MoCo queue
    queue = None
    queue_ptr = 0
    if opt.method == "moco":
        queue = torch.randn(opt.queue_size, opt.proj_dim, device=device)
        queue = F.normalize(queue, dim=1)

    # Dataset
    dataset = CombinedDataset(opt.pos_dir, opt.neg_dir, cpu_transform, rank=rank, world_size=world_size, img_size=opt.img_size)
    if len(dataset) == 0:
        if rank == 0:
            print("Dataset is empty, exiting.")
        return

    sampler = RatioBatchSampler(dataset, opt.num_positives, opt.batch_size, drop_last=True)
    if len(sampler) == 0:
        if rank == 0:
            print("Not enough data for one batch, exiting.")
        return

    loader = DataLoader(
        dataset,
        batch_sampler=sampler,
        num_workers=opt.num_workers,
        pin_memory=torch.cuda.is_available(),
    )

    # Build training phases
    phases = build_phases(opt)
    total_epochs = sum(p["epochs"] for p in phases)
    if rank == 0:
        print(f"[Plan] total_epochs={total_epochs} phases={[(p['name'], p['epochs']) for p in phases]}")
        print("-" * 40)

    # Training loop
    loss_log = []
    phase_boundaries = []
    global_epoch = 0

    for phase in phases:
        if rank == 0:
            phase_boundaries.append((global_epoch + 1, phase["name"]))

        # Apply freezing
        if "freeze_patterns" in phase:
            freeze_by_name_regex(online, phase["freeze_patterns"],
                                 freeze_bn_stats=opt.freeze_bn_stats, verbose=(rank == 0))
        else:
            freeze_backbone_upto_index(online, upto=phase["freeze_upto"],
                                       freeze_bn_stats=opt.freeze_bn_stats, verbose=(rank == 0))

        if rank == 0:
            model_ref = online.module if hasattr(online, "module") else online
            print_trainable_report(model_ref)

        # Build optimizer
        optimizer = build_optimizer_layerwise(
            online,
            base_lr=phase["base_lr"],
            weight_decay=opt.weight_decay,
            layerwise_decay=opt.layerwise_lr_decay
        )

        if rank == 0:
            print(f"[{phase['name']}] start: epochs={phase['epochs']} base_lr={phase['base_lr']}")

        gpu_aug = build_gpu_augmenter(opt.img_size, device)

        # Train epochs
        for _ in range(phase["epochs"]):
            global_epoch += 1
            model = online.module if hasattr(online, "module") else online
            model.train()

            total = 0.0
            pbar = tqdm(loader, desc=f"{phase['name']} (epoch {global_epoch}/{total_epochs})", disable=(rank != 0))
            feature_level = phase.get("feature_level", "p5")
            domain_weight_scale = phase.get("domain_weight_scale", 1.0)
            adv_weight_scale = phase.get("adv_weight_scale", 1.0)

            for batch in pbar:
                imgs, domain_labels = batch
                imgs = imgs.to(device, non_blocking=True)
                domain_labels = domain_labels.to(device, non_blocking=True)
                v1 = gpu_aug(imgs)
                v2 = gpu_aug(imgs)
                adv_loss = None

                if opt.method == "simclr":
                    images = torch.cat([v1, v2], dim=0)
                    proj = online(images, feature_level=feature_level)
                    if opt.use_domain_loss:
                        loss = domain_aware_contrastive_loss(
                            proj,
                            domain_labels=domain_labels,
                            temperature=opt.temperature,
                            lambda_domain=opt.lambda_domain * domain_weight_scale,
                        )
                    else:
                        loss = info_nce_loss(proj, temperature=opt.temperature, device=device)
                else:  # moco
                    q = online(v1, feature_level=feature_level)
                    q = F.normalize(q, dim=1)

                    with torch.no_grad():
                        if is_dist:
                            v2s, idx_unshuffle = batch_shuffle_ddp(v2)
                            k = momentum(v2s, feature_level=feature_level)
                            k = F.normalize(k, dim=1)
                            k = batch_unshuffle_ddp(k, idx_unshuffle)
                        else:
                            k = momentum(v2, feature_level=feature_level)
                            k = F.normalize(k, dim=1)

                    loss = moco_loss(q, k, queue, opt.temperature)

                if opt.use_domain_adversarial:
                    domain_model = online.module if hasattr(online, "module") else online
                    domain_logits = domain_model.forward_domain(imgs).squeeze(-1)
                    adv_targets = domain_labels.float()
                    adv_loss = F.binary_cross_entropy_with_logits(domain_logits, adv_targets)
                    loss = loss + (opt.lambda_adv * adv_weight_scale) * adv_loss

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                # MoCo: update momentum encoder and queue
                if opt.method == "moco":
                    with torch.no_grad():
                        q_params = online.module.parameters() if hasattr(online, "module") else online.parameters()
                        for pq, pk in zip(q_params, momentum.parameters()):
                            pk.data.mul_(opt.momentum_coef).add_(pq.data, alpha=1.0 - opt.momentum_coef)
                        queue, queue_ptr = dequeue_and_enqueue(queue, k.detach(), queue_ptr)

                total += loss.item()
                if rank == 0:
                    postfix = {"loss": f"{loss.item():.4f}", "feat": feature_level}
                    if adv_loss is not None:
                        postfix["adv"] = f"{adv_loss.item():.4f}"
                    pbar.set_postfix(postfix)

            avg = total / max(1, len(loader))
            if rank == 0:
                loss_log.append((global_epoch, phase["name"], avg))
                print(f"[{phase['name']}] epoch_done avg_loss={avg:.4f}")

    # Save results
    if rank == 0:
        save_path = opt.save_path
        backbone_src = online.module.backbone_neck if hasattr(online, "module") else online.backbone_neck

        # Save backbone+neck state_dict
        torch.save(backbone_src.state_dict(), save_path)
        print(f"Saved backbone+neck state_dict to: {save_path}")

        # Optionally save as Ultralytics-compatible format
        if opt.save_full_pt:
            from ultralytics import YOLO
            full_model = YOLO(opt.yolo_weights_path)
            full_model.model.load_state_dict(backbone_src.state_dict(), strict=False)
            full_save = save_path.replace(".pt", "_full.pt")
            torch.save({"model": full_model.model}, full_save)
            print(f"Saved full model to: {full_save}")

        save_loss_curve(loss_log, phase_boundaries, opt.loss_png, opt.loss_csv)

    if is_dist:
        torch.distributed.barrier()
        torch.distributed.destroy_process_group()


# ----------------------------
# CLI
# ----------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Unified Contrastive Pretraining for YOLO11-UDA (SimCLR / MoCo)"
    )

    # Method selection
    parser.add_argument("--method", type=str, default="simclr", choices=["simclr", "moco"],
                        help="Contrastive learning method: simclr or moco")

    # Device
    parser.add_argument("--device", default="0", help='cuda device like "0" or "0,1" or "cpu"')

    # Training / Data
    parser.add_argument("--img-size", type=int, default=640, help="Input image size (H=W)")
    parser.add_argument("--batch-size", type=int, default=8, help="Batch size")
    parser.add_argument("--num-positives", type=int, default=4, help="Positive samples per batch")
    parser.add_argument("--num-workers", type=int, default=0,
                        help="DataLoader worker count. Use 0 on restricted Windows setups.")
    parser.add_argument("--temperature", type=float, default=0.1, help="Temperature coefficient")
    parser.add_argument("--proj-dim", type=int, default=128, help="Projection head output dimension")
    parser.add_argument("--epochs", type=int, default=10, help="Total training epochs")
    parser.add_argument("--use-cbam", action="store_true", default=False,
                        help="Apply CBAM to the P5 feature before the projection head")
    parser.add_argument("--cbam-reduction", type=int, default=16,
                        help="CBAM channel reduction ratio")
    parser.add_argument("--use-domain-loss", action="store_true", default=False,
                        help="Use domain-aware contrastive loss (SimCLR path only)")
    parser.add_argument("--lambda-domain", type=float, default=0.5,
                        help="Weight for the domain-aware contrastive term")
    parser.add_argument("--use-domain-adversarial", action="store_true", default=False,
                        help="Enable GRL-based domain adversarial training on P5 features")
    parser.add_argument("--lambda-adv", type=float, default=0.5,
                        help="Weight for the domain adversarial loss term")
    parser.add_argument("--domain-hidden-dim", type=int, default=256,
                        help="Hidden dimension for the domain discriminator")
    parser.add_argument("--grl-alpha", type=float, default=1.0,
                        help="Gradient reversal strength for domain adversarial training")

    # Optimizer / Layer-wise LR
    parser.add_argument("--lr", type=float, default=1e-4, help="Base learning rate")
    parser.add_argument("--weight-decay", type=float, default=1e-5, help="AdamW weight decay")
    parser.add_argument("--lr-drop", type=float, default=0.1, help="LR multiplier for later stages")
    parser.add_argument("--layerwise-lr-decay", type=float, default=0.85,
                        help="Layer-wise LR decay factor (0.8~0.95)")

    # Freeze strategy
    parser.add_argument("--freeze-target", type=str, default="backbone",
                        choices=["none", "backbone", "neck", "both"],
                        help="What to freeze during training")
    parser.add_argument("--use-layerlock", action="store_true", default=False,
                        help="LayerLock-inspired mode: progressively freeze shallow layers and transition feature levels")
    parser.add_argument("--layerlock-freeze-steps", type=_csv_ints, default="-1,2,6,10",
                        help='Comma-separated freeze steps for LayerLock mode, e.g. "-1,2,6,10"')
    parser.add_argument("--layerlock-feature-levels", type=_csv_list, default="p3,p4,p5,p5",
                        help='Comma-separated feature levels for LayerLock phases, e.g. "p3,p4,p5,p5"')
    parser.add_argument("--layerlock-phase-epochs", type=_csv_ints, default="",
                        help="Optional comma-separated epoch counts for each LayerLock phase")
    parser.add_argument("--layerlock-min-loss-scale", type=float, default=0.25,
                        help="Minimum scaling factor for domain and adversarial losses in the earliest LayerLock phase")
    parser.add_argument("--staged-unfreeze", action="store_true", default=False,
                        help="Use staged unfreezing strategy")
    parser.add_argument("--stage1-epochs", type=int, default=4,
                        help="Epochs for stage 1 (frozen training)")
    parser.add_argument("--stage2-epochs", type=int, default=2,
                        help="Epochs per unfreeze step in stage 2")
    parser.add_argument("--unfreeze-steps", type=_csv_ints, default="8,4,0",
                        help='Comma-separated block indices to unfreeze, e.g. "8,4,0"')
    parser.add_argument("--backbone-freeze-upto", type=int, default=YOLO11_BACKBONE_END,
                        help="Freeze backbone blocks 0..N (YOLO11 default: 10)")
    parser.add_argument("--neck-start-idx", type=int, default=YOLO11_NECK_START,
                        help="Neck start block index (YOLO11 default: 11)")
    parser.add_argument("--detect-idx", type=int, default=YOLO11_DETECT_IDX,
                        help="Detect head block index (YOLO11 default: 23)")
    parser.add_argument("--freeze-bn-stats", action="store_true", default=False,
                        help="Set BN in frozen modules to eval and freeze affine")

    # MoCo specific
    parser.add_argument("--queue-size", type=int, default=65536, help="MoCo queue size")
    parser.add_argument("--momentum-coef", type=float, default=0.999,
                        help="MoCo momentum encoder update coefficient")

    # Paths
    parser.add_argument("--yolo-weights-path", type=str, default="./weights/yolo11n.pt",
                        help="YOLO11 pretrained weights path")
    parser.add_argument("--pos-dir", type=str, default="./contra_img/positive_img",
                        help="Positive samples directory")
    parser.add_argument("--neg-dir", type=str, default="./contra_img/negative_img",
                        help="Negative samples directory")
    parser.add_argument("--save-path", type=str, default="contrastive_pretrained.pt",
                        help="Output weights path")
    parser.add_argument("--save-full-pt", action="store_true", default=False,
                        help="Also save complete model with Detect head")

    # Logging
    parser.add_argument("--loss-png", type=str, default="loss_curve.png",
                        help="Loss curve PNG filename")
    parser.add_argument("--loss-csv", type=str, default=None,
                        help="Loss log CSV filename")

    opt = parser.parse_args()

    if opt.use_domain_loss and opt.method != "simclr":
        raise ValueError("--use-domain-loss is currently supported only with --method simclr")

    # Launch
    if "," in opt.device and opt.device != "cpu":
        devs = opt.device.split(",")
        nprocs = len(devs)
        os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(devs)
        os.environ["WORLD_SIZE"] = str(nprocs)
        os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
        os.environ.setdefault("MASTER_PORT", "29500")
        mp.spawn(main_worker, args=(opt,), nprocs=nprocs, join=True)
    else:
        main_worker(0, opt)

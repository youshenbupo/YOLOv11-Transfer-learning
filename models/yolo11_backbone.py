"""
YOLO11 Backbone Adapter for Contrastive Learning.

Provides functions to load YOLO11 (Ultralytics) and extract backbone+neck
for contrastive pretraining, replacing the Detect head with Identity_YOLO.

YOLO11-nano architecture (24 blocks):
  Backbone (0-10): Conv, Conv, C3k2, Conv, C3k2, Conv, C3k2, Conv, C3k2, SPPF, C2PSA
  Neck     (11-22): Upsample, Concat, C3k2, Upsample, Concat, C3k2, Conv, Concat, C3k2, Conv, Concat, C3k2
  Head     (23):     Detect
"""

import torch
import torch.nn as nn


# YOLO11 variant configs: (backbone_end_idx, neck_p5_dim)
YOLO11_CONFIGS = {
    "yolo11n": {"backbone_end": 10, "p5_dim": 256, "blocks": 24},
    "yolo11s": {"backbone_end": 10, "p5_dim": 256, "blocks": 24},
    "yolo11m": {"backbone_end": 10, "p5_dim": 512, "blocks": 24},
    "yolo11l": {"backbone_end": 10, "p5_dim": 512, "blocks": 24},
    "yolo11x": {"backbone_end": 10, "p5_dim": 512, "blocks": 24},
}


class IdentityDetect(nn.Module):
    """No-op replacement for the Detect head during contrastive pretraining."""
    def __init__(self, original=None):
        super().__init__()
        if original is not None:
            for attr in ("f", "i", "type"):
                if hasattr(original, attr):
                    setattr(self, attr, getattr(original, attr))

    def forward(self, x):
        return x


def load_yolo11_backbone_neck(weights_path: str, device, rank=0):
    """
    Load a YOLO11 model from Ultralytics weights, replace the Detect head
    with IdentityDetect, and return the modified model.

    Args:
        weights_path: Path to YOLO11 .pt weights (e.g. 'weights/yolo11n.pt')
        device: torch device
        rank: process rank (0 = main)

    Returns:
        model: nn.Module with backbone+neck (Detect head replaced)
        p5_dim: int, the output channel dimension of the last feature map (P5)
        config: dict with architecture metadata
    """
    from ultralytics import YOLO

    if rank == 0:
        print(f"--- Loading YOLO11 backbone+neck from {weights_path} ---")

    # Load model
    yolo = YOLO(weights_path)
    model = yolo.model.to(device)

    # Detect architecture variant
    variant = None
    for key in YOLO11_CONFIGS:
        if key in weights_path.lower():
            variant = key
            break
    if variant is None:
        variant = "yolo11n"  # default
        if rank == 0:
            print(f"[Warning] Unknown variant, defaulting to {variant}")

    config = YOLO11_CONFIGS[variant]

    # Find and replace Detect head
    detect_idx = None
    for i, m in enumerate(model.model):
        if m.__class__.__name__ == "Detect":
            detect_idx = i
            break

    if detect_idx is None:
        detect_idx = len(model.model) - 1
        if rank == 0:
            print("[Warning] Detect head not found by name, using last block")

    # Replace Detect with IdentityDetect
    model.model[detect_idx] = IdentityDetect(model.model[detect_idx])

    # Infer P3/P4/P5 output dimensions directly from the modified model.
    model.eval()
    with torch.no_grad():
        dummy = torch.zeros(1, 3, 64, 64, device=device)
        feats = model(dummy)
    if not isinstance(feats, (list, tuple)) or len(feats) < 3:
        raise RuntimeError("Expected YOLO11 backbone+neck to return [P3, P4, P5] features.")

    feature_dims = {
        "p3": int(feats[0].shape[1]),
        "p4": int(feats[1].shape[1]),
        "p5": int(feats[2].shape[1]),
    }
    p5_dim = feature_dims["p5"]

    if rank == 0:
        print(f"  Architecture: {variant}")
        print(f"  Total blocks: {len(model.model)}")
        print(f"  Backbone: blocks 0-{config['backbone_end']}")
        print(f"  Neck: blocks {config['backbone_end']+1}-{detect_idx-1}")
        print(f"  Detect head replaced at block {detect_idx}")
        print(f"  Feature dims: {feature_dims}")
        print(f"--- Loading complete ---\n")

    config = dict(config)
    config["feature_dims"] = feature_dims
    return model, p5_dim, config


def get_backbone_neck_block_range(config: dict):
    """Return (backbone_start, backbone_end, neck_start, neck_end) block indices."""
    backbone_end = config["backbone_end"]
    return 0, backbone_end, backbone_end + 1, config["blocks"] - 2

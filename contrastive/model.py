"""
Model components for contrastive learning with YOLO11.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from modules.attention import CBAM
from modules.domain import GradientReversalLayer, DomainDiscriminator


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


class ProjectionHead(nn.Module):
    """MLP projection head that maps P5 features to a lower-dimensional embedding space."""
    def __init__(self, input_dim, hidden_dim=2048, output_dim=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, x):
        x = F.adaptive_avg_pool2d(x, (1, 1)).view(x.size(0), -1)
        return self.net(x)


class ContrastiveYOLO(nn.Module):
    """Wrapper that combines YOLO11 backbone+neck with a projection head."""
    def __init__(self, modified_yolo_model, feature_dims, proj_dim: int,
                 use_cbam: bool = False, cbam_reduction: int = 16,
                 use_domain_adversarial: bool = False,
                 domain_hidden_dim: int = 256,
                 grl_alpha: float = 1.0):
        super().__init__()
        self.backbone_neck = modified_yolo_model
        self.feature_dims = dict(feature_dims)
        p5_dim = self.feature_dims["p5"]
        self.cbam = CBAM(p5_dim, reduction=cbam_reduction) if use_cbam else None
        self.projection_heads = nn.ModuleDict({
            name: ProjectionHead(input_dim=dim, output_dim=proj_dim)
            for name, dim in self.feature_dims.items()
        })
        self.grl = GradientReversalLayer(alpha=grl_alpha) if use_domain_adversarial else None
        self.domain_discriminator = (
            DomainDiscriminator(p5_dim, hidden_dim=domain_hidden_dim)
            if use_domain_adversarial else None
        )

    def extract_features(self, x):
        feats = self.backbone_neck(x)  # [P3, P4, P5]
        feature_map = {
            "p3": feats[0],
            "p4": feats[1],
            "p5": feats[2],
        }
        if self.cbam is not None:
            feature_map["p5"] = self.cbam(feature_map["p5"])
        return feature_map

    def extract_p5(self, x):
        return self.extract_features(x)["p5"]

    def forward(self, x, feature_level: str = "p5"):
        feature_level = feature_level.lower()
        features = self.extract_features(x)
        if feature_level not in features:
            raise ValueError(f"Unsupported feature level: {feature_level}")
        return self.projection_heads[feature_level](features[feature_level])

    def forward_domain(self, x):
        if self.domain_discriminator is None or self.grl is None:
            raise RuntimeError("Domain adversarial branch is not enabled for this model.")
        p5 = self.extract_p5(x)
        return self.domain_discriminator(self.grl(p5))


def get_yolo_backbone_neck(weights_path: str, device, rank=0):
    """
    Load YOLO11 model, replace Detect head with IdentityDetect, return modified model.

    This is the unified entry point for contrastive pretraining.
    Uses Ultralytics API to load YOLO11 weights.

    Args:
        weights_path: Path to YOLO11 .pt weights (e.g. 'weights/yolo11n.pt')
        device: torch device
        rank: process rank (0 = main)

    Returns:
        model: nn.Module with backbone+neck (Detect head replaced)
        p5_dim: int, output channel dimension of P5 feature map
        config: dict with architecture metadata
    """
    from models.yolo11_backbone import load_yolo11_backbone_neck
    return load_yolo11_backbone_neck(weights_path, device, rank)

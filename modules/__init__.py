"""
Innovation modules for YOLO11-UDA.

Contains attention mechanisms, domain discriminators, and improved loss functions.
"""

from .attention import CBAM, ChannelAttention, SpatialAttention, SEBlock
from .domain import DomainDiscriminator, GradientReversalLayer, domain_contrastive_loss
from .losses import domain_aware_contrastive_loss, progressive_pseudo_label_loss

__all__ = [
    "CBAM",
    "ChannelAttention",
    "SpatialAttention",
    "SEBlock",
    "DomainDiscriminator",
    "GradientReversalLayer",
    "domain_contrastive_loss",
    "domain_aware_contrastive_loss",
    "progressive_pseudo_label_loss",
]

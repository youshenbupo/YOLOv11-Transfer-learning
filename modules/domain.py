"""
Domain adaptation modules for YOLO11-UDA.

Provides domain discriminator for domain adversarial training
and domain-aware contrastive loss.

Reference:
    - DANN: Ganin et al., "Domain-Adversarial Training of Neural Networks", JMLR 2016
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class GradientReversalFunction(torch.autograd.Function):
    """Gradient Reversal Layer (GRL) for domain adversarial training."""

    @staticmethod
    def forward(ctx, x, alpha):
        ctx.alpha = alpha
        return x.clone()

    @staticmethod
    def backward(ctx, grad_output):
        return -ctx.alpha * grad_output, None


class GradientReversalLayer(nn.Module):
    """Wrapper for gradient reversal."""
    def __init__(self, alpha=1.0):
        super().__init__()
        self.alpha = alpha

    def forward(self, x):
        return GradientReversalFunction.apply(x, self.alpha)

    def set_alpha(self, alpha):
        self.alpha = alpha


class DomainDiscriminator(nn.Module):
    """
    Domain discriminator for domain adversarial training.

    Takes feature maps from the backbone and predicts whether
    they come from the source or target domain.

    Architecture:
        Global Avg Pool -> FC -> ReLU -> FC -> Sigmoid
    """
    def __init__(self, in_channels, hidden_dim=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(in_channels, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, x):
        return self.net(x)


class DomainAdversarialModel(nn.Module):
    """
    Wraps a YOLO11 backbone with a domain discriminator for DANN-style training.

    During forward pass:
        - Extract features from backbone
        - Pass features through gradient reversal layer
        - Domain discriminator predicts domain label

    Args:
        backbone: YOLO11 backbone+neck (without Detect head)
        feature_dim: Channel dimension of the feature map used for domain classification
        grl_alpha: Gradient reversal coefficient (higher = stronger adversarial)
    """
    def __init__(self, backbone, feature_dim, grl_alpha=1.0):
        super().__init__()
        self.backbone = backbone
        self.grl = GradientReversalLayer(alpha=grl_alpha)
        self.discriminator = DomainDiscriminator(feature_dim)

    def forward(self, x):
        feats = self.backbone(x)
        p5 = feats[-1]  # Use P5 features for domain classification
        reversed_feats = self.grl(p5)
        domain_pred = self.discriminator(reversed_feats)
        return domain_pred

    def set_alpha(self, alpha):
        self.grl.set_alpha(alpha)


def domain_contrastive_loss(features, domain_labels, temperature=0.1):
    """
    Domain-aware contrastive loss.

    Encourages features from the same domain to be similar
    and features from different domains to be dissimilar,
    while preserving instance-level discrimination.

    Args:
        features: (N, D) normalized feature embeddings
        domain_labels: (N,) domain labels (0=source, 1=target)
        temperature: temperature scaling

    Returns:
        loss: scalar tensor
    """
    features = F.normalize(features, dim=1)
    N = features.shape[0]

    # Cosine similarity matrix
    sim = torch.matmul(features, features.T) / temperature

    # Mask for same-domain pairs (excluding self)
    domain_mask = (domain_labels.unsqueeze(0) == domain_labels.unsqueeze(1)).float()
    eye = torch.eye(N, device=features.device)
    same_domain_mask = domain_mask * (1 - eye)

    # Mask for different-domain pairs
    diff_domain_mask = 1 - domain_mask

    # For each anchor, compute loss
    # Same-domain positives should be close
    # Different-domain negatives should be far
    exp_sim = torch.exp(sim)

    # Positive: same domain, different instance
    pos_sum = (exp_sim * same_domain_mask).sum(dim=1)
    # All negatives (different domain)
    neg_sum = (exp_sim * diff_domain_mask).sum(dim=1)

    # Avoid division by zero
    eps = 1e-8
    loss = -torch.log(pos_sum / (pos_sum + neg_sum + eps) + eps).mean()

    return loss

"""
Improved loss functions for YOLO11-UDA.

Includes:
- Domain-aware contrastive loss (combines instance-level and domain-level contrast)
- Progressive pseudo-label loss (quality-weighted pseudo label training)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


def domain_aware_contrastive_loss(features, domain_labels, temperature=0.1, lambda_domain=0.5):
    """
    Domain-Aware Contrastive Loss (Innovation Point 2).

    Combines standard InfoNCE instance-level contrast with domain-level contrast:
        L = L_instance + lambda * L_domain

    - L_instance: Standard InfoNCE (same instance across different augmentations)
    - L_domain: Domain contrast (pull same-class across domains, push different-class)

    Args:
        features: (2N, D) concatenated features from two augmented views
        domain_labels: (N,) domain labels for each sample (0=source, 1=target)
        temperature: temperature scaling for contrastive loss
        lambda_domain: weight for domain contrastive term

    Returns:
        loss: scalar tensor
    """
    batch_size = features.shape[0] // 2
    features = F.normalize(features, dim=1)

    # === Instance-level contrast (SimCLR style) ===
    sim_matrix = torch.matmul(features, features.T)
    mask = torch.eye(2 * batch_size, dtype=torch.bool, device=features.device)
    sim_matrix = sim_matrix[~mask].view(2 * batch_size, -1)

    pos = torch.exp(torch.sum(features[:batch_size] * features[batch_size:], dim=-1) / temperature)
    pos = torch.cat([pos, pos], dim=0)
    neg = torch.exp(sim_matrix / temperature).sum(dim=-1)
    loss_instance = -torch.log(pos / (pos + neg)).mean()

    # === Domain-level contrast ===
    # Get features for each view
    feat_v1 = features[:batch_size]  # (N, D)
    feat_v2 = features[batch_size:]  # (N, D)

    # Domain mask: same domain pairs
    domain_mask = (domain_labels.unsqueeze(0) == domain_labels.unsqueeze(1)).float()
    eye = torch.eye(batch_size, device=features.device)
    same_domain = domain_mask * (1 - eye)
    diff_domain = 1 - domain_mask

    # Cross-view similarity
    cross_sim = torch.matmul(feat_v1, feat_v2.T) / temperature
    exp_cross = torch.exp(cross_sim)

    # Same domain should be similar (positive), different domain should be dissimilar (negative)
    pos_domain = (exp_cross * same_domain).sum(dim=1)
    neg_domain = (exp_cross * diff_domain).sum(dim=1)
    eps = 1e-8
    loss_domain = -torch.log(pos_domain / (pos_domain + neg_domain + eps) + eps).mean()

    return loss_instance + lambda_domain * loss_domain


def compute_pseudo_label_quality(current_preds, previous_preds, iou_threshold=0.5):
    """
    Compute pseudo label quality based on temporal consistency.

    Pseudo labels that are consistent across iterations are more reliable.

    Args:
        current_preds: (N, 6) current predictions [x1, y1, x2, y2, conf, class]
        previous_preds: (M, 6) previous predictions
        iou_threshold: IoU threshold for consistency matching

    Returns:
        quality_scores: (N,) quality score for each prediction
    """
    if previous_preds is None or len(previous_preds) == 0:
        return torch.ones(len(current_preds), device=current_preds.device)

    # Compute pairwise IoU
    ious = _box_iou(current_preds[:, :4], previous_preds[:, :4])

    # For each current prediction, find best matching previous prediction
    best_ious, _ = ious.max(dim=1)

    # Quality score: high IoU with previous = consistent = reliable
    quality_scores = (best_ious >= iou_threshold).float()

    return quality_scores


def _box_iou(boxes1, boxes2):
    """Compute pairwise IoU between two sets of boxes (x1, y1, x2, y2)."""
    area1 = (boxes1[:, 2] - boxes1[:, 0]) * (boxes1[:, 3] - boxes1[:, 1])
    area2 = (boxes2[:, 2] - boxes2[:, 0]) * (boxes2[:, 3] - boxes2[:, 1])

    lt = torch.max(boxes1[:, None, :2], boxes2[None, :, :2])
    rb = torch.min(boxes1[:, None, 2:], boxes2[None, :, 2:])

    wh = (rb - lt).clamp(min=0)
    inter = wh[:, :, 0] * wh[:, :, 1]

    iou = inter / (area1[:, None] + area2[None, :] - inter + 1e-6)
    return iou


def progressive_pseudo_label_loss(pred_boxes, pred_scores, pseudo_boxes,
                                   pseudo_labels, quality_scores,
                                   conf_weight=1.0, quality_weight=0.5):
    """
    Progressive Pseudo-Label Loss (Innovation Point 3).

    Instead of binary filtering, uses quality-weighted loss:
    - High quality + high confidence -> strong supervision
    - High quality + low confidence -> weak supervision
    - Low quality -> down-weighted or ignored

    Args:
        pred_boxes: (N, 4) predicted boxes from student model
        pred_scores: (N,) prediction confidence scores
        pseudo_boxes: (M, 4) pseudo label boxes
        pseudo_labels: (M,) pseudo label classes
        quality_scores: (M,) quality score for each pseudo label
        conf_weight: weight for confidence-based supervision
        quality_weight: weight for quality-based supervision

    Returns:
        loss: scalar tensor
    """
    if len(pseudo_boxes) == 0:
        return torch.tensor(0.0, device=pred_boxes.device)

    # Match predictions to pseudo labels
    ious = _box_iou(pred_boxes, pseudo_boxes)
    best_iou, match_idx = ious.max(dim=1)

    # Only consider matches above IoU threshold
    matched_mask = best_iou > 0.5

    if not matched_mask.any():
        return torch.tensor(0.0, device=pred_boxes.device)

    # Get matched predictions and pseudo labels
    matched_pred_boxes = pred_boxes[matched_mask]
    matched_pred_scores = pred_scores[matched_mask]
    matched_pseudo_boxes = pseudo_boxes[match_idx[matched_mask]]
    matched_quality = quality_scores[match_idx[matched_mask]]

    # Box regression loss (CIoU or L1)
    box_loss = F.smooth_l1_loss(matched_pred_boxes, matched_pseudo_boxes, reduction="none")

    # Quality-weighted loss
    quality_weights = matched_quality * quality_weight + (1 - quality_weight)
    weighted_box_loss = (box_loss.mean(dim=1) * quality_weights).mean()

    return weighted_box_loss

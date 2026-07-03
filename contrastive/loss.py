"""
Loss functions for contrastive learning: InfoNCE (SimCLR) and MoCo.
"""

import torch
import torch.nn.functional as F


def info_nce_loss(features, temperature, device):
    """
    InfoNCE loss for SimCLR-style contrastive learning.
    Assumes features are concatenated [view1, view2] along batch dimension.
    """
    batch_size = features.shape[0] // 2
    features = F.normalize(features, dim=1)
    sim = torch.matmul(features, features.T)
    mask = torch.eye(2 * batch_size, dtype=torch.bool, device=device)
    sim = sim[~mask].view(2 * batch_size, -1)

    pos = torch.exp(torch.sum(features[:batch_size] * features[batch_size:], dim=-1) / temperature)
    pos = torch.cat([pos, pos], dim=0)
    neg = torch.exp(sim / temperature).sum(dim=-1)
    loss = -torch.log(pos / (pos + neg)).mean()
    return loss


def moco_loss(query, key, queue, temperature):
    """
    MoCo v1/v2 style contrastive loss.
    query: (N, C) from online encoder
    key: (N, C) from momentum encoder
    queue: (K, C) negative samples queue
    """
    l_pos = torch.einsum("nc,nc->n", [query, key]).unsqueeze(-1)
    l_neg = torch.einsum("nc,kc->nk", [query, queue.clone().detach()])
    logits = torch.cat([l_pos, l_neg], dim=1) / temperature
    labels = torch.zeros(logits.shape[0], dtype=torch.long, device=query.device)
    return F.cross_entropy(logits, labels)


@torch.no_grad()
def dequeue_and_enqueue(queue, keys, queue_ptr):
    """Update the MoCo queue with new keys."""
    bsz = keys.shape[0]
    K = queue.shape[0]
    if bsz > K:
        keys = keys[:K]
        bsz = K
    end = queue_ptr + bsz
    if end <= K:
        queue[queue_ptr:end] = keys
    else:
        first = K - queue_ptr
        queue[queue_ptr:K] = keys[:first]
        queue[0:end - K] = keys[first:]
    queue_ptr = (queue_ptr + bsz) % K
    return queue, queue_ptr

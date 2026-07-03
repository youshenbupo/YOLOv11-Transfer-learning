"""
Distributed Data Parallel (DDP) utilities for contrastive learning.
"""

import os
from typing import Tuple
import torch
import torch.distributed as dist


def init_distributed_mode(local_rank_cli: int = 0) -> Tuple[bool, int, int, int]:
    """Initialize distributed training mode. Returns (is_dist, rank, world_size, local_rank)."""
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        rank = int(os.environ["RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        local_rank = int(os.environ.get("LOCAL_RANK", 0))
        backend = "nccl" if torch.cuda.is_available() else "gloo"
        os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
        os.environ.setdefault("MASTER_PORT", "29500")
        if not dist.is_initialized():
            dist.init_process_group(backend=backend, init_method="env://")
        return True, rank, world_size, local_rank

    if "LOCAL_RANK" not in os.environ:
        os.environ["LOCAL_RANK"] = str(local_rank_cli)
        os.environ["RANK"] = str(local_rank_cli)

    rank = int(os.environ.get("RANK", "0"))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    is_dist = world_size > 1

    if is_dist and not dist.is_initialized():
        backend = "nccl" if torch.cuda.is_available() else "gloo"
        os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
        os.environ.setdefault("MASTER_PORT", "29500")
        dist.init_process_group(
            backend=backend, init_method="env://", rank=rank, world_size=world_size
        )
    return is_dist, rank, world_size, local_rank


@torch.no_grad()
def concat_all_gather(tensor):
    """Gather tensors from all processes."""
    ws = dist.get_world_size()
    buf = [torch.zeros_like(tensor) for _ in range(ws)]
    dist.all_gather(buf, tensor, async_op=False)
    return torch.cat(buf, dim=0)


@torch.no_grad()
def batch_shuffle_ddp(x):
    """Shuffle batch for DDP (used in MoCo key encoder)."""
    x_g = concat_all_gather(x)
    N = x_g.shape[0]
    idx_shuffle = torch.randperm(N, device=x.device)
    dist.broadcast(idx_shuffle, src=0)
    idx_unshuffle = torch.argsort(idx_shuffle)
    x_g = x_g[idx_shuffle]
    r = dist.get_rank()
    ws = dist.get_world_size()
    per = N // ws
    s = r * per
    e = s + per
    return x_g[s:e], idx_unshuffle


@torch.no_grad()
def batch_unshuffle_ddp(x, idx_unshuffle):
    """Unshuffle batch after DDP shuffling."""
    x_g = concat_all_gather(x)
    x_g = x_g[idx_unshuffle]
    r = dist.get_rank()
    ws = dist.get_world_size()
    per = x_g.shape[0] // ws
    s = r * per
    e = s + per
    return x_g[s:e]

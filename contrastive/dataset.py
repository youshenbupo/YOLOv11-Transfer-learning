"""
Dataset and sampler for contrastive learning.
"""

import os
import numpy as np
import torch
from torch.utils.data import Dataset
from torch.utils.data.sampler import Sampler
from PIL import Image


class CombinedDataset(Dataset):
    """Dataset that combines positive (target domain) and negative (source domain) images."""
    def __init__(self, positive_dir, negative_dir, cpu_transform, rank=0, world_size=1, img_size: int = 640):
        self.cpu_transform = cpu_transform
        self.img_size = img_size
        pos_all = [os.path.join(positive_dir, f) for f in os.listdir(positive_dir)
                   if f.lower().endswith(("png", "jpg", "jpeg"))]
        neg_all = [os.path.join(negative_dir, f) for f in os.listdir(negative_dir)
                   if f.lower().endswith(("png", "jpg", "jpeg"))]

        if world_size > 1:
            self.pos_paths = pos_all[rank::world_size]
            self.neg_paths = neg_all[rank::world_size]
        else:
            self.pos_paths = pos_all
            self.neg_paths = neg_all

        self.all_image_paths = self.pos_paths + self.neg_paths
        if rank == 0:
            print(f"Dataset: {len(self.pos_paths)} positive, {len(self.neg_paths)} negative, "
                  f"{len(self.all_image_paths)} total images.")

    def __len__(self):
        return len(self.all_image_paths)

    def __getitem__(self, idx):
        p = self.all_image_paths[idx]
        domain_label = 1 if idx < len(self.pos_paths) else 0
        try:
            img = Image.open(p).convert("RGB")
            return self.cpu_transform(img), torch.tensor(domain_label, dtype=torch.long)
        except Exception:
            # Fallback to random noise on load failure
            return torch.randn(3, self.img_size, self.img_size), torch.tensor(domain_label, dtype=torch.long)


class RatioBatchSampler(Sampler):
    """Sampler that ensures each batch has a fixed ratio of positive/negative samples."""
    def __init__(self, dataset, num_positives_per_batch, batch_size, drop_last=True):
        self.dataset = dataset
        self.num_positives_per_batch = num_positives_per_batch
        self.batch_size = batch_size
        self.drop_last = drop_last
        self.num_negatives_per_batch = batch_size - num_positives_per_batch
        if self.num_negatives_per_batch < 0:
            raise ValueError("num_positives_per_batch cannot be larger than batch_size")

        num_pos = len(getattr(dataset, "pos_paths", []))
        num_neg = len(getattr(dataset, "neg_paths", []))
        self.pos_indices = list(range(num_pos))
        self.neg_indices = list(range(num_pos, num_pos + num_neg))

        if drop_last:
            if num_pos < self.num_positives_per_batch or num_neg < self.num_negatives_per_batch:
                self.num_batches = 0
            else:
                self.num_batches = min(
                    num_pos // self.num_positives_per_batch,
                    num_neg // self.num_negatives_per_batch
                )
        else:
            raise NotImplementedError("drop_last=False is not implemented")

    def __iter__(self):
        np.random.shuffle(self.pos_indices)
        np.random.shuffle(self.neg_indices)
        pos_iter = iter(self.pos_indices)
        neg_iter = iter(self.neg_indices)
        for _ in range(self.num_batches):
            batch = []
            for _ in range(self.num_positives_per_batch):
                batch.append(next(pos_iter))
            for _ in range(self.num_negatives_per_batch):
                batch.append(next(neg_iter))
            np.random.shuffle(batch)
            yield batch

    def __len__(self):
        return self.num_batches

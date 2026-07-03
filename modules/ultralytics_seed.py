"""Make Ultralytics dataloader worker randomness depend on the experiment seed."""

import os


def patch_ultralytics_dataloader_seed(seed):
    """Patch the detection trainer's loader without modifying site-packages.

    Ultralytics 8.4.82 seeds its loader generator with a constant plus RANK,
    so runs with different ``model.train(seed=...)`` values can become
    parameter-identical when workers > 0. This wrapper preserves the original
    loader behavior while adding the recorded experiment seed.
    """
    import torch
    from ultralytics.data import build as data_build
    from ultralytics.models.yolo.detect import train as detect_train

    experiment_seed = int(seed)

    def seeded_build_dataloader(
        dataset,
        batch,
        workers,
        shuffle=True,
        rank=-1,
        drop_last=False,
        pin_memory=True,
    ):
        batch = min(batch, len(dataset))
        nd = torch.cuda.device_count()
        nw = min(os.cpu_count() // max(nd, 1), workers)
        sampler = (
            None
            if rank == -1
            else data_build.distributed.DistributedSampler(dataset, shuffle=shuffle)
            if shuffle
            else data_build.ContiguousDistributedSampler(dataset)
        )
        generator = torch.Generator()
        generator.manual_seed(6148914691236517205 + data_build.RANK + experiment_seed)
        return data_build.InfiniteDataLoader(
            dataset=dataset,
            batch_size=batch,
            shuffle=shuffle and sampler is None,
            num_workers=nw,
            sampler=sampler,
            prefetch_factor=4 if nw > 0 else None,
            pin_memory=nd > 0 and pin_memory,
            collate_fn=getattr(dataset, "collate_fn", None),
            worker_init_fn=data_build.seed_worker,
            generator=generator,
            drop_last=drop_last and len(dataset) % batch != 0,
        )

    detect_train.build_dataloader = seeded_build_dataloader

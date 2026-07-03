#!/usr/bin/env python
"""Train a reproducible source-domain-only YOLO detector."""

import argparse
import json
import os
import time


def main(opt):
    from ultralytics import YOLO
    from modules.ultralytics_seed import patch_ultralytics_dataloader_seed

    patch_ultralytics_dataloader_seed(opt.seed)
    model = YOLO(opt.weights)
    started = time.perf_counter()
    model.train(
        data=opt.data,
        epochs=opt.epochs,
        batch=opt.batch_size,
        imgsz=opt.img_size,
        device=opt.device,
        workers=opt.workers,
        seed=opt.seed,
        project=os.path.abspath(opt.project),
        name=opt.name,
        exist_ok=True,
        plots=False,
        verbose=True,
    )
    save_dir = os.path.abspath(str(model.trainer.save_dir))
    best = os.path.join(save_dir, "weights", "best.pt")
    if not os.path.exists(best):
        best = os.path.join(save_dir, "weights", "last.pt")
    summary = {
        "weights": os.path.abspath(best),
        "data": os.path.abspath(opt.data),
        "epochs": opt.epochs,
        "batch_size": opt.batch_size,
        "workers": opt.workers,
        "seed": opt.seed,
        "duration_sec": time.perf_counter() - started,
    }
    with open(os.path.join(save_dir, "source_training_summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(f"SOURCE_WEIGHTS={summary['weights']}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--weights", default="weights/yolo11n.pt")
    parser.add_argument("--data", required=True)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=24)
    parser.add_argument("--img-size", type=int, default=640)
    parser.add_argument("--device", default="0")
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--project", default="runs/source_only")
    parser.add_argument("--name", required=True)
    main(parser.parse_args())

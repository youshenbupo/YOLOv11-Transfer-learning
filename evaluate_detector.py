#!/usr/bin/env python
"""Evaluate a YOLO detector on a labeled dataset YAML and emit JSON metrics."""

import argparse
import json
import os


def main(opt):
    from ultralytics import YOLO

    model = YOLO(opt.weights)
    result = model.val(
        data=opt.data,
        imgsz=opt.img_size,
        batch=opt.batch_size,
        device=opt.device,
        workers=opt.workers,
        plots=False,
        verbose=False,
    )
    summary = {
        "weights": os.path.abspath(opt.weights),
        "data": os.path.abspath(opt.data),
        "map50": float(result.box.map50),
        "map50_95": float(result.box.map),
        "map75": float(result.box.map75),
        "precision": float(result.box.mp),
        "recall": float(result.box.mr),
    }
    os.makedirs(os.path.dirname(os.path.abspath(opt.output)), exist_ok=True)
    with open(opt.output, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(f"TARGET_MAP50={summary['map50']:.6f}")
    print(f"TARGET_MAP50_95={summary['map50_95']:.6f}")
    print(f"TARGET_MAP75={summary['map75']:.6f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate YOLO detector metrics")
    parser.add_argument("--weights", required=True)
    parser.add_argument("--data", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--img-size", type=int, default=640)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--device", default="0")
    parser.add_argument("--workers", type=int, default=0)
    main(parser.parse_args())

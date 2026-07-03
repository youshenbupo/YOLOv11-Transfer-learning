#!/usr/bin/env python
"""
Standalone test-time consistency calibration for YOLO11-UDA.
"""

import argparse
import os

from ultralytics import YOLO

from modules.test_time_calibration import run_calibrated_inference
from self_training import patch_ultralytics_threadpool, resolve_yolo_weights


def main(opt):
    patch_ultralytics_threadpool()
    weights_path = resolve_yolo_weights(opt.weights, opt.yolo_base_weights)
    model = YOLO(weights_path)

    save_dir = os.path.join(opt.project, opt.name)
    summary = run_calibrated_inference(
        model,
        image_folder=opt.source,
        save_dir=save_dir,
        conf_thres=opt.conf_thres,
        img_size=opt.img_size,
        device=opt.device,
        alpha=opt.calibration_alpha,
        beta=opt.calibration_beta,
        match_iou_threshold=opt.calibration_iou_threshold,
        views=opt.calibration_views,
        calibration_mode=opt.calibration_mode,
        consistency_threshold=opt.consistency_threshold,
        rank_top_fraction=opt.rank_top_fraction,
        rank_min_confidence=opt.rank_min_confidence,
        rank_class_aware=opt.rank_class_aware,
    )

    print("=" * 60)
    print("TEST-TIME CALIBRATION COMPLETE")
    print("=" * 60)
    print(f"  Save dir: {summary['save_dir']}")
    print(f"  Images: {summary['num_images']}")
    print(f"  Raw predictions: {summary['raw_predictions']}")
    print(f"  Mean consistency: {summary['mean_consistency']:.4f}")
    print(f"  Views: {', '.join(summary['views'])}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test-time consistency calibration for YOLO11-UDA")
    parser.add_argument("--weights", type=str, default="weights/yolo11n.pt",
                        help="YOLO11 weights or a contrastive checkpoint")
    parser.add_argument("--yolo-base-weights", type=str, default="weights/yolo11n.pt",
                        help="Base YOLO11 checkpoint used when merging a backbone-only checkpoint")
    parser.add_argument("--source", type=str, default="target_dataset/unlabels_img",
                        help="Folder of images to calibrate")
    parser.add_argument("--project", type=str, default="runs/calibration",
                        help="Output project directory")
    parser.add_argument("--name", type=str, default="ttc_run",
                        help="Run name under the project directory")
    parser.add_argument("--img-size", type=int, default=640,
                        help="Inference image size")
    parser.add_argument("--device", type=str, default="0",
                        help="Device for Ultralytics inference")
    parser.add_argument("--conf-thres", type=float, default=0.01,
                        help="Base confidence threshold before calibration")
    parser.add_argument("--calibration-views", nargs="+",
                        default=["original", "hflip", "bright"],
                        help="Ordered list of calibration views. Must start with 'original'.")
    parser.add_argument("--calibration-alpha", type=float, default=0.5,
                        help="Base multiplier term for calibrated confidence")
    parser.add_argument("--calibration-beta", type=float, default=0.5,
                        help="Consistency multiplier term for calibrated confidence")
    parser.add_argument("--calibration-iou-threshold", type=float, default=0.5,
                        help="IoU threshold used when matching boxes across views")
    parser.add_argument("--calibration-mode", type=str, default="global",
                        choices=["global", "selective", "rank"],
                        help="Calibration rule: global rescales every box, selective only downweights uncertain boxes, rank targets risky high-confidence boxes")
    parser.add_argument("--consistency-threshold", type=float, default=0.5,
                        help="Consistency threshold used by selective calibration mode")
    parser.add_argument("--rank-top-fraction", type=float, default=0.25,
                        help="Fraction of detections considered high-risk in rank mode")
    parser.add_argument("--rank-min-confidence", type=float, default=0.1,
                        help="Minimum raw confidence for a detection to be eligible in rank mode")
    parser.add_argument("--rank-class-aware", action="store_true", default=False,
                        help="In rank mode, select risky boxes separately inside each predicted class")
    main(parser.parse_args())

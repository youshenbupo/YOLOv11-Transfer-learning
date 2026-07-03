#!/usr/bin/env python
"""
Evaluate raw and calibrated detection confidence calibration on a labeled target split.
"""

import argparse
import os
from pathlib import Path

from ultralytics import YOLO

from modules.calibration_metrics import summarize_detection_calibration, save_summary
from modules.test_time_calibration import run_calibrated_inference
from self_training import patch_ultralytics_threadpool, resolve_yolo_weights


def run_raw_inference(model, image_dir, save_dir, conf_thres, img_size, device):
    project_dir = os.path.abspath(os.path.dirname(save_dir))
    run_name = os.path.basename(save_dir)
    os.makedirs(project_dir, exist_ok=True)
    results = model.predict(
        source=image_dir,
        conf=conf_thres,
        imgsz=img_size,
        device=device,
        project=project_dir,
        name=run_name,
        exist_ok=True,
        save_txt=True,
        save_conf=True,
        verbose=False,
    )
    return str(results[0].save_dir) if results else save_dir


def print_summary(title, summary):
    print(f"[{title}] images={summary['num_images']} events={summary['num_events']}")
    print(
        f"[{title}] ECE={summary['ece']:.4f} "
        f"NLL={summary['nll']:.4f} "
        f"Brier={summary['brier']:.4f}"
    )


def main(opt):
    patch_ultralytics_threadpool()
    weights_path = resolve_yolo_weights(opt.weights, opt.yolo_base_weights)
    model = YOLO(weights_path)

    output_root = os.path.abspath(opt.output_dir)
    raw_dir = os.path.join(output_root, "raw_predictions")
    calibrated_dir = os.path.join(output_root, "calibrated_predictions")
    os.makedirs(output_root, exist_ok=True)

    raw_save_dir = run_raw_inference(
        model,
        image_dir=opt.image_dir,
        save_dir=raw_dir,
        conf_thres=opt.conf_thres,
        img_size=opt.img_size,
        device=opt.device,
    )
    calibrated_summary = run_calibrated_inference(
        model,
        image_folder=opt.image_dir,
        save_dir=calibrated_dir,
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

    raw_summary = summarize_detection_calibration(
        pred_dir=os.path.join(raw_save_dir, "labels"),
        image_dir=opt.image_dir,
        label_dir=opt.label_dir,
        iou_threshold=opt.match_iou_threshold,
        num_bins=opt.num_bins,
    )
    calibrated_eval = summarize_detection_calibration(
        pred_dir=os.path.join(calibrated_summary["save_dir"], "labels"),
        image_dir=opt.image_dir,
        label_dir=opt.label_dir,
        iou_threshold=opt.match_iou_threshold,
        num_bins=opt.num_bins,
    )

    comparison = {
        "weights": os.path.abspath(weights_path),
        "image_dir": os.path.abspath(opt.image_dir),
        "label_dir": os.path.abspath(opt.label_dir),
        "raw": raw_summary,
        "calibrated": calibrated_eval,
        "delta": {
            "ece": calibrated_eval["ece"] - raw_summary["ece"],
            "nll": calibrated_eval["nll"] - raw_summary["nll"],
            "brier": calibrated_eval["brier"] - raw_summary["brier"],
        },
        "calibration_config": {
            "views": opt.calibration_views,
            "alpha": opt.calibration_alpha,
            "beta": opt.calibration_beta,
            "calibration_iou_threshold": opt.calibration_iou_threshold,
            "calibration_mode": opt.calibration_mode,
            "consistency_threshold": opt.consistency_threshold,
            "rank_top_fraction": opt.rank_top_fraction,
            "rank_min_confidence": opt.rank_min_confidence,
            "rank_class_aware": opt.rank_class_aware,
            "match_iou_threshold": opt.match_iou_threshold,
            "num_bins": opt.num_bins,
        },
    }

    save_summary(raw_summary, os.path.join(output_root, "raw_metrics.json"))
    save_summary(calibrated_eval, os.path.join(output_root, "calibrated_metrics.json"))
    save_summary(comparison, os.path.join(output_root, "comparison.json"))

    print("=" * 60)
    print("CALIBRATION EVALUATION COMPLETE")
    print("=" * 60)
    print_summary("raw", raw_summary)
    print_summary("calibrated", calibrated_eval)
    print(
        f"[delta] ECE={comparison['delta']['ece']:+.4f} "
        f"NLL={comparison['delta']['nll']:+.4f} "
        f"Brier={comparison['delta']['brier']:+.4f}"
    )
    print(f"[files] {os.path.join(output_root, 'comparison.json')}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate raw vs calibrated detection confidence metrics")
    parser.add_argument("--weights", type=str, default="weights/yolo11n.pt",
                        help="YOLO11 weights or a contrastive checkpoint")
    parser.add_argument("--yolo-base-weights", type=str, default="weights/yolo11n.pt",
                        help="Base YOLO11 checkpoint used when merging a backbone-only checkpoint")
    parser.add_argument("--image-dir", type=str, default="target_dataset/val_img/images",
                        help="Evaluation image directory")
    parser.add_argument("--label-dir", type=str, default="target_dataset/val_img/labels",
                        help="Evaluation label directory in YOLO format")
    parser.add_argument("--output-dir", type=str, default="runs/calibration_eval/default",
                        help="Directory for raw/calibrated predictions and metrics JSON files")
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
                        help="IoU threshold used when matching boxes across views during calibration")
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
    parser.add_argument("--match-iou-threshold", type=float, default=0.5,
                        help="IoU threshold used to decide whether a prediction is correct against ground truth")
    parser.add_argument("--num-bins", type=int, default=10,
                        help="Number of confidence bins for ECE")
    main(parser.parse_args())

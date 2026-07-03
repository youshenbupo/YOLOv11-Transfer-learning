#!/usr/bin/env python
"""
Run a calibration ablation sweep and aggregate ECE / NLL / Brier results.
"""

import argparse
import csv
import itertools
import json
import os
import subprocess
import sys


def parse_csv_floats(text):
    return [float(item) for item in text.split(",") if item.strip()]


def parse_view_sets(text):
    groups = []
    for group in text.split(";"):
        views = [item.strip() for item in group.split(",") if item.strip()]
        if views:
            groups.append(views)
    return groups


def load_comparison(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def write_markdown_report(rows, output_path):
    lines = [
        "# Calibration Sweep Summary",
        "",
        "| Rank | Run | Views | Alpha | Beta | Cal IoU | Raw ECE | Cal ECE | Delta ECE | Raw NLL | Cal NLL | Delta NLL | Raw Brier | Cal Brier | Delta Brier |",
        "| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]

    rank = 0
    for row in rows:
        if row.get("status") != "ok":
            continue
        rank += 1
        lines.append(
            "| {rank} | {run_name} | {views} | {alpha:.2f} | {beta:.2f} | {calibration_iou_threshold:.2f} | "
            "{raw_ece:.4f} | {cal_ece:.4f} | {delta_ece:+.4f} | "
            "{raw_nll:.4f} | {cal_nll:.4f} | {delta_nll:+.4f} | "
            "{raw_brier:.4f} | {cal_brier:.4f} | {delta_brier:+.4f} |".format(
                rank=rank,
                **row,
            )
        )

    if rank == 0:
        lines.append("| - | no successful runs | - | - | - | - | - | - | - | - | - | - | - | - | - |")

    with open(output_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


def main(opt):
    python_exe = opt.python_exe or sys.executable
    output_root = os.path.abspath(opt.output_root)
    os.makedirs(output_root, exist_ok=True)

    view_sets = parse_view_sets(opt.view_sets)
    alphas = parse_csv_floats(opt.alphas)
    betas = parse_csv_floats(opt.betas)
    cal_ious = parse_csv_floats(opt.calibration_ious)
    consistency_thresholds = parse_csv_floats(opt.consistency_thresholds)
    rank_top_fractions = parse_csv_floats(opt.rank_top_fractions)
    rank_min_confidences = parse_csv_floats(opt.rank_min_confidences)
    rank_class_aware_values = [item.strip().lower() in {"1", "true", "yes", "y"} for item in opt.rank_class_aware_values.split(",") if item.strip()]

    jobs = []
    for views, alpha, beta, cal_iou, consistency_threshold, rank_top_fraction, rank_min_confidence, rank_class_aware in itertools.product(
        view_sets, alphas, betas, cal_ious, consistency_thresholds, rank_top_fractions, rank_min_confidences, rank_class_aware_values
    ):
        jobs.append(
            {
                "views": views,
                "alpha": alpha,
                "beta": beta,
                "calibration_iou_threshold": cal_iou,
                "consistency_threshold": consistency_threshold,
                "rank_top_fraction": rank_top_fraction,
                "rank_min_confidence": rank_min_confidence,
                "rank_class_aware": rank_class_aware,
            }
        )

    if opt.max_runs > 0:
        jobs = jobs[:opt.max_runs]

    print(f"[Sweep] planned runs: {len(jobs)}")
    rows = []

    for idx, job in enumerate(jobs, start=1):
        run_name = (
            f"run_{idx:03d}_"
            f"views-{'-'.join(job['views'])}_"
            f"a-{job['alpha']:.2f}_"
            f"b-{job['beta']:.2f}_"
            f"ciou-{job['calibration_iou_threshold']:.2f}"
        )
        run_dir = os.path.join(output_root, run_name)
        cmd = [
            python_exe,
            "evaluate_calibration.py",
            "--weights", opt.weights,
            "--yolo-base-weights", opt.yolo_base_weights,
            "--image-dir", opt.image_dir,
            "--label-dir", opt.label_dir,
            "--output-dir", run_dir,
            "--img-size", str(opt.img_size),
            "--device", opt.device,
            "--conf-thres", str(opt.conf_thres),
            "--calibration-alpha", str(job["alpha"]),
            "--calibration-beta", str(job["beta"]),
            "--calibration-iou-threshold", str(job["calibration_iou_threshold"]),
            "--calibration-mode", opt.calibration_mode,
            "--consistency-threshold", str(job["consistency_threshold"]),
            "--rank-top-fraction", str(job["rank_top_fraction"]),
            "--rank-min-confidence", str(job["rank_min_confidence"]),
            "--match-iou-threshold", str(opt.match_iou_threshold),
            "--num-bins", str(opt.num_bins),
            "--calibration-views",
            *job["views"],
        ]
        if job["rank_class_aware"]:
            cmd.append("--rank-class-aware")

        print(f"[Sweep] ({idx}/{len(jobs)}) {' '.join(cmd)}")
        completed = subprocess.run(cmd, cwd=os.getcwd(), check=False)
        if completed.returncode != 0:
            print(f"[Sweep] run failed: {run_name}")
            rows.append(
                {
                    "run_name": run_name,
                    "status": "failed",
                    "views": " ".join(job["views"]),
                    "alpha": job["alpha"],
                    "beta": job["beta"],
                    "calibration_iou_threshold": job["calibration_iou_threshold"],
                    "consistency_threshold": job["consistency_threshold"],
                    "rank_top_fraction": job["rank_top_fraction"],
                    "rank_min_confidence": job["rank_min_confidence"],
                    "rank_class_aware": job["rank_class_aware"],
                }
            )
            continue

        comparison_path = os.path.join(run_dir, "comparison.json")
        comparison = load_comparison(comparison_path)
        rows.append(
            {
                "run_name": run_name,
                "status": "ok",
                "views": " ".join(job["views"]),
                "alpha": job["alpha"],
                "beta": job["beta"],
                "calibration_iou_threshold": job["calibration_iou_threshold"],
                "consistency_threshold": job["consistency_threshold"],
                "rank_top_fraction": job["rank_top_fraction"],
                "rank_min_confidence": job["rank_min_confidence"],
                "rank_class_aware": job["rank_class_aware"],
                "raw_ece": comparison["raw"]["ece"],
                "raw_nll": comparison["raw"]["nll"],
                "raw_brier": comparison["raw"]["brier"],
                "cal_ece": comparison["calibrated"]["ece"],
                "cal_nll": comparison["calibrated"]["nll"],
                "cal_brier": comparison["calibrated"]["brier"],
                "delta_ece": comparison["delta"]["ece"],
                "delta_nll": comparison["delta"]["nll"],
                "delta_brier": comparison["delta"]["brier"],
            }
        )

    rows_sorted = sorted(
        rows,
        key=lambda row: (
            row.get("status") != "ok",
            row.get("delta_ece", float("inf")),
            row.get("delta_nll", float("inf")),
            row.get("delta_brier", float("inf")),
        ),
    )

    summary_json = os.path.join(output_root, "sweep_summary.json")
    with open(summary_json, "w", encoding="utf-8") as f:
        json.dump(rows_sorted, f, indent=2)

    summary_csv = os.path.join(output_root, "sweep_summary.csv")
    fieldnames = sorted({key for row in rows_sorted for key in row.keys()}) if rows_sorted else []
    if fieldnames:
        with open(summary_csv, "w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows_sorted)
    summary_md = os.path.join(output_root, "sweep_summary.md")
    write_markdown_report(rows_sorted, summary_md)

    print("=" * 60)
    print("CALIBRATION SWEEP COMPLETE")
    print("=" * 60)
    if rows_sorted:
        best = next((row for row in rows_sorted if row.get("status") == "ok"), None)
        if best:
            print(
                f"[Best] {best['run_name']} "
                f"delta_ece={best['delta_ece']:+.4f} "
                f"delta_nll={best['delta_nll']:+.4f} "
                f"delta_brier={best['delta_brier']:+.4f}"
            )
    print(f"[Files] {summary_json}")
    if fieldnames:
        print(f"[Files] {summary_csv}")
    print(f"[Files] {summary_md}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run a calibration ablation sweep")
    parser.add_argument("--python-exe", type=str, default="",
                        help="Optional Python executable to use for each run")
    parser.add_argument("--weights", type=str, default="weights/yolo11n.pt",
                        help="YOLO11 weights or a contrastive checkpoint")
    parser.add_argument("--yolo-base-weights", type=str, default="weights/yolo11n.pt",
                        help="Base YOLO11 checkpoint used when merging a backbone-only checkpoint")
    parser.add_argument("--image-dir", type=str, default="target_dataset/val_img/images",
                        help="Evaluation image directory")
    parser.add_argument("--label-dir", type=str, default="target_dataset/val_img/labels",
                        help="Evaluation label directory")
    parser.add_argument("--output-root", type=str, default="runs/calibration_eval/sweep",
                        help="Root directory for sweep outputs")
    parser.add_argument("--img-size", type=int, default=64,
                        help="Inference image size")
    parser.add_argument("--device", type=str, default="0",
                        help="Device for Ultralytics inference")
    parser.add_argument("--conf-thres", type=float, default=0.01,
                        help="Base confidence threshold before calibration")
    parser.add_argument("--match-iou-threshold", type=float, default=0.5,
                        help="IoU threshold used to mark predictions correct against ground truth")
    parser.add_argument("--num-bins", type=int, default=10,
                        help="Number of confidence bins for ECE")
    parser.add_argument("--view-sets", type=str,
                        default="original,hflip;original,hflip,bright;original,hflip,bright,dark",
                        help='Semicolon-separated view groups, each group comma-separated')
    parser.add_argument("--alphas", type=str, default="0.7,0.5,0.3",
                        help="Comma-separated alpha values")
    parser.add_argument("--betas", type=str, default="0.3,0.5,0.7",
                        help="Comma-separated beta values")
    parser.add_argument("--calibration-ious", type=str, default="0.3,0.5,0.7",
                        help="Comma-separated calibration IoU thresholds")
    parser.add_argument("--calibration-mode", type=str, default="global",
                        choices=["global", "selective", "rank"],
                        help="Calibration rule evaluated in the sweep")
    parser.add_argument("--consistency-thresholds", type=str, default="0.5",
                        help="Comma-separated selective-calibration consistency thresholds")
    parser.add_argument("--rank-top-fractions", type=str, default="0.25",
                        help="Comma-separated high-risk fractions for rank mode")
    parser.add_argument("--rank-min-confidences", type=str, default="0.1",
                        help="Comma-separated minimum raw confidence thresholds for rank mode")
    parser.add_argument("--rank-class-aware-values", type=str, default="false",
                        help='Comma-separated booleans for rank mode, e.g. "false,true"')
    parser.add_argument("--max-runs", type=int, default=0,
                        help="Optional cap on total runs for quick smoke tests")
    main(parser.parse_args())

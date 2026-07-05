#!/usr/bin/env python
"""
Run formal paper-oriented experiment matrices for prepared UDA benchmarks.
"""

import argparse
import csv
import json
import os
import re
import subprocess
import sys
import time
from statistics import mean, stdev
from copy import deepcopy


BENCHMARKS = {
    "cityscapes_to_foggy": {
        "display_name": "Cityscapes -> Foggy Cityscapes",
        "source_dataset": "benchmarks/cityscapes_to_foggy/source_dataset",
        "target_dataset": "benchmarks/cityscapes_to_foggy/target_dataset",
        "target_val_image_dir": "benchmarks/cityscapes_to_foggy/target_dataset/val_img/images",
        "target_val_label_dir": "benchmarks/cityscapes_to_foggy/target_dataset/val_img/labels",
        "target_val_yaml": "benchmarks/cityscapes_to_foggy/target_dataset/target_val.yaml",
        "self_training": {
            "img_size": 640,
            "batch_size": 24,
            "iterations": 1,
            "epochs_per_iter": 3,
        },
        "contrastive": {
            "img_size": 224,
            "batch_size": 24,
            "num_positives": 4,
            "epochs": 5,
        },
    },
    "sim10k_to_cityscapes": {
        "display_name": "SIM10K -> Cityscapes",
        "source_dataset": "benchmarks/sim10k_to_cityscapes/source_dataset",
        "target_dataset": "benchmarks/sim10k_to_cityscapes/target_dataset",
        "target_val_image_dir": "benchmarks/sim10k_to_cityscapes/target_dataset/val_img/images",
        "target_val_label_dir": "benchmarks/sim10k_to_cityscapes/target_dataset/val_img/labels",
        "target_val_yaml": "benchmarks/sim10k_to_cityscapes/target_dataset/target_val.yaml",
        "self_training": {
            "img_size": 640,
            "batch_size": 8,
            "iterations": 1,
            "epochs_per_iter": 3,
        },
        "contrastive": {
            "img_size": 224,
            "batch_size": 8,
            "num_positives": 4,
            "epochs": 5,
        },
    },
}


def _fmt(value):
    if value is None:
        return "-"
    if isinstance(value, float):
        return f"{value:.4f}"
    return str(value)


def _parse_metric(pattern, text, cast=float, default=None):
    matches = re.findall(pattern, text)
    if not matches:
        return default
    value = matches[-1]
    if isinstance(value, tuple):
        value = value[-1]
    return cast(value)


def _run_command(cmd, workdir, log_path):
    start = time.time()
    result = subprocess.run(
        cmd,
        cwd=workdir,
        text=True,
        encoding="utf-8",
        errors="replace",
        capture_output=True,
        check=False,
    )
    duration = time.time() - start
    with open(log_path, "w", encoding="utf-8") as f:
        f.write("COMMAND:\n")
        f.write(" ".join(cmd) + "\n\n")
        f.write("STDOUT:\n")
        f.write(result.stdout or "")
        f.write("\nSTDERR:\n")
        f.write(result.stderr or "")
    return result, duration


def _is_memory_retryable(text):
    haystack = (text or "").lower()
    needles = [
        "out of memory",
        "cuda out of memory",
        "arraymemoryerror",
        "unable to allocate",
    ]
    return any(token in haystack for token in needles)


def _find_flag_value(cmd, flag):
    for idx, token in enumerate(cmd[:-1]):
        if token == flag:
            return idx + 1, cmd[idx + 1]
    return None, None


def _retry_with_lower_batch(cmd, workdir, log_prefix, min_batch_size=1):
    total_duration = 0.0
    attempt = 0
    current_cmd = list(cmd)

    while True:
        log_path = f"{log_prefix}.attempt{attempt + 1}.log"
        result, duration = _run_command(current_cmd, workdir, log_path)
        total_duration += duration
        if result.returncode == 0:
            return result, total_duration, log_path, attempt

        output_text = (result.stdout or "") + "\n" + (result.stderr or "")
        batch_idx, batch_val = _find_flag_value(current_cmd, "--batch-size")
        if batch_idx is None or not _is_memory_retryable(output_text):
            return result, total_duration, log_path, attempt

        batch_size = int(batch_val)
        if batch_size <= min_batch_size:
            return result, total_duration, log_path, attempt

        next_batch = max(min_batch_size, batch_size // 2)
        if next_batch == batch_size:
            return result, total_duration, log_path, attempt

        current_cmd[batch_idx] = str(next_batch)
        attempt += 1


def _load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _load_existing_rows(summary_json):
    if not os.path.exists(summary_json):
        return []
    try:
        data = _load_json(summary_json)
    except (OSError, json.JSONDecodeError):
        return []
    return data if isinstance(data, list) else []


def _write_markdown(rows, output_path, title="Paper Experiment Summary"):
    lines = [
        f"# {title}",
        "",
        "| Benchmark | Experiment | Status | Source mAP50 | Source mAP50-95 | Target mAP50 | Target mAP50-95 | Target ECE | dECE | Target NLL | dNLL | Target Brier | dBrier | Notes |",
        "| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |",
    ]
    for row in rows:
        lines.append(
            "| {benchmark} | {name} | {status} | {source_map50} | {source_map5095} | {target_map50} | {target_map5095} | {target_ece} | {target_delta_ece} | {target_nll} | {target_delta_nll} | {target_brier} | {target_delta_brier} | {notes} |".format(
                benchmark=row.get("benchmark", "-"),
                name=row["name"],
                status=row["status"],
                source_map50=_fmt(row.get("source_map50")),
                source_map5095=_fmt(row.get("source_map5095")),
                target_map50=_fmt(row.get("target_map50")),
                target_map5095=_fmt(row.get("target_map5095")),
                target_ece=_fmt(row.get("target_ece")),
                target_delta_ece=_fmt(row.get("target_delta_ece")),
                target_nll=_fmt(row.get("target_nll")),
                target_delta_nll=_fmt(row.get("target_delta_nll")),
                target_brier=_fmt(row.get("target_brier")),
                target_delta_brier=_fmt(row.get("target_delta_brier")),
                notes=row.get("notes", ""),
            )
        )
    with open(output_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


def _write_conclusion(rows, output_path):
    by_benchmark = {}
    for row in rows:
        by_benchmark.setdefault(row["benchmark"], []).append(row)

    lines = ["# Formal Experiment Conclusions", ""]

    for benchmark, items in by_benchmark.items():
        ok_items = [row for row in items if row.get("status") == "ok"]
        best_map = max(ok_items, key=lambda r: r.get("target_map50") or -1) if ok_items else None
        best_cal = min(
            [row for row in ok_items if row.get("target_delta_ece") is not None],
            key=lambda r: r.get("target_delta_ece"),
            default=None,
        )
        pretrain_rows = [row for row in ok_items if "pretrain" in row["name"]]
        best_pretrain = max(pretrain_rows, key=lambda r: r.get("source_map50") or -1) if pretrain_rows else None

        lines.append(f"## {benchmark}")
        lines.append("")
        if best_map:
            lines.append(
                f"- Best target mAP50: `{best_map['name']}` = `{_fmt(best_map.get('target_map50'))}` "
                f"(mAP50-95 `{_fmt(best_map.get('target_map5095'))}`)"
            )
        else:
            lines.append("- No successful experiments.")
        if best_cal:
            lines.append(
                f"- Best calibration delta ECE: `{best_cal['name']}` = `{_fmt(best_cal.get('target_delta_ece'))}` "
                f"(NLL `{_fmt(best_cal.get('target_delta_nll'))}`, Brier `{_fmt(best_cal.get('target_delta_brier'))}`)"
            )
        else:
            lines.append("- No calibration result available.")
        if best_pretrain:
            lines.append(
                f"- Most promising pretraining variant: `{best_pretrain['name']}` "
                f"with source mAP50 `{_fmt(best_pretrain.get('source_map50'))}`"
            )
        else:
            lines.append("- No successful pretraining variant to keep yet.")
        lines.append("")

    with open(output_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines).rstrip() + "\n")


def _sort_rows(rows):
    return sorted(
        rows,
        key=lambda row: (
            row.get("status") != "ok",
            -(row.get("target_map50") or -1),
            row.get("target_delta_ece", float("inf")) if row.get("target_delta_ece") is not None else float("inf"),
        ),
    )


def _merge_rows(existing_rows, new_rows):
    merged = {}
    for row in existing_rows + new_rows:
        key = (row.get("benchmark"), row.get("name"))
        merged[key] = row
    return list(merged.values())


def _parse_seeds(seed_text):
    seeds = []
    for token in str(seed_text).replace(",", " ").split():
        token = token.strip()
        if not token:
            continue
        seeds.append(int(token))
    return seeds or [0]


def _with_seed(base_flags, seed):
    return [*base_flags, "--seed", str(seed)]


def _expand_seeded_experiments(experiments, seeds):
    if len(seeds) == 1:
        seed = seeds[0]
        expanded = []
        for exp in experiments:
            clone = deepcopy(exp)
            clone["seed"] = seed
            clone["self_training_flags"] = _with_seed(clone["self_training_flags"], seed)
            expanded.append(clone)
        return expanded

    expanded = []
    for exp in experiments:
        for seed in seeds:
            clone = deepcopy(exp)
            clone["base_name"] = exp["name"]
            clone["name"] = f"{exp['name']}_seed{seed}"
            clone["seed"] = seed
            clone["self_training_flags"] = _with_seed(clone["self_training_flags"], seed)
            expanded.append(clone)
    return expanded


def _calibration_flags(mode="selective", consistency="0.35"):
    return [
        "--use-test-time-calibration",
        "--calibration-mode", mode,
        "--calibration-consistency-threshold", consistency,
        "--calibration-alpha", "0.9",
        "--calibration-beta", "0.1",
        "--calibration-iou-threshold", "0.3",
        "--calibration-views", "original", "hflip",
    ]


def _calibration_eval_flags(mode="selective", consistency="0.35"):
    return [
        "--calibration-mode", mode,
        "--consistency-threshold", consistency,
        "--calibration-alpha", "0.9",
        "--calibration-beta", "0.1",
        "--calibration-iou-threshold", "0.3",
        "--calibration-views", "original", "hflip",
    ]


def build_ablation_experiments(base_self):
    geom_flags = _calibration_flags("selective", "0.35")
    geom_eval = _calibration_eval_flags("selective", "0.35")
    drone_base = [
        "--use-drone-aware-pseudo",
        "--drone-quality-threshold", "0.05",
        "--drone-min-consistency", "0.25",
        "--drone-small-area-threshold", "0.0025",
    ]
    return [
        {
            "name": "baseline",
            "self_training_flags": base_self,
            "calibration_eval": False,
            "notes": "YOLO11 self-training baseline",
        },
        {
            "name": "geometry_quality",
            "self_training_flags": base_self + geom_flags + drone_base + [
                "--drone-small-conf-offset", "0.0",
            ],
            "calibration_eval": True,
            "calibration_flags": geom_eval,
            "notes": "Multi-view geometry consistency and confidence quality, without small-object threshold offset",
        },
        {
            "name": "small_object_threshold",
            "self_training_flags": base_self + geom_flags + [
                "--use-drone-aware-pseudo",
                "--drone-quality-threshold", "0.05",
                "--drone-confidence-power", "1.0",
                "--drone-consistency-power", "0.0",
                "--drone-temporal-power", "0.0",
                "--drone-min-consistency", "0.0",
                "--drone-small-area-threshold", "0.0025",
                "--drone-small-conf-offset", "0.03",
            ],
            "calibration_eval": True,
            "calibration_flags": geom_eval,
            "notes": "Small-object adaptive confidence floor, with geometry metadata used only to enable area-aware pseudo selection",
        },
        {
            "name": "small_object_ema",
            "self_training_flags": base_self + geom_flags + [
                "--use-drone-aware-pseudo",
                "--drone-quality-threshold", "0.05",
                "--drone-confidence-power", "1.0",
                "--drone-consistency-power", "0.0",
                "--drone-temporal-power", "0.0",
                "--drone-min-consistency", "0.0",
                "--drone-small-area-threshold", "0.0025",
                "--drone-small-conf-offset", "0.03",
                "--use-ema-teacher",
                "--ema-decay", "0.999",
            ],
            "calibration_eval": True,
            "calibration_flags": geom_eval,
            "notes": "Small-object adaptive threshold with EMA teacher for multi-round self-training",
        },
        {
            "name": "small_object_offset_001",
            "self_training_flags": base_self + geom_flags + [
                "--use-drone-aware-pseudo",
                "--drone-quality-threshold", "0.05",
                "--drone-confidence-power", "1.0",
                "--drone-consistency-power", "0.0",
                "--drone-temporal-power", "0.0",
                "--drone-min-consistency", "0.0",
                "--drone-small-area-threshold", "0.0025",
                "--drone-small-conf-offset", "0.01",
            ],
            "calibration_eval": True,
            "calibration_flags": geom_eval,
            "notes": "Small-object confidence offset sensitivity: 0.01",
        },
        {
            "name": "small_object_offset_005",
            "self_training_flags": base_self + geom_flags + [
                "--use-drone-aware-pseudo",
                "--drone-quality-threshold", "0.05",
                "--drone-confidence-power", "1.0",
                "--drone-consistency-power", "0.0",
                "--drone-temporal-power", "0.0",
                "--drone-min-consistency", "0.0",
                "--drone-small-area-threshold", "0.0025",
                "--drone-small-conf-offset", "0.05",
            ],
            "calibration_eval": True,
            "calibration_flags": geom_eval,
            "notes": "Small-object confidence offset sensitivity: 0.05",
        },
        {
            "name": "uncertainty_aware_small_object",
            "self_training_flags": base_self + geom_flags + [
                "--use-uncertainty-aware-pseudo",
                "--drone-small-area-threshold", "0.0025",
                "--drone-small-conf-offset", "0.03",
                "--uncertainty-max-std", "0.20",
                "--uncertainty-penalty-scale", "1.0",
            ],
            "calibration_eval": True,
            "calibration_flags": geom_eval,
            "notes": "Small-object adaptive threshold with uncertainty-aware offset reduction via TTA confidence std",
        },
        {
            "name": "uncertainty_aware_4views",
            "self_training_flags": base_self + [
                "--use-test-time-calibration",
                "--calibration-mode", "selective",
                "--calibration-consistency-threshold", "0.35",
                "--calibration-alpha", "0.9",
                "--calibration-beta", "0.1",
                "--calibration-iou-threshold", "0.3",
                "--calibration-views", "original", "hflip", "bright", "dark",
                "--use-uncertainty-aware-pseudo",
                "--drone-small-area-threshold", "0.0025",
                "--drone-small-conf-offset", "0.03",
                "--uncertainty-max-std", "0.20",
                "--uncertainty-penalty-scale", "1.0",
            ],
            "calibration_eval": True,
            "calibration_flags": [
                "--calibration-mode", "selective",
                "--consistency-threshold", "0.35",
                "--calibration-alpha", "0.9",
                "--calibration-beta", "0.1",
                "--calibration-iou-threshold", "0.3",
                "--calibration-views", "original", "hflip", "bright", "dark",
            ],
            "notes": "Uncertainty-aware small-object threshold with 4 diverse TTA views",
        },
        {
            "name": "ema_teacher",
            "self_training_flags": base_self + [
                "--use-ema-teacher",
                "--ema-decay", "0.5",
            ],
            "calibration_eval": False,
            "notes": "Iteration-level EMA teacher for next-round pseudo-label generation",
        },
        {
            "name": "geometry_small",
            "self_training_flags": base_self + geom_flags + drone_base + [
                "--drone-small-conf-offset", "0.03",
            ],
            "calibration_eval": True,
            "calibration_flags": geom_eval,
            "notes": "Geometry consistency + uncertainty quality + small-object confidence floor",
        },
        {
            "name": "full_method",
            "self_training_flags": base_self + geom_flags + drone_base + [
                "--drone-small-conf-offset", "0.03",
                "--use-ema-teacher",
                "--ema-decay", "0.5",
            ],
            "calibration_eval": True,
            "calibration_flags": geom_eval,
            "notes": "Full method: geometry consistency + uncertainty quality + small-object floor + EMA teacher",
        },
    ]


def build_experiments(mode, benchmark_cfg, opt):
    self_batch = opt.self_batch_size or benchmark_cfg["self_training"]["batch_size"]
    self_img = opt.self_img_size or benchmark_cfg["self_training"]["img_size"]
    self_iters = opt.self_iterations or benchmark_cfg["self_training"]["iterations"]
    self_epochs = opt.self_epochs_per_iter or benchmark_cfg["self_training"]["epochs_per_iter"]

    contrast_batch = opt.contrastive_batch_size or benchmark_cfg["contrastive"]["batch_size"]
    contrast_img = opt.contrastive_img_size or benchmark_cfg["contrastive"]["img_size"]
    contrast_epochs = opt.contrastive_epochs or benchmark_cfg["contrastive"]["epochs"]
    contrast_pos = opt.contrastive_num_positives or benchmark_cfg["contrastive"]["num_positives"]

    if mode == "smoke":
        self_batch = opt.self_batch_size or min(self_batch, 2)
        self_img = opt.self_img_size or 64
        self_iters = opt.self_iterations or 1
        self_epochs = opt.self_epochs_per_iter or 1
        contrast_batch = opt.contrastive_batch_size or min(contrast_batch, 2)
        contrast_img = opt.contrastive_img_size or 64
        contrast_epochs = opt.contrastive_epochs or 1
        contrast_pos = opt.contrastive_num_positives or 1

    base_self = [
        "--iterations", str(self_iters),
        "--epochs-per-iter", str(self_epochs),
        "--batch-size", str(self_batch),
        "--img-size", str(self_img),
        "--device", opt.device,
        "--workers", str(opt.workers),
        "--inference-batch-size", str(opt.inference_batch_size),
        "--initial-conf-threshold", str(opt.initial_conf_threshold),
    ]

    if opt.suite == "ablation":
        return _expand_seeded_experiments(build_ablation_experiments(base_self), _parse_seeds(opt.seeds))

    base_contrastive = [
        "--method", "simclr",
        "--device", opt.device,
        "--epochs", str(contrast_epochs),
        "--batch-size", str(contrast_batch),
        "--num-positives", str(contrast_pos),
        "--num-workers", str(opt.contrastive_num_workers),
        "--img-size", str(contrast_img),
        "--freeze-target", "backbone",
        "--stage1-epochs", str(min(4, contrast_epochs)),
        "--save-full-pt",
    ]

    experiments = [
        {
            "name": "baseline",
            "self_training_flags": base_self,
            "calibration_eval": False,
            "notes": "YOLO11 self-training baseline",
        },
        {
            "name": "rank_calibration",
            "self_training_flags": base_self + [
                "--use-test-time-calibration",
                "--calibration-mode", "rank",
                "--calibration-consistency-threshold", "0.5",
                "--calibration-rank-top-fraction", "0.25",
                "--calibration-rank-min-confidence", "0.1",
                "--calibration-alpha", "0.9",
                "--calibration-beta", "0.1",
                "--calibration-iou-threshold", "0.3",
                "--calibration-views", "original", "hflip",
            ],
            "calibration_eval": True,
            "calibration_flags": [
                "--calibration-mode", "rank",
                "--consistency-threshold", "0.5",
                "--rank-top-fraction", "0.25",
                "--rank-min-confidence", "0.1",
                "--calibration-alpha", "0.9",
                "--calibration-beta", "0.1",
                "--calibration-iou-threshold", "0.3",
                "--calibration-views", "original", "hflip",
            ],
            "notes": "Baseline + rank-based test-time calibration",
        },
        {
            "name": "drone_geometry_uncertainty",
            "self_training_flags": base_self + [
                "--use-test-time-calibration",
                "--calibration-mode", "selective",
                "--calibration-consistency-threshold", "0.35",
                "--calibration-alpha", "0.9",
                "--calibration-beta", "0.1",
                "--calibration-iou-threshold", "0.3",
                "--calibration-views", "original", "hflip",
                "--use-drone-aware-pseudo",
                "--drone-quality-threshold", "0.05",
                "--drone-min-consistency", "0.25",
                "--drone-small-area-threshold", "0.0025",
                "--drone-small-conf-offset", "0.03",
                "--use-ema-teacher",
                "--ema-decay", "0.5",
            ],
            "calibration_eval": True,
            "calibration_flags": [
                "--calibration-mode", "selective",
                "--consistency-threshold", "0.35",
                "--calibration-alpha", "0.9",
                "--calibration-beta", "0.1",
                "--calibration-iou-threshold", "0.3",
                "--calibration-views", "original", "hflip",
            ],
            "notes": "Geometry consistency + uncertainty quality + small-object floor + EMA teacher",
        },
        {
            "name": "progressive_pseudo",
            "self_training_flags": base_self + [
                "--use-progressive-pseudo",
                "--consistency-threshold", "0.5",
            ],
            "calibration_eval": False,
            "notes": "Baseline + temporal pseudo refinement",
        },
        {
            "name": "calibration_guided_pseudo",
            "self_training_flags": base_self + [
                "--use-test-time-calibration",
                "--calibration-mode", "rank",
                "--calibration-consistency-threshold", "0.5",
                "--calibration-rank-top-fraction", "0.25",
                "--calibration-rank-min-confidence", "0.1",
                "--calibration-alpha", "0.9",
                "--calibration-beta", "0.1",
                "--calibration-iou-threshold", "0.3",
                "--calibration-views", "original", "hflip",
                "--use-calibration-guided-pseudo",
                "--pseudo-calibration-min-consistency", "0.5",
                "--pseudo-calibration-max-risk", "0.25",
            ],
            "calibration_eval": True,
            "calibration_flags": [
                "--calibration-mode", "rank",
                "--consistency-threshold", "0.5",
                "--rank-top-fraction", "0.25",
                "--rank-min-confidence", "0.1",
                "--calibration-alpha", "0.9",
                "--calibration-beta", "0.1",
                "--calibration-iou-threshold", "0.3",
                "--calibration-views", "original", "hflip",
            ],
            "notes": "Rank calibration reused to gate pseudo labels by consistency and risk",
        },
        {
            "name": "ema_teacher",
            "self_training_flags": base_self + [
                "--use-ema-teacher",
                "--ema-decay", "0.999",
            ],
            "calibration_eval": False,
            "notes": "Iteration-level EMA teacher-student self-training",
        },
        {
            "name": "adaptive_pseudo_threshold",
            "self_training_flags": base_self + [
                "--use-test-time-calibration",
                "--calibration-mode", "rank",
                "--calibration-consistency-threshold", "0.5",
                "--calibration-rank-top-fraction", "0.25",
                "--calibration-rank-min-confidence", "0.1",
                "--calibration-alpha", "0.9",
                "--calibration-beta", "0.1",
                "--calibration-iou-threshold", "0.3",
                "--calibration-views", "original", "hflip",
                "--use-adaptive-pseudo-threshold",
                "--pseudo-adaptive-target-consistency", "0.5",
                "--pseudo-adaptive-threshold-scale", "0.3",
            ],
            "calibration_eval": True,
            "calibration_flags": [
                "--calibration-mode", "rank",
                "--consistency-threshold", "0.5",
                "--rank-top-fraction", "0.25",
                "--rank-min-confidence", "0.1",
                "--calibration-alpha", "0.9",
                "--calibration-beta", "0.1",
                "--calibration-iou-threshold", "0.3",
                "--calibration-views", "original", "hflip",
            ],
            "notes": "Adaptive pseudo thresholding driven by calibration consistency",
        },
    ]

    if mode in {"default", "extended"}:
        contrastive_dataset_flags = [
            "--pos-dir", benchmark_cfg["target_dataset"] + "/unlabels_img",
            "--neg-dir", benchmark_cfg["source_dataset"] + "/images/train",
            "--yolo-weights-path", os.path.join("weights", "yolo11n.pt"),
        ]
        experiments.extend(
            [
                {
                    "name": "cbam_pretrain",
                    "contrastive_flags": base_contrastive + contrastive_dataset_flags + [
                        "--use-cbam",
                        "--cbam-reduction", "16",
                    ],
                    "self_training_flags": base_self,
                    "calibration_eval": False,
                    "notes": "Contrastive pretraining with CBAM",
                },
                {
                    "name": "domain_loss_pretrain",
                    "contrastive_flags": base_contrastive + contrastive_dataset_flags + [
                        "--use-domain-loss",
                        "--lambda-domain", "0.5",
                    ],
                    "self_training_flags": base_self,
                    "calibration_eval": False,
                    "notes": "Contrastive pretraining with domain-aware loss",
                },
                {
                    "name": "layerlock_pretrain",
                    "contrastive_flags": base_contrastive + contrastive_dataset_flags + [
                        "--use-layerlock",
                        "--layerlock-freeze-steps=-1,10",
                        "--layerlock-feature-levels", "p3,p5",
                        "--layerlock-phase-epochs", f"{max(1, contrast_epochs // 2)},{contrast_epochs - max(1, contrast_epochs // 2)}",
                    ],
                    "self_training_flags": base_self,
                    "calibration_eval": False,
                    "notes": "LayerLock-inspired contrastive pretraining",
                },
            ]
        )

    return _expand_seeded_experiments(experiments, _parse_seeds(opt.seeds))


def _filter_experiments(experiments, include_names):
    if not include_names:
        return experiments
    include_set = {name.strip() for name in include_names if name.strip()}
    return [
        exp for exp in experiments
        if exp["name"] in include_set or exp.get("base_name") in include_set
    ]


def _write_summary_files(rows_sorted, output_root, title):
    summary_json = os.path.join(output_root, "experiment_summary.json")
    with open(summary_json, "w", encoding="utf-8") as f:
        json.dump(rows_sorted, f, indent=2)

    fieldnames = sorted({key for row in rows_sorted for key in row.keys()}) if rows_sorted else []
    summary_csv = os.path.join(output_root, "experiment_summary.csv")
    if fieldnames:
        with open(summary_csv, "w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows_sorted)

    summary_md = os.path.join(output_root, "experiment_summary.md")
    _write_markdown(rows_sorted, summary_md, title=title)
    _write_seed_aggregate(rows_sorted, os.path.join(output_root, "seed_aggregate.csv"))
    return summary_json, summary_csv, summary_md


def _mean_std(values):
    clean = [float(v) for v in values if v is not None]
    if not clean:
        return None, None
    if len(clean) == 1:
        return mean(clean), 0.0
    return mean(clean), stdev(clean)


def _write_seed_aggregate(rows, output_path):
    groups = {}
    for row in rows:
        if row.get("status") != "ok":
            continue
        key = (row.get("benchmark"), row.get("base_name") or row.get("name"))
        groups.setdefault(key, []).append(row)
    if not groups:
        return

    metrics = [
        "target_map50",
        "target_map5095",
        "target_map75",
        "target_precision",
        "target_recall",
        "source_map50",
        "source_map5095",
    ]
    fieldnames = ["benchmark", "experiment", "num_seeds", "seeds"]
    for metric in metrics:
        fieldnames.extend([f"{metric}_mean", f"{metric}_std"])

    rows_out = []
    for (benchmark, experiment), items in sorted(groups.items()):
        out = {
            "benchmark": benchmark,
            "experiment": experiment,
            "num_seeds": len(items),
            "seeds": " ".join(str(item.get("seed", "")) for item in items),
        }
        for metric in metrics:
            avg, sd = _mean_std([item.get(metric) for item in items])
            out[f"{metric}_mean"] = avg
            out[f"{metric}_std"] = sd
        rows_out.append(out)

    with open(output_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows_out)


def run_benchmark(opt, benchmark_name):
    python_exe = opt.python_exe or sys.executable
    workdir = os.path.abspath(opt.workdir)
    benchmark_cfg = deepcopy(BENCHMARKS[benchmark_name])

    if opt.source_dataset:
        benchmark_cfg["source_dataset"] = opt.source_dataset
    if opt.target_dataset:
        benchmark_cfg["target_dataset"] = opt.target_dataset
    if opt.target_val_image_dir:
        benchmark_cfg["target_val_image_dir"] = opt.target_val_image_dir
    if opt.target_val_label_dir:
        benchmark_cfg["target_val_label_dir"] = opt.target_val_label_dir

    output_root = os.path.abspath(os.path.join(opt.output_root, benchmark_name))
    os.makedirs(output_root, exist_ok=True)
    summary_json = os.path.join(output_root, "experiment_summary.json")
    existing_rows = _load_existing_rows(summary_json)
    completed_names = {
        row.get("name") for row in existing_rows
        if row.get("status") == "ok" and row.get("target_map50") is not None
    }

    def persist_rows(new_rows):
        merged = _sort_rows(_merge_rows(existing_rows, new_rows))
        _write_summary_files(
            merged, output_root, f"{benchmark_cfg['display_name']} Experiment Summary"
        )
        return merged

    experiments = build_experiments(opt.mode, benchmark_cfg, opt)
    experiments = _filter_experiments(experiments, opt.include_experiments)
    if opt.max_experiments > 0:
        experiments = experiments[:opt.max_experiments]

    rows = []
    print(f"[Experiments] benchmark={benchmark_name} planned={len(experiments)}")

    for exp in experiments:
        if exp["name"] in completed_names and not opt.rerun_completed:
            print(f"[Experiments] skipping completed {benchmark_name}/{exp['name']}")
            continue

        exp_row = deepcopy(exp)
        exp_row["benchmark"] = benchmark_name
        exp_row["benchmark_display_name"] = benchmark_cfg["display_name"]
        exp_row["status"] = "ok"
        exp_dir = os.path.join(output_root, exp["name"])
        os.makedirs(exp_dir, exist_ok=True)
        print(f"[Experiments] running {benchmark_name}/{exp['name']}")

        current_weights = (
            os.path.abspath(opt.initial_weights)
            if opt.initial_weights
            else os.path.join(workdir, "weights", "yolo11n.pt")
        )

        contrastive_flags = exp.get("contrastive_flags")
        if contrastive_flags:
            contrastive_save = os.path.join(exp_dir, f"{exp['name']}_contrastive.pt")
            contrastive_cmd = [
                python_exe,
                "train_contrastive.py",
                *contrastive_flags,
                "--save-path", contrastive_save,
                "--loss-png", os.path.join(exp_dir, "contrastive_loss.png"),
                "--loss-csv", os.path.join(exp_dir, "contrastive_loss.csv"),
            ]
            result, duration, contrastive_log, contrastive_attempts = _retry_with_lower_batch(
                contrastive_cmd, workdir, os.path.join(exp_dir, "contrastive"), min_batch_size=1
            )
            exp_row["contrastive_duration_sec"] = round(duration, 2)
            exp_row["contrastive_returncode"] = result.returncode
            exp_row["contrastive_log"] = contrastive_log
            exp_row["contrastive_attempts"] = contrastive_attempts + 1
            if result.returncode != 0:
                exp_row["status"] = "failed"
                exp_row["notes"] = exp.get("notes", "") + " | contrastive failed"
                rows.append(exp_row)
                persist_rows(rows)
                continue
            current_weights = contrastive_save.replace(".pt", "_full.pt")
            exp_row["contrastive_ckpt"] = current_weights

        self_cmd = [
            python_exe,
            "self_training.py",
            "--initial-weights", current_weights,
            "--yolo-base-weights", os.path.join("weights", "yolo11n.pt"),
            "--source-dataset", benchmark_cfg["source_dataset"],
            "--target-dataset", benchmark_cfg["target_dataset"],
            *exp["self_training_flags"],
        ]
        result, duration, self_log, self_attempts = _retry_with_lower_batch(
            self_cmd, workdir, os.path.join(exp_dir, "self_training"), min_batch_size=1
        )
        exp_row["self_training_duration_sec"] = round(duration, 2)
        exp_row["self_training_returncode"] = result.returncode
        exp_row["self_training_log"] = self_log
        exp_row["self_training_attempts"] = self_attempts + 1
        if result.returncode != 0:
            exp_row["status"] = "failed"
            exp_row["notes"] = exp.get("notes", "") + " | self-training failed"
            rows.append(exp_row)
            persist_rows(rows)
            continue

        output_text = result.stdout + "\n" + result.stderr
        exp_row["source_map50"] = _parse_metric(r"mAP@0\.5:\s+([0-9.]+)", output_text, float)
        exp_row["source_map5095"] = _parse_metric(r"mAP@0\.5:0\.95:\s+([0-9.]+)", output_text, float)
        final_weights = _parse_metric(r"Final weights:\s+(.+)", output_text, str)
        exp_row["final_weights"] = final_weights

        if final_weights:
            target_eval_json = os.path.join(exp_dir, "target_detection_metrics.json")
            target_eval_log = os.path.join(exp_dir, "target_detection_eval.log")
            target_eval_cmd = [
                python_exe,
                "evaluate_detector.py",
                "--weights", final_weights,
                "--data", benchmark_cfg["target_val_yaml"],
                "--output", target_eval_json,
                "--img-size", str(opt.eval_img_size or benchmark_cfg["self_training"]["img_size"]),
                "--batch-size", str(opt.self_batch_size or benchmark_cfg["self_training"]["batch_size"]),
                "--device", opt.device,
                "--workers", str(opt.workers),
            ]
            target_result, target_duration = _run_command(
                target_eval_cmd, workdir, target_eval_log
            )
            exp_row["target_detection_duration_sec"] = round(target_duration, 2)
            exp_row["target_detection_returncode"] = target_result.returncode
            exp_row["target_detection_log"] = target_eval_log
            if target_result.returncode == 0:
                target_metrics = _load_json(target_eval_json)
                exp_row["target_map50"] = target_metrics["map50"]
                exp_row["target_map5095"] = target_metrics["map50_95"]
                exp_row["target_map75"] = target_metrics["map75"]
                exp_row["target_precision"] = target_metrics["precision"]
                exp_row["target_recall"] = target_metrics["recall"]
            else:
                exp_row["status"] = "failed"
                exp_row["notes"] = exp.get("notes", "") + " | target detection eval failed"

        if exp.get("calibration_eval") and final_weights:
            eval_dir = os.path.join(exp_dir, "calibration_eval")
            eval_log = os.path.join(exp_dir, "calibration_eval.log")
            eval_cmd = [
                python_exe,
                "evaluate_calibration.py",
                "--weights", final_weights,
                "--yolo-base-weights", os.path.join("weights", "yolo11n.pt"),
                "--image-dir", benchmark_cfg["target_val_image_dir"],
                "--label-dir", benchmark_cfg["target_val_label_dir"],
                "--output-dir", eval_dir,
                "--img-size", str(opt.eval_img_size or benchmark_cfg["self_training"]["img_size"]),
                "--device", opt.device,
                *exp.get("calibration_flags", []),
            ]
            result, duration = _run_command(eval_cmd, workdir, eval_log)
            exp_row["calibration_duration_sec"] = round(duration, 2)
            exp_row["calibration_returncode"] = result.returncode
            exp_row["calibration_log"] = eval_log
            if result.returncode == 0:
                comparison = _load_json(os.path.join(eval_dir, "comparison.json"))
                exp_row["target_ece"] = comparison["calibrated"]["ece"]
                exp_row["target_nll"] = comparison["calibrated"]["nll"]
                exp_row["target_brier"] = comparison["calibrated"]["brier"]
                exp_row["target_delta_ece"] = comparison["delta"]["ece"]
                exp_row["target_delta_nll"] = comparison["delta"]["nll"]
                exp_row["target_delta_brier"] = comparison["delta"]["brier"]
            else:
                exp_row["status"] = "failed"
                exp_row["notes"] = exp.get("notes", "") + " | calibration eval failed"

        rows.append(exp_row)
        persist_rows(rows)

    rows_sorted = persist_rows(rows)
    return rows_sorted


def write_cross_benchmark_summary(all_rows, output_root):
    rows_sorted = _sort_rows(all_rows)
    cross_root = os.path.abspath(os.path.join(output_root, "cross_benchmark"))
    os.makedirs(cross_root, exist_ok=True)
    _write_summary_files(rows_sorted, cross_root, "Cross-Benchmark Experiment Summary")
    _write_conclusion(rows_sorted, os.path.join(cross_root, "conclusions.md"))
    return cross_root


def main(opt):
    benchmark_names = [opt.benchmark] if opt.benchmark != "all" else list(BENCHMARKS.keys())
    all_rows = []

    for benchmark_name in benchmark_names:
        rows = run_benchmark(opt, benchmark_name)
        all_rows.extend(rows)

    cross_root = write_cross_benchmark_summary(all_rows, os.path.abspath(opt.output_root))

    print("=" * 60)
    print("PAPER EXPERIMENTS COMPLETE")
    print("=" * 60)
    if all_rows:
        best = _sort_rows(all_rows)[0]
        print(
            f"[Best] {best['benchmark']}/{best['name']} "
            f"status={best['status']} "
            f"source_mAP50={_fmt(best.get('source_map50'))} "
            f"target_dECE={_fmt(best.get('target_delta_ece'))}"
        )
    print(f"[Files] {cross_root}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run formal paper-oriented experiment matrices")
    parser.add_argument("--python-exe", type=str, default="", help="Optional Python executable to use")
    parser.add_argument("--workdir", type=str, default=".", help="Project working directory")
    parser.add_argument("--output-root", type=str, default="runs/paper_experiments_formal",
                        help="Output directory for experiment logs and summaries")
    parser.add_argument("--mode", type=str, default="default", choices=["smoke", "default", "extended"],
                        help="Experiment matrix size")
    parser.add_argument("--suite", type=str, default="standard", choices=["standard", "ablation"],
                        help="Experiment suite: standard keeps the broad matrix; ablation runs method-component ablations")
    parser.add_argument("--seeds", type=str, default="0",
                        help="Comma- or space-separated random seeds, e.g. '0,1,2'")
    parser.add_argument("--rerun-completed", action="store_true",
                        help="Run experiments again even when a successful summary row already exists")
    parser.add_argument("--benchmark", type=str, default="all",
                        choices=["all", *BENCHMARKS.keys()],
                        help="Prepared benchmark to run")
    parser.add_argument("--source-dataset", type=str, default="", help="Override source dataset directory")
    parser.add_argument("--initial-weights", type=str, default="",
                        help="Common initial detector weights for all experiments")
    parser.add_argument("--target-dataset", type=str, default="", help="Override target dataset directory")
    parser.add_argument("--target-val-image-dir", type=str, default="", help="Override target validation image directory")
    parser.add_argument("--target-val-label-dir", type=str, default="", help="Override target validation label directory")
    parser.add_argument("--device", type=str, default="0", help="CUDA device or CPU target")
    parser.add_argument("--workers", type=int, default=1,
                        help="Ultralytics dataloader workers; 1 balances throughput and RAM use on Windows")
    parser.add_argument("--inference-batch-size", type=int, default=16,
                        help="Batch size for pseudo-label inference and calibrated views")
    parser.add_argument("--contrastive-num-workers", type=int, default=0, help="Dataloader workers for contrastive pretraining")
    parser.add_argument("--initial-conf-threshold", type=float, default=0.1, help="Initial pseudo-label threshold")
    parser.add_argument("--self-img-size", type=int, default=0, help="Override self-training image size")
    parser.add_argument("--self-batch-size", type=int, default=0, help="Override self-training batch size")
    parser.add_argument("--self-iterations", type=int, default=0, help="Override self-training iterations")
    parser.add_argument("--self-epochs-per-iter", type=int, default=0, help="Override self-training epochs per iteration")
    parser.add_argument("--contrastive-img-size", type=int, default=0, help="Override contrastive image size")
    parser.add_argument("--contrastive-batch-size", type=int, default=0, help="Override contrastive batch size")
    parser.add_argument("--contrastive-epochs", type=int, default=0, help="Override contrastive epochs")
    parser.add_argument("--contrastive-num-positives", type=int, default=0, help="Override positives per contrastive batch")
    parser.add_argument("--eval-img-size", type=int, default=0, help="Override calibration evaluation image size")
    parser.add_argument("--max-experiments", type=int, default=0, help="Optional cap on total experiments per benchmark")
    parser.add_argument("--include-experiments", nargs="*", default=[],
                        help="Optional experiment names to run, e.g. baseline rank_calibration")
    main(parser.parse_args())

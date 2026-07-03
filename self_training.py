#!/usr/bin/env python
"""
Unified Self-Training Loop for YOLO11-UDA

Iterative pseudo-labeling and fine-tuning using Ultralytics YOLO11.
Replaces subprocess-based YOLOv7 calls with direct Python API.

Usage:
    # Dynamic threshold with plotting (recommended)
    python self_training.py --dynamic-threshold --plot-results --iterations 5

    # Fixed threshold
    python self_training.py --initial-conf-threshold 0.7 --iterations 3

    # With contrastive pretrained weights
    python self_training.py --initial-weights contrastive_pretrained.pt --iterations 3
"""

import os
import sys
import shutil
import argparse
import time
import json
import random
from pathlib import Path

import yaml
import torch
import numpy as np
from PIL import Image
from modules.losses import compute_pseudo_label_quality
from modules.test_time_calibration import run_calibrated_inference

os.environ.setdefault("POLARS_SKIP_CPU_CHECK", "1")

# Optional dependencies for plotting
try:
    import pandas as pd
    HAS_PANDAS = True
except ImportError:
    HAS_PANDAS = False

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False


# --- Core functions ---

def set_reproducible_seed(seed):
    """Seed Python, NumPy and PyTorch for repeatable experiment runs."""
    if seed is None or seed < 0:
        return
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

class _SequentialPool:
    """Small context-manager compatible substitute for multiprocessing.pool.ThreadPool."""
    def __init__(self, *_args, **_kwargs):
        pass

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False

    def imap(self, func, iterable):
        for item in iterable:
            yield func(item)


def _pil_imread(filename, flags=1):
    """
    More stable Windows fallback for Ultralytics' imread wrapper.

    The default implementation uses np.fromfile(), which has triggered
    intermittent ArrayMemoryError failures in long-running experiments
    on this machine.
    """
    with Image.open(filename) as img:
        if flags == 0:
            img = img.convert("L")
            return np.array(img)
        img = img.convert("RGB")
        arr = np.array(img)
        return arr[:, :, ::-1]  # RGB -> BGR


def patch_ultralytics_threadpool():
    """
    Avoid WinError 5 in restricted Windows environments where Ultralytics'
    cache-building ThreadPool cannot allocate multiprocessing pipes.
    """
    import ultralytics.data.base as data_base
    import ultralytics.data.dataset as data_dataset
    import ultralytics.utils.patches as patches

    data_base.ThreadPool = _SequentialPool
    data_dataset.ThreadPool = _SequentialPool
    patches.imread = _pil_imread


def resolve_yolo_weights(weights_path, base_weights_path="weights/yolo11n.pt", cache_dir="runs/prepared_weights"):
    """
    Return a YOLO-loadable weights file.

    Supports either:
      1. a normal Ultralytics .pt checkpoint, or
      2. a contrastive-pretrained backbone+neck state_dict saved by train_contrastive.py
    """
    from ultralytics import YOLO
    patch_ultralytics_threadpool()

    try:
        YOLO(weights_path)
        return weights_path
    except Exception:
        pass

    print(f"[Weights] '{weights_path}' is not directly loadable by YOLO. Trying backbone+neck merge...")
    state = torch.load(weights_path, map_location="cpu")
    if not isinstance(state, dict):
        raise ValueError(f"Unsupported weights format: {weights_path}")

    os.makedirs(cache_dir, exist_ok=True)
    stem = Path(weights_path).stem
    merged_path = os.path.join(cache_dir, f"{stem}_full.pt")

    if os.path.exists(merged_path):
        print(f"[Weights] Reusing merged YOLO checkpoint: {merged_path}")
        return merged_path

    model = YOLO(base_weights_path)
    missing = model.model.load_state_dict(state, strict=False)
    print(f"[Weights] Loaded backbone state_dict with strict=False")
    print(f"[Weights] Missing keys: {len(missing.missing_keys)}, unexpected keys: {len(missing.unexpected_keys)}")
    model.save(merged_path)
    print(f"[Weights] Saved merged YOLO checkpoint to: {merged_path}")
    return merged_path

def run_inference(weights_path, image_folder, conf_thres=0.01, img_size=640, device="0",
                  project="runs/predict", name="predict",
                  use_test_time_calibration=False,
                  calibration_views=None,
                  calibration_alpha=0.50,
                  calibration_beta=0.50,
                  calibration_iou_threshold=0.5,
                  calibration_mode="global",
                  calibration_consistency_threshold=0.5,
                  calibration_rank_top_fraction=0.25,
                  calibration_rank_min_confidence=0.1,
                  calibration_rank_class_aware=False,
                  inference_batch_size=16):
    """
    Run YOLO11 inference on unlabeled target domain images.

    Returns:
        predictions: list of detection results
    """
    from ultralytics import YOLO
    patch_ultralytics_threadpool()

    print(f"\n---> Step 1: Running inference with weights {weights_path}...")
    model = YOLO(weights_path)
    project_dir = os.path.abspath(project)
    os.makedirs(project_dir, exist_ok=True)
    if use_test_time_calibration:
        save_dir = os.path.join(project_dir, name)
        print(f"    Using test-time consistency calibration with views={calibration_views}.")
        summary = run_calibrated_inference(
            model,
            image_folder=image_folder,
            save_dir=save_dir,
            conf_thres=conf_thres,
            img_size=img_size,
            device=device,
            alpha=calibration_alpha,
            beta=calibration_beta,
            match_iou_threshold=calibration_iou_threshold,
            views=calibration_views,
            calibration_mode=calibration_mode,
            consistency_threshold=calibration_consistency_threshold,
            rank_top_fraction=calibration_rank_top_fraction,
            rank_min_confidence=calibration_rank_min_confidence,
            rank_class_aware=calibration_rank_class_aware,
            inference_batch_size=inference_batch_size,
        )
        print(
            "    Calibrated inference complete on "
            f"{summary['num_images']} images, mean consistency={summary['mean_consistency']:.4f}."
        )
        return summary, save_dir

    results = model.predict(
        source=image_folder,
        conf=conf_thres,
        imgsz=img_size,
        device=device,
        project=project_dir,
        name=name,
        exist_ok=True,
        save_txt=True,
        save_conf=True,
        verbose=False,
        stream=True,
        batch=inference_batch_size,
    )
    save_dir = None
    num_images = 0
    for result in results:
        num_images += 1
        if save_dir is None:
            save_dir = str(result.save_dir)
    print(f"    Inference complete on {num_images} images.")
    return {"num_images": num_images}, save_dir


def _yolo_xywh_to_xyxy(xc, yc, w, h):
    x1 = xc - w / 2.0
    y1 = yc - h / 2.0
    x2 = xc + w / 2.0
    y2 = yc + h / 2.0
    return [x1, y1, x2, y2]


def _load_prediction_tensor(txt_path):
    rows = []
    if not txt_path or not os.path.exists(txt_path):
        return None

    with open(txt_path, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 5:
                continue
            cls = float(parts[0])
            xc, yc, w, h = map(float, parts[1:5])
            conf = float(parts[5]) if len(parts) >= 6 else 1.0
            x1, y1, x2, y2 = _yolo_xywh_to_xyxy(xc, yc, w, h)
            rows.append([x1, y1, x2, y2, conf, cls])

    if not rows:
        return None
    return torch.tensor(rows, dtype=torch.float32)


def _load_calibration_detections(metadata_dir, fname):
    if not metadata_dir:
        return None
    metadata_path = os.path.join(metadata_dir, Path(fname).with_suffix(".json").name)
    if not os.path.exists(metadata_path):
        return None
    with open(metadata_path, "r", encoding="utf-8") as f:
        payload = json.load(f)
    return payload.get("detections", [])


def _build_calibration_mask(num_preds, calibration_detections, min_consistency, max_risk):
    mask = torch.ones(num_preds, dtype=torch.bool)
    if not calibration_detections:
        return mask

    for idx in range(min(num_preds, len(calibration_detections))):
        det = calibration_detections[idx]
        consistency = float(det.get("consistency", 1.0))
        risk_score = float(det.get("risk_score", 0.0))
        if consistency < min_consistency or risk_score > max_risk:
            mask[idx] = False
    return mask


def _adaptive_conf_threshold(base_threshold, calibration_det, target_consistency, scale):
    if not calibration_det:
        return base_threshold
    consistency = float(calibration_det.get("consistency", target_consistency))
    penalty = max(0.0, target_consistency - consistency)
    return min(0.999, max(0.0, base_threshold + scale * penalty))


def _same_class_temporal_scores(current_preds, previous_preds, iou_threshold=0.5):
    """Return continuous same-class cross-iteration stability scores in [0, 1]."""
    scores = torch.ones(len(current_preds), dtype=torch.float32)
    if previous_preds is None or len(previous_preds) == 0:
        return scores

    scores.zero_()
    for cls in current_preds[:, 5].unique():
        current_idx = torch.where(current_preds[:, 5] == cls)[0]
        previous_idx = torch.where(previous_preds[:, 5] == cls)[0]
        if len(previous_idx) == 0:
            continue
        current_boxes = current_preds[current_idx, :4]
        previous_boxes = previous_preds[previous_idx, :4]
        lt = torch.maximum(current_boxes[:, None, :2], previous_boxes[None, :, :2])
        rb = torch.minimum(current_boxes[:, None, 2:4], previous_boxes[None, :, 2:4])
        wh = (rb - lt).clamp(min=0)
        inter = wh[:, :, 0] * wh[:, :, 1]
        current_area = (
            (current_boxes[:, 2] - current_boxes[:, 0]).clamp(min=0)
            * (current_boxes[:, 3] - current_boxes[:, 1]).clamp(min=0)
        )
        previous_area = (
            (previous_boxes[:, 2] - previous_boxes[:, 0]).clamp(min=0)
            * (previous_boxes[:, 3] - previous_boxes[:, 1]).clamp(min=0)
        )
        iou = inter / (current_area[:, None] + previous_area[None, :] - inter + 1e-6)
        best_iou = iou.max(dim=1).values
        # Below-threshold matches contribute proportionally instead of becoming false certainty.
        scores[current_idx] = (best_iou / max(iou_threshold, 1e-6)).clamp(max=1.0)
    return scores


def _filter_drone_aware_pseudo(
    current_txt,
    previous_txt,
    calibration_detections,
    base_conf_threshold,
    quality_threshold=0.35,
    confidence_power=1.0,
    consistency_power=1.0,
    temporal_power=0.5,
    temporal_iou_threshold=0.5,
    min_consistency=0.35,
    small_area_threshold=0.0025,
    small_conf_offset=0.10,
):
    """
    Geometry- and uncertainty-aware pseudo-label selection for aerial imagery.

    Small objects receive a lower confidence floor, but still have to pass the
    multi-view consistency floor and the joint quality criterion.
    """
    current_preds = _load_prediction_tensor(current_txt)
    if current_preds is None:
        return [], {"total": 0, "kept": 0, "small_total": 0, "small_kept": 0}
    if not calibration_detections:
        raise ValueError(
            "Drone-aware pseudo labeling requires calibration metadata. "
            "Enable --use-test-time-calibration."
        )

    previous_preds = _load_prediction_tensor(previous_txt)
    temporal_scores = _same_class_temporal_scores(
        current_preds, previous_preds, temporal_iou_threshold
    )
    filtered = []
    stats = {"total": len(current_preds), "kept": 0, "small_total": 0, "small_kept": 0}

    for idx, pred in enumerate(current_preds):
        if idx >= len(calibration_detections):
            continue
        x1, y1, x2, y2, raw_conf, cls = pred.tolist()
        area = max(0.0, x2 - x1) * max(0.0, y2 - y1)
        is_small = area <= small_area_threshold
        stats["small_total"] += int(is_small)

        det = calibration_detections[idx]
        consistency = max(0.0, min(1.0, float(det.get("consistency", 0.0))))
        calibrated_conf = max(
            0.0, min(1.0, float(det.get("calibrated_conf", raw_conf)))
        )
        temporal = float(temporal_scores[idx].item())
        confidence_floor = max(
            0.0,
            base_conf_threshold - (small_conf_offset if is_small else 0.0),
        )
        quality = (
            calibrated_conf ** confidence_power
            * consistency ** consistency_power
            * temporal ** temporal_power
        )
        keep = (
            calibrated_conf >= confidence_floor
            and consistency >= min_consistency
            and quality >= quality_threshold
        )
        if not keep:
            continue

        xc = (x1 + x2) / 2.0
        yc = (y1 + y2) / 2.0
        w = x2 - x1
        h = y2 - y1
        filtered.append(f"{int(cls)} {xc:.6f} {yc:.6f} {w:.6f} {h:.6f}")
        stats["kept"] += 1
        stats["small_kept"] += int(is_small)

    return filtered, stats


def _filter_with_temporal_consistency(current_txt, previous_txt, conf_threshold, consistency_threshold,
                                      calibration_detections=None,
                                      calibration_min_consistency=0.0,
                                      calibration_max_risk=1.0,
                                      use_adaptive_pseudo_threshold=False,
                                      adaptive_target_consistency=0.5,
                                      adaptive_threshold_scale=0.3):
    current_preds = _load_prediction_tensor(current_txt)
    if current_preds is None:
        return [], 0, 0

    conf_mask = current_preds[:, 4] >= conf_threshold
    calibration_mask = _build_calibration_mask(
        len(current_preds),
        calibration_detections,
        calibration_min_consistency,
        calibration_max_risk,
    )
    quality_scores = torch.ones(len(current_preds), dtype=torch.float32)

    prev_preds = _load_prediction_tensor(previous_txt)
    if prev_preds is not None:
        quality_scores = compute_pseudo_label_quality(
            current_preds,
            prev_preds,
            iou_threshold=consistency_threshold,
        ).cpu()

    reliable_mask = conf_mask & calibration_mask & (quality_scores >= 0.5)
    filtered_lines = []
    kept = 0
    for idx, keep in enumerate(reliable_mask.tolist()):
        if keep and use_adaptive_pseudo_threshold and calibration_detections and idx < len(calibration_detections):
            effective_threshold = _adaptive_conf_threshold(
                conf_threshold,
                calibration_detections[idx],
                adaptive_target_consistency,
                adaptive_threshold_scale,
            )
            keep = bool(current_preds[idx, 4].item() >= effective_threshold)
        if not keep:
            continue
        x1, y1, x2, y2, _conf, cls = current_preds[idx].tolist()
        xc = (x1 + x2) / 2.0
        yc = (y1 + y2) / 2.0
        w = x2 - x1
        h = y2 - y1
        filtered_lines.append(f"{int(cls)} {xc:.6f} {yc:.6f} {w:.6f} {h:.6f}")
        kept += 1

    consistent = int((quality_scores >= 0.5).sum().item())
    return filtered_lines, kept, consistent


def generate_pseudo_labels(labels_dir, output_dir, conf_threshold,
                           previous_labels_dir=None,
                           use_progressive_refinement=False,
                           consistency_threshold=0.5,
                           calibration_metadata_dir=None,
                           use_calibration_guided_pseudo=False,
                           calibration_min_consistency=0.0,
                           calibration_max_risk=1.0,
                           use_adaptive_pseudo_threshold=False,
                           adaptive_target_consistency=0.5,
                           adaptive_threshold_scale=0.3,
                           use_drone_aware_pseudo=False,
                           drone_quality_threshold=0.35,
                           drone_confidence_power=1.0,
                           drone_consistency_power=1.0,
                           drone_temporal_power=0.5,
                           drone_min_consistency=0.35,
                           drone_small_area_threshold=0.0025,
                           drone_small_conf_offset=0.10):
    """
    Filter detection predictions by confidence threshold to generate pseudo labels.

    Args:
        labels_dir: Directory containing YOLO-format prediction .txt files
        output_dir: Directory to save filtered pseudo labels
        conf_threshold: Minimum confidence to keep a detection
        previous_labels_dir: Previous iteration prediction directory for temporal consistency
        use_progressive_refinement: Whether to apply consistency filtering
        consistency_threshold: IoU threshold for cross-iteration consistency

    Returns:
        num_labels: Number of pseudo labels generated
    """
    mode_bits = []
    if use_progressive_refinement:
        mode_bits.append("temporal consistency")
    if use_calibration_guided_pseudo:
        mode_bits.append(
            f"calibration gate(cons>={calibration_min_consistency}, risk<={calibration_max_risk})"
        )
    if use_adaptive_pseudo_threshold:
        mode_bits.append(
            f"adaptive threshold(target_cons={adaptive_target_consistency}, scale={adaptive_threshold_scale})"
        )
    if use_drone_aware_pseudo:
        mode_bits.append(
            f"drone-aware quality(q>={drone_quality_threshold}, "
            f"small_area<={drone_small_area_threshold})"
        )
    mode_msg = f" + {' + '.join(mode_bits)}" if mode_bits else ""
    print(f"\n---> Step 2: Generating pseudo labels (conf >= {conf_threshold}{mode_msg})...")
    os.makedirs(output_dir, exist_ok=True)

    num_labels = 0
    num_consistent = 0
    drone_stats = {"total": 0, "kept": 0, "small_total": 0, "small_kept": 0}
    if not os.path.exists(labels_dir):
        print(f"    Warning: labels dir not found: {labels_dir}")
        return 0

    for fname in os.listdir(labels_dir):
        if not fname.endswith(".txt"):
            continue
        src = os.path.join(labels_dir, fname)
        dst = os.path.join(output_dir, fname)
        calibration_detections = _load_calibration_detections(calibration_metadata_dir, fname)

        if use_drone_aware_pseudo:
            prev_src = os.path.join(previous_labels_dir, fname) if previous_labels_dir else None
            filtered, image_stats = _filter_drone_aware_pseudo(
                src,
                prev_src,
                calibration_detections,
                conf_threshold,
                quality_threshold=drone_quality_threshold,
                confidence_power=drone_confidence_power,
                consistency_power=drone_consistency_power,
                temporal_power=drone_temporal_power,
                temporal_iou_threshold=consistency_threshold,
                min_consistency=drone_min_consistency,
                small_area_threshold=drone_small_area_threshold,
                small_conf_offset=drone_small_conf_offset,
            )
            kept = image_stats["kept"]
            for key in drone_stats:
                drone_stats[key] += image_stats[key]
        elif use_progressive_refinement:
            prev_src = os.path.join(previous_labels_dir, fname) if previous_labels_dir else None
            filtered, kept, consistent = _filter_with_temporal_consistency(
                src,
                prev_src,
                conf_threshold,
                consistency_threshold,
                calibration_detections=calibration_detections if use_calibration_guided_pseudo else None,
                calibration_min_consistency=calibration_min_consistency,
                calibration_max_risk=calibration_max_risk,
                use_adaptive_pseudo_threshold=use_adaptive_pseudo_threshold,
                adaptive_target_consistency=adaptive_target_consistency,
                adaptive_threshold_scale=adaptive_threshold_scale,
            )
            num_consistent += consistent
        else:
            filtered = []
            kept = 0
            with open(src, "r", encoding="utf-8") as f:
                for idx, line in enumerate(f):
                    parts = line.strip().split()
                    if len(parts) >= 6:
                        conf = float(parts[5])
                        effective_threshold = conf_threshold
                        if use_adaptive_pseudo_threshold and calibration_detections and idx < len(calibration_detections):
                            effective_threshold = _adaptive_conf_threshold(
                                conf_threshold,
                                calibration_detections[idx],
                                adaptive_target_consistency,
                                adaptive_threshold_scale,
                            )
                        keep = conf >= effective_threshold
                        if keep and use_calibration_guided_pseudo and calibration_detections and idx < len(calibration_detections):
                            det = calibration_detections[idx]
                            consistency = float(det.get("consistency", 1.0))
                            risk_score = float(det.get("risk_score", 0.0))
                            keep = consistency >= calibration_min_consistency and risk_score <= calibration_max_risk
                        if keep:
                            filtered.append(" ".join(parts[:5]))
                            kept += 1
                    elif len(parts) == 5:
                        keep = True
                        if use_calibration_guided_pseudo and calibration_detections and idx < len(calibration_detections):
                            det = calibration_detections[idx]
                            consistency = float(det.get("consistency", 1.0))
                            risk_score = float(det.get("risk_score", 0.0))
                            keep = consistency >= calibration_min_consistency and risk_score <= calibration_max_risk
                        if keep:
                            filtered.append(line.strip())
                            kept += 1

        if filtered:
            with open(dst, "w", encoding="utf-8") as f:
                f.write("\n".join(filtered) + "\n")
            num_labels += kept

    if use_progressive_refinement and previous_labels_dir:
        print(f"    Consistent predictions kept: {num_consistent}")
    if use_drone_aware_pseudo:
        print(
            "    Drone-aware selection: "
            f"{drone_stats['kept']}/{drone_stats['total']} boxes kept; "
            f"small objects {drone_stats['small_kept']}/{drone_stats['small_total']} kept."
        )
        with open(os.path.join(output_dir, "selection_summary.json"), "w", encoding="utf-8") as f:
            json.dump(drone_stats, f, indent=2)
    print(f"    Generated {num_labels} pseudo labels from {len(os.listdir(labels_dir)) if os.path.exists(labels_dir) else 0} files.")
    return num_labels


def _link_or_copy(src, dst):
    """Create a same-volume hard link when possible, otherwise copy the file."""
    if os.path.exists(dst):
        return
    try:
        os.link(src, dst)
    except OSError:
        shutil.copy2(src, dst)


def prepare_mixed_dataset(source_path, target_images_path, pseudo_labels_path, output_path):
    """
    Combine source domain (labeled) and pseudo-labeled target domain into a mixed dataset.

    Args:
        source_path: Path to source dataset directory (has images/ and labels/)
        target_images_path: Path to target domain images
        pseudo_labels_path: Path to pseudo label .txt files
        output_path: Path to create mixed dataset

    Returns:
        data_yaml_path: Path to the mixed dataset YAML config
    """
    print(f"\n---> Step 3: Preparing mixed dataset...")
    os.makedirs(output_path, exist_ok=True)

    # Create directories
    for subdir in ["images/train", "labels/train", "images/val", "labels/val"]:
        os.makedirs(os.path.join(output_path, subdir), exist_ok=True)

    # Copy source domain data
    src_img_train = os.path.join(source_path, "images", "train")
    src_lbl_train = os.path.join(source_path, "labels", "train")
    dst_img_train = os.path.join(output_path, "images", "train")
    dst_lbl_train = os.path.join(output_path, "labels", "train")

    if os.path.exists(src_img_train):
        for f in os.listdir(src_img_train):
            _link_or_copy(os.path.join(src_img_train, f), os.path.join(dst_img_train, f))
    if os.path.exists(src_lbl_train):
        for f in os.listdir(src_lbl_train):
            _link_or_copy(os.path.join(src_lbl_train, f), os.path.join(dst_lbl_train, f))

    # Copy source val data
    src_img_val = os.path.join(source_path, "images", "val")
    src_lbl_val = os.path.join(source_path, "labels", "val")
    dst_img_val = os.path.join(output_path, "images", "val")
    dst_lbl_val = os.path.join(output_path, "labels", "val")

    if os.path.exists(src_img_val):
        for f in os.listdir(src_img_val):
            _link_or_copy(os.path.join(src_img_val, f), os.path.join(dst_img_val, f))
    if os.path.exists(src_lbl_val):
        for f in os.listdir(src_lbl_val):
            _link_or_copy(os.path.join(src_lbl_val, f), os.path.join(dst_lbl_val, f))

    # Copy target domain images and pseudo labels
    target_imgs = [f for f in os.listdir(target_images_path)
                   if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff'))]
    for fname in target_imgs:
        _link_or_copy(os.path.join(target_images_path, fname), os.path.join(dst_img_train, fname))
        # Copy corresponding pseudo label
        label_name = os.path.splitext(fname)[0] + ".txt"
        pseudo_src = os.path.join(pseudo_labels_path, label_name)
        if os.path.exists(pseudo_src):
            _link_or_copy(pseudo_src, os.path.join(dst_lbl_train, label_name))

    # Create data YAML
    # Try to read nc from source dataset yaml
    source_yaml = None
    for yaml_name in ["source_data.yaml", "data.yaml"]:
        p = os.path.join(source_path, yaml_name)
        if os.path.exists(p):
            source_yaml = p
            break

    nc = 80  # default
    names = []
    if source_yaml:
        with open(source_yaml) as f:
            src_cfg = yaml.safe_load(f)
        nc = src_cfg.get("nc", 80)
        names = src_cfg.get("names", [])

    data_yaml = {
        "train": os.path.abspath(os.path.join(output_path, "images", "train")),
        "val": os.path.abspath(os.path.join(output_path, "images", "val")),
        "nc": nc,
        "names": names if names else [f"class{i}" for i in range(nc)],
    }
    data_yaml_path = os.path.join(output_path, "mixed_data.yaml")
    with open(data_yaml_path, "w") as f:
        yaml.dump(data_yaml, f, default_flow_style=False)

    print(f"    Mixed dataset: {len(target_imgs)} target images + source images")
    print(f"    Config saved to: {data_yaml_path}")
    return data_yaml_path


def make_absolute_data_yaml(data_yaml_path, output_dir="runs/prepared_data"):
    """
    Normalize a dataset YAML so Ultralytics sees absolute train/val/test paths.
    """
    os.makedirs(output_dir, exist_ok=True)
    with open(data_yaml_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    yaml_dir = os.path.dirname(os.path.abspath(data_yaml_path))
    path_root = cfg.get("path")
    if path_root:
        path_root = path_root if os.path.isabs(path_root) else os.path.abspath(os.path.join(yaml_dir, path_root))
    else:
        path_root = yaml_dir

    def resolve_one(value):
        if os.path.isabs(value):
            return value

        candidates = [
            os.path.abspath(os.path.join(path_root, value)),
            os.path.abspath(os.path.join(yaml_dir, value)),
            os.path.abspath(value),
        ]
        for candidate in candidates:
            if os.path.exists(candidate):
                return candidate
        return candidates[0]

    def resolve_split(key):
        value = cfg.get(key)
        if not value:
            return value
        if isinstance(value, list):
            return [resolve_one(v) for v in value]
        return resolve_one(value)

    normalized = dict(cfg)
    normalized["train"] = resolve_split("train")
    normalized["val"] = resolve_split("val")
    if "test" in normalized:
        normalized["test"] = resolve_split("test")
    normalized["path"] = path_root

    out_path = os.path.join(output_dir, Path(data_yaml_path).stem + "_abs.yaml")
    with open(out_path, "w", encoding="utf-8") as f:
        yaml.safe_dump(normalized, f, sort_keys=False, allow_unicode=True)
    return out_path


def run_retraining(weights_path, data_yaml_path, epochs=5, batch_size=16,
                   img_size=640, device="0", project="runs/train", name="self_train",
                   workers=0, seed=0):
    """
    Fine-tune YOLO11 on the mixed dataset using Ultralytics API.

    Args:
        weights_path: Initial weights
        data_yaml_path: Path to mixed dataset YAML
        epochs: Training epochs
        batch_size: Batch size
        img_size: Image size
        device: Device
        project: Project directory
        name: Run name

    Returns:
        best_weights: Path to best weights after training
    """
    from ultralytics import YOLO
    from modules.ultralytics_seed import patch_ultralytics_dataloader_seed
    patch_ultralytics_threadpool()
    patch_ultralytics_dataloader_seed(seed)

    print(f"\n---> Step 4: Fine-tuning on mixed dataset...")
    print(f"    weights={weights_path}, data={data_yaml_path}, epochs={epochs}")

    run_name = f"{name}_{int(time.time())}"
    project_dir = os.path.abspath(project)
    os.makedirs(project_dir, exist_ok=True)
    model = YOLO(weights_path)
    results = model.train(
        data=data_yaml_path,
        epochs=epochs,
        batch=batch_size,
        imgsz=img_size,
        device=device,
        workers=workers,
        seed=seed,
        project=project_dir,
        name=run_name,
        exist_ok=False,
        verbose=True,
    )

    save_dir = Path(getattr(model.trainer, "save_dir", Path(project_dir) / run_name))
    best_weights = str(save_dir / "weights" / "best.pt")
    if not os.path.exists(best_weights):
        best_weights = str(save_dir / "weights" / "last.pt")

    print(f"    Training complete. Best weights: {best_weights}")
    return best_weights


def update_ema_teacher(teacher_weights, student_weights, decay=0.999, output_dir="runs/ema_teacher"):
    """
    Update teacher weights with an iteration-level exponential moving average.
    """
    from ultralytics import YOLO

    os.makedirs(output_dir, exist_ok=True)
    teacher_model = YOLO(teacher_weights)
    student_model = YOLO(student_weights)

    teacher_state = teacher_model.model.state_dict()
    student_state = student_model.model.state_dict()
    blended_state = {}

    for key, teacher_tensor in teacher_state.items():
        student_tensor = student_state.get(key)
        if student_tensor is None or teacher_tensor.shape != student_tensor.shape:
            blended_state[key] = teacher_tensor
            continue

        if torch.is_floating_point(teacher_tensor) and torch.is_floating_point(student_tensor):
            blended_state[key] = teacher_tensor * decay + student_tensor * (1.0 - decay)
        else:
            blended_state[key] = student_tensor

    teacher_model.model.load_state_dict(blended_state, strict=False)
    ema_path = os.path.join(output_dir, f"ema_teacher_{int(time.time())}.pt")
    teacher_model.save(ema_path)
    print(f"    EMA teacher updated: {ema_path}")
    return ema_path


def evaluate_model(weights_path, data_yaml_path, img_size=640, device="0", workers=0):
    """Evaluate model on validation set and return mAP metrics."""
    from ultralytics import YOLO
    patch_ultralytics_threadpool()

    model = YOLO(weights_path)
    results = model.val(
        data=data_yaml_path,
        imgsz=img_size,
        device=device,
        workers=workers,
        verbose=False,
    )

    map50 = results.box.map50
    map50_95 = results.box.map
    print(f"    mAP@0.5={map50:.4f}, mAP@0.5:0.95={map50_95:.4f}")
    return map50, map50_95


def plot_training_results(iterations, map50_list, map50_95_list, output_path="self_training_results.png"):
    """Plot mAP curves over self-training iterations."""
    if not HAS_MATPLOTLIB:
        print("[Plot] matplotlib not available, skipping plot.")
        return

    fig, ax = plt.subplots(1, 1, figsize=(8, 5))
    ax.plot(iterations, map50_list, marker="o", label="mAP@0.5")
    ax.plot(iterations, map50_95_list, marker="s", label="mAP@0.5:0.95")
    ax.set_xlabel("Iteration")
    ax.set_ylabel("mAP")
    ax.set_title("Self-Training Progress")
    ax.legend()
    ax.grid(True, linestyle="--", alpha=0.4)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    print(f"[Plot] Saved to {output_path}")


# --- Main ---

def main(opt):
    """Main self-training loop."""
    project_root = Path(".")
    set_reproducible_seed(opt.seed)

    # Validate paths
    if not os.path.exists(opt.source_dataset):
        raise FileNotFoundError(f"Source dataset not found: {opt.source_dataset}")
    if not os.path.exists(opt.target_dataset):
        raise FileNotFoundError(f"Target dataset not found: {opt.target_dataset}")
    if not os.path.exists(opt.initial_weights):
        raise FileNotFoundError(f"Initial weights not found: {opt.initial_weights}")
    if opt.use_drone_aware_pseudo and not opt.use_test_time_calibration:
        raise ValueError(
            "--use-drone-aware-pseudo requires --use-test-time-calibration "
            "because geometry consistency comes from multi-view inference."
        )

    # Find target images directory
    target_img_dir = os.path.join(opt.target_dataset, "unlabels_img")
    if not os.path.exists(target_img_dir):
        # Try alternative names
        for alt in ["images", "unlabeled", "unlabel"]:
            alt_path = os.path.join(opt.target_dataset, alt)
            if os.path.exists(alt_path):
                target_img_dir = alt_path
                break

    print("=" * 60)
    print("YOLO11-UDA Self-Training")
    print("=" * 60)
    print(f"  Source dataset: {opt.source_dataset}")
    print(f"  Target dataset: {opt.target_dataset}")
    print(f"  Target images:  {target_img_dir}")
    print(f"  Initial weights: {opt.initial_weights}")
    print(f"  Iterations: {opt.iterations}")
    print(f"  Seed: {opt.seed}")
    print(f"  Dynamic threshold: {opt.dynamic_threshold}")
    print("=" * 60)

    current_weights = resolve_yolo_weights(opt.initial_weights, opt.yolo_base_weights)
    teacher_weights = current_weights
    all_map50 = []
    all_map50_95 = []
    conf_threshold = opt.initial_conf_threshold
    session_stamp = int(time.time())
    previous_labels_dir = None

    # Find validation data for evaluation
    val_yaml = None
    for yaml_name in ["source_data.yaml", "data.yaml"]:
        p = os.path.join(opt.source_dataset, yaml_name)
        if os.path.exists(p):
            val_yaml = p
            break

    # Initial evaluation
    if val_yaml:
        print("\n--- Initial evaluation ---")
        val_yaml = make_absolute_data_yaml(val_yaml)
        m50, m50_95 = evaluate_model(current_weights, val_yaml, opt.img_size, opt.device, opt.workers)
        all_map50.append(m50)
        all_map50_95.append(m50_95)

    # Self-training iterations
    for iteration in range(1, opt.iterations + 1):
        print(f"\n{'='*60}")
        print(f"  ITERATION {iteration}/{opt.iterations}")
        print(f"  Confidence threshold: {conf_threshold:.2f}")
        print(f"{'='*60}")

        # Step 1: Inference on target domain
        run_name = f"self_train_iter_{iteration}_{session_stamp}"
        _, inference_dir = run_inference(
            teacher_weights if opt.use_ema_teacher else current_weights,
            target_img_dir,
            conf_thres=0.01,
            img_size=opt.img_size,
            device=opt.device,
            project="runs/predict",
            name=run_name,
            use_test_time_calibration=opt.use_test_time_calibration,
            calibration_views=opt.calibration_views,
            calibration_alpha=opt.calibration_alpha,
            calibration_beta=opt.calibration_beta,
            calibration_iou_threshold=opt.calibration_iou_threshold,
            calibration_mode=opt.calibration_mode,
            calibration_consistency_threshold=opt.calibration_consistency_threshold,
            calibration_rank_top_fraction=opt.calibration_rank_top_fraction,
            calibration_rank_min_confidence=opt.calibration_rank_min_confidence,
            calibration_rank_class_aware=opt.calibration_rank_class_aware,
            inference_batch_size=opt.inference_batch_size,
        )

        # Step 2: Generate pseudo labels
        labels_dir = os.path.join(inference_dir, "labels") if inference_dir else None
        calibration_metadata_dir = (
            os.path.join(inference_dir, "metadata")
            if inference_dir and opt.use_test_time_calibration
            else None
        )
        pseudo_dir = os.path.join("runs", "pseudo_labels", f"iter_{iteration}_{session_stamp}")
        num_labels = generate_pseudo_labels(
            labels_dir,
            pseudo_dir,
            conf_threshold,
            previous_labels_dir=previous_labels_dir,
            use_progressive_refinement=opt.use_progressive_pseudo,
            consistency_threshold=opt.consistency_threshold,
            calibration_metadata_dir=calibration_metadata_dir,
            use_calibration_guided_pseudo=opt.use_calibration_guided_pseudo,
            calibration_min_consistency=opt.pseudo_calibration_min_consistency,
            calibration_max_risk=opt.pseudo_calibration_max_risk,
            use_adaptive_pseudo_threshold=opt.use_adaptive_pseudo_threshold,
            adaptive_target_consistency=opt.pseudo_adaptive_target_consistency,
            adaptive_threshold_scale=opt.pseudo_adaptive_threshold_scale,
            use_drone_aware_pseudo=opt.use_drone_aware_pseudo,
            drone_quality_threshold=opt.drone_quality_threshold,
            drone_confidence_power=opt.drone_confidence_power,
            drone_consistency_power=opt.drone_consistency_power,
            drone_temporal_power=opt.drone_temporal_power,
            drone_min_consistency=opt.drone_min_consistency,
            drone_small_area_threshold=opt.drone_small_area_threshold,
            drone_small_conf_offset=opt.drone_small_conf_offset,
        )

        if num_labels == 0:
            print(f"    No pseudo labels generated at threshold {conf_threshold:.2f}, stopping.")
            break

        # Step 3: Prepare mixed dataset
        mixed_dir = os.path.join("runs", "mixed_dataset", f"iter_{iteration}_{session_stamp}")
        data_yaml_path = prepare_mixed_dataset(
            opt.source_dataset, target_img_dir, pseudo_dir, mixed_dir
        )

        # Step 4: Retrain
        student_weights = run_retraining(
            current_weights, data_yaml_path,
            epochs=opt.epochs_per_iter,
            batch_size=opt.batch_size,
            img_size=opt.img_size,
            device=opt.device,
            project="runs/train",
            name=run_name,
            workers=opt.workers,
            seed=opt.seed,
        )
        if opt.use_ema_teacher:
            teacher_weights = update_ema_teacher(
                teacher_weights,
                student_weights,
                decay=opt.ema_decay,
            )
            # Keep optimizing and returning the student. The EMA teacher is only
            # used to generate pseudo labels at the start of the next iteration.
            current_weights = student_weights
        else:
            current_weights = student_weights

        # Step 5: Evaluate
        if val_yaml:
            m50, m50_95 = evaluate_model(current_weights, val_yaml, opt.img_size, opt.device, opt.workers)
            all_map50.append(m50)
            all_map50_95.append(m50_95)

        # Update threshold
        if opt.dynamic_threshold:
            conf_threshold = min(conf_threshold + opt.conf_increment, 0.95)
            print(f"    Threshold updated to: {conf_threshold:.2f}")

        previous_labels_dir = labels_dir

    # Final results
    print(f"\n{'='*60}")
    print("  SELF-TRAINING COMPLETE")
    print(f"{'='*60}")
    print(f"  Final weights: {current_weights}")
    if all_map50:
        print(f"  mAP@0.5: {all_map50[-1]:.4f}")
        print(f"  mAP@0.5:0.95: {all_map50_95[-1]:.4f}")

    # Plot results
    if opt.plot_results and all_map50:
        iterations = list(range(0, len(all_map50)))
        plot_training_results(iterations, all_map50, all_map50_95)

    return current_weights


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Self-Training Loop for YOLO11-UDA")

    # Paths
    parser.add_argument("--source-dataset", type=str, default="source_dataset",
                        help="Source dataset directory")
    parser.add_argument("--target-dataset", type=str, default="target_dataset",
                        help="Target dataset directory")
    parser.add_argument("--initial-weights", type=str, default="weights/yolo11n.pt",
                        help="Initial weights (YOLO11 pretrained or contrastive pretrained)")
    parser.add_argument("--yolo-base-weights", type=str, default="weights/yolo11n.pt",
                        help="Base YOLO11 checkpoint used when merging contrastive backbone weights")

    # Training
    parser.add_argument("--iterations", type=int, default=3,
                        help="Number of self-training iterations")
    parser.add_argument("--epochs-per-iter", type=int, default=5,
                        help="Training epochs per iteration")
    parser.add_argument("--batch-size", type=int, default=16,
                        help="Batch size for training")
    parser.add_argument("--inference-batch-size", type=int, default=16,
                        help="Batch size for pseudo-label inference and calibrated views")
    parser.add_argument("--workers", type=int, default=0,
                        help="Ultralytics dataloader workers. Use 0 on restricted Windows setups.")
    parser.add_argument("--img-size", type=int, default=640,
                        help="Image size")
    parser.add_argument("--device", type=str, default="0",
                        help="Device (cuda or cpu)")
    parser.add_argument("--seed", type=int, default=0,
                        help="Random seed for reproducible self-training")

    # Pseudo label threshold
    parser.add_argument("--initial-conf-threshold", type=float, default=0.7,
                        help="Initial confidence threshold for pseudo labels")
    parser.add_argument("--dynamic-threshold", action="store_true", default=False,
                        help="Use dynamic threshold (increasing each iteration)")
    parser.add_argument("--conf-increment", type=float, default=0.1,
                        help="Threshold increment per iteration (for dynamic)")
    parser.add_argument("--use-progressive-pseudo", action="store_true", default=False,
                        help="Filter pseudo labels using cross-iteration temporal consistency")
    parser.add_argument("--consistency-threshold", type=float, default=0.5,
                        help="IoU threshold used for progressive pseudo-label consistency")
    parser.add_argument("--use-test-time-calibration", action="store_true", default=False,
                        help="Run consistency-based confidence calibration during target-domain inference")
    parser.add_argument("--calibration-views", nargs="+",
                        default=["original", "hflip", "bright"],
                        help="Ordered list of views for test-time calibration. Must start with 'original'.")
    parser.add_argument("--calibration-alpha", type=float, default=0.5,
                        help="Base multiplier term for calibrated confidence")
    parser.add_argument("--calibration-beta", type=float, default=0.5,
                        help="Consistency multiplier term for calibrated confidence")
    parser.add_argument("--calibration-iou-threshold", type=float, default=0.5,
                        help="IoU threshold used when matching boxes across augmented views")
    parser.add_argument("--calibration-mode", type=str, default="global",
                        choices=["global", "selective", "rank"],
                        help="Calibration rule: global rescales every box, selective only downweights uncertain boxes, rank targets risky high-confidence boxes")
    parser.add_argument("--calibration-consistency-threshold", type=float, default=0.5,
                        help="Consistency threshold used by selective calibration mode")
    parser.add_argument("--calibration-rank-top-fraction", type=float, default=0.25,
                        help="Fraction of detections considered high-risk in rank mode")
    parser.add_argument("--calibration-rank-min-confidence", type=float, default=0.1,
                        help="Minimum raw confidence for a detection to be eligible in rank mode")
    parser.add_argument("--calibration-rank-class-aware", action="store_true", default=False,
                        help="In rank mode, select risky boxes separately inside each predicted class")
    parser.add_argument("--use-calibration-guided-pseudo", action="store_true", default=False,
                        help="Use test-time calibration consistency/risk to filter pseudo labels")
    parser.add_argument("--pseudo-calibration-min-consistency", type=float, default=0.5,
                        help="Minimum calibration consistency required for a pseudo label when calibration-guided filtering is enabled")
    parser.add_argument("--pseudo-calibration-max-risk", type=float, default=0.25,
                        help="Maximum calibration risk score allowed for a pseudo label when calibration-guided filtering is enabled")
    parser.add_argument("--use-adaptive-pseudo-threshold", action="store_true", default=False,
                        help="Adapt pseudo-label confidence thresholds using calibration consistency")
    parser.add_argument("--pseudo-adaptive-target-consistency", type=float, default=0.5,
                        help="Target consistency level used when computing adaptive pseudo thresholds")
    parser.add_argument("--pseudo-adaptive-threshold-scale", type=float, default=0.3,
                        help="How strongly low-consistency detections increase the pseudo-label confidence threshold")
    parser.add_argument("--use-ema-teacher", action="store_true", default=False,
                        help="Use an iteration-level EMA teacher for pseudo-label generation and final checkpoint tracking")
    parser.add_argument("--ema-decay", type=float, default=0.90,
                        help="Iteration-level EMA decay used to update teacher weights")
    parser.add_argument("--use-drone-aware-pseudo", action="store_true", default=False,
                        help="Use aerial small-object and geometry-consistency aware pseudo-label selection")
    parser.add_argument("--drone-quality-threshold", type=float, default=0.35,
                        help="Minimum joint confidence/consistency/temporal quality score")
    parser.add_argument("--drone-confidence-power", type=float, default=1.0,
                        help="Confidence exponent in the drone-aware quality score")
    parser.add_argument("--drone-consistency-power", type=float, default=1.0,
                        help="Multi-view consistency exponent in the drone-aware quality score")
    parser.add_argument("--drone-temporal-power", type=float, default=0.5,
                        help="Cross-iteration stability exponent in the drone-aware quality score")
    parser.add_argument("--drone-min-consistency", type=float, default=0.35,
                        help="Hard multi-view consistency floor for every pseudo label")
    parser.add_argument("--drone-small-area-threshold", type=float, default=0.0025,
                        help="Normalized box area below which a detection is treated as a small aerial object")
    parser.add_argument("--drone-small-conf-offset", type=float, default=0.10,
                        help="Amount subtracted from the confidence floor for consistent small objects")

    # Plotting
    parser.add_argument("--plot-results", action="store_true", default=False,
                        help="Plot mAP curves after training")

    opt = parser.parse_args()
    main(opt)

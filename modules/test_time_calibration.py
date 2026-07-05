import json
import os
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
import torch


IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


def list_image_files(image_folder):
    return sorted(
        str(path)
        for path in Path(image_folder).iterdir()
        if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
    )


def load_image(image_path):
    image = cv2.imread(image_path)
    if image is None:
        raise FileNotFoundError(f"Failed to read image: {image_path}")
    return image


def apply_view(image, view_name):
    if view_name == "original":
        return image
    if view_name == "hflip":
        return cv2.flip(image, 1)
    if view_name == "bright":
        return cv2.convertScaleAbs(image, alpha=1.05, beta=12)
    if view_name == "dark":
        return cv2.convertScaleAbs(image, alpha=0.90, beta=-10)
    if view_name == "contrast":
        return cv2.convertScaleAbs(image, alpha=1.15, beta=0)
    if view_name == "blur":
        return cv2.GaussianBlur(image, (5, 5), sigmaX=0.8)
    raise ValueError(f"Unsupported calibration view: {view_name}")


def remap_boxes_to_original(boxes_xyxy, view_name, width, height):
    remapped = boxes_xyxy.clone()
    if view_name == "hflip" and len(remapped):
        x1 = remapped[:, 0].clone()
        x2 = remapped[:, 2].clone()
        remapped[:, 0] = width - x2
        remapped[:, 2] = width - x1
    remapped[:, 0::2] = remapped[:, 0::2].clamp_(0, width)
    remapped[:, 1::2] = remapped[:, 1::2].clamp_(0, height)
    return remapped


def result_to_tensor(result, view_name, width, height):
    boxes = result.boxes
    if boxes is None or boxes.xyxy is None or len(boxes) == 0:
        return torch.empty((0, 6), dtype=torch.float32)

    xyxy = boxes.xyxy.detach().cpu().to(torch.float32)
    xyxy = remap_boxes_to_original(xyxy, view_name, width, height)
    conf = boxes.conf.detach().cpu().to(torch.float32).unsqueeze(1)
    cls = boxes.cls.detach().cpu().to(torch.float32).unsqueeze(1)
    return torch.cat([xyxy, conf, cls], dim=1)


def box_iou(boxes1, boxes2):
    if boxes1.numel() == 0 or boxes2.numel() == 0:
        return torch.zeros((boxes1.shape[0], boxes2.shape[0]), dtype=torch.float32)

    x11, y11, x12, y12 = boxes1[:, 0:1], boxes1[:, 1:2], boxes1[:, 2:3], boxes1[:, 3:4]
    x21, y21, x22, y22 = boxes2[:, 0], boxes2[:, 1], boxes2[:, 2], boxes2[:, 3]

    inter_x1 = torch.maximum(x11, x21)
    inter_y1 = torch.maximum(y11, y21)
    inter_x2 = torch.minimum(x12, x22)
    inter_y2 = torch.minimum(y12, y22)

    inter_w = (inter_x2 - inter_x1).clamp(min=0)
    inter_h = (inter_y2 - inter_y1).clamp(min=0)
    inter = inter_w * inter_h

    area1 = ((x12 - x11).clamp(min=0) * (y12 - y11).clamp(min=0))
    area2 = ((x22 - x21).clamp(min=0) * (y22 - y21).clamp(min=0))
    union = area1 + area2 - inter
    return inter / union.clamp(min=1e-6)


def _match_view_score(base_det, aug_tensor, match_iou_threshold):
    if aug_tensor.numel() == 0:
        return 0.0

    same_class_mask = aug_tensor[:, 5] == base_det[5]
    candidates = aug_tensor[same_class_mask]
    if candidates.numel() == 0:
        return 0.0

    ious = box_iou(base_det[:4].unsqueeze(0), candidates[:, :4]).squeeze(0)
    best_iou, best_idx = torch.max(ious, dim=0)
    best_iou_value = float(best_iou.item())
    if best_iou_value < match_iou_threshold:
        return 0.0

    matched_conf = float(candidates[best_idx, 4].item())
    base_conf = float(base_det[4].item())
    conf_agreement = max(0.0, 1.0 - abs(base_conf - matched_conf))
    return 0.5 * best_iou_value + 0.5 * conf_agreement


def _match_view_confidences(base_det, aug_tensors, match_iou_threshold):
    """
    Return the matched raw confidence from each augmented view.

    If no same-class box with IoU >= match_iou_threshold is found, the view
    contributes a confidence of 0.0, which inflates the apparent disagreement
    and lets downstream filtering treat the detection as uncertain.
    """
    confidences = []
    for aug_tensor in aug_tensors:
        if aug_tensor.numel() == 0:
            confidences.append(0.0)
            continue

        same_class_mask = aug_tensor[:, 5] == base_det[5]
        candidates = aug_tensor[same_class_mask]
        if candidates.numel() == 0:
            confidences.append(0.0)
            continue

        ious = box_iou(base_det[:4].unsqueeze(0), candidates[:, :4]).squeeze(0)
        best_iou, best_idx = torch.max(ious, dim=0)
        if float(best_iou.item()) < match_iou_threshold:
            confidences.append(0.0)
            continue

        confidences.append(float(candidates[best_idx, 4].item()))
    return confidences


def calibrate_predictions(
    base_tensor,
    aug_tensors,
    alpha=0.50,
    beta=0.50,
    match_iou_threshold=0.5,
    mode="global",
    consistency_threshold=0.5,
    rank_top_fraction=0.25,
    rank_min_confidence=0.1,
    rank_class_aware=False,
):
    calibrated_rows = []
    image_scores = []
    risk_scores = []
    consistency_scores = []
    raw_confidences = []
    conf_stds = []

    for base_det in base_tensor:
        per_view_scores = [
            _match_view_score(base_det, aug_tensor, match_iou_threshold)
            for aug_tensor in aug_tensors
        ]
        per_view_confs = _match_view_confidences(
            base_det, aug_tensors, match_iou_threshold
        )
        consistency_score = float(sum(per_view_scores) / max(len(per_view_scores), 1))
        conf_std = float(np.std(per_view_confs)) if per_view_confs else 0.0
        raw_conf = float(base_det[4].item())
        risk_score = raw_conf * (1.0 - consistency_score)
        consistency_scores.append(consistency_score)
        raw_confidences.append(raw_conf)
        risk_scores.append(risk_score)
        conf_stds.append(conf_std)

    selected_rank_indices = set()
    if mode == "rank":
        if rank_class_aware:
            class_to_indices = {}
            for idx, (base_det, raw_conf) in enumerate(zip(base_tensor, raw_confidences)):
                if raw_conf < rank_min_confidence:
                    continue
                cls = int(base_det[5].item())
                class_to_indices.setdefault(cls, []).append(idx)

            for class_indices in class_to_indices.values():
                class_indices = sorted(
                    class_indices,
                    key=lambda idx: risk_scores[idx],
                    reverse=True,
                )
                top_k = max(1, int(len(class_indices) * rank_top_fraction)) if class_indices else 0
                selected_rank_indices.update(class_indices[:top_k])
        else:
            candidate_indices = [
                idx for idx, raw_conf in enumerate(raw_confidences)
                if raw_conf >= rank_min_confidence
            ]
            candidate_indices = sorted(
                candidate_indices,
                key=lambda idx: risk_scores[idx],
                reverse=True,
            )
            top_k = max(1, int(len(base_tensor) * rank_top_fraction)) if len(base_tensor) else 0
            selected_rank_indices = set(candidate_indices[:top_k])

    for idx, base_det in enumerate(base_tensor):
        consistency_score = consistency_scores[idx]
        raw_conf = float(base_det[4].item())
        if mode == "selective":
            if consistency_score < consistency_threshold:
                multiplier = alpha + beta * consistency_score
                calibrated_conf = max(0.0, min(1.0, raw_conf * multiplier))
            else:
                multiplier = 1.0
                calibrated_conf = raw_conf
        elif mode == "rank":
            if idx in selected_rank_indices and consistency_score < consistency_threshold:
                multiplier = alpha + beta * consistency_score
                calibrated_conf = max(0.0, min(1.0, raw_conf * multiplier))
            else:
                multiplier = 1.0
                calibrated_conf = raw_conf
        else:
            multiplier = alpha + beta * consistency_score
            calibrated_conf = max(0.0, min(1.0, raw_conf * multiplier))
        calibrated_rows.append(
            {
                "cls": int(base_det[5].item()),
                "xyxy": [float(v) for v in base_det[:4].tolist()],
                "raw_conf": raw_conf,
                "consistency": consistency_score,
                "risk_score": risk_scores[idx],
                "conf_std": conf_stds[idx],
                "per_view_confs": per_view_confs,
                "multiplier": multiplier,
                "calibrated_conf": calibrated_conf,
            }
        )
        image_scores.append(consistency_score)

    mean_consistency = float(sum(image_scores) / max(len(image_scores), 1)) if image_scores else 0.0
    return calibrated_rows, mean_consistency


def compute_class_uncertainty_stats(
    model,
    image_folder,
    img_size=640,
    device="0",
    views=None,
    conf_thres=0.01,
    alpha=0.9,
    beta=0.1,
    consistency_threshold=0.35,
    calibration_mode="selective",
    match_iou_threshold=0.5,
    inference_batch_size=16,
    small_area_threshold=0.0025,
):
    """
    Run calibrated inference on a folder of images and return per-class
    uncertainty statistics.

    Uncertainty for each detection is defined as max(conf_std, 1-consistency)
    so that it remains meaningful even with only two TTA views.
    """
    views = views or ["original", "hflip"]
    if views[0] != "original":
        raise ValueError("views must start with 'original'")

    image_paths = list_image_files(image_folder)
    class_uncertainties = defaultdict(list)
    class_small_uncertainties = defaultdict(list)

    inference_batch_size = max(1, int(inference_batch_size))
    for batch_start in range(0, len(image_paths), inference_batch_size):
        batch_paths = image_paths[batch_start : batch_start + inference_batch_size]
        batch_images = [load_image(path) for path in batch_paths]
        batch_shapes = [image.shape[:2] for image in batch_images]
        batch_view_tensors = [{} for _ in batch_images]

        for view_name in views:
            augmented_batch = [apply_view(image, view_name) for image in batch_images]
            results = model.predict(
                source=augmented_batch,
                conf=conf_thres,
                imgsz=img_size,
                device=device,
                batch=len(augmented_batch),
                save=False,
                save_txt=False,
                save_conf=False,
                verbose=False,
            )
            for idx, result in enumerate(results):
                height, width = batch_shapes[idx]
                batch_view_tensors[idx][view_name] = result_to_tensor(
                    result, view_name, width, height
                )

        for image_path, (height, width), view_tensors in zip(
            batch_paths, batch_shapes, batch_view_tensors
        ):
            base_tensor = view_tensors["original"]
            aug_tensors = [view_tensors[name] for name in views[1:]]
            calibrated_rows, _ = calibrate_predictions(
                base_tensor,
                aug_tensors,
                alpha=alpha,
                beta=beta,
                match_iou_threshold=match_iou_threshold,
                mode=calibration_mode,
                consistency_threshold=consistency_threshold,
            )
            for det in calibrated_rows:
                x1, y1, x2, y2 = det["xyxy"]
                area_norm = max(0.0, x2 - x1) * max(0.0, y2 - y1) / (width * height)
                conf_std = det.get("conf_std", 0.0)
                consistency = det.get("consistency", 1.0)
                uncertainty = max(conf_std, 1.0 - consistency)
                cls = int(det["cls"])
                class_uncertainties[cls].append(uncertainty)
                if area_norm <= small_area_threshold:
                    class_small_uncertainties[cls].append(uncertainty)

    stats = {}
    all_uncertainties = []
    for cls, vals in class_uncertainties.items():
        all_uncertainties.extend(vals)
        arr = np.array(vals)
        small_vals = class_small_uncertainties.get(cls, [])
        small_arr = np.array(small_vals) if small_vals else arr
        stats[cls] = {
            "mean_uncertainty": float(np.mean(arr)),
            "std_uncertainty": float(np.std(arr)),
            "mean_small_uncertainty": float(np.mean(small_arr)),
            "count": int(len(arr)),
        }

    global_mean = float(np.mean(all_uncertainties)) if all_uncertainties else 0.0
    stats["__global__"] = {"mean_uncertainty": global_mean}
    return stats


def xyxy_to_yolo_line(row, width, height, include_conf=True):
    x1, y1, x2, y2 = row["xyxy"]
    xc = ((x1 + x2) / 2.0) / width
    yc = ((y1 + y2) / 2.0) / height
    w = (x2 - x1) / width
    h = (y2 - y1) / height
    xc = min(max(xc, 0.0), 1.0)
    yc = min(max(yc, 0.0), 1.0)
    w = min(max(w, 0.0), 1.0)
    h = min(max(h, 0.0), 1.0)

    fields = [
        str(int(row["cls"])),
        f"{xc:.6f}",
        f"{yc:.6f}",
        f"{w:.6f}",
        f"{h:.6f}",
    ]
    if include_conf:
        fields.append(f"{row['calibrated_conf']:.6f}")
    return " ".join(fields)


def run_calibrated_inference(
    model,
    image_folder,
    save_dir,
    conf_thres=0.01,
    img_size=640,
    device="0",
    alpha=0.50,
    beta=0.50,
    match_iou_threshold=0.5,
    views=None,
    calibration_mode="global",
    consistency_threshold=0.5,
    rank_top_fraction=0.25,
    rank_min_confidence=0.1,
    rank_class_aware=False,
    inference_batch_size=16,
):
    views = views or ["original", "hflip", "bright"]
    if not views or views[0] != "original":
        raise ValueError("views must start with 'original'")

    save_dir = os.path.abspath(save_dir)
    labels_dir = os.path.join(save_dir, "labels")
    metadata_dir = os.path.join(save_dir, "metadata")
    os.makedirs(labels_dir, exist_ok=True)
    os.makedirs(metadata_dir, exist_ok=True)

    image_paths = list_image_files(image_folder)
    summary = {
        "save_dir": save_dir,
        "views": views,
        "alpha": alpha,
        "beta": beta,
        "calibration_mode": calibration_mode,
        "consistency_threshold": consistency_threshold,
        "rank_top_fraction": rank_top_fraction,
        "rank_min_confidence": rank_min_confidence,
        "rank_class_aware": rank_class_aware,
        "match_iou_threshold": match_iou_threshold,
        "images": [],
    }
    total_raw = 0
    total_kept = 0
    consistency_scores = []

    inference_batch_size = max(1, int(inference_batch_size))
    for batch_start in range(0, len(image_paths), inference_batch_size):
        batch_paths = image_paths[batch_start:batch_start + inference_batch_size]
        batch_images = [load_image(path) for path in batch_paths]
        batch_shapes = [image.shape[:2] for image in batch_images]
        batch_view_tensors = [{} for _ in batch_images]

        for view_name in views:
            augmented_batch = [apply_view(image, view_name) for image in batch_images]
            results = model.predict(
                source=augmented_batch,
                conf=conf_thres,
                imgsz=img_size,
                device=device,
                batch=len(augmented_batch),
                save=False,
                save_txt=False,
                save_conf=False,
                verbose=False,
            )
            for idx, result in enumerate(results):
                height, width = batch_shapes[idx]
                batch_view_tensors[idx][view_name] = result_to_tensor(
                    result, view_name, width, height
                )

        for image_path, (height, width), view_tensors in zip(
            batch_paths, batch_shapes, batch_view_tensors
        ):
            base_tensor = view_tensors["original"]
            aug_tensors = [view_tensors[name] for name in views[1:]]
            calibrated_rows, mean_consistency = calibrate_predictions(
                base_tensor,
                aug_tensors,
                alpha=alpha,
                beta=beta,
                match_iou_threshold=match_iou_threshold,
                mode=calibration_mode,
                consistency_threshold=consistency_threshold,
                rank_top_fraction=rank_top_fraction,
                rank_min_confidence=rank_min_confidence,
                rank_class_aware=rank_class_aware,
            )

            stem = Path(image_path).stem
            label_path = os.path.join(labels_dir, f"{stem}.txt")
            metadata_path = os.path.join(metadata_dir, f"{stem}.json")
            with open(label_path, "w", encoding="utf-8") as f:
                for row in calibrated_rows:
                    f.write(xyxy_to_yolo_line(row, width, height, include_conf=True) + "\n")
            with open(metadata_path, "w", encoding="utf-8") as f:
                json.dump(
                    {
                        "image": image_path,
                        "label_path": label_path,
                        "width": width,
                        "height": height,
                        "views": views,
                        "detections": calibrated_rows,
                    },
                    f,
                    indent=2,
                )

            total_raw += int(base_tensor.shape[0])
            total_kept += len(calibrated_rows)
            consistency_scores.append(mean_consistency)
            summary["images"].append(
                {
                    "image": image_path,
                    "raw_predictions": int(base_tensor.shape[0]),
                    "calibrated_predictions": len(calibrated_rows),
                    "mean_consistency": mean_consistency,
                    "label_path": label_path,
                    "metadata_path": metadata_path,
                }
            )

    summary["num_images"] = len(image_paths)
    summary["raw_predictions"] = total_raw
    summary["calibrated_predictions"] = total_kept
    summary["mean_consistency"] = (
        float(sum(consistency_scores) / max(len(consistency_scores), 1))
        if consistency_scores
        else 0.0
    )

    summary_path = os.path.join(save_dir, "calibration_summary.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    return summary

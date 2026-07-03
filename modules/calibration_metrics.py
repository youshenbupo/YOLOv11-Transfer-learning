import json
import math
import os
from pathlib import Path

import torch


def _box_iou(boxes1, boxes2):
    if boxes1.numel() == 0 or boxes2.numel() == 0:
        return torch.zeros((boxes1.shape[0], boxes2.shape[0]), dtype=torch.float32)

    lt = torch.max(boxes1[:, None, :2], boxes2[None, :, :2])
    rb = torch.min(boxes1[:, None, 2:], boxes2[None, :, 2:])
    wh = (rb - lt).clamp(min=0)
    inter = wh[:, :, 0] * wh[:, :, 1]

    area1 = (boxes1[:, 2] - boxes1[:, 0]).clamp(min=0) * (boxes1[:, 3] - boxes1[:, 1]).clamp(min=0)
    area2 = (boxes2[:, 2] - boxes2[:, 0]).clamp(min=0) * (boxes2[:, 3] - boxes2[:, 1]).clamp(min=0)
    return inter / (area1[:, None] + area2[None, :] - inter + 1e-6)


def list_image_stems(image_dir):
    stems = []
    for path in sorted(Path(image_dir).iterdir()):
        if path.is_file():
            stems.append(path.stem)
    return stems


def load_yolo_boxes(txt_path, width, height, with_conf=False):
    rows = []
    if not txt_path or not os.path.exists(txt_path):
        return torch.empty((0, 6 if with_conf else 5), dtype=torch.float32)

    with open(txt_path, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split()
            min_fields = 6 if with_conf else 5
            if len(parts) < min_fields:
                continue
            cls = float(parts[0])
            xc, yc, w, h = map(float, parts[1:5])
            x1 = (xc - w / 2.0) * width
            y1 = (yc - h / 2.0) * height
            x2 = (xc + w / 2.0) * width
            y2 = (yc + h / 2.0) * height
            if with_conf:
                conf = float(parts[5])
                rows.append([x1, y1, x2, y2, conf, cls])
            else:
                rows.append([x1, y1, x2, y2, cls])

    if not rows:
        return torch.empty((0, 6 if with_conf else 5), dtype=torch.float32)
    return torch.tensor(rows, dtype=torch.float32)


def build_detection_events(preds, gts, iou_threshold=0.5):
    events = []
    if preds.numel() == 0:
        return events

    preds = preds[torch.argsort(preds[:, 4], descending=True)]
    gt_used = torch.zeros((gts.shape[0],), dtype=torch.bool) if gts.numel() else torch.zeros((0,), dtype=torch.bool)
    pred_boxes = preds[:, :4]
    gt_boxes = gts[:, :4] if gts.numel() else torch.empty((0, 4), dtype=torch.float32)
    ious = _box_iou(pred_boxes, gt_boxes) if gts.numel() else torch.zeros((preds.shape[0], 0), dtype=torch.float32)

    for pred_idx in range(preds.shape[0]):
        conf = float(preds[pred_idx, 4].item())
        cls = int(preds[pred_idx, 5].item())
        correct = 0.0

        if gts.numel():
            same_class = (gts[:, 4].to(torch.int64) == cls) & (~gt_used)
            if same_class.any():
                cls_ious = ious[pred_idx].clone()
                cls_ious[~same_class] = -1.0
                best_iou, best_idx = torch.max(cls_ious, dim=0)
                if float(best_iou.item()) >= iou_threshold:
                    correct = 1.0
                    gt_used[best_idx] = True

        events.append({"confidence": conf, "correct": correct, "class": cls})
    return events


def compute_ece(events, num_bins=10):
    if not events:
        return 0.0, []

    bins = []
    ece = 0.0
    total = len(events)
    for bin_idx in range(num_bins):
        left = bin_idx / num_bins
        right = (bin_idx + 1) / num_bins
        if bin_idx == num_bins - 1:
            bucket = [e for e in events if left <= e["confidence"] <= right]
        else:
            bucket = [e for e in events if left <= e["confidence"] < right]
        if not bucket:
            bins.append({"bin": bin_idx, "count": 0, "accuracy": 0.0, "confidence": 0.0, "gap": 0.0})
            continue

        acc = sum(e["correct"] for e in bucket) / len(bucket)
        conf = sum(e["confidence"] for e in bucket) / len(bucket)
        gap = abs(acc - conf)
        ece += (len(bucket) / total) * gap
        bins.append({"bin": bin_idx, "count": len(bucket), "accuracy": acc, "confidence": conf, "gap": gap})
    return ece, bins


def compute_nll(events, eps=1e-6):
    if not events:
        return 0.0
    losses = []
    for event in events:
        p = min(max(event["confidence"], eps), 1.0 - eps)
        y = event["correct"]
        losses.append(-(y * math.log(p) + (1.0 - y) * math.log(1.0 - p)))
    return sum(losses) / len(losses)


def compute_brier(events):
    if not events:
        return 0.0
    return sum((event["confidence"] - event["correct"]) ** 2 for event in events) / len(events)


def summarize_detection_calibration(pred_dir, image_dir, label_dir, iou_threshold=0.5, num_bins=10):
    import cv2

    stems = list_image_stems(image_dir)
    all_events = []
    image_summaries = []

    for stem in stems:
        image_path = None
        for suffix in (".jpg", ".jpeg", ".png", ".bmp", ".webp"):
            candidate = os.path.join(image_dir, stem + suffix)
            if os.path.exists(candidate):
                image_path = candidate
                break
        if image_path is None:
            continue

        image = cv2.imread(image_path)
        if image is None:
            continue
        height, width = image.shape[:2]

        pred_path = os.path.join(pred_dir, f"{stem}.txt")
        gt_path = os.path.join(label_dir, f"{stem}.txt")
        preds = load_yolo_boxes(pred_path, width, height, with_conf=True)
        gts = load_yolo_boxes(gt_path, width, height, with_conf=False)
        events = build_detection_events(preds, gts, iou_threshold=iou_threshold)
        all_events.extend(events)
        image_summaries.append(
            {
                "image": image_path,
                "num_predictions": int(preds.shape[0]),
                "num_ground_truth": int(gts.shape[0]),
                "num_correct": int(sum(event["correct"] for event in events)),
            }
        )

    ece, bins = compute_ece(all_events, num_bins=num_bins)
    summary = {
        "prediction_dir": os.path.abspath(pred_dir),
        "image_dir": os.path.abspath(image_dir),
        "label_dir": os.path.abspath(label_dir),
        "num_images": len(image_summaries),
        "num_events": len(all_events),
        "ece": ece,
        "nll": compute_nll(all_events),
        "brier": compute_brier(all_events),
        "iou_threshold": iou_threshold,
        "num_bins": num_bins,
        "bins": bins,
        "images": image_summaries,
    }
    return summary


def save_summary(summary, output_path):
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

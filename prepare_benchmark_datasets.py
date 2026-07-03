import argparse
import os
import shutil
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
import yaml
from PIL import Image


CITYSCAPES_DET_CLASSES = [
    (24, "person"),
    (25, "rider"),
    (26, "car"),
    (27, "truck"),
    (28, "bus"),
    (31, "train"),
    (32, "motorcycle"),
    (33, "bicycle"),
]
CITYSCAPES_CLASS_ID_TO_INDEX = {class_id: idx for idx, (class_id, _) in enumerate(CITYSCAPES_DET_CLASSES)}
CITYSCAPES_NAMES = [name for _, name in CITYSCAPES_DET_CLASSES]
CITYSCAPES_CAR_INDEX = CITYSCAPES_NAMES.index("car")


def ensure_dir(path: Path):
    path.mkdir(parents=True, exist_ok=True)


def reset_dir(path: Path):
    if path.exists():
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)


def link_or_copy(src: Path, dst: Path):
    ensure_dir(dst.parent)
    if dst.exists():
        return
    try:
        os.link(src, dst)
    except OSError:
        shutil.copy2(src, dst)


def write_yaml(path: Path, payload: dict):
    ensure_dir(path.parent)
    with path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(payload, f, sort_keys=False, allow_unicode=True)


def normalize_box(x1, y1, x2, y2, width, height):
    bw = (x2 - x1 + 1) / width
    bh = (y2 - y1 + 1) / height
    cx = (x1 + x2 + 1) / 2 / width
    cy = (y1 + y2 + 1) / 2 / height
    return cx, cy, bw, bh


def instance_png_to_yolo(instance_png: Path, out_txt: Path, class_filter=None, remap=None):
    arr = np.array(Image.open(instance_png))
    height, width = arr.shape[:2]
    lines = []
    for instance_id in np.unique(arr):
        instance_id = int(instance_id)
        if instance_id < 1000:
            continue
        class_id = instance_id // 1000
        if class_id not in CITYSCAPES_CLASS_ID_TO_INDEX:
            continue
        cls = CITYSCAPES_CLASS_ID_TO_INDEX[class_id]
        if class_filter is not None and cls not in class_filter:
            continue
        if remap is not None:
            cls = remap[cls]
        ys, xs = np.where(arr == instance_id)
        if xs.size == 0 or ys.size == 0:
            continue
        x1, x2 = int(xs.min()), int(xs.max())
        y1, y2 = int(ys.min()), int(ys.max())
        cx, cy, bw, bh = normalize_box(x1, y1, x2, y2, width, height)
        lines.append(f"{cls} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}")

    ensure_dir(out_txt.parent)
    out_txt.write_text("\n".join(lines), encoding="utf-8")


def prepare_cityscapes_detection(clear_root: Path, gt_root: Path, out_root: Path):
    reset_dir(out_root)
    image_root = clear_root / "leftImg8bit"
    label_root = gt_root / "gtFine"
    for split in ["train", "val"]:
        for city_dir in sorted((image_root / split).iterdir()):
            if not city_dir.is_dir():
                continue
            for image_path in sorted(city_dir.glob("*_leftImg8bit.png")):
                stem = image_path.name.replace("_leftImg8bit.png", "")
                dst_img = out_root / "images" / split / image_path.name
                dst_lbl = out_root / "labels" / split / f"{image_path.stem}.txt"
                link_or_copy(image_path, dst_img)
                instance_png = label_root / split / city_dir.name / f"{stem}_gtFine_instanceIds.png"
                instance_png_to_yolo(instance_png, dst_lbl)

    write_yaml(
        out_root / "source_data.yaml",
        {
            "train": "./images/train",
            "val": "./images/val",
            "test": "./images/val",
            "nc": len(CITYSCAPES_NAMES),
            "names": CITYSCAPES_NAMES,
        },
    )


def prepare_foggy_target(foggy_root: Path, city_labels_root: Path, out_root: Path, beta: str):
    reset_dir(out_root)
    foggy_image_root = foggy_root / "leftImg8bit_foggyDBF"
    suffix = f"_leftImg8bit_foggy_beta_{beta}.png"
    for split in ["train", "val"]:
        src_split = foggy_image_root / split
        if split == "train":
            dst_img_root = out_root / "unlabels_img"
            dst_lbl_root = None
        else:
            dst_img_root = out_root / "val_img" / "images"
            dst_lbl_root = out_root / "val_img" / "labels"
        for city_dir in sorted(src_split.iterdir()):
            if not city_dir.is_dir():
                continue
            for image_path in sorted(city_dir.glob(f"*{suffix}")):
                base_stem = image_path.name.replace(suffix, "")
                dst_img = dst_img_root / image_path.name
                link_or_copy(image_path, dst_img)
                if dst_lbl_root is not None:
                    src_lbl = city_labels_root / "labels" / split / f"{base_stem}.txt"
                    dst_lbl = dst_lbl_root / f"{image_path.stem}.txt"
                    ensure_dir(dst_lbl.parent)
                    shutil.copy2(src_lbl, dst_lbl)


def city_label_to_car_only(src_txt: Path, dst_txt: Path):
    lines = []
    if src_txt.exists():
        for raw in src_txt.read_text(encoding="utf-8").splitlines():
            parts = raw.strip().split()
            if len(parts) != 5:
                continue
            cls = int(parts[0])
            if cls != CITYSCAPES_CAR_INDEX:
                continue
            lines.append("0 " + " ".join(parts[1:]))
    ensure_dir(dst_txt.parent)
    dst_txt.write_text("\n".join(lines), encoding="utf-8")


def prepare_cityscapes_car_target(clear_root: Path, city_labels_root: Path, out_root: Path):
    reset_dir(out_root)
    image_root = clear_root / "leftImg8bit"
    for split in ["train", "val"]:
        for city_dir in sorted((image_root / split).iterdir()):
            if not city_dir.is_dir():
                continue
            for image_path in sorted(city_dir.glob("*_leftImg8bit.png")):
                base_stem = image_path.name.replace("_leftImg8bit.png", "")
                if split == "train":
                    dst_img = out_root / "unlabels_img" / image_path.name
                    link_or_copy(image_path, dst_img)
                else:
                    dst_img = out_root / "val_img" / "images" / image_path.name
                    dst_lbl = out_root / "val_img" / "labels" / f"{image_path.stem}.txt"
                    link_or_copy(image_path, dst_img)
                    src_lbl = city_labels_root / "labels" / split / f"{image_path.stem}.txt"
                    city_label_to_car_only(src_lbl, dst_lbl)


def parse_voc_box(obj, width, height):
    bnd = obj.find("bndbox")
    x1 = float(bnd.findtext("xmin"))
    y1 = float(bnd.findtext("ymin"))
    x2 = float(bnd.findtext("xmax"))
    y2 = float(bnd.findtext("ymax"))
    cx, cy, bw, bh = normalize_box(x1, y1, x2, y2, width, height)
    return cx, cy, bw, bh


def prepare_sim10k_source(sim_root: Path, out_root: Path, val_count: int):
    reset_dir(out_root)
    ann_root = sim_root / "Annotations"
    img_root = sim_root / "JPEGImages"
    stems = sorted(p.stem for p in ann_root.glob("*.xml"))
    if val_count <= 0 or val_count >= len(stems):
        train_stems = stems
        val_stems = stems[-min(1000, len(stems)) :]
    else:
        train_stems = stems[:-val_count]
        val_stems = stems[-val_count:]

    def export_split(split_name, split_stems):
        for stem in split_stems:
            src_img = img_root / f"{stem}.jpg"
            src_xml = ann_root / f"{stem}.xml"
            dst_img = out_root / "images" / split_name / src_img.name
            dst_lbl = out_root / "labels" / split_name / f"{stem}.txt"
            link_or_copy(src_img, dst_img)
            root = ET.parse(src_xml).getroot()
            size = root.find("size")
            width = float(size.findtext("width"))
            height = float(size.findtext("height"))
            lines = []
            for obj in root.findall("object"):
                if obj.findtext("name") != "car":
                    continue
                cx, cy, bw, bh = parse_voc_box(obj, width, height)
                lines.append(f"0 {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}")
            ensure_dir(dst_lbl.parent)
            dst_lbl.write_text("\n".join(lines), encoding="utf-8")

    export_split("train", train_stems)
    export_split("val", val_stems)
    write_yaml(
        out_root / "source_data.yaml",
        {
            "train": "./images/train",
            "val": "./images/val",
            "test": "./images/val",
            "nc": 1,
            "names": ["car"],
        },
    )


def write_target_val_yaml(target_root: Path, names: list[str], path: Path):
    write_yaml(
        path,
        {
            "train": str((target_root / "val_img" / "images").resolve()),
            "val": str((target_root / "val_img" / "images").resolve()),
            "test": str((target_root / "val_img" / "images").resolve()),
            "nc": len(names),
            "names": names,
        },
    )


def main():
    parser = argparse.ArgumentParser(description="Prepare formal UDA detection benchmarks for this repo.")
    parser.add_argument("--data-root", type=Path, default=Path("data"))
    parser.add_argument("--output-root", type=Path, default=Path("benchmarks"))
    parser.add_argument("--foggy-beta", type=str, default="0.02")
    parser.add_argument("--sim10k-val-count", type=int, default=1000)
    args = parser.parse_args()

    clear_root = args.data_root / "leftImg8bit_trainvaltest"
    gt_root = args.data_root / "gtFine_trainvaltest"
    foggy_root = args.data_root / "leftImg8bit_trainval_foggyDBF"
    sim_root = args.data_root / "sim10k"

    for path in [clear_root, gt_root, foggy_root, sim_root]:
        if not path.exists():
            raise FileNotFoundError(f"Required dataset path not found: {path}")

    city_source = args.output_root / "cityscapes_to_foggy" / "source_dataset"
    foggy_target = args.output_root / "cityscapes_to_foggy" / "target_dataset"
    sim_source = args.output_root / "sim10k_to_cityscapes" / "source_dataset"
    city_car_target = args.output_root / "sim10k_to_cityscapes" / "target_dataset"

    print("[1/4] Preparing Cityscapes 8-class source dataset...")
    prepare_cityscapes_detection(clear_root, gt_root, city_source)

    print("[2/4] Preparing Foggy Cityscapes target dataset...")
    prepare_foggy_target(foggy_root, city_source, foggy_target, args.foggy_beta)
    write_target_val_yaml(foggy_target, CITYSCAPES_NAMES, foggy_target / "target_val.yaml")

    print("[3/4] Preparing SIM10K source dataset...")
    prepare_sim10k_source(sim_root, sim_source, args.sim10k_val_count)

    print("[4/4] Preparing Cityscapes car-only target dataset...")
    prepare_cityscapes_car_target(clear_root, city_source, city_car_target)
    write_target_val_yaml(city_car_target, ["car"], city_car_target / "target_val.yaml")

    print("\nDone.")
    print(f"Cityscapes -> Foggy source: {city_source.resolve()}")
    print(f"Cityscapes -> Foggy target: {foggy_target.resolve()}")
    print(f"SIM10K -> Cityscapes source: {sim_source.resolve()}")
    print(f"SIM10K -> Cityscapes target: {city_car_target.resolve()}")


if __name__ == "__main__":
    main()

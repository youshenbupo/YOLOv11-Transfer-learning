# Dataset Setup

## 1. Already downloaded

For Cityscapes, the correct first two files are:

- `gtFine_trainvaltest.zip`
- `leftImg8bit_trainvaltest.zip`

These are the required base files for most detection and domain adaptation experiments.

## 2. What each file is for

- `leftImg8bit_trainvaltest.zip`
  - RGB images
  - needed for training and evaluation

- `gtFine_trainvaltest.zip`
  - fine annotations
  - needed to build labels for train/val

## 3. What you still need

For the paper plan in this repo, the next datasets are:

1. `Foggy Cityscapes`
2. `SIM10K`

Optional later:

3. `KITTI`

## 4. Current local layout

You have already extracted the official downloads under:

```text
D:\yolo\yolov7-main\data\
  gtFine_trainvaltest\
  leftImg8bit_trainvaltest\
  leftImg8bit_trainval_foggyDBF\
  sim10k\
```

This is a valid raw-data layout for the preparation script in this repo.

## 5. Recommended benchmark-ready layout

```text
benchmarks/
  cityscapes_to_foggy/
    source_dataset/
      images/
      labels/
      source_data.yaml
    target_dataset/
      unlabels_img/
      val_img/
      target_val.yaml
  sim10k_to_cityscapes/
    source_dataset/
      images/
      labels/
      source_data.yaml
    target_dataset/
      unlabels_img/
      val_img/
      target_val.yaml
```

## 6. Preparation command

Run:

```bash
python prepare_benchmark_datasets.py --data-root data --output-root benchmarks --foggy-beta 0.02 --sim10k-val-count 1000
```

This converts the raw downloads into datasets that can be used directly by
`self_training.py` and the calibration/evaluation scripts.

## 7. Current prepared benchmark counts

After running the script on this machine, the prepared datasets contain:

- `Cityscapes -> Foggy`
  - source train: `2975`
  - source val: `500`
  - target unlabeled train: `2975`
  - target val: `500`

- `SIM10K -> Cityscapes`
  - source train: `9000`
  - source val: `1000`
  - target unlabeled train: `2975`
  - target val: `500`

## 8. Notes for this repo

The old `source_dataset/` and `target_dataset/` folders are still useful for local smoke tests.
For paper-style experiments, use the prepared datasets under `benchmarks/`.

## 9. Formal experiment commands

Cityscapes -> Foggy Cityscapes:

```bash
python run_paper_experiments.py --benchmark cityscapes_to_foggy --mode default
```

SIM10K -> Cityscapes:

```bash
python run_paper_experiments.py --benchmark sim10k_to_cityscapes --mode default
```

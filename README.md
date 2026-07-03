# YOLO11-UDA

Unsupervised domain adaptation for object detection on top of Ultralytics YOLO11, using:

- contrastive pretraining on source/target images
- iterative pseudo-label self-training on the target domain

This repository is no longer a plain YOLOv7 project even though the folder name may still say `yolov7-main`. The active training code is built around YOLO11 and the Ultralytics Python API.

## What This Project Does

The workflow has two stages:

1. `train_contrastive.py`
   - loads YOLO11 backbone + neck
   - removes the detect head for representation learning
   - trains with `SimCLR` or `MoCo`
   - supports freezing and staged unfreezing

2. `self_training.py`
   - runs inference on unlabeled target images
   - filters predictions into pseudo labels
   - mixes source labels and target pseudo labels
   - fine-tunes a YOLO11 detector

## Repository Layout

```text
train_contrastive.py        Contrastive pretraining entrypoint
self_training.py            Self-training / pseudo-labeling entrypoint
TECHNICAL_DOC.md            Method notes and paper-oriented design doc

contrastive/
  model.py                  ContrastiveYOLO and projection head
  dataset.py                CombinedDataset and ratio sampler
  freeze.py                 Freeze policy and layer-wise LR
  loss.py                   InfoNCE and MoCo losses
  ddp_utils.py              DDP helpers
  utils.py                  Setup and plotting helpers

models/
  yolo11_backbone.py        YOLO11 backbone + neck adapter

modules/
  attention.py              CBAM / SE attention modules
  domain.py                 Domain adversarial modules
  losses.py                 Experimental domain-aware losses

source_dataset/             Source domain dataset with labels
target_dataset/             Target domain images and optional validation split
contra_img/                 Positive / negative images for contrastive pretraining
weights/                    YOLO checkpoints
test_time_calibration.py    Standalone test-time consistency calibration entrypoint
```

## Environment

Verified in the local conda environment:

```bash
conda activate yolo
```

Typical dependencies:

```bash
pip install -r requirements.txt
```

## Windows Notes

This repository is currently tuned for a Windows environment with restricted multiprocessing behavior.

- Use `--num-workers 0` for `train_contrastive.py`
- Use `--workers 0` for `self_training.py`
- `self_training.py` includes a Windows-safe Ultralytics cache patch
- dataset YAML paths are normalized automatically before validation/training

## Contrastive Pretraining

Minimal smoke test:

```bash
python train_contrastive.py \
    --method simclr \
    --device 0 \
    --epochs 1 \
    --batch-size 2 \
    --num-positives 1 \
    --num-workers 0 \
    --img-size 64 \
    --freeze-target backbone \
    --stage1-epochs 1 \
    --save-path smoke_contrastive.pt \
    --save-full-pt
```

Recommended first real run:

```bash
python train_contrastive.py \
    --method simclr \
    --freeze-target backbone \
    --staged-unfreeze \
    --epochs 10 \
    --batch-size 8 \
    --num-workers 0 \
    --yolo-weights-path weights/yolo11n.pt \
    --save-path contrastive_pretrained.pt \
    --save-full-pt
```

Optional CBAM on the P5 feature before the projection head:

```bash
python train_contrastive.py \
    --method simclr \
    --use-cbam \
    --cbam-reduction 16
```

Optional domain-aware contrastive loss on the SimCLR path:

```bash
python train_contrastive.py \
    --method simclr \
    --use-domain-loss \
    --lambda-domain 0.5
```

Optional domain adversarial training with GRL and a domain discriminator:

```bash
python train_contrastive.py \
    --method simclr \
    --use-domain-adversarial \
    --lambda-adv 0.5 \
    --domain-hidden-dim 256 \
    --grl-alpha 1.0
```

Optional LayerLock-inspired progressive freezing:

```bash
python train_contrastive.py \
    --method simclr \
    --epochs 8 \
    --batch-size 8 \
    --num-workers 0 \
    --use-layerlock \
    --layerlock-freeze-steps=-1,2,6,10 \
    --layerlock-feature-levels p3,p4,p5,p5
```

This mode does not reproduce the original latent-prediction LayerLock architecture. Instead, it adapts the same training idea to YOLO11 contrastive pretraining:

- early phase: learn on lower-level features with no shallow freezing
- middle phase: progressively lock shallow backbone blocks
- late phase: shift the contrastive target toward higher-level semantic features

LayerLock phases can also scale domain-aware and adversarial loss weights gradually, which makes it easier to study whether stabilization and domain alignment help each other.

## Self-Training

First self-training round from plain YOLO11 weights:

```bash
python self_training.py \
    --initial-weights weights/yolo11n.pt \
    --iterations 1 \
    --epochs-per-iter 1 \
    --batch-size 2 \
    --img-size 64 \
    --device 0 \
    --workers 0 \
    --initial-conf-threshold 0.1
```

Run self-training from a contrastive checkpoint:

```bash
python self_training.py \
    --initial-weights contrastive_pretrained.pt \
    --yolo-base-weights weights/yolo11n.pt \
    --iterations 3 \
    --epochs-per-iter 5 \
    --workers 0 \
    --dynamic-threshold \
    --plot-results
```

If `--initial-weights` points to a backbone-only contrastive checkpoint, `self_training.py` will automatically merge it into a YOLO11 checkpoint using `--yolo-base-weights`.

Optional progressive pseudo-label refinement:

```bash
python self_training.py \
    --initial-weights weights/yolo11n.pt \
    --iterations 3 \
    --workers 0 \
    --use-progressive-pseudo \
    --consistency-threshold 0.5
```

Optional test-time consistency calibration during target-domain inference:

```bash
python self_training.py \
    --initial-weights weights/yolo11n.pt \
    --iterations 1 \
    --workers 0 \
    --use-test-time-calibration \
    --calibration-views original hflip bright \
    --calibration-alpha 0.5 \
    --calibration-beta 0.5
```

### Drone Scene Adaptation

For aerial scene-1 to scene-2 adaptation, the recommended first experiment uses
multi-view geometry consistency, small-object-aware confidence floors, temporal
stability across self-training rounds, and an EMA teacher:

```bash
python self_training.py \
    --source-dataset path/to/scene1 \
    --target-dataset path/to/scene2 \
    --initial-weights weights/yolo11n.pt \
    --iterations 3 \
    --epochs-per-iter 5 \
    --batch-size 4 \
    --workers 0 \
    --device 0 \
    --initial-conf-threshold 0.50 \
    --use-test-time-calibration \
    --calibration-mode selective \
    --calibration-views original hflip bright contrast \
    --calibration-iou-threshold 0.30 \
    --use-drone-aware-pseudo \
    --drone-quality-threshold 0.35 \
    --drone-min-consistency 0.35 \
    --drone-small-area-threshold 0.0025 \
    --drone-small-conf-offset 0.10 \
    --use-ema-teacher \
    --ema-decay 0.90
```

The drone-aware quality score is
`confidence^a * multi_view_consistency^b * temporal_stability^c`. Small boxes
may use a slightly lower confidence floor, but they must still pass the geometry
consistency floor and the joint quality threshold. Each pseudo-label directory
contains `selection_summary.json` with overall and small-object retention counts.

Standalone calibration run:

```bash
python test_time_calibration.py \
    --weights weights/yolo11n.pt \
    --source target_dataset/unlabels_img \
    --project runs/calibration \
    --name ttc_smoke
```

Selective calibration variant:

```bash
python test_time_calibration.py \
    --weights weights/yolo11n.pt \
    --source target_dataset/unlabels_img \
    --project runs/calibration \
    --name ttc_selective \
    --calibration-mode selective \
    --consistency-threshold 0.5 \
    --calibration-alpha 0.9 \
    --calibration-beta 0.1 \
    --calibration-views original hflip
```

Rank-based selective calibration variant:

```bash
python test_time_calibration.py \
    --weights weights/yolo11n.pt \
    --source target_dataset/unlabels_img \
    --project runs/calibration \
    --name ttc_rank \
    --calibration-mode rank \
    --consistency-threshold 0.5 \
    --rank-top-fraction 0.25 \
    --rank-min-confidence 0.1 \
    --calibration-alpha 0.9 \
    --calibration-beta 0.1 \
    --calibration-views original hflip
```

Class-aware rank variant:

```bash
python test_time_calibration.py \
    --weights weights/yolo11n.pt \
    --source target_dataset/unlabels_img \
    --project runs/calibration \
    --name ttc_rank_class \
    --calibration-mode rank \
    --rank-class-aware \
    --consistency-threshold 0.5 \
    --rank-top-fraction 0.25 \
    --rank-min-confidence 0.1 \
    --calibration-alpha 0.9 \
    --calibration-beta 0.1 \
    --calibration-views original hflip
```

Calibration metric evaluation on a labeled target split:

```bash
python evaluate_calibration.py \
    --weights weights/yolo11n.pt \
    --image-dir target_dataset/val_img/images \
    --label-dir target_dataset/val_img/labels \
    --output-dir runs/calibration_eval/default
```

Selective evaluation:

```bash
python evaluate_calibration.py \
    --weights weights/yolo11n.pt \
    --image-dir target_dataset/val_img/images \
    --label-dir target_dataset/val_img/labels \
    --output-dir runs/calibration_eval/selective \
    --calibration-mode selective \
    --consistency-threshold 0.5 \
    --calibration-alpha 0.9 \
    --calibration-beta 0.1 \
    --calibration-views original hflip
```

Rank-based evaluation:

```bash
python evaluate_calibration.py \
    --weights weights/yolo11n.pt \
    --image-dir target_dataset/val_img/images \
    --label-dir target_dataset/val_img/labels \
    --output-dir runs/calibration_eval/rank \
    --calibration-mode rank \
    --consistency-threshold 0.5 \
    --rank-top-fraction 0.25 \
    --rank-min-confidence 0.1 \
    --calibration-alpha 0.9 \
    --calibration-beta 0.1 \
    --calibration-views original hflip
```

Class-aware rank evaluation:

```bash
python evaluate_calibration.py \
    --weights weights/yolo11n.pt \
    --image-dir target_dataset/val_img/images \
    --label-dir target_dataset/val_img/labels \
    --output-dir runs/calibration_eval/rank_class \
    --calibration-mode rank \
    --rank-class-aware \
    --consistency-threshold 0.5 \
    --rank-top-fraction 0.25 \
    --rank-min-confidence 0.1 \
    --calibration-alpha 0.9 \
    --calibration-beta 0.1 \
    --calibration-views original hflip
```

Calibration ablation sweep:

```bash
python run_calibration_sweep.py \
    --weights weights/yolo11n.pt \
    --image-dir target_dataset/val_img/images \
    --label-dir target_dataset/val_img/labels \
    --output-root runs/calibration_eval/sweep
```

Paper experiment matrix:

```bash
python run_paper_experiments.py \
    --python-exe D:\anaconda3\envs\yolo\python.exe \
    --workdir D:\yolo\yolov7-main \
    --output-root runs/paper_experiments \
    --mode default
```

Recommended conservative sweep:

```bash
python run_calibration_sweep.py \
    --python-exe D:\anaconda3\envs\yolo\python.exe \
    --weights weights/yolo11n.pt \
    --image-dir target_dataset/val_img/images \
    --label-dir target_dataset/val_img/labels \
    --output-root runs/calibration_eval/sweep_v1 \
    --img-size 64 \
    --device 0 \
    --calibration-mode selective \
    --consistency-thresholds 0.3,0.5,0.7 \
    --view-sets "original,hflip;original,hflip,bright;original,hflip,bright,dark;original,hflip,contrast" \
    --alphas 0.9,0.7,0.5 \
    --betas 0.1,0.3,0.5 \
    --calibration-ious 0.3,0.5,0.7
```

The current global calibration rule is a consistency-based confidence adjustment:

```text
conf_calibrated = conf_raw * (alpha + beta * consistency_score)
```

where `consistency_score` comes from view agreement across the original image and light augmentations such as horizontal flip and brightness shift.

The selective calibration rule only downweights uncertain boxes:

```text
if consistency_score < tau:
    conf_calibrated = conf_raw * (alpha + beta * consistency_score)
else:
    conf_calibrated = conf_raw
```

The rank-based rule only adjusts the riskiest confident boxes:

```text
risk_score = conf_raw * (1 - consistency_score)
select top-k risky boxes above a confidence floor
only downweight those selected boxes when consistency_score < tau
```

With `--rank-class-aware`, risky-box selection happens separately inside each predicted class.

The current calibration evaluation reports:

- `ECE` for confidence bin miscalibration
- `NLL` for confidence log-loss
- `Brier` score for squared confidence error

These metrics are computed over detection events, where a prediction is marked correct when it matches an unmatched ground-truth box of the same class at or above the chosen IoU threshold.

## Suggested Dataset Layout

For paper-oriented UDA detection experiments, the most practical first setup is:

- `Cityscapes -> Foggy Cityscapes`
- `SIM10K -> Cityscapes`
- optional later: `KITTI -> Cityscapes`

Suggested local layout:

```text
datasets/
  cityscapes/
    leftImg8bit/
      train/
      val/
    gtFine/
      train/
      val/
  foggy_cityscapes/
    leftImg8bit_foggy/
      train/
      val/
    gtFine/
      train/
      val/
  sim10k/
    images/
    labels/
  kitti/
    images/
    labels/
```

Recommended paper path:

1. Start with `Cityscapes -> Foggy Cityscapes` for the first calibration study.
2. Add `SIM10K -> Cityscapes` for a stronger cross-domain story.
3. Add `KITTI -> Cityscapes` when you want a fuller benchmark table.

## Output Locations

- Pseudo-label inference: `runs/predict/<run_name>/`
- Calibrated inference summaries: `runs/predict/<run_name>/calibration_summary.json`
- Calibration evaluation reports: `runs/calibration_eval/<name>/comparison.json`
- Calibration sweep summary: `runs/calibration_eval/sweep/sweep_summary.csv`
- Paper-ready sweep table: `runs/calibration_eval/sweep/sweep_summary.md`
- Paper experiment summary: `runs/paper_experiments/experiment_summary.md`
- Mixed dataset snapshots: `runs/mixed_dataset/<iteration_stamp>/`
- Fine-tuned detector: `runs/train/<run_name>/weights/best.pt`
- Prepared merged YOLO checkpoints: `runs/prepared_weights/`
- Normalized data YAMLs: `runs/prepared_data/`

## Verified Result

The current codebase has been verified to complete one full self-training round in the local `yolo` environment.

Example output checkpoint:

```text
runs/train/self_train_iter_1_1779885048_1779885058/weights/best.pt
```

## Current Status

Working now:

- contrastive pretraining smoke test
- full first-round self-training
- backbone-only contrastive checkpoint merge into YOLO11
- stable Windows execution with `workers=0`
- optional CBAM before the contrastive projection head
- optional domain-aware contrastive loss on the SimCLR path
- optional GRL-based domain adversarial training on P5 features
- optional LayerLock-inspired progressive freezing with `P3 -> P4 -> P5` feature transition
- optional progressive pseudo-label refinement across self-training iterations
- optional test-time consistency calibration for target-domain inference

Not yet fully integrated into the main training loop:

- true quality-weighted pseudo-label loss inside detector fine-tuning
- calibration-aware detector fine-tuning loss beyond pseudo-label filtering

## References

- [Ultralytics YOLO](https://github.com/ultralytics/ultralytics)
- [SimCLR](https://github.com/google-research/simclr)
- [MoCo](https://github.com/facebookresearch/moco)

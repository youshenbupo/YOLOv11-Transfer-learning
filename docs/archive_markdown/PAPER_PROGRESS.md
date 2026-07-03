# Paper Progress

## Current headline

On the current local smoke setup, the strongest and most reliable gain is coming from **rank-based test-time calibration** after self-training.

## Verified results

### 1. Self-training baseline

- Final source-side smoke result:
  - `mAP@0.5 = 0.1066`
  - `mAP@0.5:0.95 = 0.0535`

### 2. Rank-based test-time calibration

Using:

- views: `original hflip`
- mode: `rank`
- `alpha = 0.9`
- `beta = 0.1`
- `consistency_threshold = 0.5`
- `rank_top_fraction = 0.25`
- `rank_min_confidence = 0.1`
- `calibration_iou_threshold = 0.3`

Target-domain calibration evaluation improved on the local labeled target split:

- `delta ECE = -0.0108`
- `delta NLL = -0.0223`
- `delta Brier = -0.0107`

This is the clearest positive result in the current project state.

### 3. Selective / class-aware calibration

- Global multiplicative calibration was unstable.
- Selective calibration reduced the damage but did not clearly beat the baseline.
- Class-aware rank calibration was stable, but on the current small target split it did not clearly outperform class-agnostic rank calibration.

## Smoke-matrix interpretation

The smoke experiment matrix is useful for plumbing and early direction finding, but not for judging the representation-learning variants yet.

Current observation:

- `CBAM / domain-aware loss / LayerLock-inspired / domain adversarial` pretraining variants do run
- but under the current **1-epoch tiny-image smoke setup**, their self-training results collapse to `0.0` source mAP
- this should be treated as a **stress-smoke outcome**, not a research conclusion

## What looks paper-worthy now

The most coherent paper story today is:

1. **LayerLock-inspired progressive contrastive pretraining**
2. **Progressive pseudo-label refinement**
3. **Rank-based test-time calibration**

Among these, only the third one already shows a clean positive metric movement on the local target split.

## Recommended next real experiments

Run on proper benchmarks instead of the smoke subset:

1. `Cityscapes -> Foggy Cityscapes`
2. `SIM10K -> Cityscapes`
3. optionally `KITTI -> Cityscapes`

Suggested order:

1. baseline self-training
2. baseline + rank calibration
3. LayerLock-inspired pretraining + self-training
4. LayerLock-inspired pretraining + rank calibration
5. full stack

## Important caution

Do not over-claim the smoke pretraining ablations. The current evidence supports:

- the infrastructure is working
- the calibration branch is promising
- the LayerLock-inspired branch is implemented and runnable

It does **not** yet support a strong performance claim for the pretraining variants on real UDA benchmarks.

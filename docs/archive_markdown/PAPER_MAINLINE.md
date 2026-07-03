# Paper Mainline

## Recommended headline

Current evidence supports a narrow paper story:

- baseline: YOLO11 self-training for UDA detection
- main method: rank-based test-time calibration
- ablations:
  - calibration-guided pseudo labeling
  - adaptive pseudo thresholding
  - progressive pseudo refinement
  - contrastive pretraining variants

Methods that should not be kept as headline contributions on the current evidence:

- CBAM pretraining
- LayerLock-inspired pretraining
- current domain-loss contrastive pretraining
- iteration-level EMA teacher

## Main results

The main table should only compare `baseline` and `rank_calibration`.

| Benchmark | Method | Target mAP50 | Target mAP50-95 | Delta ECE | Delta NLL | Delta Brier |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| Cityscapes -> Foggy | baseline | 0.1433 | 0.0871 | - | - | - |
| Cityscapes -> Foggy | rank_calibration | 0.1418 | 0.0873 | -0.0000038 | -0.0000248 | -0.0000110 |
| SIM10K -> Cityscapes | baseline | 0.3246 | 0.1870 | - | - | - |
| SIM10K -> Cityscapes | rank_calibration | 0.3715 | 0.2117 | -0.0000184 | -0.0000462 | -0.0000134 |

Interpretation:

- On `SIM10K -> Cityscapes`, `rank_calibration` is the strongest method currently available in this project.
- On `Cityscapes -> Foggy`, `rank_calibration` is essentially tied with the baseline on detection, while keeping calibration slightly better.
- This makes `rank_calibration` defensible as a low-cost, stable improvement, but not yet a dominant gain across every benchmark.

## Ablations

### SIM10K -> Cityscapes

| Method | Target mAP50 | Target mAP50-95 | Note |
| --- | ---: | ---: | --- |
| baseline | 0.3246 | 0.1870 | reference |
| rank_calibration | 0.3715 | 0.2117 | best current result |
| calibration_guided_pseudo | 0.3715 | 0.2117 | stable, no gain beyond rank_calibration |
| adaptive_pseudo_threshold | 0.3715 | 0.2117 | stable, no gain beyond rank_calibration |
| progressive_pseudo | 0.2833 | 0.1739 | worse than baseline on target AP |
| domain_loss_pretrain | 0.0208 | 0.0127 | collapses on target AP |
| cbam_pretrain | 0.0153 | 0.0104 | collapses on target AP |
| layerlock_pretrain | 0.0035 | 0.0017 | collapses on target AP |
| ema_teacher | 0.0062 | 0.0020 | not viable in current form |

### Cityscapes -> Foggy

| Method | Target mAP50 | Target mAP50-95 | Note |
| --- | ---: | ---: | --- |
| baseline | 0.1433 | 0.0871 | best current result |
| rank_calibration | 0.1418 | 0.0873 | tied with baseline |
| progressive_pseudo | 0.1374 | 0.0834 | slightly worse |
| cbam_pretrain | 0.0516 | 0.0302 | poor target transfer |
| domain_loss_pretrain | 0.0345 | 0.0192 | poor target transfer |
| layerlock_pretrain | 0.0160 | 0.0091 | poor target transfer |

## Gap to literature

Using the `SOCCER` paper as a stable reference point:

| Benchmark | This project best AP50 | Reference method | Reference AP50 | Gap |
| --- | ---: | --- | ---: | ---: |
| Cityscapes -> Foggy | 0.1433 | SOCCER | 0.511 | 0.3677 |
| SIM10K -> Cityscapes | 0.3715 | SOCCER | 0.638 | 0.2665 |

Interpretation:

- The current project is still well below strong DAOD literature.
- The gap is smaller on `SIM10K -> Cityscapes` than on `Cityscapes -> Foggy`.
- `SIM10K -> Cityscapes` is therefore the benchmark that should anchor the current draft.

## Recommended paper structure

### Core contribution

State the contribution conservatively:

1. We study confidence calibration for UDA object detection under a YOLO11 self-training pipeline.
2. We propose a rank-based test-time calibration rule that improves target-domain confidence reliability and improves target-domain AP on `SIM10K -> Cityscapes`.
3. We analyze several pseudo-label and pretraining variants and show that most of them do not outperform the simpler calibrated baseline.

### What to avoid claiming

Do not claim:

- a strong general UDA detection breakthrough
- that contrastive pretraining is effective in the current implementation
- that EMA teacher or LayerLock-inspired freezing helps this detector

### What to emphasize

Emphasize:

- target-domain AP, not source validation AP
- calibration metrics (`ECE`, `NLL`, `Brier`)
- the consistency of `rank_calibration`
- negative results as part of the ablation story

## Next writing tasks

1. Convert the main results table into the paper's experimental results section.
2. Add one method figure for rank-based calibration:
   - multi-view inference
   - consistency estimation
   - risky high-confidence box downweighting
3. Add one ablation subsection:
   - rank_calibration
   - calibration_guided_pseudo
   - adaptive_pseudo_threshold
   - EMA teacher
4. Add one discussion subsection:
   - why source metrics were misleading
   - why target-domain AP should drive method selection

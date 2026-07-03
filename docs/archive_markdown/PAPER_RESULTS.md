# Paper Results Draft

## Experiment setup

- Runner: `run_paper_experiments.py`
- Hardware: local `RTX 5060 8GB`
- Environment: `conda` env `yolo`
- Prepared benchmarks:
  - `Cityscapes -> Foggy Cityscapes`
  - `SIM10K -> Cityscapes`

## Result validity note

- The earlier `Cityscapes -> Foggy` record with `mAP50 = 0.0000` was invalid.
- Root cause: benchmark labels were generated with mismatched filenames, so Ultralytics treated many source images as background-only samples.
- After fixing `prepare_benchmark_datasets.py` and regenerating the benchmark directories, the formal results below replaced that invalid run.

## Formal benchmark results collected so far

### 1. Cityscapes -> Foggy Cityscapes

Status: partial matrix completed.

#### baseline

- status: `ok`
- source `mAP50`: `0.2902`
- source `mAP50-95`: `0.1703`
- self-training duration: `1218.78 s`
- final weights:
  - `D:\yolo\yolov7-main\runs\train\self_train_iter_1_1779987244_1779987538\weights\best.pt`
- self-training log:
  - `D:\yolo\yolov7-main\runs\paper_experiments_formal_fixed\cityscapes_to_foggy\baseline\self_training.attempt1.log`

#### rank_calibration

- status: `ok`
- source `mAP50`: `0.2835`
- source `mAP50-95`: `0.1653`
- self-training duration: `1186.88 s`
- calibrated target `ECE`: `0.0158815536`
- calibrated target `NLL`: `0.1957286896`
- calibrated target `Brier`: `0.0538229158`
- `delta ECE`: `-0.0000038433`
- `delta NLL`: `-0.0000248403`
- `delta Brier`: `-0.0000110517`
- calibration duration: `109.72 s`
- final weights:
  - `D:\yolo\yolov7-main\runs\train\self_train_iter_1_1779988462_1779988728\weights\best.pt`
- self-training log:
  - `D:\yolo\yolov7-main\runs\paper_experiments_formal_fixed\cityscapes_to_foggy\rank_calibration\self_training.attempt1.log`
- calibration comparison:
  - `D:\yolo\yolov7-main\runs\paper_experiments_formal_fixed\cityscapes_to_foggy\rank_calibration\calibration_eval\comparison.json`

#### progressive_pseudo

- status: `ok`
- source `mAP50`: `0.2848`
- source `mAP50-95`: `0.1676`
- self-training duration: `1203.05 s`
- final weights:
  - `D:\yolo\yolov7-main\runs\train\self_train_iter_1_1779989950_1779990238\weights\best.pt`
- self-training log:
  - `D:\yolo\yolov7-main\runs\paper_experiments_formal_fixed\cityscapes_to_foggy\progressive_pseudo\self_training.attempt1.log`

#### cbam_pretrain

- status: `ok`
- source `mAP50`: `0.2672`
- source `mAP50-95`: `0.1536`
- contrastive pretraining duration: `2267.95 s`
- self-training duration: `1400.77 s`
- contrastive checkpoint:
  - `D:\yolo\yolov7-main\runs\paper_experiments_formal_fixed\cityscapes_to_foggy\cbam_pretrain\cbam_pretrain_contrastive_full.pt`
- final weights:
  - `D:\yolo\yolov7-main\runs\train\self_train_iter_1_1779993644_1779993853\weights\best.pt`
- logs:
  - `D:\yolo\yolov7-main\runs\paper_experiments_formal_fixed\cityscapes_to_foggy\cbam_pretrain\contrastive.attempt1.log`
  - `D:\yolo\yolov7-main\runs\paper_experiments_formal_fixed\cityscapes_to_foggy\cbam_pretrain\self_training.attempt1.log`

#### domain_loss_pretrain

- status: `ok`
- source `mAP50`: `0.2758`
- source `mAP50-95`: `0.1552`
- contrastive pretraining duration: `2231.94 s`
- self-training duration: `1379.50 s`
- contrastive checkpoint:
  - `D:\yolo\yolov7-main\runs\paper_experiments_formal_fixed\cityscapes_to_foggy\domain_loss_pretrain\domain_loss_pretrain_contrastive_full.pt`
- final weights:
  - `D:\yolo\yolov7-main\runs\train\self_train_iter_1_1779997335_1779997563\weights\best.pt`
- logs:
  - `D:\yolo\yolov7-main\runs\paper_experiments_formal_fixed\cityscapes_to_foggy\domain_loss_pretrain\contrastive.attempt1.log`
  - `D:\yolo\yolov7-main\runs\paper_experiments_formal_fixed\cityscapes_to_foggy\domain_loss_pretrain\self_training.attempt1.log`

#### layerlock_pretrain

- status: `ok`
- source `mAP50`: `0.2664`
- source `mAP50-95`: `0.1538`
- contrastive pretraining duration: `2227.52 s`
- self-training duration: `1341.07 s`
- contrastive checkpoint:
  - `D:\yolo\yolov7-main\runs\paper_experiments_formal_fixed\cityscapes_to_foggy\layerlock_pretrain\layerlock_pretrain_contrastive_full.pt`
- final weights:
  - `D:\yolo\yolov7-main\runs\train\self_train_iter_1_1780000936_1780001147\weights\best.pt`
- logs:
  - `D:\yolo\yolov7-main\runs\paper_experiments_formal_fixed\cityscapes_to_foggy\layerlock_pretrain\contrastive.attempt1.log`
  - `D:\yolo\yolov7-main\runs\paper_experiments_formal_fixed\cityscapes_to_foggy\layerlock_pretrain\self_training.attempt1.log`

#### interim conclusion

- On the corrected first formal pass, `baseline` is better than `rank_calibration` on source-domain detection:
  - `mAP50`: `0.2902` vs `0.2835`
  - `mAP50-95`: `0.1703` vs `0.1653`
- `progressive_pseudo` is also below the baseline:
  - `mAP50`: `0.2848`
  - `mAP50-95`: `0.1676`
- `cbam_pretrain` is clearly below the baseline and below the two lighter variants:
  - `mAP50`: `0.2672`
  - `mAP50-95`: `0.1536`
- `domain_loss_pretrain` is the best pretraining variant so far, but still below the plain baseline:
  - `mAP50`: `0.2758`
  - `mAP50-95`: `0.1552`
- `layerlock_pretrain` is also below the baseline and roughly tied with `cbam_pretrain`:
  - `mAP50`: `0.2664`
  - `mAP50-95`: `0.1538`
- `rank_calibration` gives a very small target-domain calibration improvement:
  - `delta ECE < 0`
  - `delta NLL < 0`
  - `delta Brier < 0`
- Current ordering on source detection is:
  - `baseline` > `progressive_pseudo` > `rank_calibration` > `domain_loss_pretrain` > `cbam_pretrain` > `layerlock_pretrain`
- The calibration gain exists, but the magnitude is currently very small. At this stage it is a weak positive signal, not a paper claim by itself.

### 2. SIM10K -> Cityscapes

Status: partial matrix completed.

#### baseline

- status: `ok`
- source `mAP50`: `0.5628`
- source `mAP50-95`: `0.3563`
- self-training duration: `2001.13 s`
- final weights:
  - `D:\yolo\yolov7-main\runs\train\self_train_iter_1_1780002397_1780002733\weights\best.pt`
- self-training log:
  - `D:\yolo\yolov7-main\runs\paper_experiments_formal_fixed\sim10k_to_cityscapes\baseline\self_training.attempt1.log`

#### rank_calibration

- status: `ok`
- source `mAP50`: `0.5697`
- source `mAP50-95`: `0.3585`
- self-training duration: `1946.91 s`
- `delta ECE`: `-0.0000184014`
- `delta NLL`: `-0.0000461946`
- `delta Brier`: `-0.0000133815`
- final weights:
  - `D:\yolo\yolov7-main\runs\train\self_train_iter_1_1780004416_1780004743\weights\best.pt`
- self-training log:
  - `D:\yolo\yolov7-main\runs\paper_experiments_formal_fixed\sim10k_to_cityscapes\rank_calibration\self_training.attempt1.log`

#### progressive_pseudo

- status: `ok`
- source `mAP50`: `0.5832`
- source `mAP50-95`: `0.3618`
- self-training duration: `1993.33 s`
- final weights:
  - `D:\yolo\yolov7-main\runs\train\self_train_iter_1_1780006494_1780006837\weights\best.pt`
- self-training log:
  - `D:\yolo\yolov7-main\runs\paper_experiments_formal_fixed\sim10k_to_cityscapes\progressive_pseudo\self_training.attempt1.log`

#### domain_loss_pretrain

- status: `ok`
- source `mAP50`: `0.5805`
- source `mAP50-95`: `0.3648`
- contrastive pretraining duration: `1716.23 s`
- self-training duration: `2649.00 s`
- contrastive checkpoint:
  - `D:\yolo\yolov7-main\runs\paper_experiments_formal_fixed\sim10k_to_cityscapes\domain_loss_pretrain\domain_loss_pretrain_contrastive_full.pt`
- final weights:
  - `D:\yolo\yolov7-main\runs\train\self_train_iter_1_1780010260_1780010474\weights\best.pt`
- logs:
  - `D:\yolo\yolov7-main\runs\paper_experiments_formal_fixed\sim10k_to_cityscapes\domain_loss_pretrain\contrastive.attempt1.log`
  - `D:\yolo\yolov7-main\runs\paper_experiments_formal_fixed\sim10k_to_cityscapes\domain_loss_pretrain\self_training.attempt1.log`

#### cbam_pretrain

- status: `ok`
- source `mAP50`: `0.5660`
- source `mAP50-95`: `0.3587`
- contrastive pretraining duration: `1864.61 s`
- self-training duration: `3407.90 s`
- contrastive checkpoint:
  - `D:\yolo\yolov7-main\runs\paper_experiments_formal_fixed\sim10k_to_cityscapes\cbam_pretrain\cbam_pretrain_contrastive_full.pt`
- final weights:
  - `D:\yolo\yolov7-main\runs\train\self_train_iter_1_1780017367_1780017636\weights\best.pt`
- note:
  - the original runner attempt failed during inference because predictions were being accumulated in RAM
  - after patching `self_training.py` to use `stream=True`, the self-training rerun completed successfully

#### layerlock_pretrain

- status: `ok`
- source `mAP50`: `0.5738`
- source `mAP50-95`: `0.3586`
- contrastive pretraining duration: `1848.51 s`
- self-training duration: `3375.64 s`
- contrastive checkpoint:
  - `D:\yolo\yolov7-main\runs\paper_experiments_formal_fixed\sim10k_to_cityscapes\layerlock_pretrain\layerlock_pretrain_contrastive_full.pt`
- final weights:
  - `D:\yolo\yolov7-main\runs\train\self_train_iter_1_1780022801_1780023284\weights\best.pt`
- logs:
  - `D:\yolo\yolov7-main\runs\paper_experiments_formal_fixed\sim10k_to_cityscapes\layerlock_pretrain\contrastive.attempt1.log`
  - `D:\yolo\yolov7-main\runs\paper_experiments_formal_fixed\sim10k_to_cityscapes\layerlock_pretrain\self_training.attempt1.log`

#### interim conclusion

- `progressive_pseudo` is currently the best `mAP50` variant on this benchmark:
  - `mAP50`: `0.5832`
  - `mAP50-95`: `0.3618`
- `domain_loss_pretrain` is slightly below `progressive_pseudo` on `mAP50`, but best on `mAP50-95`:
  - `mAP50`: `0.5805`
  - `mAP50-95`: `0.3648`
- `layerlock_pretrain` is mid-pack and does not beat the lighter variants:
  - `mAP50`: `0.5738`
  - `mAP50-95`: `0.3586`
- `cbam_pretrain` only recovers to near-baseline level after the memory patch:
  - `mAP50`: `0.5660`
  - `mAP50-95`: `0.3587`
- `rank_calibration` improves both detection and calibration relative to the plain baseline:
  - `mAP50`: `0.5697` vs `0.5628`
  - `delta ECE < 0`
  - `delta NLL < 0`
  - `delta Brier < 0`
- Current ordering on source detection is:
  - `progressive_pseudo` > `domain_loss_pretrain` > `layerlock_pretrain` > `rank_calibration` > `cbam_pretrain` > `baseline`

## Current paper-level takeaway

1. The formal benchmark pipeline is now valid for `Cityscapes -> Foggy`.
2. The previous `mAP=0` result should not be cited.
3. The corrected `Cityscapes -> Foggy` baseline has learned meaningful source-domain detection.
4. `rank_calibration` is currently calibration-positive but detection-neutral to slightly negative.
5. `progressive_pseudo` does not beat the current baseline on the first formal pass.
6. None of the three pretraining variants beats the plain baseline on the first formal pass.
7. `domain_loss_pretrain` is the least bad pretraining variant, but still not strong enough to justify the extra cost.
8. `SIM10K -> Cityscapes` changes the picture: both `progressive_pseudo` and `domain_loss_pretrain` are competitive, and `rank_calibration` is positive on both detection and calibration.
9. `layerlock_pretrain` is not competitive enough to carry a headline claim on the current settings.
10. `cbam_pretrain` is also not competitive, and required a memory-safety patch to complete on the larger benchmark.
11. The current evidence supports a paper story centered on `progressive pseudo refinement + rank calibration`, with `domain_loss_pretrain` as the only pretraining branch still worth keeping alive.

## Target-domain comparable AP checkpoint

The earlier `source_map50/source_map5095` values are not paper-comparable against DAOD literature. The comparable quantity is target-domain AP on the benchmark target validation split.

### Cityscapes -> Foggy Cityscapes target AP

- `baseline`: `mAP50 = 0.1433`, `mAP50-95 = 0.0871`
- `progressive_pseudo`: `mAP50 = 0.1374`, `mAP50-95 = 0.0834`
- `rank_calibration`: `mAP50 = 0.1418`, `mAP50-95 = 0.0873`
- `domain_loss_pretrain`: `mAP50 = 0.0345`, `mAP50-95 = 0.0192`
- `cbam_pretrain`: `mAP50 = 0.0516`, `mAP50-95 = 0.0302`
- `layerlock_pretrain`: `mAP50 = 0.0160`, `mAP50-95 = 0.0091`

Notes:
- These numbers come from `target_val.yaml` with the 8-class Foggy validation split.
- Current target-domain ordering on `C->F` is:
  - `baseline` > `rank_calibration` > `progressive_pseudo` > `cbam_pretrain` > `domain_loss_pretrain` > `layerlock_pretrain`

### SIM10K -> Cityscapes target AP

- `baseline`: `mAP50 = 0.3246`, `mAP50-95 = 0.1870`
- `rank_calibration`: `mAP50 = 0.3715`, `mAP50-95 = 0.2117`
- `calibration_guided_pseudo`: `mAP50 = 0.3715`, `mAP50-95 = 0.2117`
- `adaptive_pseudo_threshold`: `mAP50 = 0.3715`, `mAP50-95 = 0.2117`
- `progressive_pseudo`: `mAP50 = 0.2833`, `mAP50-95 = 0.1739`
- `domain_loss_pretrain`: `mAP50 = 0.0208`, `mAP50-95 = 0.0127`
- `cbam_pretrain`: `mAP50 = 0.0153`, `mAP50-95 = 0.0104`
- `layerlock_pretrain`: `mAP50 = 0.0035`, `mAP50-95 = 0.0017`

Interpretation:
- On target-domain AP, `rank_calibration` is currently the strongest variant on `SIM10K -> Cityscapes`.
- `calibration_guided_pseudo` completed successfully but did not improve over `rank_calibration` on this benchmark. Its calibration metrics remained slightly positive:
  - `delta ECE = -0.0000184`
  - `delta NLL = -0.0000462`
  - `delta Brier = -0.0000134`
- `adaptive_pseudo_threshold` also completed successfully and matched `rank_calibration` rather than beating it.
- `progressive_pseudo` looks strong on source validation, but does not currently convert that strength into better target-domain AP.
- The contrastive-pretraining branches (`domain_loss_pretrain`, `cbam_pretrain`, `layerlock_pretrain`) are currently not viable on target-domain AP despite some of them looking reasonable on source validation.

### Calibration-guided pseudo labeling checkpoint

Implemented method:
- run rank-based test-time calibration during target-domain inference
- save per-box `consistency` and `risk_score`
- keep pseudo labels only if:
  - calibrated confidence passes the standard pseudo threshold
  - `consistency >= 0.5`
  - `risk_score <= 0.25`

Formal `SIM10K -> Cityscapes` run:
- command family: `self_training.py` with `--use-test-time-calibration --calibration-mode rank --use-calibration-guided-pseudo`
- final weights:
  - `D:\yolo\yolov7-main\runs\train\self_train_iter_1_1780031393_1780031749\weights\best.pt`
- target AP:
  - `mAP50 = 0.3715`
  - `mAP50-95 = 0.2117`
- pseudo-label count in the first iteration:
  - `46808`

Conclusion:
- this first calibration-guided pseudo variant is stable
- it does not degrade target-domain AP
- it does not outperform plain `rank_calibration` either
- the current gating rule is therefore not strong enough to justify itself as a headline method

### Adaptive pseudo thresholding checkpoint

Implemented method:
- run rank-based test-time calibration during target-domain inference
- keep the standard pseudo-label threshold as the floor
- raise the effective threshold for low-consistency detections with:
  - `effective_threshold = base_threshold + scale * max(0, target_consistency - consistency)`

Formal `SIM10K -> Cityscapes` run:
- command family: `self_training.py` with `--use-test-time-calibration --use-adaptive-pseudo-threshold`
- final weights:
  - `D:\yolo\yolov7-main\runs\train\self_train_iter_1_1780045519_1780045946\weights\best.pt`
- target AP:
  - `mAP50 = 0.3715`
  - `mAP50-95 = 0.2117`
- pseudo-label count in the first iteration:
  - `47036`
- calibration deltas:
  - `delta ECE = -0.0000184`
  - `delta NLL = -0.0000462`
  - `delta Brier = -0.0000134`

Conclusion:
- this adaptive-threshold variant is stable
- it neither improves nor degrades the current best result
- it should be treated as an ablation of the calibration-driven pseudo-label branch, not as a stronger replacement for `rank_calibration`

### EMA teacher-student checkpoint

Implemented method:
- use the current teacher checkpoint to generate pseudo labels
- fine-tune a student on source labels + target pseudo labels
- update the teacher after each iteration with:
  - `teacher <- decay * teacher + (1 - decay) * student`

Formal `SIM10K -> Cityscapes` run:
- command family: `self_training.py` with `--use-ema-teacher --ema-decay 0.999`
- final teacher weights:
  - `D:\yolo\yolov7-main\runs\ema_teacher\ema_teacher_1780044796.pt`
- target AP:
  - `mAP50 = 0.0062`
  - `mAP50-95 = 0.0020`

Conclusion:
- this iteration-level EMA teacher is not viable in the current form
- the result is dramatically worse than:
  - `baseline`: `0.3246 / 0.1870`
  - `rank_calibration`: `0.3715 / 0.2117`
- the main reason is structural:
  - with only one self-training iteration and a very high EMA decay, the teacher stays too close to the weak initial detector
  - the EMA update happens only after retraining, so it smooths toward the student too slowly to be useful
- this should not be kept as a main method branch unless the implementation is upgraded to an in-training or per-step EMA teacher

# Rank-Based Test-Time Calibration for YOLO11 Self-Training in Unsupervised Domain Adaptive Object Detection

## Abstract

Unsupervised domain adaptive object detection (DAOD) aims to transfer a detector trained on a labeled source domain to an unlabeled target domain. In this project, we study a pragmatic question: whether a lightweight confidence calibration module can improve a YOLO11-style self-training pipeline without requiring architectural changes to the detector. We build a baseline self-training framework on top of YOLO11, evaluate it on the standard `SIM10K -> Cityscapes` and `Cityscapes -> Foggy Cityscapes` benchmarks, and introduce a rank-based test-time calibration method that estimates box reliability using multi-view consistency. The method ranks high-risk target predictions according to confidence-consistency disagreement and selectively downweights risky boxes during inference. On `SIM10K -> Cityscapes`, the proposed calibration improves target-domain AP50 from `32.46` to `37.15` and AP50-95 from `18.70` to `21.17`, while also slightly improving calibration metrics (`ECE`, `NLL`, and `Brier score`). On `Cityscapes -> Foggy Cityscapes`, the method is essentially tied with the baseline on detection and slightly better on calibration. We further evaluate several pseudo-label filtering and pretraining variants and find that most of them fail to outperform the simpler calibrated baseline. These results suggest that low-cost confidence calibration is a more reliable direction than heavier pretraining or teacher-student variants for the current YOLO11-based DAOD setup.

## 1. Introduction

Object detectors remain sensitive to domain shift. A model trained on synthetic driving images, clear-weather images, or one camera platform often degrades substantially when deployed on another domain. DAOD addresses this problem by using source labels and unlabeled target images. However, in practice, modern DAOD pipelines are often fragile for two reasons:

1. pseudo labels on the target domain are noisy;
2. model confidence is poorly calibrated under domain shift.

This project started from a broader YOLO11-based DAOD framework with contrastive pretraining, self-training, domain-aware losses, progressive pseudo-label refinement, and test-time calibration. After running formal benchmark experiments, the evidence is clear: the only consistently useful branch is rank-based test-time calibration. Several more complicated branches either failed to improve target-domain AP or collapsed entirely.

This leads to a narrower and more defensible paper question:

> Can a lightweight calibration rule improve YOLO-style DAOD self-training more reliably than heavier adaptation branches?

We answer this question by isolating a rank-based test-time calibration module that:

- generates multiple target-domain views at inference time;
- estimates per-box consistency across views;
- defines a risk score from confidence-consistency disagreement;
- selectively downweights risky high-confidence detections.

Our main contribution is not a new full-stack DAOD system. It is a focused empirical and methodological result: in the current YOLO11 self-training pipeline, rank-based calibration is a more reliable improvement than several heavier alternatives.

## 2. Related Work

Early DAOD focused on adversarial feature alignment. Domain Adaptive Faster R-CNN aligned source and target distributions at the image and instance levels and added consistency regularization to stabilize region proposals ([arXiv](https://arxiv.org/abs/1803.03243)).

Later work moved toward teacher-student and self-training paradigms. Adaptive Teacher used weak-strong augmentation, pseudo labeling, adversarial learning, and EMA updates in a teacher-student framework for cross-domain detection ([CVPR 2022 PDF](https://openaccess.thecvf.com/content/CVPR2022/papers/Li_Cross-Domain_Adaptive_Teacher_for_Object_Detection_CVPR_2022_paper.pdf)).

Consistency learning became another major line. 2PCNet introduced two-phase consistency for day-to-night DAOD, explicitly re-evaluating teacher predictions to reduce false-positive propagation ([arXiv](https://arxiv.org/abs/2303.13853)). SOCCER proposed stochastic context consistency reasoning and complementary masked views to improve pseudo labels and target adaptation ([OpenReview PDF](https://openreview.net/pdf/c74d5ab618776e5df722559174b7707314d5c680.pdf)).

Recent methods have diversified further:

- `AdvMix` combines adversarial alignment, region mixing, and uncertainty-aware self-training ([MDPI](https://www.mdpi.com/2079-9292/13/4/685));
- `CAGD` studies gradient conflict between detection and adaptation losses ([IJCAI 2024](https://www.ijcai.org/proceedings/2024/0137));
- `DA-Ada` uses domain-aware adapters with visual-language priors ([OpenReview](https://openreview.net/forum?id=hkEwwAqmCk));
- `ALDI / ALDI++` argues that benchmark hygiene matters as much as model design and reports stronger modern baselines and new SOTA under a unified protocol ([arXiv](https://arxiv.org/abs/2403.12029)).

Our work is closest in spirit to consistency-based and self-training methods, but it differs in one important way: we do not introduce another large adaptation branch. Instead, we treat target-domain confidence reliability itself as the main object of study.

## 3. Method

### 3.1 Baseline self-training

Let:

- `D_s = {(x_i^s, y_i^s)}` be the labeled source dataset;
- `D_t = {x_j^t}` be the unlabeled target dataset;
- `f_theta` be the YOLO11 detector.

The baseline runs in one self-training iteration:

1. train / initialize `f_theta` from source supervision;
2. infer pseudo labels on `D_t`;
3. filter pseudo labels by confidence threshold;
4. retrain on `D_s U \hat{D}_t`.

### 3.2 Rank-based test-time calibration

Given a target image `x`, we produce a set of augmented views:

- `v_0(x) = x` (original),
- `v_1(x), ..., v_K(x)` (e.g. horizontal flip, brightness change).

For each base detection `b` from the original image, we match it against detections from other views after remapping boxes back to the original coordinates. We compute a consistency score:

`c(b) = (1 / K) * sum_k s_k(b)`

where `s_k(b)` combines view-matched IoU and confidence agreement.

For a base detection with raw confidence `p(b)`, we define a risk score:

`r(b) = p(b) * (1 - c(b))`

This captures the cases that are most dangerous in DAOD:

- high confidence,
- low cross-view consistency.

We then rank detections by `r(b)` and only recalibrate the top risky fraction. If a risky box also has `c(b) < tau_c`, we downweight it:

`p'(b) = p(b) * (alpha + beta * c(b))`

Otherwise:

`p'(b) = p(b)`

This is the main method used in the final paper line.

### 3.3 Calibration-driven pseudo-label variants

We also tested two extensions:

1. `calibration_guided_pseudo`:
   keep pseudo labels only when calibrated detections satisfy consistency and risk constraints.
2. `adaptive_pseudo_threshold`:
   raise the confidence threshold for low-consistency boxes:

`t_eff(b) = t_0 + lambda * max(0, c_target - c(b))`

These variants were stable but did not improve over the simpler rank-calibrated baseline.

## 4. Experiments

### 4.1 Benchmarks

We evaluate on:

- `SIM10K -> Cityscapes`
- `Cityscapes -> Foggy Cityscapes`

These are standard DAOD benchmarks widely used in the literature.

### 4.2 Metrics

We report:

- target-domain `AP50`
- target-domain `AP50-95`
- `ECE`
- `NLL`
- `Brier score`

We intentionally emphasize target-domain AP rather than source validation AP, because source-domain metrics were found to be misleading in our experiments.

### 4.3 Main results

| Benchmark | Method | Target AP50 | Target AP50-95 | Delta ECE | Delta NLL | Delta Brier |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| Cityscapes -> Foggy | baseline | 14.33 | 8.71 | - | - | - |
| Cityscapes -> Foggy | rank_calibration | 14.18 | 8.73 | -0.0000038 | -0.0000248 | -0.0000110 |
| SIM10K -> Cityscapes | baseline | 32.46 | 18.70 | - | - | - |
| SIM10K -> Cityscapes | rank_calibration | 37.15 | 21.17 | -0.0000184 | -0.0000462 | -0.0000134 |

### 4.4 Ablation summary

On `SIM10K -> Cityscapes`:

- `calibration_guided_pseudo`: tied with `rank_calibration`
- `adaptive_pseudo_threshold`: tied with `rank_calibration`
- `progressive_pseudo`: worse than baseline on target AP
- `EMA teacher`: collapsed
- contrastive pretraining variants: collapsed on target AP

On `Cityscapes -> Foggy`:

- the plain baseline remained best on AP50
- `rank_calibration` was effectively tied with slightly better calibration

## 5. Discussion

### 5.1 What worked

The most robust finding is that target-domain confidence calibration matters. The rank-based calibration rule consistently avoided harming performance and produced the best target AP on `SIM10K -> Cityscapes`.

### 5.2 What failed

Several seemingly stronger directions did not hold up:

- CBAM-based pretraining
- LayerLock-inspired pretraining
- current domain-aware contrastive pretraining
- iteration-level EMA teacher

This is useful in itself: for the present YOLO11 setup, heavier adaptation mechanisms were not automatically better.

### 5.3 Gap to the literature

The project is still far from strong DAOD methods. Relative to `SOCCER`, the current gap is approximately:

- `26.65 AP50` on `SIM10K -> Cityscapes`
- `36.77 AP50` on `Cityscapes -> Foggy`

Relative to `ALDI++`, the gap is at least of similar magnitude and likely larger, since `ALDI++` reports further gains over prior SOTA ([arXiv](https://arxiv.org/abs/2403.12029)).

Therefore, the current work should not be presented as a new state-of-the-art DAOD detector. The defensible contribution is narrower: a calibration-focused study inside a YOLO11 self-training pipeline.

## 6. Limitations

1. The method is only clearly positive on one benchmark.
2. The absolute detection accuracy remains well below modern DAOD SOTA.
3. The current improvements are concentrated in confidence calibration and one target-domain benchmark rather than broad cross-benchmark dominance.
4. Several tested pretraining and teacher-student variants remain underdeveloped or ineffective.

## 7. Conclusion

This work evaluates a YOLO11-based self-training pipeline for unsupervised domain adaptive object detection and shows that a simple rank-based test-time calibration rule is the most reliable improvement among a larger pool of candidate methods. The method improves both target-domain AP and confidence reliability on `SIM10K -> Cityscapes` while remaining effectively neutral on `Cityscapes -> Foggy`. In contrast, heavier pretraining and teacher-student variants fail to consistently help. The main practical lesson is that in this setup, calibration is a more trustworthy lever than additional adaptation complexity.

## References

- Domain Adaptive Faster R-CNN for Object Detection in the Wild: [https://arxiv.org/abs/1803.03243](https://arxiv.org/abs/1803.03243)
- Cross-Domain Adaptive Teacher for Object Detection: [https://openaccess.thecvf.com/content/CVPR2022/papers/Li_Cross-Domain_Adaptive_Teacher_for_Object_Detection_CVPR_2022_paper.pdf](https://openaccess.thecvf.com/content/CVPR2022/papers/Li_Cross-Domain_Adaptive_Teacher_for_Object_Detection_CVPR_2022_paper.pdf)
- MIC: Masked Image Consistency for Context-Enhanced Domain Adaptation: [https://openaccess.thecvf.com/content/CVPR2023/papers/Hoyer_MIC_Masked_Image_Consistency_for_Context-Enhanced_Domain_Adaptation_CVPR_2023_paper.pdf](https://openaccess.thecvf.com/content/CVPR2023/papers/Hoyer_MIC_Masked_Image_Consistency_for_Context-Enhanced_Domain_Adaptation_CVPR_2023_paper.pdf)
- SOCCER: [https://openreview.net/pdf/c74d5ab618776e5df722559174b7707314d5c680.pdf](https://openreview.net/pdf/c74d5ab618776e5df722559174b7707314d5c680.pdf)
- AdvMix: [https://www.mdpi.com/2079-9292/13/4/685](https://www.mdpi.com/2079-9292/13/4/685)
- CAGD: [https://www.ijcai.org/proceedings/2024/0137](https://www.ijcai.org/proceedings/2024/0137)
- DA-Ada: [https://openreview.net/forum?id=hkEwwAqmCk](https://openreview.net/forum?id=hkEwwAqmCk)
- ALDI / ALDI++: [https://arxiv.org/abs/2403.12029](https://arxiv.org/abs/2403.12029)

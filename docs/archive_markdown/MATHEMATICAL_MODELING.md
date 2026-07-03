# Mathematical Modeling

## 1. Problem setup

We study unsupervised domain adaptive object detection.

Given:

- source domain:
  - `D_s = {(x_i^s, y_i^s)}_{i=1}^{N_s}`
- target domain:
  - `D_t = {x_j^t}_{j=1}^{N_t}`

where:

- `x_i^s` is a source image
- `y_i^s` is its detection annotation
- `x_j^t` is a target image without labels

The detector is a YOLO11 model:

- `f_theta : x -> {(b_k, p_k, c_k)}`

where:

- `b_k` is a box
- `p_k` is a confidence score
- `c_k` is a predicted class

The goal is to learn parameters `theta` such that the detector performs well on the target domain.

## 2. Baseline objective

The self-training baseline has two parts.

### 2.1 Source supervised loss

On the source domain, the detector minimizes a standard detection loss:

`L_s(theta) = L_det(f_theta(x^s), y^s)`

This includes the usual YOLO-style localization, objectness, and classification terms.

### 2.2 Target pseudo-label self-training

For each target image `x^t`, pseudo labels are generated:

`hat{y}^t = P(f_theta(x^t))`

where `P(.)` is the pseudo-label generation operator.

The mixed training objective becomes:

`L(theta) = L_s(theta) + lambda_t * L_p(theta)`

with:

`L_p(theta) = L_det(f_theta(x^t), hat{y}^t)`

The main challenge is that `hat{y}^t` is noisy under domain shift.

## 3. Test-time consistency calibration

The final paper line introduces a calibration module at target-domain inference time.

For a target image `x`, define a set of views:

`V(x) = {v_0(x), v_1(x), ..., v_K(x)}`

where:

- `v_0(x) = x`
- the other views are lightweight augmentations such as horizontal flip and brightness adjustment.

The detector predicts:

`f_theta(v_k(x)) = {d_m^(k)}`

For a base detection `d` on the original image, we match it with detections in the augmented views after remapping boxes back to the original coordinate frame.

## 4. Consistency score

For each view `k`, define a match score:

`s_k(d) = 0.5 * IoU(d, d_k^*) + 0.5 * (1 - |p(d) - p(d_k^*)|)`

where:

- `d_k^*` is the best matched detection in view `k`
- the match must satisfy an IoU threshold to be considered valid

Then the consistency score is:

`c(d) = (1 / K) * sum_(k=1)^K s_k(d)`

Properties:

- high `c(d)` means the detection is stable across views
- low `c(d)` means the detection is unstable under small perturbations

## 5. Risk score

We define risk as confidence-weighted inconsistency:

`r(d) = p(d) * (1 - c(d))`

Interpretation:

- high confidence + low consistency = dangerous false positive candidate
- low confidence + low consistency is less important
- high confidence + high consistency is relatively trustworthy

This choice is practical because it focuses correction effort on the detections most likely to harm target-domain precision.

## 6. Rank-based calibration

Let `R_alpha` be the set of top-risk detections, selected by sorting `r(d)` and keeping the highest fraction `alpha_r`.

The calibrated confidence is:

`p'(d) = p(d) * (alpha + beta * c(d))`, if `d in R_alpha` and `c(d) < tau_c`

`p'(d) = p(d)`, otherwise

where:

- `alpha` is a base multiplier
- `beta` controls the consistency contribution
- `tau_c` is a consistency threshold

This rule differs from global temperature-style calibration because it does not modify every detection. It only modifies risky detections.

## 7. Calibration-guided pseudo labeling

One tested extension uses calibration signals to gate pseudo labels.

For a target detection `d`, define a keep indicator:

`g(d) = 1[p'(d) >= tau_p] * 1[c(d) >= tau_cons] * 1[r(d) <= tau_risk]`

Then:

`hat{y}^t = {d : g(d) = 1}`

This produces a filtered pseudo-label set.

In experiments, this was stable but did not improve over the simpler rank-calibrated baseline.

## 8. Adaptive pseudo thresholding

Another tested extension uses consistency to adapt the pseudo-label threshold per detection:

`tau_eff(d) = tau_0 + gamma * max(0, c_target - c(d))`

where:

- `tau_0` is the base pseudo threshold
- `c_target` is a target consistency level
- `gamma` is a penalty scale

The keep rule becomes:

`g(d) = 1[p'(d) >= tau_eff(d)]`

This increases the threshold for unstable detections while keeping stable detections near the original threshold.

In experiments, this also tied the simpler rank-calibrated baseline.

## 9. Why target-domain AP is the right model-selection criterion

An important empirical finding in this project is that source-domain validation AP can be misleading.

Some contrastive-pretraining branches produced acceptable source metrics:

- `source mAP50`
- `source mAP50-95`

but collapsed on the target domain.

Therefore, model selection must be driven by:

- target `AP50`
- target `AP50-95`
- reliability metrics (`ECE`, `NLL`, `Brier`)

This is also consistent with the critique raised by ALDI, which argues that DAOD progress has been distorted by weak baselines and inconsistent benchmarking ([arXiv](https://arxiv.org/abs/2403.12029)).

## 10. Final mathematical interpretation

The present project can be summarized as learning a target-robust detector through:

1. source supervised training:
   - `min_theta L_s(theta)`
2. pseudo-label self-training:
   - `min_theta L_s(theta) + lambda_t L_p(theta)`
3. calibration-aware inference:
   - compute `c(d)` and `r(d)`
   - transform `p(d)` into `p'(d)`
4. optional pseudo-label refinement:
   - keep pseudo labels using `p'(d)`, `c(d)`, and `r(d)`

Among all tested variants, the only component that produced a clear and reproducible improvement is the rank-based calibration transformation on target-domain detections.

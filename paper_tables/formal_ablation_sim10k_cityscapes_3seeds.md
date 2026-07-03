# SIM10K → Cityscapes 正式消融实验

统一设置：YOLO11n，1 轮自训练，每轮 3 epochs，随机种子 0/1/2。

| 方法 | Target mAP50 | Target mAP50-95 |
| --- | ---: | ---: |
| Baseline | 0.3301 ± 0.0683 | 0.1915 ± 0.0394 |
| Geometry quality | 0.3254 ± 0.0898 | 0.1843 ± 0.0515 |
| Small-object threshold | **0.4261 ± 0.0115** | **0.2390 ± 0.0101** |
| Geometry + small-object | 0.3564 ± 0.0094 | 0.2045 ± 0.0009 |

Small-object threshold 相对 baseline 提升 0.0959 mAP50 和 0.0475 mAP50-95，同时标准差明显更小。几何一致性单独使用没有稳定收益；与小目标阈值叠加后优于 baseline，但弱于仅使用小目标阈值，表明当前几何筛选过严。

## GPU 吞吐优化

校准推理由逐图逐视角调用改为按视角批量推理。64 张图短基准中，batch=1 为 6.36 images/s，batch=16 为 8.83 images/s，提升约 38.7%。后续实验默认训练 batch=24、workers=2、推理 batch=16；显存不足时调度器自动减半训练 batch。

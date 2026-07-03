# YOLO11 无监督域适应项目交接说明

更新时间：2026-07-02  
工作目录：`D:\yolo\yolov7-main-2026-6-19`

## 1. 项目目标与论文故事线

本项目希望讲述如下故事：

> 无人机在场景1采集有标注图像并训练检测器；到达场景2后，只有无标注图像。利用无监督域适应/自训练，将场景1模型快速迁移到场景2，重点解决航拍图像中的小目标和域偏移问题。

当前方法主线是：

1. 源域有监督训练；
2. 在无标注目标域生成伪标签；
3. 使用小目标自适应置信度阈值保留容易被统一阈值过滤的小目标；
4. 可选多视角几何一致性校准；
5. 可选 EMA teacher，在第2轮开始影响伪标签；
6. 混合源域真值与目标域伪标签进行自训练。

当前最值得继续验证的论文贡献点是“小目标自适应伪标签阈值”，而不是几何一致性。早期实验显示几何筛选容易过严。

## 2. 环境

- Conda 环境：`C:\Users\jiangyh\AppData\Local\miniforge3\envs\yolo11-uda`
- Python：3.11
- PyTorch：2.11.0+cu128
- Ultralytics：8.4.82
- GPU：NVIDIA GeForce RTX 5060 Laptop GPU，8 GB
- 系统内存：约 24 GB

常用 Python：

```powershell
$python = 'C:\Users\jiangyh\AppData\Local\miniforge3\envs\yolo11-uda\python.exe'
```

推荐硬件参数：

- train batch：24
- inference batch：16
- workers：1
- image size：640

原因：

- `workers=2` 会在 Windows 派生多个 Python 子进程，系统空闲内存曾降到 1.7 GB；
- `workers=0` 内存安全，但每组训练约慢 8%；
- `workers=1` 是当前速度与内存的折中；
- inference batch=16 实测 8.83 images/s，batch=32 反而只有 8.54 images/s。

## 3. 数据集与标准评价口径

### SIM10K → Cityscapes

- 源域：`benchmarks/sim10k_to_cityscapes/source_dataset`
- 目标域：`benchmarks/sim10k_to_cityscapes/target_dataset`
- 源 YAML：`benchmarks/sim10k_to_cityscapes/source_dataset/source_data.yaml`
- 目标验证 YAML：`benchmarks/sim10k_to_cityscapes/target_dataset/target_val.yaml`
- 标准口径：单类 `car`，报告 AP50，同时本项目也报告 AP50-95。

### Cityscapes → Foggy Cityscapes

- 源域：`benchmarks/cityscapes_to_foggy/source_dataset`
- 目标域：`benchmarks/cityscapes_to_foggy/target_dataset`
- 标准口径：8类：
  `person, rider, car, truck, bus, train, motorcycle, bicycle`
- 报告8类 mAP50，同时本项目也报告 mAP50-95。

## 4. 关键代码

- `self_training.py`
  - 自训练主流程；
  - 支持 test-time calibration、小目标伪标签、EMA teacher；
  - 已修复 EMA 生命周期：最终 Student 不会被 EMA 错误覆盖；
  - 支持 `--seed` 和 `--inference-batch-size`。

- `run_paper_experiments.py`
  - 正式实验调度；
  - 支持多随机种子、消融实验、断点汇总、跳过已完成实验；
  - 支持 `--initial-weights`，可让所有方法共享同一个 source-only 起点；
  - 支持 `small_object_ema`；
  - 每完成一组立即写入 CSV/JSON，外层超时后子进程可能继续运行。

- `train_source_only.py`
  - 新增的源域有监督训练入口；
  - 写出 `source_training_summary.json`。

- `modules/test_time_calibration.py`
  - 校准推理由逐图逐视角改为按视角批量推理；
  - 64张图基准：6.36 → 8.83 images/s，约提升38.7%。

- `modules/ultralytics_seed.py`
  - 极其重要：修复 Ultralytics 8.4.82 dataloader 随机种子不随实验 seed 变化的问题。

- `evaluate_detector.py`
  - 统一目标域评估并输出 JSON。

## 5. 重要随机种子问题

### 已发现的问题

Ultralytics 8.4.82 的 `build_dataloader` 使用：

```python
generator.manual_seed(6148914691236517205 + RANK)
```

该生成器没有使用 `model.train(seed=...)`。当 `workers=1` 时，seed0/1/2 曾产生完全相同的模型张量和完全一致的 loss 轨迹。

因此，修复前所谓的多随机种子严格实验不构成独立随机重复。

### 修复

`modules/ultralytics_seed.py` 在项目内 patch detection trainer 的 dataloader，使 seed 变为：

```python
6148914691236517205 + RANK + experiment_seed
```

没有修改 conda 的 site-packages。

### 修复验证

使用 seed1/seed2 各训练1 epoch：

- seed1 源域 mAP50：0.48975
- seed2 源域 mAP50：0.50287
- 两个模型张量 SHA256 不同；
- loss 轨迹不同。

验证目录：

- `runs/seed_patch_check/seed1`
- `runs/seed_patch_check/seed2`

从现在开始，正式多种子结果只能使用 seed patch 之后重新训练的模型。

## 6. 早期探索结果：仅供方向判断

以下结果大多来自“直接以 COCO YOLO11n 为起点、源域训练不足”的短程协议，或来自 seed patch 之前。它们可以说明方向，但不能作为最终论文主表。

### SIM10K → Cityscapes，早期3种子消融

文件：

- `paper_tables/formal_ablation_sim10k_cityscapes_3seeds.md`
- `paper_tables/formal_ablation_sim10k_cityscapes_3seeds.csv`

结果：

| 方法 | mAP50 | mAP50-95 |
| --- | ---: | ---: |
| baseline | 0.3301 ± 0.0683 | 0.1915 ± 0.0394 |
| geometry quality | 0.3254 ± 0.0898 | 0.1843 ± 0.0515 |
| small-object threshold | 0.4261 ± 0.0115 | 0.2390 ± 0.0101 |
| geometry + small | 0.3564 ± 0.0094 | 0.2045 ± 0.0009 |

方向性结论：仅 small-object threshold 最好；几何筛选过严。

### Cityscapes → Foggy，早期3种子

文件：

- `paper_tables/foggy_formal_and_sota_comparison.md`
- `paper_tables/foggy_formal_results_3seeds.csv`

结果：

| 方法 | mAP50 | mAP50-95 |
| --- | ---: | ---: |
| baseline | 0.1478 ± 0.0071 | 0.0873 ± 0.0048 |
| small offset=0.03 | 0.1739 ± 0.0091 | 0.1039 ± 0.0063 |
| small offset=0.05 | 0.1668 ± 0.0118 | 0.1008 ± 0.0074 |

方向性结论：

- offset=0.03 比0.05更稳定；
- 参数敏感性 seed0 中，offset=0.01/0.03/0.05 的 mAP50 为
  0.1399/0.1634/0.1686；
- 单种子看0.05最好，但三种子均值0.03更好，说明必须做真实随机重复。

## 7. 严格协议的单次探索结果

严格协议定义：

1. 先在 SIM10K 源域训练20 epochs；
2. 以同一 source-only 权重为起点；
3. 比较 source-only、1轮 baseline、2轮 baseline、2轮 small-object+EMA；
4. 每轮自训练3 epochs。

修复随机种子前的 seed0 与 patch 后 seed0 使用相同 generator offset，因此可作为 seed0 探索参考，但最终主表仍建议全部由新三套 source-only 权重重跑。

结果：

| 方法 | Target mAP50 | Target mAP50-95 |
| --- | ---: | ---: |
| source-only，20 epochs | 0.4599 | 0.2641 |
| 1轮 baseline | 0.5218 | 0.3012 |
| 2轮 baseline | 0.5243 | 0.3019 |
| 2轮 small-object+EMA | 0.5259 | 0.3001 |

结论：

- 充分训练 source-only 后，1轮普通自训练仍提升6.19个百分点 mAP50；
- 第2轮普通自训练只增加约0.25个百分点，基本饱和；
- small-object+EMA 相对2轮 baseline 只高约0.17个百分点，mAP50-95反而低约0.18个百分点；
- 单次结果不足以证明新方法优于强 baseline。

相关目录：

- source-only：`runs/source_only/sim10k_yolo11n_e20`
- 1轮 baseline：`runs/strict_protocol_sim10k_baseline1`
- 2轮 baseline：`runs/strict_protocol_sim10k_baseline2`
- 2轮 small+EMA：`runs/strict_protocol_sim10k_small_ema2`

small+EMA 的外层进程曾被电脑中断，但两轮训练均完整：

- Iteration 1：
  `runs/train/self_train_iter_1_1782863678_1782863885`
- Iteration 2：
  `runs/train/self_train_iter_2_1782863678_1782865366`
- 手工目标域评估：
  `runs/strict_protocol_sim10k_small_ema2/sim10k_to_cityscapes/small_object_ema/target_detection_metrics.json`

## 8. 当前真正有效的训练进度

随机种子 patch 后，正在重建三套独立 source-only 模型。

### 已完成

SIM10K source seed0，20 epochs：

- 目录：`runs/source_only_seeded/sim10k_seed0_e20`
- 权重：
  `runs/source_only_seeded/sim10k_seed0_e20/weights/best.pt`
- 源域最终 epoch20 mAP50：0.64989
- 源域最终 epoch20 mAP50-95：0.42452
- 训练时长：4160秒，约69分钟
- `source_training_summary.json` 已存在。

注意：该新 seed0 权重尚未重新做 Cityscapes 目标域 source-only 评估。

### 尚未完成

1. SIM10K source seed1，20 epochs；
2. SIM10K source seed2，20 epochs；
3. 三套 source-only 权重分别在 Cityscapes target val 上评估；
4. 对每个 source seed 配对运行：
   - 2轮 baseline；
   - 2轮 small-object+EMA；
5. 汇总真实的 mean ± std 和配对差值。

当前没有存活的训练进程。

## 9. 推荐继续执行的命令

### 9.1 训练 source seed1

```powershell
& $python train_source_only.py `
  --weights weights/yolo11n.pt `
  --data benchmarks/sim10k_to_cityscapes/source_dataset/source_data.yaml `
  --epochs 20 --batch-size 24 --img-size 640 `
  --device 0 --workers 1 --seed 1 `
  --project runs/source_only_seeded `
  --name sim10k_seed1_e20
```

### 9.2 训练 source seed2

```powershell
& $python train_source_only.py `
  --weights weights/yolo11n.pt `
  --data benchmarks/sim10k_to_cityscapes/source_dataset/source_data.yaml `
  --epochs 20 --batch-size 24 --img-size 640 `
  --device 0 --workers 1 --seed 2 `
  --project runs/source_only_seeded `
  --name sim10k_seed2_e20
```

### 9.3 评估 source-only

对每个 seed 执行：

```powershell
& $python evaluate_detector.py `
  --weights runs/source_only_seeded/sim10k_seed0_e20/weights/best.pt `
  --data benchmarks/sim10k_to_cityscapes/target_dataset/target_val.yaml `
  --output runs/source_only_seeded/sim10k_seed0_e20/target_detection_metrics.json `
  --img-size 640 --batch-size 24 --device 0 --workers 1
```

seed1/2 替换路径即可。

### 9.4 两轮 baseline

每个 source seed 单独运行，确保 `--initial-weights` 与 `--seeds` 一致：

```powershell
& $python run_paper_experiments.py `
  --python-exe $python --workdir . `
  --output-root runs/strict_seeded_sim10k_seed0_baseline2 `
  --suite ablation --benchmark sim10k_to_cityscapes `
  --seeds 0 `
  --initial-weights runs/source_only_seeded/sim10k_seed0_e20/weights/best.pt `
  --self-iterations 2 --self-epochs-per-iter 3 `
  --workers 1 --self-batch-size 24 --inference-batch-size 16 `
  --device 0 --include-experiments baseline
```

### 9.5 两轮 small-object+EMA

```powershell
& $python run_paper_experiments.py `
  --python-exe $python --workdir . `
  --output-root runs/strict_seeded_sim10k_seed0_small_ema2 `
  --suite ablation --benchmark sim10k_to_cityscapes `
  --seeds 0 `
  --initial-weights runs/source_only_seeded/sim10k_seed0_e20/weights/best.pt `
  --self-iterations 2 --self-epochs-per-iter 3 `
  --workers 1 --self-batch-size 24 --inference-batch-size 16 `
  --device 0 --include-experiments small_object_ema
```

seed1/2 分别替换：

- `--seeds`
- `--initial-weights`
- `--output-root`

## 10. 运行与中断注意事项

1. 单次工具调用常在3600秒超时，但 Windows 子进程通常继续运行。
2. 超时后先检查：

```powershell
Get-CimInstance Win32_Process |
  Where-Object {$_.CommandLine -match 'train_source_only.py|self_training.py|run_paper_experiments.py'}
```

3. 不要因为外层超时立即重启同一实验。
4. 检查 `results.csv` 是否跑满目标 epoch，以及 `source_training_summary.json` 或 `experiment_summary.csv` 是否生成。
5. `run_paper_experiments.py` 已支持每组即时落盘和跳过已完成组。
6. 电脑断电后，如果两轮训练的 `results.csv` 都完整，可手工用 `evaluate_detector.py` 评估最终 Student `best.pt`，不必重训。
7. 最终模型应使用 Student；EMA teacher 只负责下一轮伪标签，不应覆盖最终 Student。

## 11. SOTA 调研定位

标准 DAOD 论文常见指标（mAP@0.5）：

- CSDA，ICCV 2023：Cityscapes→Foggy 45.3；
- NSA，ICCV 2023：Foggy 52.7，SIM10K→Cityscapes 56.3；
- CAT，CVPR 2024：Foggy 52.5；
- BlenDA，2024：Foggy 53.4；
- SEEN-DA，CVPR 2025：Foggy 57.5，SIM10K→Cityscapes 66.8；
- ALDI++，TMLR 2025：现代公平协议下更高，但其 backbone、预训练和训练预算与本项目差别很大。

参考链接：

- CAT：
  `https://openaccess.thecvf.com/content/CVPR2024/papers/Kennerley_CAT_Exploiting_Inter-Class_Dynamics_for_Domain_Adaptive_Object_Detection_CVPR_2024_paper.pdf`
- SEEN-DA：
  `https://openaccess.thecvf.com/content/CVPR2025/papers/Li_SEEN-DA_SEmantic_ENtropy_guided_Domain-aware_Attention_for_Domain_Adaptive_Object_Detection_CVPR_2025_paper.pdf`
- ALDI++：
  `https://openreview.net/pdf?id=ssXSrZ94sR`

不要将当前 YOLO11n、20 epoch source + 2×3 epoch adaptation 与8张V100、RegionCLIP/Faster R-CNN、数万 iteration 的论文直接做绝对公平排名。论文应重点报告：

1. 同代码、同起点、同预算的配对增益；
2. 轻量模型参数量、训练时间、推理速度；
3. 标准 benchmark 绝对指标作为定位；
4. 与 SOTA 的训练预算和 backbone 差异。

## 12. 未来方向，按优先级排序

### P0：完成真实三种子严格协议

这是当前最重要的工作。完成前不要写“新方法显著优于 baseline”。

### P1：分析新方法为何只略高于强 baseline

建议统计每轮：

- 伪标签总数；
- 小目标伪标签数量及保留率；
- 不同置信区间的伪标签数量；
- 第1轮到第2轮标签新增/消失；
- 按目标尺寸划分 AP_small/AP_medium/AP_large；
- precision/recall 变化。

如果提升主要来自小目标，论文故事会更扎实；如果整体指标持平但 AP_small 明显提升，也可能形成有效贡献。

### P2：将 EMA 从“轮次级”改为“step级 online teacher”

当前 EMA 只在一轮训练完成后合并一次，远弱于主流 mean-teacher：

- teacher 每个 optimizer step 更新；
- teacher 对弱增强目标图生成伪标签；
- student 对强增强图学习；
- teacher/student 同时存在于训练循环。

这是接近 SOTA 的关键工程方向。

### P3：增加 weak/strong augmentation 一致性

主流 DAOD 常使用 teacher weak view + student strong view。当前项目主要依赖离线伪标签和多视角校准，训练期一致性不足。

### P4：更强 source baseline

可尝试：

- YOLO11s 而非 YOLO11n；
- 50–100 source epochs；
- cosine LR、早停；
- 更高分辨率，例如960，重点改善航拍小目标；
- 但必须先保持当前轻量协议完成三种子，不能边跑边换配置。

### P5：复制严格协议到 Foggy

SIM10K严格三种子完成并确认方法有效后，再对 Cityscapes→Foggy 复制：

1. 训练三套 Cityscapes source-only；
2. source-only Foggy 目标评估；
3. 两轮 baseline；
4. 两轮 small-object+EMA；
5. 8类分项 AP 与尺寸分项分析。

### P6：无人机场景数据

现有 benchmark 是论文验证，不是真正无人机数据。最终故事建议增加：

- VisDrone 或 UAVDT 作为无人机公开数据；
- 自采场景1/场景2；
- 天气、光照、高度、视角中的至少一种域偏移；
- 少量人工标注的目标域 test set，仅用于评估，不能参与训练。

## 13. 交接给下一智能体的第一项动作

先训练：

1. `runs/source_only_seeded/sim10k_seed1_e20`
2. `runs/source_only_seeded/sim10k_seed2_e20`

然后评估三套 source-only 目标域指标，再逐 seed 配对运行2轮 baseline与2轮 small-object+EMA。不要继续使用 `runs/strict_protocol_sim10k_baseline2_multiseed` 的 seed1/2 作为随机重复，因为它们是在 dataloader seed 修复前生成的、模型张量完全相同。


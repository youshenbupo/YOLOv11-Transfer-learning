# Cityscapes → Foggy Cityscapes 正式实验与 SOTA 定位

## 本项目正式结果

统一设置：YOLO11n，1 轮自训练，每轮 3 epochs，train batch=24，workers=1，inference batch=16，随机种子 0/1/2。Foggy Cityscapes 按标准 8 类、mAP@0.5 评估。

| 方法 | Target mAP50 | Target mAP50-95 |
| --- | ---: | ---: |
| Self-training baseline | 0.1478 ± 0.0071 | 0.0873 ± 0.0048 |
| Small-object threshold，offset=0.03 | **0.1739 ± 0.0091** | **0.1039 ± 0.0063** |
| Small-object threshold，offset=0.05 | 0.1668 ± 0.0118 | 0.1008 ± 0.0074 |

offset=0.03 相对 baseline 提升 0.0260 mAP50 和 0.0166 mAP50-95。三个配对种子均为正提升；配对 t 检验为 t=5.28、p=0.034，但 n=3 很小，只应作为支持性证据。

## 参数敏感性

seed0 的 offset=0.01/0.03/0.05 mAP50 分别为 0.1399/0.1634/0.1686。单次实验看似 0.05 最好，但三种子均值显示 0.03 更高且更稳定，因此主方法继续采用 0.03。

## 标准 DAOD 论文指标

下表均为论文报告的 mAP@0.5。不同 detector、backbone、预训练和训练预算不能直接与本项目的轻量 YOLO11n、3 epochs 数字进行公平排名。

| 方法 | 年份/出处 | Cityscapes→Foggy，8 类 | SIM10K→Cityscapes，car |
| --- | --- | ---: | ---: |
| CSDA | ICCV 2023 | 45.3 | - |
| NSA | ICCV 2023 | 52.7 | 56.3 |
| CAT | CVPR 2024 | 52.5 | 未报告 |
| BlenDA | 2024 | 53.4 | 未报告 |
| SEEN-DA | CVPR 2025 | **57.5** | **66.8** |
| ALDI++，现代公平协议 | TMLR 2025 | R50-FPN 约 66.8；ViT-L 76.1 | 论文报告相对公平 baseline 提升 5.7 |

SEEN-DA 使用 RegionCLIP/Faster R-CNN，并在 8 张 V100 上训练；CAT 使用长达 80,000 iterations 的 teacher-student 流程。ALDI++ 进一步指出传统 DAOD 工作存在弱 baseline 和不统一实现的问题，因此论文中应同时报告“同代码同预算 baseline 增益”和“标准 benchmark 绝对指标”，不要只比较绝对数值。

## 当前结论

当前方法在两个迁移任务都取得稳定收益：

- SIM10K→Cityscapes：mAP50 从 0.3301 提升到 0.4261，提升 0.0959。
- Cityscapes→Foggy：mAP50 从 0.1478 提升到 0.1739，提升 0.0260。

跨数据集收益支持“小目标自适应伪标签阈值”具有一定泛化性，但绝对性能与 SOTA 仍有明显差距。主要原因是训练预算极低、初始模型不是完整 source-domain detector、只有一轮伪标签更新，并且尚未使用在线 EMA teacher、强弱增强和长程训练。

## 推荐的下一轮实验

使用 offset=0.03，先补 source-only 充分训练模型，再进行 2 轮 teacher-student 自训练；比较：

1. source-only；
2. 1 轮 self-training baseline；
3. 2 轮 baseline；
4. 2 轮 small-object + EMA teacher。

该实验能区分当前收益究竟来自小目标伪标签策略，还是仅来自额外训练轮次，也是与主流 mean-teacher SOTA 方法接轨的关键一步。

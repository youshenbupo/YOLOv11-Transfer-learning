# YOLO11-UDA: 基于对比学习与自训练的无监督域适应目标检测

## 1. 研究背景与动机

### 1.1 问题定义

目标检测模型在源域（有标注数据）上训练后，部署到目标域（无标注数据）时性能显著下降。这种域差距（Domain Gap）来源于：

- **外观差异**：光照、天气、相机参数不同
- **几何差异**：视角、尺度、分辨率不同
- **分布差异**：目标类别比例、背景复杂度不同

**无监督域适应（UDA）** 的目标：在目标域无标注的情况下，缩小域差距，提升目标域检测性能。

### 1.2 现有方法的局限

| 方法类别 | 代表工作 | 局限 |
|---------|---------|------|
| 域对抗训练 | DAFaster, SWDA | 训练不稳定，特征对齐粒度粗 |
| 伪标签自训练 | Mean Teacher, MTTrans | 伪标签噪声累积，误差传播 |
| 图像风格迁移 | CycleGAN-based | 计算开销大，可能引入伪影 |
| 知识蒸馏 | 多教师蒸馏 | 依赖教师模型质量 |

### 1.3 我们的动机

现有方法的共同问题：**缺乏对目标域特征的有效预学习**。

我们的思路：先通过对比学习让模型"认识"目标域的特征分布，再通过自训练"学会"在目标域上检测。两阶段互补，形成完整闭环。

---

## 2. 方法概述：YOLO11-UDA

### 2.1 整体框架

```
┌─────────────────────────────────────────────────────┐
│                    YOLO11-UDA 框架                    │
├─────────────────────────────────────────────────────┤
│                                                      │
│  Stage 1: 对比学习预训练 (Contrastive Pretraining)    │
│  ┌───────────┐    ┌───────────┐    ┌──────────────┐ │
│  │ 源域图像   │    │ 目标域图像 │    │ YOLO11       │ │
│  │ (有标签)   │    │ (无标签)   │    │ Backbone+Neck│ │
│  └─────┬─────┘    └─────┬─────┘    └──────┬───────┘ │
│        │                │                  │         │
│        └────────┬───────┘                  │         │
│                 ▼                          ▼         │
│        ┌───────────────┐         ┌──────────────┐   │
│        │ 数据增强 (×2)  │         │ ProjectionHead│   │
│        │ 两个视角       │         │ (128-dim)     │   │
│        └───────┬───────┘         └──────┬───────┘   │
│                │                         │           │
│                ▼                         ▼           │
│        ┌──────────────────────────────────────┐     │
│        │     SimCLR / MoCo 对比损失            │     │
│        │     InfoNCE / Momentum Contrast       │     │
│        └──────────────────────────────────────┘     │
│                        │                            │
│                        ▼                            │
│              预训练权重 (backbone+neck)              │
│                                                      │
├─────────────────────────────────────────────────────┤
│                                                      │
│  Stage 2: 自训练域适应 (Self-Training Adaptation)    │
│  ┌──────────────────────────────────────────────┐   │
│  │         迭代循环 (Iteration 1..N)              │   │
│  │  ┌─────────────────────────────────────────┐ │   │
│  │  │ 1. 推理: 预训练模型 → 目标域无标签数据    │ │   │
│  │  │ 2. 筛选: 高置信度检测 → 伪标签            │ │   │
│  │  │ 3. 混合: 源域真标签 + 目标域伪标签        │ │   │
│  │  │ 4. 微调: 在混合数据集上训练 YOLO11        │ │   │
│  │  │ 5. 更新: 阈值递增，进入下一轮              │ │   │
│  │  └─────────────────────────────────────────┘ │   │
│  └──────────────────────────────────────────────┘   │
│                        │                            │
│                        ▼                            │
│              最终域适应检测模型                       │
│                                                      │
└─────────────────────────────────────────────────────┘
```

### 2.2 Stage 1: 对比学习预训练

**核心思想**：将 YOLO11 的 backbone+neck 作为特征提取器，通过对比学习让模型在特征空间中拉近源域和目标域的同类特征，推远不同类特征。

**关键组件**：

| 组件 | 说明 |
|------|------|
| `ContrastiveYOLO` | YOLO11 backbone+neck + ProjectionHead 的组合体 |
| `ProjectionHead` | 2层 MLP (2048→256→128)，将检测特征映射到对比学习空间 |
| `info_nce_loss` | SimCLR 风格的 InfoNCE 损失 |
| `moco_loss` | MoCo 风格的动量对比损失 |
| `CombinedDataset` | 合并正样本（目标域）和负样本（源域）的数据集 |
| `RatioBatchSampler` | 保证每个 batch 中正负样本比例固定 |

**冻结策略**：

| 策略 | 说明 | 适用场景 |
|------|------|---------|
| `backbone` | 冻结 backbone，训练 neck + projection head | 域差距小 |
| `neck` | 冻结 neck，训练 backbone + projection head | 需要调整特征提取 |
| `both` | neck 始终冻结，backbone 分阶段解冻 | 域差距大，推荐 |
| `none` | 全部训练 | 域差距很大 |

**分阶段解冻**：
```
Stage 1: 冻结 backbone，训练 neck + projection head  [4 epochs]
Stage 2: 逐步解冻 backbone (20层 → 10层 → 0层)       [2 epochs each]
         每阶段学习率降低 10 倍
```

### 2.3 Stage 2: 自训练域适应

**核心思想**：用预训练模型在目标域无标签数据上生成伪标签，与源域真标签混合，迭代微调。

**动态阈值策略**：
- 初始阈值 0.7，每轮递增 0.1
- 高阈值保证伪标签质量，逐步提升覆盖度

**迭代流程**：
```
Iteration 1..N:
  1. 推理: 用当前模型对目标域无标签数据预测 (conf=0.01)
  2. 筛选: 高置信度阈值生成伪标签 (动态递增)
  3. 混合: 源域真实标签 + 目标域伪标签 → 混合数据集
  4. 微调: 在混合数据上训练若干轮
  5. 迭代: 更新模型，进入下一轮
```

---

## 3. YOLO11 架构分析与迁移方案

### 3.1 YOLO11 vs YOLOv7 关键差异

| 方面 | YOLOv7 | YOLO11 (Ultralytics) |
|------|--------|---------------------|
| 模型加载 | `from models.yolo import Model; Model(cfg)` | `from ultralytics import YOLO; YOLO("yolo11n.pt")` |
| 架构 | 平铺 `nn.Sequential`，anchor-based | 平铺结构，**anchor-free** |
| 检测头 | `IDetect` / `IAuxDetect` | `Detect` (解耦头) |
| 配置格式 | 用户可编辑 YAML | Ultralytics 内部管理 |
| 训练 API | `python train.py --cfg ...` | `model.train(data=..., epochs=...)` |
| 推理 API | `python detect.py --weights ...` | `model.predict(source=..., save_txt=True)` |
| DDP | 手动 `torch.distributed` | Ultralytics 内部处理 |

### 3.2 迁移策略

**核心原则**：保持 UDA 算法逻辑不变，只替换模型加载和接口调用。

**迁移层次**：

```
┌─────────────────────────────────────┐
│         UDA 算法层 (保持不变)         │  对比损失、数据集、DDP工具
├─────────────────────────────────────┤
│         适配层 (需要重写)             │  model.py, freeze.py
├─────────────────────────────────────┤
│         模型接口层 (需要替换)         │  YOLOv7 → Ultralytics
└─────────────────────────────────────┘
```

### 3.3 YOLO11 Block 结构分析

YOLO11 的 `model.model` 是一个 `nn.Sequential`，需要确定：
- backbone 结束位置（用于冻结策略）
- neck 开始位置
- Detect 头位置

```python
# 分析 YOLO11 结构的代码
from ultralytics import YOLO
model = YOLO("yolo11n.pt").model
for i, m in enumerate(model.model):
    print(f"Block {i}: {m.__class__.__name__}")
```

---

## 4. 创新点规划

### 4.1 创新点 1: 注意力增强的对比学习 (Attention-Enhanced Contrastive Learning)

**动机**：标准对比学习对所有特征一视同仁，但检测任务中目标区域的特征更重要。

**方案**：
- 在 YOLO11 的 backbone 中插入 **CBAM（Convolutional Block Attention Module）**
- 对比学习阶段，利用注意力图加权特征，让模型关注目标区域
- 公式：`z = AttentionWeightedPool(features)` 而非简单的全局池化

```python
class CBAM(nn.Module):
    """Channel + Spatial Attention"""
    def __init__(self, channels, reduction=16):
        super().__init__()
        self.channel_att = ChannelAttention(channels, reduction)
        self.spatial_att = SpatialAttention()

    def forward(self, x):
        x = self.channel_att(x) * x
        x = self.spatial_att(x) * x
        return x
```

### 4.2 创新点 2: 域感知对比损失 (Domain-Aware Contrastive Loss)

**动机**：标准 InfoNCE 只区分正负样本对，不考虑域信息。

**方案**：
- 设计新的对比损失，同时考虑**实例级相似性**和**域级差异性**
- 引入域间对比项：拉近同一实例在不同域的特征，推远不同实例

```python
def domain_aware_contrastive_loss(features, domain_labels, temperature=0.1):
    """
    L = L_instance + λ * L_domain
    - L_instance: 标准 InfoNCE (同一样本的不同增强视角)
    - L_domain: 域间对比 (同一类别在不同域的特征拉近)
    """
    L_instance = info_nce_loss(features, temperature)
    L_domain = domain_contrastive(features, domain_labels, temperature)
    return L_instance + 0.5 * L_domain
```

### 4.3 创新点 3: 渐进式伪标签精炼 (Progressive Pseudo-Label Refinement)

**动机**：传统自训练使用固定或线性递增的置信度阈值，不够精细。

**方案**：
- 引入**伪标签质量评估器**：不只看置信度，还看空间一致性（相邻迭代的检测框是否稳定）
- **课程学习策略**：先学简单样本（高置信度+高一致性），逐步引入困难样本
- **伪标签加权**：根据质量分数对伪标签加权，而非二值筛选

```python
def refine_pseudo_labels(current_preds, previous_preds, conf_threshold):
    """
    结合置信度和时间一致性筛选伪标签
    - 高置信度 + 高一致性 → 可靠伪标签
    - 高置信度 + 低一致性 → 待验证
    - 低置信度 → 丢弃
    """
    conf_mask = current_preds[:, 4] > conf_threshold
    consistency = compute_iou_consistency(current_preds, previous_preds)
    reliable = conf_mask & (consistency > 0.7)
    return current_preds[reliable]
```

### 4.4 创新点 4: 多尺度特征对齐 (Multi-Scale Feature Alignment)

**动机**：不同尺度的特征图承载不同粒度的域信息。

**方案**：
- 对 P3/P4/P5 三个尺度分别进行对比学习
- 不同尺度使用不同的 ProjectionHead 和温度系数
- 小尺度（P3）关注局部纹理域差异，大尺度（P5）关注全局语义域差异

---

## 5. 实验设计

### 5.1 数据集

| 场景 | 源域 | 目标域 | 域差距 |
|------|------|--------|--------|
| 天气迁移 | 正常天气 (BDD100K) | 雨天/雾天/雪天 | 小-中 |
| 传感器迁移 | 普通相机 (COCO) | 红外/无人机视角 | 中-大 |
| 场景迁移 | 白天城市 | 夜间/乡村 | 大 |
| 跨数据集 | COCO | VOC / 自建数据集 | 可控 |

### 5.2 实验矩阵

#### 实验 1: 消融实验（核心）

| 编号 | 方法 | 对比学习 | 自训练 | 注意力模块 | 域感知损失 |
|------|------|---------|--------|-----------|-----------|
| E1 | Source Only (Baseline) | - | - | - | - |
| E2 | 仅自训练 | - | ✓ | - | - |
| E3 | 仅对比学习 | ✓ | - | - | - |
| E4 | 对比学习 + 自训练 | ✓ | ✓ | - | - |
| E5 | + 注意力模块 | ✓ | ✓ | ✓ | - |
| E6 | + 域感知损失 | ✓ | ✓ | ✓ | ✓ |

#### 实验 2: 对比学习策略对比

| 变量 | 配置 |
|------|------|
| 方法 | SimCLR vs MoCo |
| 冻结策略 | backbone / neck / both / none |
| 温度系数 | 0.05, 0.1, 0.2, 0.5 |
| 分阶段解冻 | 是 vs 否 |

#### 实验 3: 自训练策略对比

| 变量 | 配置 |
|------|------|
| 迭代次数 | 1, 2, 3, 5 |
| 阈值策略 | 固定 / 线性递增 / 课程学习 |
| 初始阈值 | 0.5, 0.6, 0.7, 0.8 |
| 伪标签精炼 | 无 / 时间一致性 / 加权 |

#### 实验 4: 与 SOTA 方法对比

| 基线方法 | 说明 |
|---------|------|
| Source Only | 直接迁移，无适应 |
| DAFaster | 域对抗 Faster R-CNN |
| SWDA | 强弱域适应 |
| Mean Teacher | 半学习 |
| MTTrans | Transformer + 伪标签 |
| 原始 YOLO11 + 微调 | 朴素迁移 |

### 5.3 评估指标

| 指标 | 说明 | 优先级 |
|------|------|--------|
| mAP@0.5 | 主要指标 | 必须 |
| mAP@0.5:0.95 | COCO 标准 | 必须 |
| 每类 AP | 分析各类别适应效果 | 建议 |
| FPS | 推理速度，保持实时性 | 建议 |
| 伪标签准确率 | 自训练质量 | 建议 |
| 参数量/FLOPs | 模型复杂度 | 可选 |

### 5.4 可视化实验

| 内容 | 方法 |
|------|------|
| 特征分布 | t-SNE 可视化源域/目标域特征（对比学习前后） |
| 注意力图 | 可视化 CBAM 的 channel/spatial attention |
| 检测结果 | 源域模型 vs UDA 模型在目标域的检测对比 |
| 损失曲线 | 对比学习 loss、自训练 mAP 随迭代变化 |
| 伪标签演变 | 各迭代轮次的伪标签质量变化 |

---

## 6. 论文写作框架

### 6.1 标题建议

- "YOLO11-UDA: Unsupervised Domain Adaptation for Real-Time Object Detection via Contrastive Learning and Self-Training"
- "Attention-Enhanced Domain Adaptive YOLO for Cross-Domain Object Detection"

### 6.2 论文结构

```
Abstract
1. Introduction
   - 问题定义与动机
   - 现有方法局限
   - 我们的贡献（3点）
2. Related Work
   - 目标检测 (YOLO 系列)
   - 无监督域适应
   - 对比学习
3. Method
   3.1 整体框架概述
   3.2 对比学习预训练 (Stage 1)
       - 注意力增强的对比学习
       - 域感知对比损失
   3.3 渐进式自训练域适应 (Stage 2)
       - 伪标签精炼
       - 课程学习策略
4. Experiments
   4.1 实验设置
   4.2 与 SOTA 对比
   4.3 消融实验
   4.4 可视化分析
5. Conclusion
```

### 6.3 预期贡献点

1. **统一框架**：首次将对比学习预训练 + 自训练域适应统一到实时检测器 YOLO11 上
2. **注意力增强对比学习**：引入 CBAM 让对比学习关注目标区域
3. **域感知对比损失**：设计同时考虑实例级和域级的对比损失
4. **渐进式伪标签精炼**：基于时间一致性的伪标签质量评估

---

## 7. 依赖环境

```bash
# Python >= 3.10
conda create -n yolo11-uda python=3.10
conda activate yolo11-uda

# PyTorch (CUDA 11.8)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118

# Ultralytics (YOLO11)
pip install ultralytics

# 其他依赖
pip install matplotlib numpy opencv-python Pillow PyYAML requests scipy tqdm
pip install tensorboard pandas seaborn thop
pip install scikit-learn pycocotools  # 评估与可视化
```

---

## 8. 时间线

| 阶段 | 任务 | 预计时间 |
|------|------|---------|
| Phase 1 | 代码迁移 + 环境搭建 | 1 周 |
| Phase 2 | 基础实验 (Baseline + 对比学习 + 自训练) | 2 周 |
| Phase 3 | 创新模块实现 (注意力 + 域感知损失 + 伪标签精炼) | 2 周 |
| Phase 4 | 完整实验 + 消融实验 | 2 周 |
| Phase 5 | 论文撰写 + 绘图 | 2 周 |
| **总计** | | **~9 周** |

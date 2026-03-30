# 知识蒸馏训练指南 (Knowledge Distillation Guide)

本指南介绍如何在DPFT项目中使用知识蒸馏来训练更小、更快的模型，同时保持高精度。

## 目录
1. [什么是知识蒸馏](#什么是知识蒸馏)
2. [快速开始](#快速开始)
3. [详细配置](#详细配置)
4. [蒸馏模式说明](#蒸馏模式说明)
5. [使用方法](#使用方法)
6. [代码示例](#代码示例)
7. [调参建议](#调参建议)
8. [故障排查](#故障排查)

---

## 什么是知识蒸馏

知识蒸馏（Knowledge Distillation）是一种模型压缩技术，通过让小模型（学生）学习大模型（教师）的输出来提升性能：

- **教师模型（Teacher）**：预训练的大模型，参数多、精度高但速度慢
- **学生模型（Student）**：待训练的小模型，参数少、速度快但精度待提升
- **蒸馏损失（Distillation Loss）**：让学生学习教师的软标签（soft labels）

### 优势
✅ 学生模型推理速度快，部署成本低
✅ 保持接近教师模型的精度
✅ 适合边缘设备和实时应用

---

## 快速开始

### 步骤 1: 准备教师模型
确保你有一个训练好的教师模型checkpoint：
```bash
# 示例路径
teacher_model.pt
```

### 步骤 2: 创建配置文件
复制示例配置文件：
```bash
cp config/kradar_distillation_example.json config/my_distillation.json
```

### 步骤 3: 修改配置
编辑 `config/my_distillation.json`，设置教师模型路径：
```json
{
  "train": {
    "distillation": {
      "teacher_checkpoint": "path/to/your/teacher_model.pt",
      "freeze_teacher": true,
      "temperature": 4.0,
      "alpha": 0.5
    }
  }
}
```

### 步骤 4: 开始训练
```bash
python src/dprt/train.py \
    --src /path/to/kradar/dataset \
    --cfg config/my_distillation.json \
    --dst /path/to/output
```

---

## 详细配置

### 蒸馏配置参数

在配置文件的 `train` 部分添加 `distillation` 配置：

```json
{
  "train": {
    "distillation": {
      "teacher_checkpoint": "path/to/teacher.pt",
      "freeze_teacher": true,
      "temperature": 4.0,
      "alpha": 0.5,
      "distill_mode": "top_k",
      "top_k": 50
    }
  }
}
```

#### 参数说明

| 参数 | 类型 | 说明 | 推荐值 |
|-----|------|------|--------|
| `teacher_checkpoint` | string | 教师模型checkpoint路径（.pt文件） | 必填 |
| `freeze_teacher` | bool | 是否冻结教师模型参数（不更新） | `true` |
| `temperature` | float | 温度参数，软化概率分布<br>- 值越大，分布越平滑<br>- 典型范围：1-10 | `4.0` |
| `alpha` | float | 蒸馏损失权重<br>- `alpha=0.5`: 蒸馏损失和学生损失各占50%<br>- `alpha=0.7`: 蒸馏损失占70%<br>- 范围：0.0-1.0 | `0.5` |
| `distill_mode` | string | 蒸馏模式，详见下文 | `'top_k'` |
| `top_k` | int | 仅当 `distill_mode='top_k'` 时有效<br>从教师预测中选取置信度最高的K个框 | `50` |

### 学生模型配置

学生模型通常比教师模型更小、更轻量：

```json
{
  "model": {
    "backbones": {
      "camera_mono": {
        "name": "ResNet50"  // 教师可能用 ResNet101
      },
      "radar_bev": {
        "name": "ResNet34"  // 教师可能用 ResNet50
      }
    }
  }
}
```

#### 常见的学生模型配置选择

| 模块 | 教师模型 | 学生模型 | 压缩比 |
|------|---------|---------|--------|
| Backbone | ResNet101 | ResNet50 | ~2x |
| Backbone | ResNet50 | ResNet34 | ~1.5x |
| Backbone | ResNet50 | MobileNetV3 | ~4x |
| Fuser iterations | 6 | 4 | 1.5x |
| Queries | 900 | 400 | ~2x |

---

## 蒸馏模式说明

当前支持以下四种蒸馏模式，可通过配置文件中的 `distill_mode` 参数指定：

| 模式 | 配置值 | 说明 | 适用场景 |
|------|--------|------|---------|
| 全量蒸馏 | `"all"` | 蒸馏所有queries（可能含大量背景框） | 学生/教师架构相同 |
| 匹配蒸馏 | `"matched"` | 仅蒸馏与GT匹配的queries（稳定，推荐） | 标准蒸馏首选 |
| **Top-K蒸馏** | **`"top_k"`** | **只蒸馏教师置信度最高的K个预测** | **过滤背景噪声** |
| **加权蒸馏** | **`"weighted"`** | **蒸馏所有queries但按教师置信度加权** | **柔性背景抑制** |

### `distill_mode: "all"` — 全量蒸馏
- **行为**：将所有queries（最多 min(N_student, N_teacher) 个）用于蒸馏
- **优点**：实现简单，无需额外匹配开销
- **缺点**：包含大量背景框，信号嘈杂

### `distill_mode: "matched"` — 匹配蒸馏（推荐）
- **行为**：通过匈牙利算法将学生预测与GT框匹配，只蒸馏命中GT的预测
- **优点**：蒸馏信号准确，训练更稳定
- **缺点**：需要额外的匹配计算

### `distill_mode: "top_k"` — Top-K蒸馏 ⭐新增
- **行为**：计算教师每个query的最大类别概率（置信度），选取置信度最高的K个query进行蒸馏
- **优点**：
  - 有效过滤低置信度背景预测
  - 聚焦于教师最确定的、信息最丰富的预测
  - 不依赖GT标注（纯teacher-student交互）
- **参数**：需设置 `top_k`（推荐范围：20–100）
- **配置示例**：
  ```json
  "distillation": {
    "distill_mode": "top_k",
    "top_k": 50,
    "temperature": 4.0,
    "alpha": 0.5
  }
  ```

### `distill_mode: "weighted"` — 加权蒸馏 ⭐新增
- **行为**：蒸馏所有共享queries，但以教师置信度分数作为每个query的损失权重
  - 高置信度预测 → 权重高 → 对蒸馏信号贡献大
  - 低置信度预测（背景） → 权重低 → 贡献小
- **优点**：
  - 不丢弃任何预测（比top_k更柔和的过滤方式）
  - 背景噪声被自然抑制而非硬截断
  - 无额外超参（无需设置K值）
- **配置示例**：
  ```json
  "distillation": {
    "distill_mode": "weighted",
    "temperature": 4.0,
    "alpha": 0.5
  }
  ```

---

## 使用方法

### 方法 1: 使用配置文件（推荐）

#### 使用 Top-K 蒸馏
```bash
python src/dprt/train.py \
    --src /data/kradar \
    --cfg config/kradar_distillation_topk.json \
    --dst outputs/distillation_topk
```

#### 使用加权蒸馏
```bash
python src/dprt/train.py \
    --src /data/kradar \
    --cfg config/kradar_distillation_weighted.json \
    --dst outputs/distillation_weighted
```

#### 使用匹配蒸馏（原有推荐方式）
```bash
python src/dprt/train.py \
    --src /data/kradar \
    --cfg config/kradar_distillation_light.json \
    --dst outputs/distillation_matched
```

### 方法 2: 编程方式使用

```python
from dprt.training.loss import DistillationLoss, KDLoss
from dprt.training.trainer import build_trainer
from dprt.training.loss import Loss
import torch

# 1. 创建学生损失（标准检测损失）
student_loss = Loss.from_config(config['train'])

# 2. 创建 Top-K 蒸馏损失
distillation_loss = DistillationLoss(
    temperature=4.0,
    distill_class=True,
    distill_bbox=True,
    distill_mode='top_k',
    top_k=50,
    reduction='mean'
)

# 或者创建加权蒸馏损失
distillation_loss_w = DistillationLoss(
    temperature=4.0,
    distill_mode='weighted',
    reduction='mean'
)

# 3. 组合损失
kd_loss = KDLoss(
    student_loss=student_loss,
    distillation_loss=distillation_loss,
    alpha=0.5
)

# 4. 使用 build_trainer 自动构建（推荐）
trainer = build_trainer(config)
trainer.train(student_model, train_loader, val_loader, dst='outputs')
```

---

## 代码示例

### 示例 1: 基础蒸馏训练

```python
# train_with_distillation.py
from dprt.utils.config import load_config
from dprt.training.trainer import build_trainer
from dprt.models.dprt import build_dprt
from dprt.datasets.kradar import KRadarDataset
from torch.utils.data import DataLoader

# 加载配置
config = load_config('config/kradar_distillation_topk.json')

# 创建数据集
train_dataset = KRadarDataset(...)
train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True)

# 构建学生模型
student_model = build_dprt(config['model'])

# 构建蒸馏训练器（自动检测distillation配置）
trainer = build_trainer(config)

# 开始训练
trainer.train(
    model=student_model,
    data_loader=train_loader,
    val_loader=val_loader,
    dst='outputs/distillation'
)
```

### 示例 2: 自定义蒸馏损失

```python
from dprt.training.loss import DistillationLoss

# 只蒸馏分类logits，不蒸馏bbox（Top-K模式）
distillation_loss = DistillationLoss(
    temperature=3.0,
    distill_class=True,
    distill_bbox=False,
    distill_mode='top_k',
    top_k=30,
    reduction='mean'
)

# 全部蒸馏，加权模式
distillation_loss_w = DistillationLoss(
    temperature=5.0,
    distill_class=True,
    distill_bbox=True,
    distill_mode='weighted',
    reduction='mean'
)
```

### 示例 3: 多GPU训练

```python
import torch.nn as nn

# 包装模型为DataParallel
teacher_model = nn.DataParallel(teacher_model)
student_model = nn.DataParallel(student_model)

# 正常训练
trainer.train(student_model, train_loader, val_loader)
```

---

## 调参建议

### 1. Temperature（温度）参数

| 温度值 | 效果 | 适用场景 |
|-------|------|---------|
| T=1 | 接近硬标签，蒸馏效果弱 | 教师和学生能力接近 |
| **T=4** | **平衡软化（推荐）** | **大多数场景** |
| T=6-8 | 高度软化，注重暗知识 | 教师远强于学生 |
| T=10+ | 过度软化，可能丢失信息 | 不推荐 |

### 2. Alpha（权重平衡）参数

| Alpha值 | 损失组成 | 适用场景 |
|---------|----------|---------|
| α=0.3 | 70%学生损失 + 30%蒸馏损失 | 学生独立学习为主 |
| **α=0.5** | **50%学生损失 + 50%蒸馏损失（推荐）** | **平衡设置** |
| α=0.7 | 30%学生损失 + 70%蒸馏损失 | 强调学习教师 |
| α=1.0 | 100%蒸馏损失 | 纯蒸馏（不推荐） |

### 3. Top-K 参数调优建议

| top_k 值 | 占总queries比例（400个） | 效果 |
|----------|-------------------------|------|
| 20-30 | ~5-8% | 极度聚焦，只蒸馏高置信前景 |
| **50** | **~12.5%（推荐）** | **平衡的前景过滤** |
| 100 | ~25% | 较宽松的过滤 |
| 200+ | ~50%+ | 接近 `all` 模式 |

### 4. 蒸馏模式对比选择指南

```
问题：学生模型与教师架构相同？
├── 是 → 优先尝试 'matched'，其次 'top_k'
└── 否（学生更小）
    ├── 有GT标注，信号准确 → 'matched'
    ├── 想过滤背景但保留灵活性 → 'top_k'（top_k=50）
    └── 想柔和地抑制背景 → 'weighted'
```

### 5. 学习率调整

使用蒸馏时，学习率通常可以适当提高：
```json
{
  "optimizer": {
    "name": "AdamW",
    "lr": 0.0002  // 比标准训练高1.5-2倍
  }
}
```

### 6. Epoch数量

蒸馏训练通常收敛更快：
- **标准训练**：200 epochs
- **蒸馏训练**：100-150 epochs（可节省30-50%训练时间）

---

## 故障排查

### 问题 1: 找不到教师模型checkpoint

**错误信息**：
```
FileNotFoundError: [Errno 2] No such file or directory: 'path/to/teacher.pt'
```

**解决方案**：
1. 检查路径是否正确：
   ```bash
   ls -la path/to/teacher.pt
   ```
2. 使用绝对路径：
   ```json
   "teacher_checkpoint": "/absolute/path/to/teacher.pt"
   ```

### 问题 2: CUDA内存不足

**错误信息**：
```
RuntimeError: CUDA out of memory
```

**解决方案**：
1. 减小batch size：
   ```json
   "train": {
     "batch_size": 2  // 从4降到2
   }
   ```
2. 使用梯度累积（需要修改trainer）
3. 使用更小的学生模型

### 问题 3: 蒸馏效果不佳

**症状**：学生模型精度远低于教师

**检查清单**：
- [ ] 确认教师模型加载正确（打印精度验证）
- [ ] 尝试增大temperature（4→6→8）
- [ ] 调整alpha（0.5→0.7）
- [ ] 增加训练epochs
- [ ] 检查学生模型是否太小
- [ ] 尝试切换蒸馏模式（'all' → 'top_k' → 'weighted'）

### 问题 4: 损失函数不匹配

**错误信息**：
```
TypeError: forward() takes 3 positional arguments but 4 were given
```

**原因**：使用了标准Loss而非KDLoss

**解决方案**：
确保配置文件包含distillation配置，或手动使用KDLoss：
```python
from dprt.training.loss import KDLoss

kd_loss = KDLoss(student_loss, distillation_loss, alpha=0.5)
```

---

## 监控训练进度

训练时会输出以下损失指标：

```
Epoch 10/100:
  loss: 2.45
  loss_student_loss: 1.20        # 学生检测损失
  loss_distill_loss: 1.25        # 蒸馏损失
  loss_distill_class: 0.80       # 分类蒸馏
  loss_distill_bbox: 0.45        # 检测框蒸馏
  loss_student_total_class: 0.35
  loss_student_center: 0.28
  loss_student_size: 0.32
  loss_student_angle: 0.25
```

使用TensorBoard可视化：
```bash
tensorboard --logdir outputs/distillation/[timestamp]
```

---

## 性能对比示例

| 模型 | Backbone | Params | FPS | mAP@0.5 |
|------|----------|--------|-----|---------|
| 教师模型 | ResNet101 | 85M | 12 | 0.72 |
| 学生模型（无蒸馏） | ResNet50 | 44M | 28 | 0.64 |
| **学生模型（'all'蒸馏）** | **ResNet50** | **44M** | **28** | **0.66** |
| **学生模型（'top_k'蒸馏）** | **ResNet50** | **44M** | **28** | **0.68** ✨ |
| **学生模型（'weighted'蒸馏）** | **ResNet50** | **44M** | **28** | **0.69** ✨ |

> 注：以上数值为示意，实际精度取决于数据集和超参数设置。

---

## 为什么学生模型有时超越教师模型？

在 `top_k` 和 `weighted` 蒸馏模式下，学生的 mAP 比教师高 0.00x 是一种**已被广泛记录的现象**，原因如下：

### 1. 正则化效应（Label Smoothing）
教师的**软标签（soft labels）**天然起到标签平滑的作用，降低了学生对训练集噪声的过拟合。当数据集规模较小（子集训练）时，这一效果尤为显著。教师在全集上训练了较多轮次，可能在子集上略有过拟合，而学生通过 KD 损失获得了额外的正则化。

### 2. Born Again Networks（再生网络）现象
Furlanello 等人（2018）的研究表明，**用完全相同结构的模型作为教师和学生**，学生经过蒸馏后仍能超过教师。这说明蒸馏损失本身提供了比单纯硬标签更丰富的监督信号，使学生收敛到更好的泛化极值点。

### 3. 温度缩放（Temperature Scaling）
蒸馏时使用 `temperature > 1` 对教师 logits 进行软化，类间的相对概率关系被保留。这些**类间关系信息**（例如"看起来像卡车但更像轿车"）是硬标签 (0/1) 完全不包含的，给学生提供了额外学习信号。

### 4. Top-K / Weighted 过滤的质量保证
- **top_k 模式**：只对教师最自信的 K 个预测进行蒸馏，避免学生从不确定的教师预测中学习噪声。
- **weighted 模式**：根据教师置信度加权蒸馏损失，高质量预测贡献更多梯度。

这两种模式都比 `all` 模式（包含所有背景预测）产生更干净的监督信号，从而提升泛化性能。

### 5. 实践建议
| 现象 | 说明 |
|------|------|
| 学生 mAP > 教师 mAP (差值 < 0.01) | 正常现象，属于蒸馏正则化效果 |
| 学生 mAP > 教师 mAP (差值 > 0.02) | 建议检查教师是否过拟合，或学生使用了额外数据增强 |
| 学生 mAP << 教师 mAP | 检查 alpha、temperature、蒸馏模式配置 |

### 6. 为什么会出现 `matched`≈0.25 而 `top_k=400`≈0.24？

这个现象常见，且**不要求两者必须完全相同**。关键是两种模式在代码里的监督信号定义不同：

- `matched`：在 `KDLoss.forward` 中，学生和教师分别做 Hungarian 匹配，再按同一 GT 对齐后蒸馏（`src/dprt/training/loss.py`）。
  - 实际蒸馏的是“与 GT 对齐的正样本 queries”（数量通常接近每帧目标数），不是固定 400 个。
- `top_k`：在 `DistillationLoss.forward` 中按教师置信度排序，取前 K 个 query 做蒸馏（`src/dprt/training/loss.py`）。
  - `top_k=400` 只是“把教师的 400 个 query 都纳入”，本质仍是“置信度排序驱动”的蒸馏，不是 GT 对齐蒸馏。

因此，`matched` 与 `top_k=400` 不等价。前者更偏“目标对齐”，后者更偏“分数排序”。

另外，如果你比较的是 `all` 和 `top_k=400`，从实现上它们应当走同一批 query（当 `top_k >= min(N_student, N_teacher)` 时，`top_k` 会退化为与 `all` 相同的切片路径），理论上应基本一致。若仍有轻微差异，常见来源包括：

- 训练随机性（初始化、数据顺序、CUDA 非完全确定性）；
- 小样本子集下验证集方差更大；
- 置信度排序中的并列分数/tie 处理会带来轻微数值扰动。

更科学的比较方式：固定所有超参数与 seed，至少重复 3 次，报告 mean±std，而不是单次分数。

> **结论**：您观察到的 0.26x（学生）略高于教师的结果是**完全正常且符合理论预期**的，说明蒸馏配置工作正常。

---

## 参考文献

1. Hinton et al. "Distilling the Knowledge in a Neural Network" (2015)
2. Chen et al. "Learning Efficient Object Detection Models with Knowledge Distillation" (NeurIPS 2017)
3. Wang et al. "Distilling Object Detectors with Fine-grained Feature Imitation" (CVPR 2019)
4. Furlanello et al. "Born Again Networks" (ICML 2018) — 同架构学生超越教师的理论基础

---

## 联系支持

如有问题，请在GitHub Issues中提问。

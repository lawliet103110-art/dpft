# 知识蒸馏训练指南 (Knowledge Distillation Guide)

本指南介绍如何在DPFT项目中使用知识蒸馏来训练更小、更快的模型，同时保持高精度。

## 目录
1. [什么是知识蒸馏](#什么是知识蒸馏)
2. [快速开始](#快速开始)
3. [详细配置](#详细配置)
4. [使用方法](#使用方法)
5. [代码示例](#代码示例)
6. [调参建议](#调参建议)
7. [故障排查](#故障排查)

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
      "alpha": 0.5
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

## 使用方法

### 方法 1: 使用配置文件（推荐）

创建配置文件 `config/my_distillation.json`：
```json
{
  "train": {
    "distillation": {
      "teacher_checkpoint": "checkpoints/teacher_model.pt",
      "freeze_teacher": true,
      "temperature": 4.0,
      "alpha": 0.5
    }
  }
}
```

运行训练：
```bash
python src/dprt/train.py \
    --src /data/kradar \
    --cfg config/my_distillation.json \
    --dst outputs/distillation_exp1
```

### 方法 2: 编程方式使用

```python
from dprt.training.loss import DistillationLoss, KDLoss
from dprt.training.trainer import KnowledgeDistillationTrainer
from dprt.training.loss import Loss
import torch

# 1. 加载教师模型
teacher_model = torch.load("checkpoints/teacher_model.pt")
teacher_model.eval()

# 2. 创建学生损失（标准检测损失）
student_loss = Loss.from_config(config['train'])

# 3. 创建蒸馏损失
distillation_loss = DistillationLoss(
    temperature=4.0,
    alpha=0.5,
    distill_class=True,
    distill_bbox=True,
    reduction='mean'
)

# 4. 组合损失
kd_loss = KDLoss(
    student_loss=student_loss,
    distillation_loss=distillation_loss,
    alpha=0.5
)

# 5. 创建蒸馏训练器
trainer = KnowledgeDistillationTrainer(
    teacher_model=teacher_model,
    teacher_checkpoint=None,
    freeze_teacher=True,
    epochs=200,
    optimizer=optimizer,
    loss=kd_loss,
    scheduler=scheduler,
    metric=metric,
    device='cuda'
)

# 6. 开始训练
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
config = load_config('config/kradar_distillation_example.json')

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

# 只蒸馏分类logits，不蒸馏bbox
distillation_loss = DistillationLoss(
    temperature=3.0,
    alpha=0.6,
    distill_class=True,
    distill_bbox=False,
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

### 3. 学习率调整

使用蒸馏时，学习率通常可以适当提高：
```json
{
  "optimizer": {
    "name": "AdamW",
    "lr": 0.0002  // 比标准训练高1.5-2倍
  }
}
```

### 4. Epoch数量

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
| **学生模型（蒸馏）** | **ResNet50** | **44M** | **28** | **0.69** ✨ |

性能提升：+5% mAP，速度提升2.3x

---

## 参考文献

1. Hinton et al. "Distilling the Knowledge in a Neural Network" (2015)
2. Chen et al. "Learning Efficient Object Detection Models with Knowledge Distillation" (NeurIPS 2017)
3. Wang et al. "Distilling Object Detectors with Fine-grained Feature Imitation" (CVPR 2019)

---

## 联系支持

如有问题，请在GitHub Issues中提问。

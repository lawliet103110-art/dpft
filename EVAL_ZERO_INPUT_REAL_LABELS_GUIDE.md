# 全零输入 + 真实标签评估指南

## 用途

这个脚本用于测试模型在**极端输入条件**下的表现：
- **输入**：全零数据（最差情况）
- **标签**：真实标签数据（从数据集读取）

可以测试：
1. 模型在全零输入下的漏检率（应该 100% 漏检）
2. 模型的鲁棒性和稳定性
3. 与正常评估对比，了解输入质量的重要性

## 快速开始

### 基本用法

```bash
python eval_zero_input_real_labels.py \
    --src /path/to/kradar/processed \
    --checkpoint /path/to/model.pth \
    --cfg config/kradar_camera_mono.json
```

### 完整参数

```bash
python eval_zero_input_real_labels.py \
    --src /data/kradar/processed \
    --checkpoint /path/to/model.pth \
    --cfg config/kradar_camera_mono.json \
    --dst ./eval_results_zero_input
```

## 参数说明

| 参数 | 必需 | 默认值 | 说明 |
|------|------|--------|------|
| `--src` | ✅ | - | 预处理后的数据集路径（用于读取真实标签） |
| `--checkpoint` | ✅ | - | 模型checkpoint文件路径 |
| `--cfg` | ❌ | `config/kradar_camera_mono.json` | 配置文件路径 |
| `--dst` | ❌ | `./log` | 评估结果保存目录 |

## 工作原理

### 1. 数据处理流程

```python
# 真实数据集
real_data, real_label = KRadarDataset[i]

# 包装处理
zero_data = {}
for key, value in real_data.items():
    if is_input_data(key):  # 例如 'camera_mono', 'radar_bev'
        zero_data[key] = torch.zeros_like(value)  # 置零
    else:  # 例如 '*_p', '*_t', '*_shape'
        zero_data[key] = value  # 保留元数据

# 返回
return zero_data, real_label  # 全零输入 + 真实标签
```

### 2. 哪些数据会被置零？

**会被置零的输入数据**：
- `camera_mono` - 相机图像
- `camera_stereo` - 立体相机图像
- `radar_bev` - 雷达鸟瞰图
- `radar_front` - 雷达前视图
- `lidar` - 激光雷达点云

**保留的元数据**：
- `*_p` - 投影矩阵（如 `label_to_camera_mono_p`）
- `*_t` - 变换矩阵（如 `label_to_camera_mono_t`）
- `*_shape` - 形状信息（如 `camera_mono_shape`）
- `*_intrinsics` - 相机内参
- `*_extrinsics` - 相机外参

### 3. 标签数据

标签数据完全来自真实数据集：
```python
label = {
    'gt_center': torch.tensor([[x1, y1, z1], [x2, y2, z2], ...]),  # 真实中心坐标
    'gt_size': torch.tensor([[l1, w1, h1], [l2, w2, h2], ...]),    # 真实尺寸
    'gt_angle': torch.tensor([[sin1, cos1], [sin2, cos2], ...]),   # 真实角度
    'gt_class': torch.tensor([[...], [...], ...]),                  # 真实类别
    'description': torch.tensor([road, time, weather])              # 场景描述
}
```

## 预期结果

### 正常情况

由于输入全为零，模型无法检测到任何物体：

```
Evaluation Results:
  mAP: 0.00xx (接近 0)
  mGIoU: -1.00 或 NaN

说明：
  - mAP ≈ 0: 模型完全漏检了所有目标（预期行为）
  - mGIoU ≈ -1: 没有预测框与真值框匹配
```

### 异常情况

如果 **mAP > 0.1**：
- ⚠️  模型可能过拟合或存在数据泄露
- ⚠️  模型可能不依赖输入就产生固定预测
- ⚠️  需要检查模型架构

## 在云服务器上使用

### 步骤 1: 上传文件

```bash
scp eval_zero_input_real_labels.py user@server:/path/to/dpft/
```

### 步骤 2: 运行评估

```bash
# 假设数据在 /data/kradar/processed
# 模型在 /data/checkpoints/model.pth

python eval_zero_input_real_labels.py \
    --src /data/kradar/processed \
    --checkpoint /data/checkpoints/model.pth \
    --cfg config/kradar_camera_mono.json \
    --dst ./eval_zero_with_labels
```

### 步骤 3: 查看结果

```bash
# 使用 TensorBoard
tensorboard --logdir=./eval_zero_with_labels --host=0.0.0.0 --port=6006

# 或查看生成的文件
ls -lh ./eval_zero_with_labels/
```

## 完整测试矩阵

建议进行以下三种测试来全面评估模型：

### 1. 正常评估（基线）

```bash
python src/dprt/evaluate.py \
    --src /data/kradar/processed \
    --checkpoint model.pth \
    --cfg config/kradar_camera_mono.json \
    --dst ./eval_normal
```

**预期**: mAP = 正常值（如 0.65）

### 2. 全零输入 + 空标签

```bash
python eval_zero_input.py \
    --checkpoint model.pth \
    --cfg config/kradar_camera_mono.json \
    --dst ./eval_zero_empty \
    --num_samples 100
```

**预期**: mAP = 1.0（无目标 + 无预测 = 完美）

### 3. 全零输入 + 真实标签（本脚本）

```bash
python eval_zero_input_real_labels.py \
    --src /data/kradar/processed \
    --checkpoint model.pth \
    --cfg config/kradar_camera_mono.json \
    --dst ./eval_zero_real
```

**预期**: mAP ≈ 0.0（无法检测到任何真实目标）

## 结果对比表

| 测试场景 | 输入 | 标签 | 预期 mAP | 测试目的 |
|---------|------|------|----------|----------|
| 正常评估 | 真实数据 | 真实标签 | ~0.65 | 模型实际性能 |
| 全零+空标签 | 全零 | 空标签 | 1.0 | 误检测率（应该无误检） |
| 全零+真实标签 | 全零 | 真实标签 | ~0.0 | 漏检测率（应该全漏检） |

## 示例输出

```bash
$ python eval_zero_input_real_labels.py \
    --src /data/kradar/processed \
    --checkpoint model.pth \
    --cfg config/kradar_camera_mono.json

======================================================================
全零输入 + 真实标签评估
======================================================================

📁 加载配置: config/kradar_camera_mono.json
🎲 设置随机种子: 42

📦 加载真实数据集: /data/kradar/processed
   数据集大小: 2701 个样本

🔄 创建全零输入包装器...
📚 创建数据加载器...
   Batch size: 4
   批次数: 676

🚀 开始评估...
   Checkpoint: model.pth
   输出目录: ./log

======================================================================
100%|████████████████████| 676/676 [01:23<00:00,  8.10it/s]

======================================================================
✅ 评估完成!
======================================================================

预期结果：
  - mAP 应该接近 0（全零输入无法检测到物体）
  - 这测试了模型在极端输入下的鲁棒性

结果已保存到: ./log
======================================================================
```

## 常见问题

### Q1: 为什么需要保留元数据？

**答**: 元数据（投影矩阵、变换矩阵等）是模型推理必需的。即使输入是全零，模型仍然需要这些信息来：
1. 将查询点投影到输入空间
2. 进行坐标变换
3. 理解输入的几何结构

### Q2: 如果 mAP 不是 0 怎么办？

**可能原因**：
1. **模型设计问题**：模型可能不完全依赖输入，而是学习了固定的先验
2. **过拟合**：模型可能记住了数据集的统计特性
3. **数据泄露**：元数据中可能包含了位置信息

**检查方法**：
```python
# 检查模型是否产生了非零预测
# 使用 check_predictions.py 查看模型输出
```

### Q3: 与 eval_zero_input.py 的区别？

| 特性 | eval_zero_input.py | eval_zero_input_real_labels.py |
|------|-------------------|-------------------------------|
| 输入 | 全零 | 全零 |
| 标签 | 空标签（无目标） | 真实标签（有目标） |
| 数据集 | 不需要 | 需要真实数据集 |
| 样本数 | 可指定任意数量 | 与测试集相同 |
| 测试目的 | 误检测率 | 漏检测率 |
| 预期 mAP | 1.0 | ~0.0 |

## 技术细节

### _is_input_data 方法

这个方法决定哪些字段需要置零：

```python
def _is_input_data(self, key: str) -> bool:
    # 输入数据（需要置零）
    input_keywords = [
        'camera_mono',    # 相机图像
        'camera_stereo',  # 立体相机
        'radar_bev',      # 雷达BEV
        'radar_front',    # 雷达前视
        'lidar'           # 激光雷达
    ]

    # 元数据（保留）
    metadata_keywords = [
        '_p',           # 投影矩阵
        '_t',           # 变换矩阵
        '_shape',       # 形状信息
        '_intrinsics',  # 内参
        '_extrinsics'   # 外参
    ]

    # 如果包含元数据关键词，不置零
    for meta_kw in metadata_keywords:
        if meta_kw in key:
            return False

    # 如果是输入数据关键词，置零
    for input_kw in input_keywords:
        if input_kw == key:
            return True

    # 默认不置零（保守策略）
    return False
```

### 内存优化

如果数据集很大，可以只测试一个子集：

```python
# 修改脚本，添加采样参数
class ZeroInputRealLabelsDataset(Dataset):
    def __init__(self, real_dataset, max_samples=None):
        self.real_dataset = real_dataset
        self.max_samples = max_samples or len(real_dataset)

    def __len__(self):
        return min(self.max_samples, len(self.real_dataset))
```

## 总结

这个脚本提供了一个**压力测试**工具：

✅ **优点**：
- 测试模型在极端输入下的鲁棒性
- 验证模型确实依赖输入数据
- 帮助识别过拟合或数据泄露问题

📊 **配合其他测试使用**：
- 正常评估：了解实际性能
- 全零+空标签：测试误检率
- 全零+真实标签：测试漏检率

这样可以全面评估模型的性能和鲁棒性！

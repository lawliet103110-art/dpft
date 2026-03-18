# 全零输入评估使用指南

## 代码说明

`eval_zero_input.py` 完全参考以下文件实现：
- `src/dprt/evaluate.py` - 评估流程
- `src/dprt/datasets/kradar/dataset.py` - 数据格式
- `src/dprt/datasets/__init__.py` - 数据加载
- `src/dprt/datasets/loader.py` - 批处理逻辑

确保与项目原有代码完全兼容。

## 快速开始

### 最简单的用法

```bash
python eval_zero_input.py --checkpoint /path/to/your/model.pth
```

### 完整参数示例

```bash
python eval_zero_input.py \
    --checkpoint /path/to/model.pth \
    --cfg config/kradar_camera_mono.json \
    --dst ./log \
    --num_samples 100
```

## 参数说明

| 参数 | 必需 | 默认值 | 说明 |
|------|------|--------|------|
| `--checkpoint` | ✅ | - | 模型checkpoint文件路径 (.pth) |
| `--cfg` | ❌ | `config/kradar_camera_mono.json` | 配置文件路径 |
| `--dst` | ❌ | `./log` | 评估结果保存目录 |
| `--num_samples` | ❌ | `100` | 生成的全零样本数量 |

## 代码结构说明

### ZeroInputDataset 类

完全模拟 `KRadarDataset` 的数据格式：

```python
class ZeroInputDataset(Dataset):
    def __init__(self, config, num_samples):
        # 从配置读取参数，与 KRadarDataset 相同
        self.image_size = config['data']['image_size']
        self.num_classes = config['data']['num_classes']
        self.camera = config['data'].get('camera', '')
        self.radar = config['data'].get('radar', '')

    def __getitem__(self, index):
        # 返回 (data, label) 元组
        # data: 包含传感器数据和元数据的字典
        # label: 包含 gt_center, gt_size, gt_angle, gt_class 的字典
        return data, label
```

### 数据格式详解

#### 1. 相机数据 (camera='M')

根据 `kradar_camera_mono.json` 配置：

```python
data = {
    'camera_mono': torch.zeros(512, 512, 3),  # (H, W, C) 格式
    'label_to_camera_mono_p': torch.eye(4),   # 投影矩阵
    'label_to_camera_mono_t': torch.zeros(4, 4),  # 变换矩阵
    'camera_mono_shape': torch.tensor([512, 512, 3])  # 形状信息
}
```

**注意**：
- KRadarDataset 使用 (H, W, C) 格式，不是 (C, H, W)
- 这是因为 `load_sample_data` 中有 `img.movedim(0, -1)`

#### 2. 雷达数据 (radar='BF')

如果配置包含雷达：

```python
# BEV (Bird's Eye View) - Range-Azimuth
data['radar_bev'] = torch.zeros(256, 64, 6)  # (range, azimuth, features)
data['label_to_radar_bev_p'] = projection_matrix  # 3x4 投影矩阵
data['label_to_radar_bev_t'] = torch.eye(4)  # 4x4 变换矩阵
data['radar_bev_shape'] = torch.tensor([256, 64, 6])

# Front - Elevation-Azimuth
data['radar_front'] = torch.zeros(64, 20, 6)  # (azimuth, elevation, features)
data['label_to_radar_front_p'] = projection_matrix
data['label_to_radar_front_t'] = torch.eye(4)
data['radar_front_shape'] = torch.tensor([64, 20, 6])
```

#### 3. 标签数据

完全按照 `KRadarDataset.get_detection_label` 的格式：

```python
label = {
    'gt_center': torch.zeros(0, 3),      # (N, 3): [x, y, z] 坐标
    'gt_size': torch.zeros(0, 3),        # (N, 3): [l, w, h] 尺寸
    'gt_angle': torch.zeros(0, 2),       # (N, 2): [sin(θ), cos(θ)]
    'gt_class': torch.zeros(0, 2),       # (N, num_classes) one-hot编码
    'description': torch.zeros(3)        # (3,): [road_structure, time_zone, weather]
}
```

**注意**：
- 所有 gt_* 张量的第一维都是 0（无目标）
- `gt_angle` 使用 sin/cos 编码，不是原始角度
- `gt_class` 是 one-hot 编码，维度为 `num_classes`
- `description` 是 tensor，不是字典！包含场景描述信息（参考 `processor.py` 的 `map_description`）

### 数据加载流程

```python
# 1. 加载配置
config = load_config(cfg)

# 2. 设置随机种子
set_seed(config['computing']['seed'])

# 3. 创建数据集
test_dataset = ZeroInputDataset(config, num_samples=100)

# 4. 创建数据加载器
# load() 函数内部使用 listed_collating
# 将 List[(data, label)] 转换为 (batched_data, List[label])
test_loader = load(test_dataset, config=config)

# 5. 评估
# evaluate(config) 创建 CentralizedEvaluator 实例
# 调用其 evaluate() 方法
evaluate(config)(checkpoint, test_loader, dst)
```

### listed_collating 批处理

参考 `src/dprt/datasets/loader.py`：

```python
def listed_collating(data):
    inputs, targets = zip(*data)  # 分离输入和标签
    inputs = default_collate(inputs)  # 批处理输入
    targets = list(targets)  # 标签保持为列表
    return (inputs, targets)
```

**批处理后的格式**：
- `inputs`: 字典，每个值的形状 `(B, ...)`
  - 例如：`camera_mono` 从 `(H, W, C)` 变为 `(B, H, W, C)`
- `targets`: 列表，长度为 `B`，每个元素是一个标签字典

## 输出结果

评估完成后，在 `dst` 目录生成：

### 1. TensorBoard 日志

```bash
tensorboard --logdir=./log
```

可查看：
- `test/mAP`: 平均精度
- `test/mGIoU`: 平均广义IoU
- `test/Inference_time_mean_ms`: 平均推理时间
- `test/Inference_time_std_ms`: 推理时间标准差
- `test/FLOPS`: 浮点运算次数
- `test/MACS`: 乘加运算次数
- `test/Parameters`: 参数数量

### 2. 导出文件（如果配置了）

根据 `config['evaluate']['exporter']`，可能生成预测结果导出文件。

## 预期结果

由于输入为全零且标签为空（无目标），预期：

- **mAP ≈ 0**：模型不应检测到任何物体
- **mGIoU ≈ 0 或 NaN**：没有预测框与真值框匹配

### 正常的输出示例

```
Evaluating epoch 100...
100%|████████████████| 25/25 [00:15<00:00,  1.60it/s]

TensorBoard logs saved to: ./log/20241208_120000
Evaluation metrics:
  mAP: 0.0000
  mGIoU: nan
  Inference_time_mean_ms: 45.23
  Inference_time_std_ms: 2.31
  Parameters: 45,234,567
  FLOPS: 123,456,789,012
```

### 异常情况

如果 **mAP > 0**，可能表示：
1. 模型在零输入下产生了误检测
2. 模型可能存在过拟合
3. 模型的背景抑制能力不足

## 在云服务器上使用

### 步骤 1: 上传文件

将 `eval_zero_input.py` 上传到项目根目录：

```bash
scp eval_zero_input.py user@server:/path/to/dpft/
```

### 步骤 2: 确认环境

```bash
cd /path/to/dpft
pip install -r requirements.txt
```

### 步骤 3: 运行评估

假设模型文件在 `/data/checkpoints/20241208_120000_dprt_100.pth`：

```bash
python eval_zero_input.py \
    --checkpoint /data/checkpoints/20241208_120000_dprt_100.pth \
    --cfg config/kradar_camera_mono.json \
    --dst ./eval_results \
    --num_samples 100
```

### 步骤 4: 查看结果

```bash
# 查看 TensorBoard
tensorboard --logdir=./eval_results --host=0.0.0.0 --port=6006

# 或者直接查看生成的文件
ls -lh ./eval_results/
```

## 常见问题

### Q1: ModuleNotFoundError: No module named 'dprt'

**原因**：Python 找不到 dprt 模块

**解决方案**：

```bash
# 方案1: 在项目根目录运行
cd /path/to/dpft
python eval_zero_input.py --checkpoint model.pth

# 方案2: 设置 PYTHONPATH
export PYTHONPATH=$PYTHONPATH:/path/to/dpft/src
python eval_zero_input.py --checkpoint model.pth

# 方案3: 修改脚本添加路径（不推荐）
import sys
sys.path.insert(0, '/path/to/dpft/src')
```

### Q2: CUDA out of memory

**原因**：显存不足

**解决方案**：

1. 修改配置文件，减小 batch_size：
   ```json
   "train": {
       "batch_size": 2  // 从 4 改为 2
   }
   ```

2. 减少样本数量：
   ```bash
   python eval_zero_input.py --checkpoint model.pth --num_samples 10
   ```

3. 使用 CPU（慢但不会爆显存）：
   修改配置文件：
   ```json
   "computing": {
       "device": "cpu"
   }
   ```

### Q3: RuntimeError: shape mismatch

**原因**：数据形状与模型期望不匹配

**检查**：
1. 确认配置文件与模型训练时使用的一致
2. 检查 `image_size` 是否正确
3. 检查 `camera` 和 `radar` 配置是否匹配

### Q4: Checkpoint 文件格式错误

**问题**：`ValueError: not enough values to unpack`

**原因**：checkpoint 文件名格式不正确

**解决**：确保文件名格式为 `<timestamp>_<model>_<epoch>.pth`

例如：`20241208_120000_dprt_100.pth`

### Q5: 数值异常 (NaN/Inf)

**可能原因**：
1. 空标签导致某些指标无法计算
2. 模型输出异常

**检查**：
```bash
# 查看详细日志
python eval_zero_input.py --checkpoint model.pth 2>&1 | tee eval.log
```

## Docker 环境示例

如果在 Docker 容器中运行：

```bash
docker run -it --gpus all \
    -v /host/checkpoints:/app/checkpoints \
    -v /host/dpft:/app/dpft \
    your-image:tag \
    bash -c "cd /app/dpft && \
             python eval_zero_input.py \
                 --checkpoint /app/checkpoints/model.pth \
                 --cfg config/kradar_camera_mono.json \
                 --dst /app/log"
```

## 与正常评估的对比

| 特性 | evaluate.py | eval_zero_input.py |
|------|-------------|-------------------|
| 数据来源 | KRadarDataset | ZeroInputDataset |
| 需要数据集 | ✅ 需要 | ❌ 不需要 |
| 数据格式 | 真实数据 | 全零输入 |
| 标签格式 | 相同 | 相同（空标签） |
| 评估流程 | 相同 | 相同 |
| 评估指标 | 相同 | 相同 |
| 用途 | 性能评估 | 功能测试/调试 |

## 测试用途

此脚本主要用于：

1. ✅ **模型加载测试**：验证 checkpoint 能否正确加载
2. ✅ **推理流程测试**：测试完整的推理 pipeline
3. ✅ **性能基准测试**：测量推理时间、FLOPS、参数量
4. ✅ **稳定性测试**：验证模型在极端输入下的稳定性
5. ✅ **调试工具**：不需要数据集即可运行评估流程

**不适用于**：
- ❌ 实际性能评估（需要真实数据）
- ❌ 模型准确度评估（全零输入无意义）

## 代码参考

主要参考的源文件：

1. `src/dprt/evaluate.py` - 主评估流程
2. `src/dprt/datasets/kradar/dataset.py` - KRadarDataset 实现
3. `src/dprt/datasets/kradar/processor.py` - 数据预处理
4. `src/dprt/datasets/loader.py` - listed_collating 实现
5. `src/dprt/evaluation/evaluator.py` - CentralizedEvaluator 实现

所有数据格式和处理逻辑都严格遵循这些文件的实现。

# K-Radar 模型输出可视化指南

## 📁 Evaluatelog 文件结构说明

你的 `Evaluatelog` 文件夹模拟了服务器上的真实评估输出结构：

```
Evaluatelog/
├── events.out.tfevents.*         # TensorBoard 日志
└── exports/
    └── kradar/
        ├── 0.0/                   # 置信度阈值 0.0
        ├── 0.3/                   # 置信度阈值 0.3
        ├── 0.5/                   # 置信度阈值 0.5
        ├── 0.7/                   # 置信度阈值 0.7
        └── 0.9/                   # 置信度阈值 0.9 (你当前有数据)
            ├── all/               # 所有场景
            │   ├── preds/         # 模型预测 ✅ 有数据
            │   │   ├── 000000.txt
            │   │   ├── 000001.txt
            │   │   └── 012095.txt
            │   ├── gts/           # 真实标签 ✅ 有数据
            │   │   ├── 000000.txt
            │   │   ├── 000001.txt
            │   │   └── 012095.txt
            │   ├── desc/          # 场景描述 (空)
            │   └── val.txt        # 样本ID列表
            ├── highway/           # 高速公路场景 (空)
            └── rain/              # 雨天场景 (空)
```

## 📝 数据格式说明

### TXT 文件格式

每个 `.txt` 文件包含多个边界框，每行一个：

```
类别 truncated occluded alpha bbox1 bbox2 bbox3 bbox4 高度 宽度 长度 y z x 角度
```

**示例**：
```
sed 0 0 0 50 50 150 150 1.69 1.92 3.83 -1.47 0.05 44.39 0.01
```

**字段说明**：
- **类别**: `sed`(轿车), `bus`(公交/卡车), `mot`(摩托车), `bic`(自行车), `big`(自行车组), `ped`(行人), `peg`(行人组), `bg`(背景)
- **truncated, occluded, alpha**: 固定为 0 (未使用)
- **bbox1-4**: 固定为 50 50 150 150 (未使用)
- **高度, 宽度, 长度**: 边界框的三维尺寸 (米)
- **y, z, x**: 边界框中心的坐标 (米)
  - x: 前向距离 (0-72米)
  - y: 横向距离 (-6.4 到 6.4米)
  - z: 垂直高度 (-2.0 到 6.0米)
- **角度**: 边界框的朝向角 (弧度)

### 特殊值

- `dummy -1 -1 0 0 0 0 0 0 0 0 0 0 0 0 0`: 占位符，表示该样本没有目标

## 🚀 使用方法

### 1. 运行可视化脚本

```bash
python visualize_results.py
```

**输出**：
1. 统计信息（预测框和真实框的数量）
2. 3D可视化窗口（需要 Open3D）

### 2. 自定义可视化

编辑 `visualize_results.py` 的 `main()` 函数：

```python
# 可视化不同的样本
visualize_sample_3d(
    evaluatelog_root=EVALUATELOG_ROOT,
    sample_id='000001',        # 修改这里查看不同样本
    conf_threshold='0.9',      # 修改置信度阈值
    scene_type='all',          # 修改场景类型
    data_root=DATA_ROOT
)
```

### 3. 查看不同置信度阈值的结果

```python
# 比较不同阈值
for threshold in ['0.0', '0.3', '0.5', '0.7', '0.9']:
    if os.path.exists(f"Evaluatelog/exports/kradar/{threshold}/all"):
        compare_predictions(EVALUATELOG_ROOT, threshold, 'all')
```

## 🎨 可视化功能

### visu.py 提供的主要功能

1. **`visu_lidar_data(pc, boxes, xlim, ylim, cm)`**
   - 3D点云和边界框可视化
   - 使用 Open3D 显示交互式3D视图
   - 参数:
     - `pc`: 点云数据 (N, 4) - x, y, z, intensity
     - `boxes`: 边界框 (M, 8) - x, y, z, theta, l, w, h, class
     - `xlim`, `ylim`: 坐标范围限制

2. **`visu_2d_radar_data(grid, dims, boxes, points, ...)`**
   - 2D雷达热力图和边界框可视化
   - 参数:
     - `grid`: 2D雷达网格
     - `dims`: 维度 ('ra', 'ae' 等)
     - `boxes`: 边界框（可选）
     - `points`: 点云（可选）

3. **`visu_radar_tesseract(tesseract, dims, raster, ...)`**
   - 4D雷达数据可视化
   - 支持降维显示

## 🔍 进阶用法

### 1. 如果你有原始雷达/激光雷达数据

修改 `visualize_results.py` 中的 `load_radar_or_lidar_data()` 函数：

```python
def load_radar_or_lidar_data(data_root, sample_id):
    # 加载激光雷达数据
    lidar_path = os.path.join(data_root, 'lidar_os1', f'{sample_id}.npy')
    if os.path.exists(lidar_path):
        pc = np.load(lidar_path)  # (N, 4) - x, y, z, intensity
        return pc

    # 或者加载雷达数据并转换为点云格式
    radar_path = os.path.join(data_root, 'radar_bev', f'{sample_id}.npy')
    if os.path.exists(radar_path):
        radar_data = np.load(radar_path)
        # 将雷达数据转换为点云格式
        # ... 转换代码 ...
        return pc

    return np.empty((0, 4))
```

### 2. 创建对比可视化

```python
import matplotlib.pyplot as plt

def compare_pred_vs_gt_2d(evaluatelog_root, sample_id):
    """2D对比显示预测和真实标签"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))

    # 加载数据
    pred_boxes = load_boxes_from_txt(...)
    gt_boxes = load_boxes_from_txt(...)

    # 绘制真实标签
    ax1.set_title('Ground Truth')
    for box in gt_boxes:
        # 绘制边界框...

    # 绘制预测结果
    ax2.set_title('Predictions')
    for box in pred_boxes:
        # 绘制边界框...

    plt.show()
```

### 3. 批量处理和分析

```python
def analyze_all_samples(evaluatelog_root, conf_threshold='0.9'):
    """分析所有样本的预测质量"""
    results = []

    for sample_id in get_all_sample_ids():
        pred_boxes = load_boxes_from_txt(...)
        gt_boxes = load_boxes_from_txt(...)

        # 计算指标
        precision = calculate_precision(pred_boxes, gt_boxes)
        recall = calculate_recall(pred_boxes, gt_boxes)

        results.append({
            'sample_id': sample_id,
            'precision': precision,
            'recall': recall,
            'num_preds': len(pred_boxes),
            'num_gts': len(gt_boxes)
        })

    return results
```

## 📊 可视化效果说明

### 3D可视化 (visu_lidar_data)

- **点云**: 灰色/彩色点表示雷达或激光雷达检测
- **边界框**: 彩色立方体表示检测到的目标
  - 不同颜色代表不同类别
  - 框的方向表示目标朝向
- **坐标系**:
  - X轴: 前向 (车辆行驶方向)
  - Y轴: 横向 (左右)
  - Z轴: 垂直 (上下)

### 交互操作 (Open3D)

- **鼠标左键**: 旋转视角
- **鼠标右键**: 平移视图
- **滚轮**: 缩放
- **ESC**: 关闭窗口

## 🛠️ 依赖安装

确保已安装必要的库：

```bash
pip install open3d matplotlib numpy torch
```

## ⚠️ 常见问题

### 1. "没有预测/真实标签"

- 检查文件路径是否正确
- 确认 txt 文件不是空的或只包含 "dummy" 行

### 2. Open3D 窗口无法显示

- 确保你的系统支持图形界面
- 在服务器上使用 X11 转发或 VNC

### 3. 坐标系不对

- K-Radar 使用的坐标系：X前向，Y横向，Z垂直
- 确保数据转换时坐标顺序正确

## 📈 下一步

1. **添加原始数据**: 如果有雷达/激光雷达数据，可以看到完整的点云
2. **评估指标**: 使用 `src/dprt/evaluation/` 中的工具计算 AP、mAP 等
3. **错误分析**: 找出预测错误的样本进行分析
4. **可视化改进**: 添加更多可视化功能，如类别分布、距离分布等

## 📚 相关文件

- `src/dprt/utils/visu.py`: 可视化工具函数
- `src/dprt/evaluation/exporters/kradar.py`: 数据导出格式定义
- `src/dprt/datasets/kradar/`: K-Radar 数据集相关代码

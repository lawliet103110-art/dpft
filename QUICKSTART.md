# 快速入门指南 - 可视化你的模型输出

## 🎯 三种可视化方式

### 方式 1: 快速查看单个样本 (推荐入门)

**最简单的方法**，直接查看预测结果：

```bash
python quick_visualize.py 000000
```

**查看不同样本**：
```bash
python quick_visualize.py 000001
python quick_visualize.py 012095
```

**查看不同置信度阈值**：
```bash
python quick_visualize.py 000000 0.5
python quick_visualize.py 000000 0.7
```

---

### 方式 2: 完整可视化工具

**功能更全面**，包含统计分析：

```bash
python visualize_results.py
```

这会：
1. 显示所有样本的统计信息
2. 自动可视化第一个样本
3. 显示预测框和真实框的对比

**自定义**：编辑 `visualize_results.py` 的 `main()` 函数

---

### 方式 3: 直接使用 visu.py (高级)

**适合集成到自己的代码中**：

```python
import numpy as np
from dprt.utils.visu import visu_lidar_data

# 准备边界框数据: [x, y, z, theta, l, w, h, class]
boxes = np.array([
    [44.01, -1.85, 0.12, -0.01, 3.28, 1.61, 1.42, 0],  # 一辆轿车
])

# 点云数据 (可选): [x, y, z, intensity]
pc = np.array([[0, 0, 0, 1]])  # 空点云

# 可视化
visu_lidar_data(pc, boxes=boxes, xlim=(0, 72), ylim=(-6.4, 6.4))
```

---

## 📊 你的数据结构

```
Evaluatelog/
└── exports/kradar/0.9/all/
    ├── preds/          # 你的模型预测 ✅
    │   ├── 000000.txt  # 样本 000000 的预测结果
    │   ├── 000001.txt
    │   └── 012095.txt
    ├── gts/            # 真实标签 ✅
    │   ├── 000000.txt
    │   ├── 000001.txt
    │   └── 012095.txt
    └── val.txt         # 样本列表
```

**数据格式**（每行一个目标）：
```
类别 0 0 0 50 50 150 150 高 宽 长 y z x 角度
例如: sed 0 0 0 50 50 150 150 1.42 1.61 3.28 -1.85 0.12 44.01 -0.01
```

---

## 🎨 可视化窗口说明

### Open3D 3D 窗口

- **绿色/橙色/蓝色框**: 不同类别的目标
  - 0: Sedan (轿车)
  - 1: Bus/Truck (公交/卡车)
  - 2: Motorcycle (摩托车)
  - 3-6: 自行车、行人等

- **坐标系**:
  - X轴: 前向 (0-72米)
  - Y轴: 横向 (-6.4 到 6.4米)
  - Z轴: 高度 (-2 到 6米)

- **操作**:
  - 鼠标左键拖动: 旋转
  - 鼠标右键拖动: 平移
  - 滚轮: 缩放
  - ESC: 关闭

---

## 📝 示例输出

运行 `quick_visualize.py 000000`：

```
==================================================
可视化样本 000000 (置信度: 0.9)
==================================================

真实标签: 1 个目标
预测结果: 2 个目标

真实标签详情:
  1. sed 位置:(44.01, -1.85, 0.12) 尺寸:(3.28×1.61×1.42)

预测结果详情:
  1. sed 位置:(44.39, -1.47, 0.05) 尺寸:(3.83×1.92×1.69)
  2. sed 位置:(56.27, -2.35, 0.35) 尺寸:(3.84×1.89×1.73)

[1/2] 显示真实标签 (Ground Truth)
      关闭窗口后将显示预测结果...

[2/2] 显示预测结果 (Predictions)

完成！
```

---

## ⚡ 快速测试

复制粘贴以下命令测试：

```bash
# 1. 查看第一个样本
python quick_visualize.py 000000

# 2. 查看第二个样本
python quick_visualize.py 000001

# 3. 查看最后一个样本
python quick_visualize.py 012095

# 4. 运行完整分析
python visualize_results.py
```

---

## 🔧 如果遇到问题

### 1. 无法显示 3D 窗口
```bash
pip install open3d --upgrade
```

### 2. 导入错误
确保在项目根目录运行：
```bash
cd D:\dpft
python quick_visualize.py 000000
```

### 3. 文件不存在
检查你的 Evaluatelog 路径是否正确：
```python
# 在脚本中修改路径
base = r"D:\DPFT\Evaluatelog\exports\kradar\0.9\all"
```

---

## 📚 进一步学习

1. **详细文档**: 查看 `VISUALIZATION_GUIDE.md`
2. **可视化工具源码**: `src/dprt/utils/visu.py`
3. **数据格式定义**: `src/dprt/evaluation/exporters/kradar.py`

---

## 💡 常见用例

### 对比不同置信度阈值

```bash
python quick_visualize.py 000000 0.5
python quick_visualize.py 000000 0.7
python quick_visualize.py 000000 0.9
```

### 批量查看多个样本

创建一个简单的循环脚本：
```bash
for sample in 000000 000001 012095; do
    python quick_visualize.py $sample
done
```

### 统计分析

编辑 `visualize_results.py` 查看更详细的统计信息

---

**现在就试试**：
```bash
python quick_visualize.py 000000
```

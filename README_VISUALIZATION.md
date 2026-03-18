# 模型输出可视化 - 文件清单

## ✅ 已创建的文件

### 1. **quick_visualize.py** - 快速可视化脚本
- **用途**: 最简单快速的方式查看单个样本
- **运行**: `python quick_visualize.py 000000`
- **特点**:
  - 命令行友好
  - 显示边界框详情
  - 自动加载预测和真实标签

### 2. **visualize_results.py** - 完整可视化工具
- **用途**: 全面的可视化和统计分析
- **运行**: `python visualize_results.py`
- **特点**:
  - 显示所有样本的统计信息
  - 支持批量处理
  - 可扩展的分析功能
  - 支持原始数据加载（如果有）

### 3. **QUICKSTART.md** - 快速入门指南
- **内容**:
  - 三种可视化方式
  - 快速测试命令
  - 常见问题解决
  - 示例输出展示

### 4. **VISUALIZATION_GUIDE.md** - 详细使用指南
- **内容**:
  - Evaluatelog 文件结构说明
  - 数据格式详解
  - visu.py 功能介绍
  - 进阶用法和自定义
  - 完整的API说明

---

## 🚀 推荐使用流程

### 初次使用（5分钟）

1. **阅读快速入门**
   ```bash
   cat QUICKSTART.md
   # 或在编辑器中打开
   ```

2. **测试基本功能**
   ```bash
   python quick_visualize.py 000000
   ```

3. **查看统计信息**
   ```bash
   python visualize_results.py
   ```

### 日常使用

```bash
# 快速查看样本
python quick_visualize.py [sample_id] [conf_threshold]

# 例如:
python quick_visualize.py 000001
python quick_visualize.py 000000 0.7
```

### 深入分析

1. 阅读 `VISUALIZATION_GUIDE.md` 了解详细功能
2. 修改 `visualize_results.py` 添加自定义分析
3. 直接使用 `src/dprt/utils/visu.py` 集成到自己的代码

---

## 📁 你的数据结构

```
D:\DPFT\
├── Evaluatelog/                    # 你从服务器复制的数据
│   └── exports/kradar/0.9/all/
│       ├── preds/                  # 模型预测 ✅
│       │   ├── 000000.txt
│       │   ├── 000001.txt
│       │   └── 012095.txt
│       ├── gts/                    # 真实标签 ✅
│       │   ├── 000000.txt
│       │   ├── 000001.txt
│       │   └── 012095.txt
│       └── val.txt
│
├── src/dprt/utils/visu.py          # 可视化工具库
│
├── quick_visualize.py              # 快速可视化脚本 ⭐
├── visualize_results.py            # 完整可视化工具 ⭐
├── QUICKSTART.md                   # 快速入门指南 📖
├── VISUALIZATION_GUIDE.md          # 详细使用指南 📖
└── README_VISUALIZATION.md         # 本文件
```

---

## 🎯 核心功能

### visu.py 提供的函数

1. **visu_lidar_data(pc, boxes, xlim, ylim, cm)**
   - 3D可视化点云和边界框
   - 使用 Open3D 交互式显示
   - 支持多类别彩色显示

2. **visu_2d_radar_data(grid, dims, boxes, points, ...)**
   - 2D雷达热力图
   - 支持叠加边界框和点云
   - 支持笛卡尔/极坐标转换

3. **visu_radar_tesseract(tesseract, dims, raster, ...)**
   - 4D雷达数据可视化
   - 支持维度降维和聚合

---

## 📊 数据格式速查

### TXT 文件格式
```
类别 truncated occluded alpha bbox1 bbox2 bbox3 bbox4 h w l y z x theta
```

### visu.py 边界框格式
```python
boxes = np.array([
    [x, y, z, theta, l, w, h, class],
    # x, y, z: 中心坐标
    # theta: 朝向角 (弧度)
    # l, w, h: 长、宽、高
    # class: 类别ID (0-7)
])
```

### K-Radar 类别
```python
0: Sedan (轿车)
1: Bus or Truck (公交/卡车)
2: Motorcycle (摩托车)
3: Bicycle (自行车)
4: Bicycle Group (自行车组)
5: Pedestrian (行人)
6: Pedestrian Group (行人组)
7: Background (背景)
```

---

## 💡 使用技巧

### 1. 快速浏览所有样本
```python
# 修改 quick_visualize.py
for sample_id in ['000000', '000001', '012095']:
    visualize(sample_id, '0.9')
```

### 2. 对比不同置信度
```python
for conf in ['0.5', '0.7', '0.9']:
    visualize('000000', conf)
```

### 3. 查看检测范围
K-Radar 的检测范围：
- X: 0 - 72米 (前向)
- Y: -6.4 - 6.4米 (横向)
- Z: -2.0 - 6.0米 (高度)

### 4. 如果有原始数据
修改 `visualize_results.py` 中的 `load_radar_or_lidar_data()` 函数来加载原始雷达或激光雷达数据。

---

## ⚠️ 常见问题

### 无法显示 3D 窗口
```bash
pip install open3d matplotlib numpy
```

### 中文显示乱码
这是终端编码问题，不影响可视化功能

### "没有预测/真实标签"
- 检查文件路径
- 确认 txt 文件非空
- 检查是否有 "dummy" 行

---

## 🔗 相关资源

- **K-Radar 数据集**: https://github.com/kaist-avelab/K-Radar
- **Open3D 文档**: http://www.open3d.org/
- **DPRT 项目**: 你的当前项目

---

## 📞 下一步

1. ✅ 运行 `python quick_visualize.py 000000` 测试
2. ✅ 查看 `QUICKSTART.md` 了解基本用法
3. ✅ 阅读 `VISUALIZATION_GUIDE.md` 深入学习
4. ✅ 根据需要修改脚本添加自定义功能

---

**立即开始**: `python quick_visualize.py 000000` 🚀

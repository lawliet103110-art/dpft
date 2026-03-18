# LiDAR参考Radar的FOV机制处理方案

## 概述

参考radar_info.py的设计，为LiDAR创建类似的FOV（Field of View）参数管理机制。

## 为什么要参考Radar的FOV方式？

### Radar的优势设计

查看 `src/dprt/datasets/kradar/utils/radar_info.py`:

```python
# Radar使用预定义的传感器参数
azimuth_raster = [-53, ..., +53]  # 方位角FOV: ±53°
elevation_raster = [-18, ..., +18]  # 俯仰角FOV: ±18°
range_raster = [0.0, ..., 118.03710938]  # 距离范围: 0-118米

# 固定归一化范围
max_power = 200  # dB
min_power = 100  # dB
```

**在dataset.py中的使用**（第304-325行）：

```python
def _get_radar_ra_projection(self) -> torch.Tensor:
    return torch.Tensor([
        [0, -1, 0, (len(radar_info.azimuth_raster) - 1) / 2],
        [len(radar_info.range_raster) / max(radar_info.range_raster), 0, 0, 0],
        [0, 0, 0, 1]
    ])

def scale_radar_data(self, sample: Dict[str, torch.Tensor]):
    sample[k] = (v - radar_info.min_power) / (radar_info.max_power - radar_info.min_power) * 255
```

### LiDAR采用相同设计的优势

1. **科学性**：基于传感器规格，而非硬编码
2. **一致性**：与radar保持相同的设计模式
3. **可维护性**：参数集中管理，修改方便
4. **稳定性**：固定范围归一化，避免batch间差异

---

## 新的文件结构

### 1. lidar_info.py（新建）

路径：`src/dprt/datasets/kradar/utils/lidar_info.py`

```python
"""LiDAR sensor (OS1-128) data rasterization information."""

# OS1-128规格
azimuth_fov = (-180.0, 180.0)  # 水平FOV
elevation_fov = (-16.6, 16.6)  # 垂直FOV
max_range = 120.0  # 最大距离

# BEV投影范围（自动检测）
x_range_default = (-30.0, 30.0)  # 前后范围
y_range_default = (-30.0, 30.0)  # 左右范围

# 归一化参数
min_intensity = 0.0
max_intensity = 100.0
min_range = 0.0
max_range_norm = 50000.0

# 其他参数
bev_resolution = 256
z_min = -2.0
z_max = 6.0
```

### 2. dataset.py（修改）

**导入lidar_info**（第17行）：
```python
from dprt.datasets.kradar.utils import radar_info
from dprt.datasets.kradar.utils import lidar_info  # 新增
```

**使用lidar_info的归一化**（第352-390行）：
```python
def scale_lidar_data(self, sample):
    # 使用固定范围（类似radar）
    intensity_scaled = \
        (intensity_channels - lidar_info.min_intensity) \
        / (lidar_info.max_intensity - lidar_info.min_intensity) * 255

    range_scaled = \
        (range_channels - lidar_info.min_range) \
        / (lidar_info.max_range_norm - lidar_info.min_range) * 255
```

**使用lidar_info的投影矩阵**（第392-415行）：
```python
def _get_lidar_bev_projection(self) -> torch.Tensor:
    # 从lidar_info读取参数
    img_width = lidar_info.bev_resolution
    img_height = lidar_info.bev_resolution
    x_min, x_max = lidar_info.x_range_default
    y_min, y_max = lidar_info.y_range_default

    # 计算投影矩阵
    return torch.Tensor([...])
```

**project_lidar_to_bev使用默认值**（第417-441行）：
```python
def project_lidar_to_bev(self, point_cloud,
                        img_size=None, x_range=None, y_range=None):
    # 使用lidar_info的默认值
    if img_size is None:
        img_size = (lidar_info.bev_resolution, lidar_info.bev_resolution)
    if x_range is None:
        x_range = lidar_info.x_range_default
    if y_range is None:
        y_range = lidar_info.y_range_default
```

---

## 对比：修改前 vs 修改后

### 修改前（硬编码方式）

```python
# dataset.py第419-420行
x_range: Tuple[float, float] = (0, 100),      # ❌ 硬编码
y_range: Tuple[float, float] = (-50, 50)      # ❌ 硬编码

# 第372-387行：自适应归一化
intensity_min = intensity_channels.min()       # ❌ 每个batch不同
intensity_max = intensity_channels.max()       # ❌ 每个batch不同
```

**问题**：
- 投影范围(0, 100)不匹配你的数据(-17.7, -11.2)
- 自适应归一化导致batch间值域不一致
- 参数分散，难以维护

### 修改后（FOV机制）

```python
# lidar_info.py
x_range_default = (-30.0, 30.0)               # ✓ 基于数据分析
y_range_default = (-30.0, 30.0)               # ✓ 可配置

# dataset.py
if x_range is None:
    x_range = lidar_info.x_range_default      # ✓ 使用配置

# 固定范围归一化
intensity_scaled = (v - lidar_info.min_intensity) / \
                   (lidar_info.max_intensity - lidar_info.min_intensity) * 255
```

**优势**：
- 参数基于实际数据统计
- 固定归一化保证一致性
- 集中管理，易于调整

---

## 使用流程

### 步骤1：自动分析数据并更新lidar_info.py

```bash
# 在服务器上运行
cd /root/autodl-tmp/autodl-tmp/DPFT-main

# 分析数据范围（预览）
python update_lidar_info.py --src /root/autodl-tmp/autodl-tmp/data/kradar --samples 20

# 应用更新
python update_lidar_info.py --src /root/autodl-tmp/autodl-tmp/data/kradar --samples 20 --apply
```

**输出示例**：
```
分析点云数据范围
======================================================================
✓ 分析了 20 个点云文件

实际数据范围:
  x         : [  -18.50,   -10.20]
  y         : [  -15.30,    16.80]
  z         : [   -1.85,     5.20]
  intensity : [    5.20,    18.50]
  range     : [ 12583.00, 19686.00]

建议的配置参数:
  x_range_default: (-21.33, -7.37)
  y_range_default: (-18.51, 20.01)
  z_min: -2.24, z_max: 5.87
  min_intensity: 4.54, max_intensity: 19.16
  min_range: 12071.44, max_range_norm: 20241.15

✓ lidar_info.py 更新完成
```

### 步骤2：验证配置

```bash
# Python快速验证
python -c "
from dprt.datasets.kradar.utils import lidar_info
print('X range:', lidar_info.x_range_default)
print('Y range:', lidar_info.y_range_default)
print('Intensity:', lidar_info.min_intensity, '-', lidar_info.max_intensity)
"
```

### 步骤3：重新训练

```bash
python -m dprt.train \
    --src /root/autodl-tmp/autodl-tmp/data/kradar \
    --cfg config/kradar_4modality.json \
    --dst log/4modality_with_fov
```

---

## 技术细节

### 投影矩阵计算

参考radar的方式（dataset.py第309-325行），LiDAR的投影矩阵为：

```python
# Radar方式
P_radar = [
    [0, -1, 0, (len(azimuth_raster) - 1) / 2],
    [len(range_raster) / max(range_raster), 0, 0, 0],
    [0, 0, 0, 1]
]

# LiDAR方式（类似）
x_min, x_max = lidar_info.x_range_default
y_min, y_max = lidar_info.y_range_default
x_range = x_max - x_min
y_range = y_max - y_min

P_lidar = [
    [0, -img_width / y_range, 0, img_width * (-y_min / y_range)],
    [img_height / x_range, 0, 0, img_height * (-x_min / x_range)],
    [0, 0, 0, 1]
]
```

**数学原理**：
- 第1行：Y坐标映射到图像宽度
- 第2行：X坐标映射到图像高度
- 偏移量考虑范围的最小值（支持负坐标）

### 归一化一致性

**Radar归一化**（dataset.py第342-344行）：
```python
sample[k] = (v - radar_info.min_power) / \
            (radar_info.max_power - radar_info.min_power) * 255
```

**LiDAR归一化**（新）：
```python
intensity_scaled = (v - lidar_info.min_intensity) / \
                   (lidar_info.max_intensity - lidar_info.min_intensity) * 255

range_scaled = (v - lidar_info.min_range) / \
               (lidar_info.max_range_norm - lidar_info.min_range) * 255
```

**关键优势**：
- 所有batch使用相同的归一化范围
- 训练和测试一致
- 避免极值样本扭曲分布

---

## 故障排除

### 问题1：导入错误
```
ImportError: cannot import name 'lidar_info'
```

**解决**：
```bash
# 确保文件存在
ls src/dprt/datasets/kradar/utils/lidar_info.py

# 检查语法
python -m py_compile src/dprt/datasets/kradar/utils/lidar_info.py
```

### 问题2：投影后图像全黑

**原因**：x_range_default/y_range_default不匹配数据

**解决**：
```bash
# 重新运行分析
python update_lidar_info.py --src /data/kradar --samples 50 --apply

# 或手动修改lidar_info.py
```

### 问题3：归一化值异常

**原因**：min_intensity/max_intensity范围不准确

**解决**：
```bash
# 增加样本数重新分析
python update_lidar_info.py --src /data/kradar --samples 100 --apply
```

---

## 与原方案的兼容性

dataset.py的project_lidar_to_bev方法仍然支持手动指定参数：

```python
# 使用lidar_info默认值
bev = self.project_lidar_to_bev(point_cloud)

# 或手动覆盖（用于实验）
bev = self.project_lidar_to_bev(
    point_cloud,
    x_range=(-50, 50),
    y_range=(-40, 40)
)
```

---

## 总结

### 改进点

1. ✅ 参考radar_info.py设计
2. ✅ 创建lidar_info.py集中管理参数
3. ✅ 固定范围归一化（替代自适应）
4. ✅ 基于实际数据自动生成配置
5. ✅ 保持与radar一致的代码风格

### 预期效果

- **性能提升**：正确的投影范围覆盖数据
- **稳定训练**：固定归一化消除batch间波动
- **易于维护**：参数集中管理，调整方便
- **科学规范**：基于传感器规格和数据统计

### 文件清单

**新建文件**：
- `src/dprt/datasets/kradar/utils/lidar_info.py` - FOV参数配置
- `update_lidar_info.py` - 自动分析工具

**修改文件**：
- `src/dprt/datasets/kradar/dataset.py` - 使用lidar_info

**现在立即运行**：
```bash
python update_lidar_info.py \
    --src /root/autodl-tmp/autodl-tmp/data/kradar \
    --samples 20 \
    --apply
```

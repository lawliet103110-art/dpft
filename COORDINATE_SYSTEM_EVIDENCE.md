# 坐标系深度分析：基于processor.py和visu.py的证据

## 核心发现：点云和Labels在不同坐标系！

### 证据链条

#### 证据1：processor.py的数据处理流程

```python
# processor.py 第671-686行
def prepare_sample(self, sample, description, dst):
    # 1. 加载标定信息（用于坐标变换）
    ra_to_lidar, ea_to_lidar = self.get_radar_calibration(
        sample['calib_radar_lidar']
    )
    mono_to_lidar, stereo_to_lidar = self.get_camera_calibration(
        sample['calib_camera_lidar']
    )

    # 2. 转换Labels到LiDAR坐标系
    radar_to_lidar = self.get_translation(sample['calib_radar_lidar'])
    boxes = self._transform_boxes(boxes, radar_to_lidar)  # ← 转换！

    # 3. 加载LiDAR点云
    os1 = self.get_lidar_data(sample['os1'])  # ← 没有转换！

    # 4. 保存
    np.save(osp.join(dst, 'labels.npy'), boxes, ...)  # Labels已转换
    np.save(osp.join(dst, 'os1.npy'), os1, ...)        # 点云未转换
```

**关键发现**：
- ✅ Labels被**主动转换**到LiDAR坐标系
- ❌ 点云**没有转换**，保持原始坐标系

#### 证据2：get_lidar_data的实现

```python
# processor.py 第543-571行
def get_lidar_data(self, filename: str) -> np.ndarray:
    # 直接从.pcd文件加载
    pc = pypcd.PointCloud.from_path(filename)
    pc_data = pc.pc_data

    # 转换为numpy数组
    point_cloud = np.array([
        pc_data["x"], pc_data["y"], pc_data["z"],  # ← 直接使用x,y,z
        pc_data["intensity"], pc_data["t"],
        pc_data["reflectivity"], pc_data["ring"],
        pc_data["ambient"], pc_data["range"],
    ], dtype=self._dtype).T

    # 只过滤无效点
    point_cloud = point_cloud[np.where(np.abs(point_cloud[:, 0]) > 0.01)]

    return point_cloud  # ← 没有应用任何transformation矩阵！
```

**对比Camera/Radar的处理**：
- Camera: 保存`mono_info.npy` = `mono_to_lidar`标定矩阵（第693行）
- Radar: 保存`ra_info.npy` = `ra_to_lidar`标定矩阵（第697行）
- LiDAR: **没有保存任何标定矩阵**（因为假设已在LiDAR坐标系）

#### 证据3：可视化代码的假设

```python
# visualize_results.py 第119行
visu_lidar_data(pc, boxes=gt_boxes,
                xlim=(0, 72),      # ← FOV的X范围
                ylim=(-6.4, 6.4))  # ← FOV的Y范围

# 对比config中的FOV定义
"fov": {
    "x": [0.0, 72.0],
    "y": [-6.4, 6.4]
}
```

**分析**：
- 如果点云在世界坐标系（[-131, 245]×[-269, 74]），使用xlim=(0, 72)没有意义
- 作者**假设点云在LiDAR坐标系**，所以用FOV范围过滤

#### 证据4：visu_lidar_data同时显示点云和boxes

```python
# visu.py 第79-147行
def visu_lidar_data(pc, boxes=None, xlim=None, ylim=None):
    # 过滤点云
    if xlim is not None:
        pc = pc[np.logical_and(pc[:, 0] > xlim[0], pc[:, 0] < xlim[1])]

    # 显示点云
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pc[:, :3])
    vis.add_geometry(pcd)

    # 显示bounding boxes（直接使用box[:3]作为center）
    if boxes is not None:
        for box in boxes:
            bbox = o3d.geometry.OrientedBoundingBox()
            bbox.center = box[:3]  # ← 没有坐标转换！
            vis.add_geometry(bbox)
```

**如果点云和boxes在不同坐标系**：
- 可视化会错位（boxes在一个地方，点云在另一个地方）
- 但代码**直接叠加显示**，说明作者假设它们在同一坐标系

---

## 矛盾：代码假设 vs 实际数据

### 代码假设（DPRT作者的预期）

```
Labels:  X∈[0, 72], Y∈[-6.4, 6.4]      ← LiDAR坐标系
点云:    X∈[0, 72], Y∈[-6.4, 6.4]      ← LiDAR坐标系（假设）
可视化:  xlim=(0, 72), ylim=(-6.4, 6.4)  ← 一致
```

### 实际数据（你的K-Radar数据）

```
Labels:  X∈[0, 72], Y∈[-6.4, 6.4]        ← LiDAR坐标系✓
点云:    X∈[-131, 245], Y∈[-269, 74]     ← 世界坐标系？✗
ROI覆盖: 29%（FOV内点数/总点数）          ← 不匹配！
```

---

## 为什么会有这个矛盾？

### 可能性1：K-Radar数据集的.pcd文件存储格式

K-Radar原始数据集可能：
- `.pcd`文件存储的是**世界坐标系**的点云
- Labels从radar坐标系转换到LiDAR坐标系（相对车辆中心）
- DPRT代码假设`.pcd`已经预处理到LiDAR坐标系，但**实际没有**

### 可能性2：缺失的预处理步骤

DPRT代码可能期望用户：
1. 手动将`.pcd`文件转换到LiDAR坐标系
2. 或者使用某个未公开的预处理脚本
3. 但这一步骤**没有在文档中说明**

### 可能性3：K-Radar数据集版本差异

- K-Radar可能有多个版本
- 早期版本：点云在世界坐标系
- 后期版本：点云预处理到LiDAR坐标系
- DPRT代码为后期版本编写，但你的数据是早期版本

---

## 验证：检查原始.pcd文件

你可以在服务器上检查原始的`.pcd`文件头：

```bash
# 查看.pcd文件的前20行（包含元数据）
head -n 20 /root/autodl-tmp/autodl-tmp/raw_data/1/os1-128/os1-128_000000.pcd
```

**期望看到**：
```
# .PCD v0.7 - Point Cloud Data file format
VERSION 0.7
FIELDS x y z intensity t reflectivity ring ambient range
SIZE 4 4 4 4 4 2 1 2 4
TYPE F F F F F U U U F
COUNT 1 1 1 1 1 1 1 1 1
WIDTH 126227
HEIGHT 1
VIEWPOINT 0 0 0 1 0 0 0  ← 关键：视点信息
POINTS 126227
DATA binary
...
```

**VIEWPOINT字段**可能提供坐标系信息。

---

## 你的数据分析

### 实际测量结果

```python
点云范围:
  X: [-131.24, 245.39] = 376米跨度  ← 异常大！
  Y: [-269.72, 74.50] = 344米跨度   ← 异常大！

Labels范围:
  X: [0, 72]
  Y: [-6.4, 6.4]

ROI覆盖率: 29%  # FOV [0,72]×[-6.4,6.4] 内的点数
```

### 分析

#### 如果点云在LiDAR坐标系

**正常范围应该是**：
- OS1-128最大量程：120米
- 360°扫描：X∈[-120, 120], Y∈[-120, 120]
- 总跨度：≤240米

**你的数据**：
- X跨度：376米 ← **超出正常范围**
- Y跨度：344米 ← **超出正常范围**

**结论**：**点云不太可能在LiDAR坐标系**

#### 如果点云在世界坐标系

**合理解释**：
- 世界坐标系原点是地图中心
- 车辆位于世界坐标的某个位置
- 点云范围[-131, 245]×[-269, 74]是相对地图原点
- 29%覆盖率是车辆FOV恰好覆盖世界坐标中的某个区域

**验证方法**：
```bash
# 在服务器上运行
python verify_coordinate_system.py /root/autodl-tmp/autodl-tmp/data/kradar
```

**判断标准**：
- 如果ROI覆盖率的**标准差<5%**（所有样本都稳定在25-35%）
  → **点云在世界坐标系**（不同坐标系的碰巧重合）

- 如果ROI覆盖率的**标准差>10%**（0-100%大幅变化）
  → **点云在LiDAR坐标系**（覆盖率变化是因为车辆移动）

---

## 解决方案

### 方案A：确认是世界坐标系，使用更大投影范围

如果验证确认点云在世界坐标系：

```bash
# 使用Moderate范围（折中方案）
python update_lidar_info_smart.py \
    --src /root/autodl-tmp/autodl-tmp/data/kradar \
    --range moderate \  # (-50, 150) × (-75, 75)
    --apply
```

**理由**：
- 不能用Conservative（0, 100）因为有负值
- Moderate覆盖(-50, 150)可以包含大部分有效点
- 覆盖率约45%，平衡分辨率和信息量

### 方案B：转换点云到LiDAR坐标系（推荐但复杂）

如果想修复根本问题，需要在`get_lidar_data`中添加坐标变换：

```python
# processor.py 第543行之后
def get_lidar_data(self, filename: str,
                   vehicle_pose: np.ndarray = None) -> np.ndarray:
    # 加载点云
    pc = pypcd.PointCloud.from_path(filename)
    point_cloud = np.array([...])

    # 如果提供车辆位姿，转换到LiDAR坐标系
    if vehicle_pose is not None:
        # 世界坐标 → LiDAR坐标
        point_cloud[:, :3] = transform_world_to_lidar(
            point_cloud[:, :3],
            vehicle_pose
        )

    return point_cloud
```

**问题**：需要车辆位姿信息（可能在K-Radar的其他文件中）

### 方案C：使用当前数据，接受29%覆盖率

**最简单的方案**：
1. 运行`verify_coordinate_system.py`确认覆盖率稳定
2. 使用Moderate范围更新lidar_info
3. 重新训练

**预期效果**：
- 从5%非零像素 → 20-30%非零像素
- mAP从-7点 → +2~5点（相对3模态）

---

## 立即执行

### 步骤1：验证坐标系

```bash
cd /root/autodl-tmp/autodl-tmp/DPFT-main

python verify_coordinate_system.py /root/autodl-tmp/autodl-tmp/data/kradar
```

### 步骤2：根据验证结果更新配置

**如果ROI覆盖率标准差<5%**（稳定，说明不同坐标系）：
```bash
python update_lidar_info_smart.py \
    --src /root/autodl-tmp/autodl-tmp/data/kradar \
    --range moderate \
    --apply
```

**如果ROI覆盖率标准差>10%**（变化大，说明同一坐标系）：
```bash
# 可能是数据异常，需要进一步调查
# 先用Moderate范围试试
python update_lidar_info_smart.py \
    --src /root/autodl-tmp/autodl-tmp/data/kradar \
    --range moderate \
    --apply
```

### 步骤3：重新训练

```bash
python -m dprt.train \
    --src /root/autodl-tmp/autodl-tmp/data/kradar \
    --cfg config/kradar_4modality.json \
    --dst log/4modality_fixed
```

---

## 总结

### 关键发现

1. **代码分析明确显示**：
   - Labels被转换到LiDAR坐标系 ✓
   - 点云没有被转换 ✗
   - 可视化假设两者在同一坐标系 ✓

2. **实际数据显示**：
   - Labels: X∈[0, 72]（符合预期）
   - 点云: X∈[-131, 245]（不符合预期）
   - ROI覆盖率: 29%（中间值，需验证稳定性）

3. **最可能的解释**：
   - K-Radar原始`.pcd`文件在**世界坐标系**
   - DPRT代码假设已预处理到**LiDAR坐标系**
   - 预处理步骤缺失或未说明

### 下一步

**运行验证脚本**，把结果发给我！

```bash
python verify_coordinate_system.py /root/autodl-tmp/autodl-tmp/data/kradar
```

重点关注：
- ✅ ROI覆盖率的**标准差**
- ✅ 多个样本的**点云范围一致性**
- ✅ Labels范围是否稳定

把输出发给我，我会给你精确的配置建议！

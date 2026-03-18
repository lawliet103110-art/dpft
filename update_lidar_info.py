#!/usr/bin/env python3
"""
自动分析点云范围并更新lidar_info.py配置文件

这个脚本会：
1. 分析实际点云数据的坐标范围
2. 自动生成合适的FOV参数
3. 更新lidar_info.py中的配置（类似radar_info）
"""

import sys
import numpy as np
import os
from pathlib import Path


def analyze_point_cloud_range(data_path, num_samples=20):
    """分析点云数据的实际范围"""

    print("="*70)
    print("分析点云数据范围")
    print("="*70)

    train_path = Path(data_path) / 'train'
    if not train_path.exists():
        print(f"✗ 训练数据路径不存在: {train_path}")
        return None

    all_x, all_y, all_z = [], [], []
    all_intensity, all_range = [], []

    # 遍历样本
    count = 0
    for seq_dir in sorted(train_path.iterdir()):
        if not seq_dir.is_dir():
            continue
        for sample_dir in sorted(seq_dir.iterdir())[:5]:  # 每个序列取5个
            os1_file = sample_dir / 'os1.npy'
            if os1_file.exists():
                try:
                    data = np.load(os1_file)
                    all_x.append(data[:, 0])
                    all_y.append(data[:, 1])
                    all_z.append(data[:, 2])
                    all_intensity.append(data[:, 3])
                    all_range.append(data[:, 8])
                    count += 1
                    if count >= num_samples:
                        break
                except Exception as e:
                    print(f"✗ 加载失败 {os1_file}: {e}")
        if count >= num_samples:
            break

    if count == 0:
        print("✗ 未找到任何os1.npy文件")
        return None

    print(f"✓ 分析了 {count} 个点云文件")

    # 合并所有数据
    all_x = np.concatenate(all_x)
    all_y = np.concatenate(all_y)
    all_z = np.concatenate(all_z)
    all_intensity = np.concatenate(all_intensity)
    all_range = np.concatenate(all_range)

    # 计算范围
    ranges = {
        'x': (float(all_x.min()), float(all_x.max())),
        'y': (float(all_y.min()), float(all_y.max())),
        'z': (float(all_z.min()), float(all_z.max())),
        'intensity': (float(all_intensity.min()), float(all_intensity.max())),
        'range': (float(all_range.min()), float(all_range.max()))
    }

    print("\n实际数据范围:")
    for key, (min_val, max_val) in ranges.items():
        print(f"  {key:10s}: [{min_val:8.2f}, {max_val:8.2f}]")

    return ranges


def generate_lidar_info_config(ranges):
    """根据实际范围生成lidar_info.py配置"""

    x_min, x_max = ranges['x']
    y_min, y_max = ranges['y']
    z_min, z_max = ranges['z']
    intensity_min, intensity_max = ranges['intensity']
    range_min, range_max = ranges['range']

    # 扩展范围10%以确保覆盖所有点
    x_margin = (x_max - x_min) * 0.1
    y_margin = (y_max - y_min) * 0.1
    z_margin = (z_max - z_min) * 0.1
    intensity_margin = (intensity_max - intensity_min) * 0.05
    range_margin = (range_max - range_min) * 0.05

    x_range = (x_min - x_margin, x_max + x_margin)
    y_range = (y_min - y_margin, y_max + y_margin)
    z_range = (z_min - z_margin, z_max + z_margin)
    intensity_range = (max(0, intensity_min - intensity_margin), intensity_max + intensity_margin)
    range_range = (max(0, range_min - range_margin), range_max + range_margin)

    print("\n建议的配置参数:")
    print(f"  x_range_default: ({x_range[0]:.2f}, {x_range[1]:.2f})")
    print(f"  y_range_default: ({y_range[0]:.2f}, {y_range[1]:.2f})")
    print(f"  z_min: {z_range[0]:.2f}, z_max: {z_range[1]:.2f}")
    print(f"  min_intensity: {intensity_range[0]:.2f}, max_intensity: {intensity_range[1]:.2f}")
    print(f"  min_range: {range_range[0]:.2f}, max_range_norm: {range_range[1]:.2f}")

    return {
        'x_range_default': x_range,
        'y_range_default': y_range,
        'z_min': z_range[0],
        'z_max': z_range[1],
        'min_intensity': intensity_range[0],
        'max_intensity': intensity_range[1],
        'min_range': range_range[0],
        'max_range_norm': range_range[1]
    }


def update_lidar_info(config):
    """更新lidar_info.py文件"""

    lidar_info_file = 'src/dprt/datasets/kradar/utils/lidar_info.py'

    if not os.path.exists(lidar_info_file):
        print(f"✗ 文件不存在: {lidar_info_file}")
        return False

    print("\n修改 lidar_info.py...")

    # 生成新的配置文件内容
    content = f'''"""LiDAR sensor (OS1-128) data rasterization information.

Auto-generated from actual point cloud data analysis.
Similar to radar_info.py structure.
"""

# OS1-128 LiDAR sensor specifications
# Reference: https://ouster.com/products/scanning-lidar/os1-sensor/

# Horizontal field of view (azimuth): 360 degrees
# For BEV projection, we typically focus on front sector
azimuth_fov = (-180.0, 180.0)  # degrees, full 360°

# Vertical field of view (elevation): 33.2 degrees
# OS1-128: -16.6° to +16.6°
elevation_fov = (-16.6, 16.6)  # degrees

# Maximum range: 120 meters (typical)
# But practical range for vehicle detection is shorter
max_range = 120.0  # meters

# BEV projection parameters (auto-detected from data)
# X-axis: forward direction (range in meters)
# Y-axis: lateral direction (range in meters)

# Auto-detected ranges with 10% margin
x_range_default = ({config['x_range_default'][0]:.2f}, {config['x_range_default'][1]:.2f})  # meters
y_range_default = ({config['y_range_default'][0]:.2f}, {config['y_range_default'][1]:.2f})  # meters

# Intensity normalization range (from OS1-128 data statistics)
# Typical intensity range for OS1-128
min_intensity = {config['min_intensity']:.2f}
max_intensity = {config['max_intensity']:.2f}

# Range normalization (raw range values from point cloud column 8)
min_range = {config['min_range']:.2f}
max_range_norm = {config['max_range_norm']:.2f}  # maximum observed range value

# Grid resolution for BEV projection
bev_resolution = 256  # pixels (256x256 grid)

# Z-axis filtering for BEV projection
# Only keep points within certain height range
z_min = {config['z_min']:.2f}  # meters, ground level
z_max = {config['z_max']:.2f}  # meters, typical vehicle/object height
'''

    # 备份
    if os.path.exists(lidar_info_file):
        with open(lidar_info_file, 'r', encoding='utf-8') as f:
            old_content = f.read()
        with open(lidar_info_file + '.backup', 'w', encoding='utf-8') as f:
            f.write(old_content)
        print(f"✓ 已备份到 {lidar_info_file}.backup")

    # 保存新配置
    with open(lidar_info_file, 'w', encoding='utf-8') as f:
        f.write(content)

    print(f"✓ {lidar_info_file} 更新完成")

    # 验证语法
    try:
        compile(content, lidar_info_file, 'exec')
        print("✓ Python语法验证通过")
        return True
    except SyntaxError as e:
        print(f"✗ 语法错误: {e}")
        print("  恢复备份...")
        if os.path.exists(lidar_info_file + '.backup'):
            os.rename(lidar_info_file + '.backup', lidar_info_file)
        return False


def main():
    import argparse

    parser = argparse.ArgumentParser(
        description='自动分析点云并更新lidar_info.py（参考radar_info机制）'
    )
    parser.add_argument('--src', type=str, required=True,
                       help='数据集路径（包含train子目录）')
    parser.add_argument('--samples', type=int, default=20,
                       help='分析的样本数量')
    parser.add_argument('--apply', action='store_true',
                       help='应用更新到lidar_info.py（默认只分析）')

    args = parser.parse_args()

    # 分析数据范围
    ranges = analyze_point_cloud_range(args.src, args.samples)

    if ranges is None:
        print("\n✗ 分析失败")
        return 1

    # 生成配置
    config = generate_lidar_info_config(ranges)

    if args.apply:
        print("\n" + "="*70)
        print("应用更新")
        print("="*70)

        success = update_lidar_info(config)

        if success:
            print("\n" + "="*70)
            print("✓ 更新完成！")
            print("="*70)
            print("\nlidar_info.py已更新为基于实际数据的FOV参数")
            print("类似radar_info.py的机制")
            print("\n现在可以重新训练模型：")
            print("  python -m dprt.train --src /data/kradar --cfg config/kradar_4modality.json --dst log/4modality_fixed")
        else:
            print("\n✗ 更新失败，请检查错误信息")
            return 1
    else:
        print("\n" + "="*70)
        print("分析完成（未应用更新）")
        print("="*70)
        print("\n如果参数看起来合理，运行以下命令应用更新：")
        print(f"  python {sys.argv[0]} --src {args.src} --apply")

    return 0


if __name__ == '__main__':
    sys.exit(main())

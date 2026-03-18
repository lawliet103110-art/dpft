#!/usr/bin/env python3
"""
检查LiDAR点云的坐标范围分布
用于判断是否需要过滤远距离点
"""

import numpy as np
from pathlib import Path
import sys


def analyze_point_cloud_distribution(data_path, num_samples=20):
    """分析多个样本的点云分布"""

    print("="*70)
    print("LiDAR点云坐标范围分析")
    print("="*70)

    train_path = Path(data_path) / 'train'
    if not train_path.exists():
        print(f"✗ 路径不存在: {train_path}")
        return

    all_samples = []
    count = 0

    for seq_dir in sorted(train_path.iterdir()):
        if not seq_dir.is_dir():
            continue
        for sample_dir in sorted(seq_dir.iterdir())[:3]:
            os1_file = sample_dir / 'os1.npy'
            if os1_file.exists():
                try:
                    data = np.load(os1_file)
                    x, y, z = data[:, 0], data[:, 1], data[:, 2]
                    intensity, range_vals = data[:, 3], data[:, 8]

                    sample_info = {
                        'file': os1_file.relative_to(train_path),
                        'n_points': len(data),
                        'x_min': x.min(), 'x_max': x.max(),
                        'y_min': y.min(), 'y_max': y.max(),
                        'z_min': z.min(), 'z_max': z.max(),
                        'intensity_min': intensity.min(), 'intensity_max': intensity.max(),
                        'range_min': range_vals.min(), 'range_max': range_vals.max()
                    }

                    all_samples.append(sample_info)

                    if count < 5:
                        print(f"\n样本 {count}: {sample_info['file']}")
                        print(f"  点数: {sample_info['n_points']:,}")
                        print(f"  X: [{sample_info['x_min']:8.2f}, {sample_info['x_max']:8.2f}]")
                        print(f"  Y: [{sample_info['y_min']:8.2f}, {sample_info['y_max']:8.2f}]")
                        print(f"  Z: [{sample_info['z_min']:8.2f}, {sample_info['z_max']:8.2f}]")
                        print(f"  Intensity: [{sample_info['intensity_min']:8.2f}, {sample_info['intensity_max']:8.2f}]")
                        print(f"  Range: [{sample_info['range_min']:8.2f}, {sample_info['range_max']:8.2f}]")

                    count += 1
                    if count >= num_samples:
                        break
                except Exception as e:
                    print(f"✗ 加载失败 {os1_file}: {e}")
        if count >= num_samples:
            break

    if not all_samples:
        print("✗ 未找到任何点云数据")
        return

    print(f"\n\n{'='*70}")
    print(f"统计分析（基于 {len(all_samples)} 个样本）")
    print("="*70)

    # 提取所有样本的范围
    x_mins = [s['x_min'] for s in all_samples]
    x_maxs = [s['x_max'] for s in all_samples]
    y_mins = [s['y_min'] for s in all_samples]
    y_maxs = [s['y_max'] for s in all_samples]

    print(f"\nX坐标范围:")
    print(f"  所有样本最小值: [{np.min(x_mins):.2f}, {np.max(x_mins):.2f}]")
    print(f"  所有样本最大值: [{np.min(x_maxs):.2f}, {np.max(x_maxs):.2f}]")
    print(f"  平均范围: [{np.mean(x_mins):.2f}, {np.mean(x_maxs):.2f}]")

    print(f"\nY坐标范围:")
    print(f"  所有样本最小值: [{np.min(y_mins):.2f}, {np.max(y_mins):.2f}]")
    print(f"  所有样本最大值: [{np.min(y_maxs):.2f}, {np.max(y_maxs):.2f}]")
    print(f"  平均范围: [{np.mean(y_mins):.2f}, {np.mean(y_maxs):.2f}]")

    # 分析点云在不同距离范围内的分布
    print(f"\n\n{'='*70}")
    print("点云距离分布分析")
    print("="*70)

    # 检查第一个样本的详细分布
    sample_file = all_samples[0]['file']
    full_path = train_path / sample_file
    data = np.load(full_path)
    x, y = data[:, 0], data[:, 1]

    total_points = len(x)

    # 定义不同的范围区域
    ranges = [
        ("前方0-50m, 左右±25m", (0, 50), (-25, 25)),
        ("前方0-100m, 左右±50m", (0, 100), (-50, 50)),
        ("前方-50-100m, 左右±50m", (-50, 100), (-50, 50)),
        ("前后±100m, 左右±100m", (-100, 100), (-100, 100)),
        ("前后±150m, 左右±150m", (-150, 150), (-150, 150)),
    ]

    print(f"\n示例样本: {sample_file} (总点数: {total_points:,})")
    print(f"\n不同投影范围的点云保留率:")

    for name, x_range, y_range in ranges:
        mask = (x >= x_range[0]) & (x < x_range[1]) & \
               (y >= y_range[0]) & (y < y_range[1])
        kept_points = mask.sum()
        percentage = kept_points / total_points * 100

        print(f"  {name:30s}: {kept_points:6,} / {total_points:6,} ({percentage:5.1f}%)")

    # 建议
    print(f"\n\n{'='*70}")
    print("建议")
    print("="*70)

    # 计算90%点云覆盖所需的范围
    x_sorted = np.sort(x)
    y_sorted = np.sort(y)

    # 90%分位数
    x_5th = x_sorted[int(len(x_sorted) * 0.05)]
    x_95th = x_sorted[int(len(x_sorted) * 0.95)]
    y_5th = y_sorted[int(len(y_sorted) * 0.05)]
    y_95th = y_sorted[int(len(y_sorted) * 0.95)]

    print(f"\n90%点云覆盖范围（排除5%极值）:")
    print(f"  X: [{x_5th:.2f}, {x_95th:.2f}]")
    print(f"  Y: [{y_5th:.2f}, {y_95th:.2f}]")

    # 计算合理的投影范围
    x_margin = (x_95th - x_5th) * 0.1
    y_margin = (y_95th - y_5th) * 0.1

    suggested_x = (x_5th - x_margin, x_95th + x_margin)
    suggested_y = (y_5th - y_margin, y_95th + y_margin)

    print(f"\n建议的投影范围（添加10%边距）:")
    print(f"  x_range_default = ({suggested_x[0]:.2f}, {suggested_x[1]:.2f})")
    print(f"  y_range_default = ({suggested_y[0]:.2f}, {suggested_y[1]:.2f})")

    print(f"\n或者使用常见的车载LiDAR范围:")
    print(f"  x_range_default = (0, 100)      # 前方100米")
    print(f"  y_range_default = (-50, 50)     # 左右各50米")
    print(f"  保留率约: {((x >= 0) & (x < 100) & (y >= -50) & (y < 50)).sum() / total_points * 100:.1f}%")

    print(f"\n或者更宽松的范围:")
    print(f"  x_range_default = (-50, 150)    # 后50米到前150米")
    print(f"  y_range_default = (-75, 75)     # 左右各75米")
    print(f"  保留率约: {((x >= -50) & (x < 150) & (y >= -75) & (y < 75)).sum() / total_points * 100:.1f}%")


if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("用法: python check_lidar_coordinate_range.py <数据集路径>")
        print("示例: python check_lidar_coordinate_range.py /root/autodl-tmp/autodl-tmp/data/kradar")
        sys.exit(1)

    data_path = sys.argv[1]
    analyze_point_cloud_distribution(data_path, num_samples=20)

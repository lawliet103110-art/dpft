#!/usr/bin/env python3
"""检查LiDAR点云的实际intensity和range分布"""

import numpy as np
import os
from pathlib import Path
import argparse

def check_intensity_range(data_path, num_samples=100):
    """统计LiDAR实际数据范围"""

    print("="*70)
    print("LiDAR Intensity & Range 实际分布统计")
    print("="*70)

    train_path = Path(data_path) / 'train'
    if not train_path.exists():
        print(f"错误：训练数据路径不存在: {train_path}")
        return None

    all_intensity = []
    all_range = []
    all_x, all_y, all_z = [], [], []

    count = 0
    print(f"\n正在扫描 {num_samples} 个样本...")

    for seq_dir in sorted(train_path.iterdir()):
        if not seq_dir.is_dir():
            continue

        for sample_dir in sorted(seq_dir.iterdir()):
            if count >= num_samples:
                break

            os1_file = sample_dir / 'os1.npy'
            if not os1_file.exists():
                continue

            try:
                data = np.load(os1_file)
                if data.ndim != 2 or data.shape[1] != 9:
                    continue

                # 提取各列
                x, y, z = data[:, 0], data[:, 1], data[:, 2]
                intensity = data[:, 3]
                range_val = data[:, 8]

                # 应用FOV过滤（匹配训练代码）
                x_min, x_max = 0, 72
                y_min, y_max = -6.4, 6.4
                z_min, z_max = -2, 6

                mask = (x >= x_min) & (x < x_max) & \
                       (y >= y_min) & (y < y_max) & \
                       (z >= z_min) & (z < z_max)

                all_intensity.append(intensity[mask])
                all_range.append(range_val[mask])
                all_x.append(x[mask])
                all_y.append(y[mask])
                all_z.append(z[mask])

                count += 1

                if count % 20 == 0:
                    print(f"  已处理 {count}/{num_samples} 个样本")

            except Exception as e:
                print(f"  跳过文件 {os1_file}: {e}")
                continue

        if count >= num_samples:
            break

    if count == 0:
        print("错误：未找到任何有效的os1.npy文件")
        return None

    print(f"✓ 成功分析 {count} 个样本\n")

    # 合并所有数据
    all_intensity = np.concatenate(all_intensity)
    all_range = np.concatenate(all_range)
    all_x = np.concatenate(all_x)
    all_y = np.concatenate(all_y)
    all_z = np.concatenate(all_z)

    print("="*70)
    print("统计结果（FOV过滤后）")
    print("="*70)

    # Intensity统计
    print(f"\n【Intensity (反射强度)】")
    print(f"  总点数: {len(all_intensity):,}")
    print(f"  最小值: {all_intensity.min():.2f}")
    print(f"  最大值: {all_intensity.max():.2f}")
    print(f"  均值:   {all_intensity.mean():.2f}")
    print(f"  中位数: {np.median(all_intensity):.2f}")
    print(f"  标准差: {all_intensity.std():.2f}")

    print(f"\n  分位数:")
    percentiles = [0.1, 1, 5, 25, 50, 75, 95, 99, 99.9]
    for p in percentiles:
        val = np.percentile(all_intensity, p)
        print(f"    {p:5.1f}%: {val:8.2f}")

    # Range统计
    print(f"\n【Range (距离原始值, column 8)】")
    print(f"  总点数: {len(all_range):,}")
    print(f"  最小值: {all_range.min():.2f}")
    print(f"  最大值: {all_range.max():.2f}")
    print(f"  均值:   {all_range.mean():.2f}")
    print(f"  中位数: {np.median(all_range):.2f}")
    print(f"  标准差: {all_range.std():.2f}")

    print(f"\n  分位数:")
    for p in percentiles:
        val = np.percentile(all_range, p)
        print(f"    {p:5.1f}%: {val:8.2f}")

    # XYZ范围（验证FOV设置）
    print(f"\n【空间坐标范围】")
    print(f"  X: [{all_x.min():.2f}, {all_x.max():.2f}]")
    print(f"  Y: [{all_y.min():.2f}, {all_y.max():.2f}]")
    print(f"  Z: [{all_z.min():.2f}, {all_z.max():.2f}]")

    # 推荐配置
    print("\n" + "="*70)
    print("推荐配置参数")
    print("="*70)

    # 使用0.1-99.9分位数避免极端异常值
    intensity_min = np.percentile(all_intensity, 0.1)
    intensity_max = np.percentile(all_intensity, 99.9)
    range_min = np.percentile(all_range, 0.1)
    range_max = np.percentile(all_range, 99.9)

    print(f"\n【推荐方案1：基于0.1-99.9分位数（保守）】")
    print(f"min_intensity = {intensity_min:.2f}")
    print(f"max_intensity = {intensity_max:.2f}")
    print(f"min_range = {range_min:.2f}")
    print(f"max_range_norm = {range_max:.2f}")

    # 计算覆盖率
    coverage_intensity = ((all_intensity >= intensity_min) &
                          (all_intensity <= intensity_max)).mean() * 100
    coverage_range = ((all_range >= range_min) &
                      (all_range <= range_max)).mean() * 100
    print(f"  覆盖率: intensity {coverage_intensity:.1f}%, range {coverage_range:.1f}%")

    # 与可视化代码对比
    print(f"\n【推荐方案2：基于可视化代码（简单）】")
    print(f"min_intensity = 0.0")
    print(f"max_intensity = 2048.0")
    print(f"min_range = {range_min:.2f}  # 需要根据实际分布调整")
    print(f"max_range_norm = {range_max:.2f}")

    coverage_intensity_vis = ((all_intensity >= 0) &
                              (all_intensity <= 2048)).mean() * 100
    print(f"  覆盖率: intensity {coverage_intensity_vis:.1f}%")

    # 当前配置的问题
    print(f"\n【当前配置问题诊断】")
    current_intensity_min = 2.0
    current_intensity_max = 299.0
    current_range_min = 4680.0
    current_range_max = 16880.04

    below_min_intensity = (all_intensity < current_intensity_min).mean() * 100
    above_max_intensity = (all_intensity > current_intensity_max).mean() * 100
    below_min_range = (all_range < current_range_min).mean() * 100
    above_max_range = (all_range > current_range_max).mean() * 100

    print(f"  当前配置: intensity [{current_intensity_min}, {current_intensity_max}]")
    print(f"    ❌ {below_min_intensity:.1f}% 的点 < {current_intensity_min} (会被裁剪到0)")
    print(f"    ❌ {above_max_intensity:.1f}% 的点 > {current_intensity_max} (会被裁剪到255)")
    print(f"    ✓ 仅 {100-below_min_intensity-above_max_intensity:.1f}% 的点在有效范围")

    print(f"\n  当前配置: range [{current_range_min}, {current_range_max}]")
    print(f"    ❌ {below_min_range:.1f}% 的点 < {current_range_min} (会被裁剪到0)")
    print(f"    ❌ {above_max_range:.1f}% 的点 > {current_range_max} (会被裁剪到255)")
    print(f"    ✓ 仅 {100-below_min_range-above_max_range:.1f}% 的点在有效范围")

    print("\n" + "="*70)
    print("下一步操作")
    print("="*70)
    print("\n1. 更新 src/dprt/datasets/kradar/utils/lidar_info.py:")
    print("   - 根据上述推荐方案选择一个")
    print("   - 推荐使用方案1（更准确）或方案2（更简单）")
    print("\n2. 验证数据质量:")
    print("   python diagnose_lidar_bev.py /data/kradar/processed config/kradar_4modality.json")
    print("\n3. 如果诊断结果改善（稀疏性降低、均值提升），重新训练模型")

    return {
        'intensity_range': (intensity_min, intensity_max),
        'range_range': (range_min, range_max),
        'coverage': (coverage_intensity, coverage_range)
    }

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='检查LiDAR intensity和range的实际分布')
    parser.add_argument('data_path', type=str, help='数据集路径（包含train子目录）')
    parser.add_argument('--samples', type=int, default=100,
                       help='分析的样本数量（默认100）')

    args = parser.parse_args()
    check_intensity_range(args.data_path, args.samples)

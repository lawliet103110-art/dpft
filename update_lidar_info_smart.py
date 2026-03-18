#!/usr/bin/env python3
"""
智能LiDAR配置更新工具
根据实际点云分布自动选择合理的投影范围
"""

import sys
import numpy as np
import os
from pathlib import Path


def analyze_and_recommend(data_path, num_samples=20):
    """分析点云并推荐配置"""

    print("="*70)
    print("智能LiDAR配置分析")
    print("="*70)

    train_path = Path(data_path) / 'train'
    if not train_path.exists():
        print(f"✗ 训练数据路径不存在: {train_path}")
        return None

    all_x, all_y, all_z = [], [], []
    all_intensity, all_range = [], []

    count = 0
    for seq_dir in sorted(train_path.iterdir()):
        if not seq_dir.is_dir():
            continue
        for sample_dir in sorted(seq_dir.iterdir())[:3]:
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

    # 合并数据
    all_x = np.concatenate(all_x)
    all_y = np.concatenate(all_y)
    all_z = np.concatenate(all_z)
    all_intensity = np.concatenate(all_intensity)
    all_range = np.concatenate(all_range)

    # 基础统计
    print(f"\n实际数据范围（所有点）:")
    print(f"  X: [{all_x.min():.2f}, {all_x.max():.2f}]")
    print(f"  Y: [{all_y.min():.2f}, {all_y.max():.2f}]")
    print(f"  Z: [{all_z.min():.2f}, {all_z.max():.2f}]")
    print(f"  Intensity: [{all_intensity.min():.2f}, {all_intensity.max():.2f}]")
    print(f"  Range: [{all_range.min():.2f}, {all_range.max():.2f}]")

    # 分位数分析
    x_percentiles = np.percentile(all_x, [5, 25, 50, 75, 95])
    y_percentiles = np.percentile(all_y, [5, 25, 50, 75, 95])

    print(f"\nX坐标分位数 [5%, 25%, 50%, 75%, 95%]:")
    print(f"  {x_percentiles}")
    print(f"\nY坐标分位数 [5%, 25%, 50%, 75%, 95%]:")
    print(f"  {y_percentiles}")

    # 计算不同范围选项的覆盖率
    total_points = len(all_x)

    range_options = {
        'conservative': {
            'x_range': (0, 100),
            'y_range': (-50, 50),
            'description': '保守范围（前方100m，左右各50m）- 适合城市道路'
        },
        'moderate': {
            'x_range': (-50, 150),
            'y_range': (-75, 75),
            'description': '适中范围（后50m到前150m，左右各75m）- 推荐'
        },
        'wide': {
            'x_range': (-100, 200),
            'y_range': (-100, 100),
            'description': '宽松范围（后100m到前200m，左右各100m）- 高速公路'
        },
        'percentile_90': {
            'x_range': (float(x_percentiles[0]), float(x_percentiles[4])),
            'y_range': (float(y_percentiles[0]), float(y_percentiles[4])),
            'description': '90%分位数范围（排除极值）- 数据驱动'
        }
    }

    print(f"\n{'='*70}")
    print("投影范围选项对比")
    print("="*70)

    for option_name, option in range_options.items():
        x_min, x_max = option['x_range']
        y_min, y_max = option['y_range']

        mask = (all_x >= x_min) & (all_x < x_max) & \
               (all_y >= y_min) & (all_y < y_max)
        coverage = mask.sum() / total_points * 100

        print(f"\n选项: {option_name}")
        print(f"  {option['description']}")
        print(f"  X范围: [{x_min:.2f}, {x_max:.2f}]")
        print(f"  Y范围: [{y_min:.2f}, {y_max:.2f}]")
        print(f"  点云覆盖率: {coverage:.1f}%")

        # 计算归一化参数
        intensity_subset = all_intensity[mask] if mask.sum() > 0 else all_intensity
        range_subset = all_range[mask] if mask.sum() > 0 else all_range

        option['intensity_range'] = (
            float(np.percentile(intensity_subset, 1)),
            float(np.percentile(intensity_subset, 99))
        )
        option['range_range'] = (
            float(np.percentile(range_subset, 1)),
            float(np.percentile(range_subset, 99))
        )
        option['z_range'] = (
            float(np.percentile(all_z, 1)),
            float(np.percentile(all_z, 99))
        )
        option['coverage'] = coverage

    # 推荐选项（选择覆盖率在70-90%之间的最优选项）
    recommended = None
    for name, option in range_options.items():
        if 70 <= option['coverage'] <= 90:
            if recommended is None or option['coverage'] > recommended['coverage']:
                recommended = {'name': name, **option}

    if recommended is None:
        # 如果没有合适的，选择moderate
        recommended = {'name': 'moderate', **range_options['moderate']}

    print(f"\n{'='*70}")
    print(f"推荐配置: {recommended['name']}")
    print("="*70)
    print(f"  {recommended['description']}")
    print(f"  覆盖率: {recommended['coverage']:.1f}%")
    print(f"\n配置参数:")
    print(f"  x_range_default = {recommended['x_range']}")
    print(f"  y_range_default = {recommended['y_range']}")
    print(f"  z_min = {recommended['z_range'][0]:.2f}")
    print(f"  z_max = {recommended['z_range'][1]:.2f}")
    print(f"  min_intensity = {recommended['intensity_range'][0]:.2f}")
    print(f"  max_intensity = {recommended['intensity_range'][1]:.2f}")
    print(f"  min_range = {recommended['range_range'][0]:.2f}")
    print(f"  max_range_norm = {recommended['range_range'][1]:.2f}")

    return recommended, range_options


def update_lidar_info(config):
    """更新lidar_info.py文件"""

    lidar_info_file = '/root/autodl-tmp/autodl-tmp/DPFT-main/src/dprt/datasets/kradar/utils/lidar_info.py'

    if not os.path.exists(lidar_info_file):
        print(f"\n⚠ 文件不存在: {lidar_info_file}")
        print("  将创建新文件")

    print(f"\n修改 {lidar_info_file}...")

    # 生成配置文件
    x_min, x_max = config['x_range']
    y_min, y_max = config['y_range']

    content = f'''"""LiDAR sensor (OS1-128) data rasterization information.

Auto-generated from actual point cloud data analysis.
Similar to radar_info.py structure.

Configuration: {config.get('description', 'Custom')}
Point cloud coverage: {config['coverage']:.1f}%
"""

# OS1-128 LiDAR sensor specifications
# Reference: https://ouster.com/products/scanning-lidar/os1-sensor/

# Horizontal field of view (azimuth): 360 degrees
azimuth_fov = (-180.0, 180.0)  # degrees, full 360°

# Vertical field of view (elevation): 33.2 degrees
# OS1-128: -16.6° to +16.6°
elevation_fov = (-16.6, 16.6)  # degrees

# Maximum range: 120 meters (typical for OS1-128)
max_range = 120.0  # meters

# BEV projection parameters (auto-detected from data)
# X-axis: forward/backward direction (meters)
# Y-axis: left/right direction (meters)

# Selected range based on data analysis
x_range_default = ({x_min:.2f}, {x_max:.2f})  # meters
y_range_default = ({y_min:.2f}, {y_max:.2f})  # meters

# Intensity normalization range (from OS1-128 data statistics)
# Based on 1-99 percentile to avoid outliers
min_intensity = {config['intensity_range'][0]:.2f}
max_intensity = {config['intensity_range'][1]:.2f}

# Range normalization (raw range values from point cloud column 8)
# Based on 1-99 percentile to avoid outliers
min_range = {config['range_range'][0]:.2f}
max_range_norm = {config['range_range'][1]:.2f}

# Grid resolution for BEV projection
bev_resolution = 256  # pixels (256x256 grid)

# Z-axis filtering for BEV projection
# Only keep points within certain height range (based on 1-99 percentile)
z_min = {config['z_range'][0]:.2f}  # meters
z_max = {config['z_range'][1]:.2f}  # meters
'''

    # 备份
    if os.path.exists(lidar_info_file):
        with open(lidar_info_file, 'r', encoding='utf-8') as f:
            old_content = f.read()
        backup_file = lidar_info_file + '.backup'
        with open(backup_file, 'w', encoding='utf-8') as f:
            f.write(old_content)
        print(f"✓ 已备份到 {backup_file}")

    # 保存
    os.makedirs(os.path.dirname(lidar_info_file), exist_ok=True)
    with open(lidar_info_file, 'w', encoding='utf-8') as f:
        f.write(content)

    print(f"✓ {lidar_info_file} 更新完成")

    # 验证
    try:
        compile(content, lidar_info_file, 'exec')
        print("✓ Python语法验证通过")
        return True
    except SyntaxError as e:
        print(f"✗ 语法错误: {e}")
        if os.path.exists(lidar_info_file + '.backup'):
            os.rename(lidar_info_file + '.backup', lidar_info_file)
            print("  已恢复备份")
        return False


def main():
    import argparse

    parser = argparse.ArgumentParser(
        description='智能LiDAR配置工具 - 自动选择合理的投影范围'
    )
    parser.add_argument('--src', type=str, required=True,
                       help='数据集路径（包含train子目录）')
    parser.add_argument('--samples', type=int, default=20,
                       help='分析的样本数量（默认20）')
    parser.add_argument('--range', type=str,
                       choices=['conservative', 'moderate', 'wide', 'percentile_90', 'auto'],
                       default='auto',
                       help='选择范围策略（默认auto自动选择）')
    parser.add_argument('--apply', action='store_true',
                       help='应用更新到lidar_info.py（默认只分析）')

    args = parser.parse_args()

    # 分析数据
    result = analyze_and_recommend(args.src, args.samples)

    if result is None:
        print("\n✗ 分析失败")
        return 1

    recommended, all_options = result

    # 选择配置
    if args.range == 'auto':
        selected_config = recommended
        print(f"\n✓ 自动选择: {selected_config['name']}")
    else:
        selected_config = {'name': args.range, **all_options[args.range]}
        print(f"\n✓ 手动选择: {args.range}")

    if args.apply:
        print("\n" + "="*70)
        print("应用更新")
        print("="*70)

        success = update_lidar_info(selected_config)

        if success:
            print("\n" + "="*70)
            print("✓ 配置更新完成！")
            print("="*70)
            print(f"\n已应用配置: {selected_config['name']}")
            print(f"  {selected_config['description']}")
            print(f"  点云覆盖率: {selected_config['coverage']:.1f}%")
            print("\n下一步:")
            print("  1. 验证配置:")
            print("     python -c \"from dprt.datasets.kradar.utils import lidar_info; print('X:', lidar_info.x_range_default)\"")
            print("\n  2. 重新训练:")
            print("     python -m dprt.train --src /data/kradar --cfg config/kradar_4modality.json --dst log/4modality_fixed")
        else:
            print("\n✗ 更新失败")
            return 1
    else:
        print("\n" + "="*70)
        print("分析完成（未应用更新）")
        print("="*70)
        print("\n如果要应用推荐配置，运行:")
        print(f"  python {sys.argv[0]} --src {args.src} --apply")
        print(f"\n或选择特定范围:")
        print(f"  python {sys.argv[0]} --src {args.src} --range moderate --apply")

    return 0


if __name__ == '__main__':
    sys.exit(main())

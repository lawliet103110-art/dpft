#!/usr/bin/env python3
"""
自动分析点云范围并修复投影参数

这个脚本会：
1. 分析实际点云数据的坐标范围
2. 自动生成合适的投影参数
3. 修改dataset.py中的投影方法
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


def generate_projection_params(ranges):
    """根据实际范围生成投影参数"""

    x_min, x_max = ranges['x']
    y_min, y_max = ranges['y']

    # 扩展范围10%以确保覆盖所有点
    x_margin = (x_max - x_min) * 0.1
    y_margin = (y_max - y_min) * 0.1

    x_range = (x_min - x_margin, x_max + x_margin)
    y_range = (y_min - y_margin, y_max + y_margin)

    print("\n建议的投影范围:")
    print(f"  x_range: ({x_range[0]:.2f}, {x_range[1]:.2f})")
    print(f"  y_range: ({y_range[0]:.2f}, {y_range[1]:.2f})")

    return x_range, y_range


def fix_dataset_projection(x_range, y_range, intensity_range, range_range):
    """修改dataset.py中的投影和归一化参数"""

    dataset_file = 'src/dprt/datasets/kradar/dataset.py'

    if not os.path.exists(dataset_file):
        print(f"✗ 文件不存在: {dataset_file}")
        return False

    print("\n修改 dataset.py...")

    with open(dataset_file, 'r', encoding='utf-8') as f:
        content = f.read()

    # 备份
    with open(dataset_file + '.backup_projection', 'w', encoding='utf-8') as f:
        f.write(content)
    print(f"✓ 已备份到 {dataset_file}.backup_projection")

    # 修改project_lidar_to_bev的默认参数
    old_signature = '''    def project_lidar_to_bev(self, point_cloud: np.ndarray,
                            img_size: Tuple[int, int] = (256, 256),
                            x_range: Tuple[float, float] = (0, 100),
                            y_range: Tuple[float, float] = (-50, 50)) -> torch.Tensor:'''

    new_signature = f'''    def project_lidar_to_bev(self, point_cloud: np.ndarray,
                            img_size: Tuple[int, int] = (256, 256),
                            x_range: Tuple[float, float] = {x_range},
                            y_range: Tuple[float, float] = {y_range}) -> torch.Tensor:'''

    if old_signature in content:
        content = content.replace(old_signature, new_signature)
        print("✓ 更新了 project_lidar_to_bev 的默认范围")
    else:
        print("⚠ 未找到需要替换的函数签名，可能已经修改过")

    # 修改scale_lidar_data使用固定范围
    # 查找scale_lidar_data方法
    import re

    scale_method_pattern = r'def scale_lidar_data\(self.*?\n(?:.*?\n)*?        return sample'

    def generate_new_scale_method():
        intensity_min, intensity_max = intensity_range
        range_min, range_max = range_range

        return f'''def scale_lidar_data(self, sample: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Scales the lidar data to a range of 0 to 255 using fixed ranges

        Arguments:
            sample: Dictionary mapping the sample items to their data tensors.

        Returns:
            sample: Dictionary mapping the sample items to their scaled data tensors.
        """
        for k, v in sample.items():
            if k == 'lidar_top':
                # LiDAR BEV projection image has 6 channels
                # Intensity channels (0-2): using fixed range [{intensity_min:.1f}, {intensity_max:.1f}]
                intensity_channels = v[:, :, :3]
                intensity_scaled = torch.clip(
                    (intensity_channels - {intensity_min}) / ({intensity_max} - {intensity_min}) * 255,
                    0, 255
                )

                # Range channels (3-5): using fixed range [{range_min:.1f}, {range_max:.1f}]
                range_channels = v[:, :, 3:]
                range_scaled = torch.clip(
                    (range_channels - {range_min}) / ({range_max} - {range_min}) * 255,
                    0, 255
                )

                sample[k] = torch.cat([intensity_scaled, range_scaled], dim=-1)

        return sample'''

    new_scale_method = generate_new_scale_method()

    # 尝试替换
    match = re.search(scale_method_pattern, content, re.DOTALL)
    if match:
        content = content[:match.start()] + new_scale_method + content[match.end():]
        print("✓ 更新了 scale_lidar_data 使用固定归一化范围")
    else:
        print("⚠ 未找到 scale_lidar_data 方法")

    # 保存修改
    with open(dataset_file, 'w', encoding='utf-8') as f:
        f.write(content)

    print(f"✓ {dataset_file} 修改完成")

    # 验证语法
    try:
        compile(content, dataset_file, 'exec')
        print("✓ Python语法验证通过")
        return True
    except SyntaxError as e:
        print(f"✗ 语法错误: {e}")
        print("  恢复备份...")
        os.rename(dataset_file + '.backup_projection', dataset_file)
        return False


def main():
    import argparse

    parser = argparse.ArgumentParser(description='自动修复LiDAR投影参数')
    parser.add_argument('--src', type=str, required=True,
                       help='数据集路径（包含train子目录）')
    parser.add_argument('--samples', type=int, default=20,
                       help='分析的样本数量')
    parser.add_argument('--apply', action='store_true',
                       help='应用修复（默认只分析）')

    args = parser.parse_args()

    # 分析数据范围
    ranges = analyze_point_cloud_range(args.src, args.samples)

    if ranges is None:
        print("\n✗ 分析失败")
        return 1

    # 生成建议参数
    x_range, y_range = generate_projection_params(ranges)

    if args.apply:
        print("\n" + "="*70)
        print("应用修复")
        print("="*70)

        success = fix_dataset_projection(
            x_range, y_range,
            ranges['intensity'],
            ranges['range']
        )

        if success:
            print("\n" + "="*70)
            print("✓ 修复完成！")
            print("="*70)
            print("\n现在需要重新训练模型：")
            print("  python -m dprt.train --src /data/kradar --cfg config/kradar_4modality.json --dst log/4modality_fixed")
        else:
            print("\n✗ 修复失败，请检查错误信息")
            return 1
    else:
        print("\n" + "="*70)
        print("分析完成（未应用修复）")
        print("="*70)
        print("\n如果范围看起来合理，运行以下命令应用修复：")
        print(f"  python {sys.argv[0]} --src {args.src} --apply")

    return 0


if __name__ == '__main__':
    sys.exit(main())

#!/usr/bin/env python3
"""
LiDAR数据质量诊断脚本
用于诊断为什么加入LiDAR后性能下降
"""

import sys
import numpy as np
import torch
from pathlib import Path

try:
    from dprt.datasets import init as init_dataset
    from dprt.utils.config import load_config
except ImportError as e:
    print(f"导入错误: {e}")
    sys.exit(1)


def diagnose_lidar_data(data_path, num_samples=50):
    """诊断LiDAR数据质量"""

    print("="*70)
    print("LiDAR数据质量诊断")
    print("="*70)

    # 加载配置
    try:
        config = load_config('config/kradar_4modality.json')
        print("✓ 加载配置文件成功")
    except Exception as e:
        print(f"✗ 加载配置失败: {e}")
        return

    # 创建数据集
    try:
        dataset = init_dataset(
            dataset='kradar',
            src=data_path,
            split='train',
            config=config
        )
        print(f"✓ 数据集加载成功，共 {len(dataset)} 个样本")
    except Exception as e:
        print(f"✗ 数据集加载失败: {e}")
        import traceback
        traceback.print_exc()
        return

    print(f"\n检查前 {num_samples} 个样本的LiDAR数据...")
    print("-"*70)

    # 统计信息
    stats = {
        'empty_lidar': 0,           # 全零LiDAR数据
        'sparse_lidar': 0,          # 稀疏LiDAR（<5%非零像素）
        'normal_lidar': 0,          # 正常LiDAR
        'lidar_value_ranges': [],   # 数值范围
        'non_zero_ratios': [],      # 非零比例
        'channel_stats': {i: [] for i in range(6)}  # 每个通道的统计
    }

    errors = []

    for i in range(min(num_samples, len(dataset))):
        try:
            sample, label = dataset[i]

            # 检查LiDAR数据
            if 'lidar_top' not in sample:
                errors.append(f"样本 {i}: 缺少 lidar_top 键")
                continue

            lidar = sample['lidar_top']

            # 检查形状
            if lidar.shape != (256, 256, 6):
                errors.append(f"样本 {i}: LiDAR形状错误 {lidar.shape}")
                continue

            # 统计非零像素
            non_zero_pixels = (lidar.sum(dim=-1) > 0).sum().item()
            total_pixels = 256 * 256
            non_zero_ratio = non_zero_pixels / total_pixels

            stats['non_zero_ratios'].append(non_zero_ratio)

            # 分类
            if non_zero_ratio == 0:
                stats['empty_lidar'] += 1
            elif non_zero_ratio < 0.05:
                stats['sparse_lidar'] += 1
            else:
                stats['normal_lidar'] += 1

            # 数值范围
            lidar_min = lidar.min().item()
            lidar_max = lidar.max().item()
            stats['lidar_value_ranges'].append((lidar_min, lidar_max))

            # 每个通道的统计
            for ch in range(6):
                channel_data = lidar[:, :, ch]
                stats['channel_stats'][ch].append({
                    'min': channel_data.min().item(),
                    'max': channel_data.max().item(),
                    'mean': channel_data.mean().item(),
                    'std': channel_data.std().item()
                })

            # 显示前几个样本的详细信息
            if i < 5:
                print(f"\n样本 {i}:")
                print(f"  形状: {lidar.shape}")
                print(f"  非零像素: {non_zero_pixels}/{total_pixels} ({non_zero_ratio*100:.1f}%)")
                print(f"  数值范围: [{lidar_min:.2f}, {lidar_max:.2f}]")
                print(f"  各通道均值: {[lidar[:,:,ch].mean().item() for ch in range(6)]}")

        except Exception as e:
            errors.append(f"样本 {i}: {str(e)}")

    # 打印统计结果
    print("\n" + "="*70)
    print("统计结果")
    print("="*70)

    print(f"\n✓ 成功检查 {len(stats['non_zero_ratios'])} 个样本")
    print(f"  - 空LiDAR数据（全零）: {stats['empty_lidar']} ({stats['empty_lidar']/num_samples*100:.1f}%)")
    print(f"  - 稀疏LiDAR（<5%非零）: {stats['sparse_lidar']} ({stats['sparse_lidar']/num_samples*100:.1f}%)")
    print(f"  - 正常LiDAR（≥5%非零）: {stats['normal_lidar']} ({stats['normal_lidar']/num_samples*100:.1f}%)")

    if stats['non_zero_ratios']:
        print(f"\n非零像素比例:")
        print(f"  平均: {np.mean(stats['non_zero_ratios'])*100:.2f}%")
        print(f"  中位数: {np.median(stats['non_zero_ratios'])*100:.2f}%")
        print(f"  范围: [{np.min(stats['non_zero_ratios'])*100:.2f}%, {np.max(stats['non_zero_ratios'])*100:.2f}%]")

    if stats['lidar_value_ranges']:
        all_mins = [r[0] for r in stats['lidar_value_ranges']]
        all_maxs = [r[1] for r in stats['lidar_value_ranges']]
        print(f"\nLiDAR数值范围:")
        print(f"  最小值范围: [{np.min(all_mins):.2f}, {np.max(all_mins):.2f}]")
        print(f"  最大值范围: [{np.min(all_maxs):.2f}, {np.max(all_maxs):.2f}]")

    # 通道统计
    print(f"\n各通道统计（前3个为intensity，后3个为range）:")
    channel_names = ['intensity_max', 'intensity_median', 'intensity_var',
                     'range_max', 'range_median', 'range_var']
    for ch in range(6):
        if stats['channel_stats'][ch]:
            means = [s['mean'] for s in stats['channel_stats'][ch]]
            maxs = [s['max'] for s in stats['channel_stats'][ch]]
            print(f"  通道{ch} ({channel_names[ch]}):")
            print(f"    均值范围: [{np.min(means):.2f}, {np.max(means):.2f}]")
            print(f"    最大值范围: [{np.min(maxs):.2f}, {np.max(maxs):.2f}]")

    # 错误报告
    if errors:
        print(f"\n⚠ 发现 {len(errors)} 个错误:")
        for err in errors[:10]:  # 只显示前10个
            print(f"  - {err}")

    # 问题诊断
    print("\n" + "="*70)
    print("问题诊断")
    print("="*70)

    issues = []

    # 检查1：过多空数据
    if stats['empty_lidar'] > num_samples * 0.3:
        issues.append(f"⚠ 严重问题：{stats['empty_lidar']/num_samples*100:.0f}% 的LiDAR数据为全零！")
        issues.append("  可能原因：点云投影失败或数据文件损坏")

    # 检查2：稀疏数据
    if stats['sparse_lidar'] > num_samples * 0.5:
        issues.append(f"⚠ 问题：{stats['sparse_lidar']/num_samples*100:.0f}% 的LiDAR数据过于稀疏")
        issues.append("  可能原因：投影范围设置不当或点云密度低")

    # 检查3：数值范围
    if stats['lidar_value_ranges']:
        max_val = max([r[1] for r in stats['lidar_value_ranges']])
        if max_val < 10:
            issues.append(f"⚠ 问题：LiDAR最大值仅为 {max_val:.2f}，归一化可能有问题")
            issues.append("  期望范围应该在 [0, 255]")
        elif max_val > 300:
            issues.append(f"⚠ 问题：LiDAR最大值为 {max_val:.2f}，超出预期范围")
            issues.append("  期望范围应该在 [0, 255]")

    # 检查4：通道不平衡
    for ch in range(6):
        if stats['channel_stats'][ch]:
            maxs = [s['max'] for s in stats['channel_stats'][ch]]
            if max(maxs) < 1.0:
                issues.append(f"⚠ 问题：通道{ch} ({channel_names[ch]}) 数值过小（最大值<1）")

    if not issues:
        print("✓ 未发现明显问题")
        print("\n建议检查：")
        print("  1. 训练轮数是否足够（4模态可能需要更长训练时间）")
        print("  2. 学习率是否需要调整")
        print("  3. batch_size是否因内存限制而减小")
    else:
        print("发现以下问题：\n")
        for issue in issues:
            print(issue)

        print("\n建议修复方案：")
        if stats['empty_lidar'] > num_samples * 0.3:
            print("  1. 检查点云文件是否存在且格式正确")
            print("  2. 验证投影范围是否覆盖点云数据")
        if max_val < 10 or max_val > 300:
            print("  3. 修改 scale_lidar_data 的归一化逻辑")
            print("  4. 检查点云的原始数值范围")

    return stats, issues


def compare_with_without_lidar(data_path):
    """对比加载LiDAR前后的数据"""
    print("\n" + "="*70)
    print("对比3模态 vs 4模态数据加载")
    print("="*70)

    # 加载3模态配置
    try:
        config_3mod = load_config('config/kradar.json')
        dataset_3mod = init_dataset(
            dataset='kradar',
            src=data_path,
            split='train',
            config=config_3mod
        )
        sample_3mod, _ = dataset_3mod[0]
        print("\n✓ 3模态配置:")
        print(f"  输入模态: {config_3mod['model']['inputs']}")
        print(f"  键: {sample_3mod.keys()}")
    except Exception as e:
        print(f"✗ 加载3模态配置失败: {e}")
        sample_3mod = None

    # 加载4模态配置
    try:
        config_4mod = load_config('config/kradar_4modality.json')
        dataset_4mod = init_dataset(
            dataset='kradar',
            src=data_path,
            split='train',
            config=config_4mod
        )
        sample_4mod, _ = dataset_4mod[0]
        print("\n✓ 4模态配置:")
        print(f"  输入模态: {config_4mod['model']['inputs']}")
        print(f"  键: {sample_4mod.keys()}")
    except Exception as e:
        print(f"✗ 加载4模态配置失败: {e}")
        sample_4mod = None

    if sample_3mod and sample_4mod:
        print("\n差异:")
        keys_3mod = set(sample_3mod.keys())
        keys_4mod = set(sample_4mod.keys())
        new_keys = keys_4mod - keys_3mod
        print(f"  新增键: {new_keys}")


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='诊断LiDAR数据质量')
    parser.add_argument('--src', type=str, required=True,
                       help='数据集路径')
    parser.add_argument('--samples', type=int, default=50,
                       help='检查的样本数量')

    args = parser.parse_args()

    # 运行诊断
    stats, issues = diagnose_lidar_data(args.src, args.samples)

    # 对比3模态和4模态
    compare_with_without_lidar(args.src)

    print("\n" + "="*70)
    print("诊断完成")
    print("="*70)

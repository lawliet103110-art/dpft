"""
检查子集数据量和分布
"""
import os
import json
from pathlib import Path

# 修改为你的路径
subset_path = Path("/root/autodl-tmp/autodl-tmp/data/kradar_subset")
full_path = Path("/root/autodl-tmp/autodl-tmp/data/kradar/processed")  # 如果有的话

def count_samples(root_path):
    """统计数据集样本数"""
    counts = {}

    for split in ['train', 'test', 'val']:
        split_path = root_path / split
        if not split_path.exists():
            counts[split] = 0
            continue

        sample_count = 0
        for seq_dir in split_path.iterdir():
            if seq_dir.is_dir():
                for sample_dir in seq_dir.iterdir():
                    if sample_dir.is_dir():
                        sample_count += 1

        counts[split] = sample_count

    return counts

print("=" * 60)
print("子集数据统计")
print("=" * 60)

if subset_path.exists():
    subset_counts = count_samples(subset_path)
    print(f"训练集样本数: {subset_counts.get('train', 0)}")
    print(f"验证集样本数: {subset_counts.get('val', 0)}")
    print(f"测试集样本数: {subset_counts.get('test', 0)}")
    print(f"总计: {sum(subset_counts.values())}")
else:
    print(f"路径不存在: {subset_path}")

print("\n" + "=" * 60)
print("完整数据集统计（如果存在）")
print("=" * 60)

if full_path.exists():
    full_counts = count_samples(full_path)
    print(f"训练集样本数: {full_counts.get('train', 0)}")
    print(f"验证集样本数: {full_counts.get('val', 0)}")
    print(f"测试集样本数: {full_counts.get('test', 0)}")
    print(f"总计: {sum(full_counts.values())}")
else:
    print(f"路径不存在: {full_path}")

# 检查标签分布
print("\n" + "=" * 60)
print("检查子集标签分布")
print("=" * 60)

if subset_path.exists():
    test_path = subset_path / "test"
    if test_path.exists():
        label_stats = []
        sample_count = 0

        for seq_dir in test_path.iterdir():
            if seq_dir.is_dir():
                for sample_dir in seq_dir.iterdir():
                    if sample_dir.is_dir():
                        label_file = sample_dir / "labels.npy"
                        if label_file.exists():
                            import numpy as np
                            labels = np.load(label_file)
                            label_stats.append(len(labels))  # 每个样本的目标数
                            sample_count += 1

                        if sample_count >= 10:  # 只检查前10个样本
                            break
            if sample_count >= 10:
                break

        if label_stats:
            import numpy as np
            print(f"前{len(label_stats)}个样本的目标数量: {label_stats}")
            print(f"平均每样本目标数: {np.mean(label_stats):.2f}")
            print(f"最大目标数: {np.max(label_stats)}")
            print(f"最小目标数: {np.min(label_stats)}")

            if all(x == 0 for x in label_stats):
                print("⚠️  警告: 所有样本都没有标注目标!")
        else:
            print("没有找到标签文件")

print("=" * 60)

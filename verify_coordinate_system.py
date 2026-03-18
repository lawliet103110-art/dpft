#!/usr/bin/env python3
"""
综合验证点云坐标系和覆盖率
"""

import numpy as np
from pathlib import Path

def analyze_multiple_samples(data_path, num_samples=10):
    """分析多个样本的坐标系和覆盖率"""

    print("="*70)
    print("点云坐标系验证")
    print("="*70)

    train_path = Path(data_path) / 'train'

    # 收集样本
    all_samples = []
    for seq_dir in sorted(train_path.iterdir()):
        if not seq_dir.is_dir():
            continue
        for sample_dir in sorted(seq_dir.iterdir())[:num_samples]:
            os1_file = sample_dir / 'os1.npy'
            labels_file = sample_dir / 'labels.npy'
            if os1_file.exists() and labels_file.exists():
                all_samples.append((os1_file, labels_file))
                if len(all_samples) >= num_samples:
                    break
        if len(all_samples) >= num_samples:
            break

    print(f"\n检查 {len(all_samples)} 个样本:")
    print("-"*70)

    roi_coverages = []
    pc_x_ranges = []
    pc_y_ranges = []
    label_x_ranges = []
    label_y_ranges = []

    for i, (pc_file, label_file) in enumerate(all_samples):
        # 加载数据
        pc = np.load(pc_file)
        labels = np.load(label_file)

        # 点云范围
        pc_x_min, pc_x_max = pc[:, 0].min(), pc[:, 0].max()
        pc_y_min, pc_y_max = pc[:, 1].min(), pc[:, 1].max()

        # Labels范围
        if len(labels) > 0:
            label_x_min, label_x_max = labels[:, 0].min(), labels[:, 0].max()
            label_y_min, label_y_max = labels[:, 1].min(), labels[:, 1].max()
        else:
            label_x_min = label_x_max = label_y_min = label_y_max = 0

        # ROI覆盖率（FOV范围）
        roi_mask = (pc[:, 0] >= 0) & (pc[:, 0] < 72) & \
                   (pc[:, 1] >= -6.4) & (pc[:, 1] < 6.4)
        coverage = roi_mask.sum() / len(pc) * 100

        # 收集统计
        roi_coverages.append(coverage)
        pc_x_ranges.append((pc_x_min, pc_x_max))
        pc_y_ranges.append((pc_y_min, pc_y_max))
        label_x_ranges.append((label_x_min, label_x_max))
        label_y_ranges.append((label_y_min, label_y_max))

        # 打印前5个样本
        if i < 5:
            print(f"\n样本 {i}: {pc_file.parent.name}")
            print(f"  点云X: [{pc_x_min:8.2f}, {pc_x_max:8.2f}]")
            print(f"  点云Y: [{pc_y_min:8.2f}, {pc_y_max:8.2f}]")
            print(f"  Label X: [{label_x_min:6.2f}, {label_x_max:6.2f}]")
            print(f"  Label Y: [{label_y_min:6.2f}, {label_y_max:6.2f}]")
            print(f"  ROI覆盖: {coverage:5.1f}% ({roi_mask.sum()}/{len(pc)})")

    # 统计分析
    print(f"\n{'='*70}")
    print("统计分析")
    print("="*70)

    print(f"\nROI覆盖率分布:")
    print(f"  最小值: {min(roi_coverages):.1f}%")
    print(f"  最大值: {max(roi_coverages):.1f}%")
    print(f"  平均值: {np.mean(roi_coverages):.1f}%")
    print(f"  标准差: {np.std(roi_coverages):.1f}%")

    print(f"\n点云X坐标范围:")
    all_pc_x_min = min([r[0] for r in pc_x_ranges])
    all_pc_x_max = max([r[1] for r in pc_x_ranges])
    print(f"  全局最小: {all_pc_x_min:.2f}")
    print(f"  全局最大: {all_pc_x_max:.2f}")
    print(f"  跨度: {all_pc_x_max - all_pc_x_min:.2f}米")

    print(f"\n点云Y坐标范围:")
    all_pc_y_min = min([r[0] for r in pc_y_ranges])
    all_pc_y_max = max([r[1] for r in pc_y_ranges])
    print(f"  全局最小: {all_pc_y_min:.2f}")
    print(f"  全局最大: {all_pc_y_max:.2f}")
    print(f"  跨度: {all_pc_y_max - all_pc_y_min:.2f}米")

    print(f"\nLabel X坐标范围:")
    all_label_x_min = min([r[0] for r in label_x_ranges if r[0] != 0])
    all_label_x_max = max([r[1] for r in label_x_ranges if r[1] != 0])
    print(f"  全局最小: {all_label_x_min:.2f}")
    print(f"  全局最大: {all_label_x_max:.2f}")

    print(f"\nLabel Y坐标范围:")
    all_label_y_min = min([r[0] for r in label_y_ranges if r[0] != 0])
    all_label_y_max = max([r[1] for r in label_y_ranges if r[1] != 0])
    print(f"  全局最小: {all_label_y_min:.2f}")
    print(f"  全局最大: {all_label_y_max:.2f}")

    # 诊断结论
    print(f"\n{'='*70}")
    print("诊断结论")
    print("="*70)

    # 判断1：覆盖率的稳定性
    coverage_std = np.std(roi_coverages)
    if coverage_std < 5:
        print("\n✓ 发现1：ROI覆盖率稳定（标准差<5%）")
        print("  → 说明点云和Labels可能在不同坐标系")
        print("  → 所有样本都恰好有~29%的点落在FOV内（碰巧）")
    else:
        print("\n✓ 发现1：ROI覆盖率变化大（标准差>5%）")
        print("  → 说明点云和Labels在同一坐标系")
        print("  → 覆盖率变化是因为车辆位置/方向变化")

    # 判断2：点云范围是否合理
    if all_pc_x_max - all_pc_x_min > 250:
        print("\n⚠ 发现2：点云X跨度>250米（异常大）")
        print(f"  → 实际跨度: {all_pc_x_max - all_pc_x_min:.2f}米")
        print("  → OS1-128最大量程120米，正常跨度应<240米")
        print("  → 可能原因：")
        print("     1. 点云在世界坐标系（相对地图原点）")
        print("     2. 多帧累积点云（SLAM）")
        print("     3. 数据标定错误")
    else:
        print("\n✓ 发现2：点云范围合理（<250米）")
        print("  → 可能已在LiDAR坐标系")

    # 判断3：Labels是否在合理范围
    if 0 <= all_label_x_min <= 5 and 60 <= all_label_x_max <= 80:
        print("\n✓ 发现3：Labels X∈[0~5, 60~80]，符合FOV [0, 72]")
        print("  → Labels确实在LiDAR坐标系（车辆前方）")
    else:
        print(f"\n⚠ 发现3：Labels X∈[{all_label_x_min:.1f}, {all_label_x_max:.1f}]，不在预期范围")

    # 最终建议
    print(f"\n{'='*70}")
    print("配置建议")
    print("="*70)

    avg_coverage = np.mean(roi_coverages)

    if coverage_std < 5 and all_pc_x_max - all_pc_x_min > 250:
        print("\n📌 推荐方案：扩大投影范围（Moderate或Wide）")
        print("\n理由：")
        print("  1. 点云很可能在世界坐标系")
        print("  2. ROI覆盖率稳定在29%（不同坐标系的碰巧重合）")
        print("  3. 点云范围异常大（>250米）")
        print("\n建议配置:")
        print("  python update_lidar_info_smart.py \\")
        print("      --src /root/autodl-tmp/autodl-tmp/data/kradar \\")
        print("      --range moderate \\  # (-50, 150) × (-75, 75)")
        print("      --apply")

    elif avg_coverage > 50:
        print("\n📌 推荐方案：使用Conservative范围")
        print("\n理由：")
        print("  1. ROI覆盖率>50%，说明大部分点在FOV内")
        print("  2. 点云和Labels很可能在同一坐标系")
        print("\n建议配置:")
        print("  python update_lidar_info_smart.py \\")
        print("      --src /root/autodl-tmp/autodl-tmp/data/kradar \\")
        print("      --range conservative \\  # (0, 100) × (-50, 50)")
        print("      --apply")

    else:
        print("\n📌 推荐方案：使用Moderate范围（平衡）")
        print("\n理由：")
        print(f"  1. ROI覆盖率 {avg_coverage:.1f}%（中等水平）")
        print("  2. 需要在分辨率和覆盖率之间平衡")
        print("\n建议配置:")
        print("  python update_lidar_info_smart.py \\")
        print("      --src /root/autodl-tmp/autodl-tmp/data/kradar \\")
        print("      --range moderate \\  # (-50, 150) × (-75, 75)")
        print("      --apply")

    return roi_coverages, coverage_std


if __name__ == '__main__':
    import sys

    if len(sys.argv) < 2:
        print("用法: python verify_coordinate_system.py <数据集路径>")
        print("示例: python verify_coordinate_system.py /root/autodl-tmp/autodl-tmp/data/kradar")
        sys.exit(1)

    data_path = sys.argv[1]
    analyze_multiple_samples(data_path, num_samples=10)

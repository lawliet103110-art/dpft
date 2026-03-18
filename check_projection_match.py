#!/usr/bin/env python3
"""
快速检查：当前投影参数是否匹配实际数据
Quick check if current projection parameters match your actual data
"""

import numpy as np
from pathlib import Path
import sys

def check_projection_match(data_path, num_samples=10):
    """检查投影范围是否覆盖实际点云数据"""

    print("=" * 70)
    print("投影参数匹配检查")
    print("=" * 70)

    # 当前代码中的投影范围
    CURRENT_X_RANGE = (0, 100)
    CURRENT_Y_RANGE = (-50, 50)

    print(f"\n当前代码使用的投影范围:")
    print(f"  x_range: {CURRENT_X_RANGE}")
    print(f"  y_range: {CURRENT_Y_RANGE}")

    # 分析实际数据
    train_path = Path(data_path) / 'train'
    if not train_path.exists():
        print(f"\n✗ 路径不存在: {train_path}")
        return False

    all_x, all_y = [], []
    count = 0

    print(f"\n正在分析 {num_samples} 个点云文件...")

    for seq_dir in sorted(train_path.iterdir()):
        if not seq_dir.is_dir():
            continue
        for sample_dir in sorted(seq_dir.iterdir())[:2]:
            os1_file = sample_dir / 'os1.npy'
            if os1_file.exists():
                try:
                    data = np.load(os1_file)
                    all_x.append(data[:, 0])
                    all_y.append(data[:, 1])
                    count += 1
                    if count >= num_samples:
                        break
                except Exception as e:
                    print(f"✗ 加载失败 {os1_file}: {e}")
        if count >= num_samples:
            break

    if count == 0:
        print("✗ 未找到任何 os1.npy 文件")
        return False

    all_x = np.concatenate(all_x)
    all_y = np.concatenate(all_y)

    actual_x_min, actual_x_max = float(all_x.min()), float(all_x.max())
    actual_y_min, actual_y_max = float(all_y.min()), float(all_y.max())

    print(f"\n实际数据范围（来自 {count} 个文件）:")
    print(f"  x: [{actual_x_min:8.2f}, {actual_x_max:8.2f}]")
    print(f"  y: [{actual_y_min:8.2f}, {actual_y_max:8.2f}]")

    # 检查覆盖情况
    print("\n" + "=" * 70)
    print("匹配分析")
    print("=" * 70)

    # X轴检查
    x_outside = ((all_x < CURRENT_X_RANGE[0]) | (all_x >= CURRENT_X_RANGE[1])).sum()
    x_outside_pct = x_outside / len(all_x) * 100

    # Y轴检查
    y_outside = ((all_y < CURRENT_Y_RANGE[0]) | (all_y >= CURRENT_Y_RANGE[1])).sum()
    y_outside_pct = y_outside / len(all_y) * 100

    # 综合检查
    both_inside = ((all_x >= CURRENT_X_RANGE[0]) & (all_x < CURRENT_X_RANGE[1]) &
                   (all_y >= CURRENT_Y_RANGE[0]) & (all_y < CURRENT_Y_RANGE[1]))
    points_kept_pct = both_inside.sum() / len(all_x) * 100

    print(f"\nX轴覆盖:")
    if x_outside_pct > 50:
        print(f"  ✗ {x_outside_pct:.1f}% 的点在X轴范围外！")
        print(f"    实际: [{actual_x_min:.1f}, {actual_x_max:.1f}]")
        print(f"    代码: {CURRENT_X_RANGE}")
    elif x_outside_pct > 10:
        print(f"  ⚠ {x_outside_pct:.1f}% 的点在X轴范围外")
    else:
        print(f"  ✓ X轴覆盖良好 (仅{x_outside_pct:.1f}%在范围外)")

    print(f"\nY轴覆盖:")
    if y_outside_pct > 50:
        print(f"  ✗ {y_outside_pct:.1f}% 的点在Y轴范围外！")
        print(f"    实际: [{actual_y_min:.1f}, {actual_y_max:.1f}]")
        print(f"    代码: {CURRENT_Y_RANGE}")
    elif y_outside_pct > 10:
        print(f"  ⚠ {y_outside_pct:.1f}% 的点在Y轴范围外")
    else:
        print(f"  ✓ Y轴覆盖良好 (仅{y_outside_pct:.1f}%在范围外)")

    print(f"\n综合评估:")
    print(f"  保留的点: {points_kept_pct:.1f}%")

    has_problem = False

    if points_kept_pct < 50:
        print("\n" + "!" * 70)
        print("严重问题：超过50%的点被过滤掉了！")
        print("!" * 70)
        print("\n这就是为什么加入LiDAR后mAP下降的原因：")
        print("  1. 大部分点云数据被过滤")
        print("  2. 投影后的BEV图像几乎为空")
        print("  3. 模型学到的是噪声而非有用特征")
        print("  4. 导致性能下降")
        has_problem = True
    elif points_kept_pct < 85:
        print("\n⚠ 警告：超过15%的点被过滤，建议调整投影范围")
        has_problem = True
    else:
        print("\n✓ 投影范围覆盖良好")

    # 建议修复
    if has_problem:
        print("\n" + "=" * 70)
        print("建议的修复方案")
        print("=" * 70)

        # 计算建议范围（添加10%边距）
        x_margin = (actual_x_max - actual_x_min) * 0.1
        y_margin = (actual_y_max - actual_y_min) * 0.1

        suggested_x = (actual_x_min - x_margin, actual_x_max + x_margin)
        suggested_y = (actual_y_min - y_margin, actual_y_max + y_margin)

        print(f"\n1. 手动修改 src/dprt/datasets/kradar/dataset.py")
        print(f"   在 project_lidar_to_bev 方法中 (约第406-409行):")
        print(f"   将:")
        print(f"     x_range: Tuple[float, float] = {CURRENT_X_RANGE},")
        print(f"     y_range: Tuple[float, float] = {CURRENT_Y_RANGE}")
        print(f"   改为:")
        print(f"     x_range: Tuple[float, float] = ({suggested_x[0]:.1f}, {suggested_x[1]:.1f}),")
        print(f"     y_range: Tuple[float, float] = ({suggested_y[0]:.1f}, {suggested_y[1]:.1f})")

        print(f"\n2. 或者使用自动修复工具:")
        print(f"   python auto_fix_lidar_projection.py --src {data_path} --apply")

        print("\n3. 修复后重新训练模型")

    print("\n" + "=" * 70)

    return not has_problem


if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("用法: python check_projection_match.py <数据集路径>")
        print("示例: python check_projection_match.py /root/autodl-tmp/autodl-tmp/data/kradar")
        sys.exit(1)

    data_path = sys.argv[1]
    success = check_projection_match(data_path)

    sys.exit(0 if success else 1)

#!/usr/bin/env python3
"""
修复LiDAR集成的语法和导入问题
"""

import os
import sys

def fix_loader_syntax():
    """修复loader.py的语法错误"""

    loader_path = 'src/dprt/datasets/loader.py'

    if not os.path.exists(loader_path):
        print(f"✗ 文件不存在: {loader_path}")
        return False

    print(f"正在修复 {loader_path}...")

    with open(loader_path, 'r', encoding='utf-8') as f:
        content = f.read()

    # 修复可能的语法错误
    # 确保default_keys_shapes字典格式正确
    fixes = [
        # 修复1：确保所有键值对都有逗号
        ("'radar_front': (1, 256, 256, 6)    # 假设的雷达前视图形状\n            'lidar_top':",
         "'radar_front': (1, 256, 256, 6),   # 假设的雷达前视图形状\n            'lidar_top':"),

        # 修复2：确保lidar_top行格式正确
        ("'lidar_top': (1, 256, 256, 6)      # LiDAR BEV投影形状\n        }",
         "'lidar_top': (1, 256, 256, 6)       # LiDAR BEV投影形状\n        }"),
    ]

    modified = False
    for old, new in fixes:
        if old in content:
            content = content.replace(old, new)
            modified = True
            print(f"  ✓ 应用修复")

    if modified:
        # 备份原文件
        backup_path = loader_path + '.backup'
        with open(backup_path, 'w', encoding='utf-8') as f:
            f.write(content)
        print(f"  ✓ 已备份到: {backup_path}")

    # 验证语法
    try:
        compile(content, loader_path, 'exec')
        print("  ✓ 语法验证通过")

        # 写入修复后的文件
        with open(loader_path, 'w', encoding='utf-8') as f:
            f.write(content)
        print(f"✓ {loader_path} 修复完成")
        return True

    except SyntaxError as e:
        print(f"  ✗ 语法验证失败: {e}")
        print(f"  错误位置: 第{e.lineno}行")
        return False


def fix_kradar_init():
    """修复kradar __init__.py的导入"""

    init_path = 'src/dprt/datasets/kradar/__init__.py'

    if not os.path.exists(init_path):
        print(f"✗ 文件不存在: {init_path}")
        return False

    print(f"\n正在检查 {init_path}...")

    with open(init_path, 'r', encoding='utf-8') as f:
        content = f.read()

    # 确保有正确的导出
    if 'from dprt.datasets.kradar.dataset import KRadarDataset' not in content:
        # 添加导入
        if 'from' not in content:
            content = 'from dprt.datasets.kradar.dataset import KRadarDataset\n\n' + content
        else:
            # 在其他导入后添加
            lines = content.split('\n')
            import_end = 0
            for i, line in enumerate(lines):
                if line.startswith('from ') or line.startswith('import '):
                    import_end = i + 1

            lines.insert(import_end, 'from dprt.datasets.kradar.dataset import KRadarDataset')
            content = '\n'.join(lines)

        with open(init_path, 'w', encoding='utf-8') as f:
            f.write(content)
        print(f"  ✓ 添加了 KRadarDataset 导入")
    else:
        print(f"  ✓ KRadarDataset 导入已存在")

    return True


def verify_dataset_import():
    """验证dataset.py的语法"""

    dataset_path = 'src/dprt/datasets/kradar/dataset.py'

    if not os.path.exists(dataset_path):
        print(f"✗ 文件不存在: {dataset_path}")
        return False

    print(f"\n正在验证 {dataset_path}...")

    with open(dataset_path, 'r', encoding='utf-8') as f:
        content = f.read()

    try:
        compile(content, dataset_path, 'exec')
        print("  ✓ 语法验证通过")
        return True
    except SyntaxError as e:
        print(f"  ✗ 语法错误: {e}")
        print(f"  错误位置: 第{e.lineno}行")
        return False


def main():
    print("="*60)
    print("LiDAR集成修复脚本")
    print("="*60)

    # 检查是否在正确的目录
    if not os.path.exists('src/dprt'):
        print("\n✗ 错误：请在项目根目录运行此脚本")
        print("  当前目录:", os.getcwd())
        print("  应该包含: src/dprt/")
        sys.exit(1)

    success = True

    # 修复loader.py
    if not fix_loader_syntax():
        success = False

    # 修复__init__.py
    if not fix_kradar_init():
        success = False

    # 验证dataset.py
    if not verify_dataset_import():
        success = False

    print("\n" + "="*60)
    if success:
        print("✓ 所有修复完成！")
        print("\n现在可以运行:")
        print("  python test_lidar_integration.py --src /path/to/data")
    else:
        print("✗ 修复过程中遇到错误，请检查上面的输出")
    print("="*60)

    return 0 if success else 1


if __name__ == '__main__':
    sys.exit(main())

#!/bin/bash
# 完整的LiDAR集成修复脚本
# 在DPFT-main目录下运行: bash fix_all_lidar_issues.sh

echo "=========================================="
echo "LiDAR集成完整修复脚本"
echo "=========================================="

# 检查是否在正确的目录
if [ ! -d "src/dprt" ]; then
    echo "✗ 错误：请在DPFT-main目录下运行此脚本"
    echo "  当前目录: $(pwd)"
    exit 1
fi

echo ""
echo "步骤 1/4: 修复 loader.py 语法错误..."
echo "----------------------------------------"

# 修复loader.py的逗号问题
python3 << 'PYTHON_SCRIPT'
import re

loader_file = 'src/dprt/datasets/loader.py'

try:
    with open(loader_file, 'r', encoding='utf-8') as f:
        content = f.read()

    # 备份原文件
    with open(loader_file + '.backup', 'w', encoding='utf-8') as f:
        f.write(content)
    print(f"✓ 已备份到: {loader_file}.backup")

    # 修复可能缺失的逗号
    # 方法1：使用正则表达式确保逗号存在
    pattern = r"'radar_front':\s*\(1,\s*256,\s*256,\s*6\)\s*#[^\n]*\n\s*'lidar_top':"
    replacement = "'radar_front': (1, 256, 256, 6),   # 假设的雷达前视图形状\n            'lidar_top':"

    if re.search(pattern, content):
        content = re.sub(pattern, replacement, content)
        print("✓ 修复了 radar_front 后的逗号")

    # 保存修复后的文件
    with open(loader_file, 'w', encoding='utf-8') as f:
        f.write(content)

    # 验证语法
    compile(content, loader_file, 'exec')
    print("✓ loader.py 语法验证通过")

except SyntaxError as e:
    print(f"✗ loader.py 语法错误: {e}")
    print(f"  错误位置: 第{e.lineno}行")
    exit(1)
except Exception as e:
    print(f"✗ 处理 loader.py 时出错: {e}")
    exit(1)

print("✓ loader.py 修复完成")
PYTHON_SCRIPT

if [ $? -ne 0 ]; then
    echo "✗ loader.py 修复失败"
    exit 1
fi

echo ""
echo "步骤 2/4: 确保 kradar/__init__.py 正确..."
echo "----------------------------------------"

# kradar/__init__.py 应该保持为空（这是正确的）
init_file='src/dprt/datasets/kradar/__init__.py'

if [ ! -f "$init_file" ]; then
    touch "$init_file"
    echo "✓ 创建了空的 $init_file"
else
    # 确保文件为空或只有注释
    python3 << 'PYTHON_SCRIPT'
init_file = 'src/dprt/datasets/kradar/__init__.py'
with open(init_file, 'r') as f:
    content = f.read().strip()

if content and not content.startswith('#'):
    # 文件不为空且不只是注释，清空它
    with open(init_file, 'w') as f:
        f.write('')
    print(f"✓ 清空了 {init_file}（应该为空）")
else:
    print(f"✓ {init_file} 已经是空的（正确）")
PYTHON_SCRIPT
fi

echo ""
echo "步骤 3/4: 修复测试脚本导入方式..."
echo "----------------------------------------"

# 修复test_lidar_integration.py的导入方式
python3 << 'PYTHON_SCRIPT'
test_file = 'test_lidar_integration.py'

try:
    with open(test_file, 'r', encoding='utf-8') as f:
        content = f.read()

    # 备份
    with open(test_file + '.backup', 'w', encoding='utf-8') as f:
        f.write(content)

    # 修复导入部分
    old_import = '''try:
    from dprt.datasets.kradar import KRadarDataset
    from dprt.utils.config import load_config'''

    new_import = '''try:
    from dprt.datasets import init as init_dataset
    from dprt.datasets.kradar.dataset import KRadarDataset
    from dprt.utils.config import load_config'''

    if old_import in content:
        content = content.replace(old_import, new_import)
        print("✓ 更新了导入语句")

    # 修复dataset初始化方式（test_dataset_loading_with_lidar函数）
    # 查找并替换KRadarDataset的直接初始化
    old_init = '''        # Create dataset with LiDAR enabled
        dataset = KRadarDataset(
            src=data_path,
            split='train',
            camera='M',
            radar='BF',
            lidar=1,
            lidar_dropout=0.0
        )'''

    new_init = '''        # Load config
        config = load_config('config/kradar_4modality.json')

        # Create dataset with LiDAR enabled (using the official way)
        dataset = init_dataset(
            dataset='kradar',
            src=data_path,
            split='train',
            config=config
        )'''

    if old_init in content:
        content = content.replace(old_init, new_init)
        print("✓ 更新了数据集初始化方式")

    # 保存
    with open(test_file, 'w', encoding='utf-8') as f:
        f.write(content)

    print(f"✓ {test_file} 修复完成")

except Exception as e:
    print(f"⚠ 处理 {test_file} 时出错: {e}")
    print("  可以手动修复或跳过测试脚本")
PYTHON_SCRIPT

echo ""
echo "步骤 4/4: 验证所有Python文件语法..."
echo "----------------------------------------"

echo "验证 loader.py..."
python3 -m py_compile src/dprt/datasets/loader.py
if [ $? -eq 0 ]; then
    echo "  ✓ loader.py 语法正确"
else
    echo "  ✗ loader.py 语法错误"
    exit 1
fi

echo "验证 dataset.py..."
python3 -m py_compile src/dprt/datasets/kradar/dataset.py
if [ $? -eq 0 ]; then
    echo "  ✓ dataset.py 语法正确"
else
    echo "  ✗ dataset.py 语法错误"
    exit 1
fi

echo "验证 dprt.py..."
python3 -m py_compile src/dprt/models/dprt.py
if [ $? -eq 0 ]; then
    echo "  ✓ dprt.py 语法正确"
else
    echo "  ✗ dprt.py 语法错误"
    exit 1
fi

echo ""
echo "=========================================="
echo "✓ 所有修复完成！"
echo "=========================================="
echo ""
echo "现在可以运行测试："
echo "  python test_lidar_integration.py --src /root/autodl-tmp/autodl-tmp/data/kradar"
echo ""
echo "或者开始训练："
echo "  python -m dprt.train --src /root/autodl-tmp/autodl-tmp/data/kradar --cfg config/kradar_4modality.json --dst log/test"
echo ""

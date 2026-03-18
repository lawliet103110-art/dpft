#!/bin/bash
# LiDAR集成快速修复脚本
# 在DPFT-main目录下运行

echo "=========================================="
echo "LiDAR集成快速修复"
echo "=========================================="

# 进入项目目录
cd /root/autodl-tmp/autodl-tmp/DPFT-main || exit 1

echo "1. 检查并修复 loader.py..."

# 修复loader.py的语法错误（确保有逗号）
python3 << 'EOF'
import re

with open('src/dprt/datasets/loader.py', 'r') as f:
    content = f.read()

# 备份
with open('src/dprt/datasets/loader.py.backup', 'w') as f:
    f.write(content)

# 确保字典格式正确
content = content.replace(
    "'radar_front': (1, 256, 256, 6)    # 假设的雷达前视图形状\n            'lidar_top':",
    "'radar_front': (1, 256, 256, 6),   # 假设的雷达前视图形状\n            'lidar_top':"
)

# 保存
with open('src/dprt/datasets/loader.py', 'w') as f:
    f.write(content)

print("✓ loader.py 已修复")
EOF

echo "2. 检查并修复 kradar/__init__.py..."

# 确保__init__.py有正确的导入
python3 << 'EOF'
init_file = 'src/dprt/datasets/kradar/__init__.py'

try:
    with open(init_file, 'r') as f:
        content = f.read()
except FileNotFoundError:
    content = ''

# 确保有KRadarDataset导入
if 'from dprt.datasets.kradar.dataset import KRadarDataset' not in content and \
   'from .dataset import KRadarDataset' not in content:
    # 添加导入
    content = 'from .dataset import KRadarDataset\n\n' + content

    with open(init_file, 'w') as f:
        f.write(content)
    print("✓ 添加了 KRadarDataset 导入")
else:
    print("✓ KRadarDataset 导入已存在")
EOF

echo "3. 验证Python语法..."
python3 -m py_compile src/dprt/datasets/loader.py && echo "  ✓ loader.py 语法正确"
python3 -m py_compile src/dprt/datasets/kradar/dataset.py && echo "  ✓ dataset.py 语法正确"

echo ""
echo "=========================================="
echo "✓ 修复完成！"
echo "=========================================="
echo ""
echo "现在可以运行测试："
echo "  python test_lidar_integration.py --src /root/autodl-tmp/autodl-tmp/data/kradar"

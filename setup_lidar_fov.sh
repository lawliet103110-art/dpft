#!/bin/bash
# LiDAR FOV配置更新脚本
# 参考radar_info机制，自动配置LiDAR参数

echo "======================================================================="
echo "LiDAR FOV配置工具（参考Radar机制）"
echo "======================================================================="
echo ""
echo "这个工具将："
echo "  1. 分析实际点云数据的坐标范围"
echo "  2. 生成适合你数据的FOV参数"
echo "  3. 更新lidar_info.py（类似radar_info.py）"
echo "  4. 使用固定范围归一化（避免batch间差异）"
echo ""
echo "======================================================================="
echo ""

# 数据路径
DATA_PATH="/root/autodl-tmp/autodl-tmp/data/kradar"

# 步骤1：分析数据（预览模式）
echo "步骤1: 分析点云数据范围"
echo "-----------------------------------------------------------------------"
python update_lidar_info.py --src $DATA_PATH --samples 20

echo ""
echo "======================================================================="
echo ""
read -p "配置看起来合理吗？按Enter继续应用更新，或Ctrl+C取消... " REPLY
echo ""

# 步骤2：应用更新
echo "步骤2: 更新lidar_info.py"
echo "-----------------------------------------------------------------------"
python update_lidar_info.py --src $DATA_PATH --samples 20 --apply

if [ $? -eq 0 ]; then
    echo ""
    echo "======================================================================="
    echo "✓ 配置更新成功！"
    echo "======================================================================="
    echo ""
    echo "lidar_info.py已更新，现在使用："
    echo "  - 基于实际数据的投影范围"
    echo "  - 固定范围归一化（类似radar）"
    echo "  - 集中化参数管理"
    echo ""
    echo "下一步："
    echo "  1. 验证配置："
    echo "     python -c \"from dprt.datasets.kradar.utils import lidar_info; print('X:', lidar_info.x_range_default); print('Y:', lidar_info.y_range_default)\""
    echo ""
    echo "  2. 重新训练模型："
    echo "     python -m dprt.train \\"
    echo "         --src $DATA_PATH \\"
    echo "         --cfg config/kradar_4modality.json \\"
    echo "         --dst log/4modality_with_fov"
    echo ""
    echo "  3. 对比之前的结果，应该看到mAP显著提升！"
    echo ""
else
    echo ""
    echo "✗ 更新失败，请检查错误信息"
    exit 1
fi

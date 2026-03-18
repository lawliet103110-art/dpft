"""
渐进式训练策略

步骤1：先用3模态训练到收敛
步骤2：加载预训练权重，只训练LiDAR分支
步骤3：微调整个网络
"""

# 训练脚本示例

# 步骤1：3模态预训练（已完成）
# python -m dprt.train --src /data/kradar --cfg config/kradar.json --dst log/3modality

# 步骤2：加载3模态权重，只训练LiDAR
# 需要修改训练脚本冻结其他模态

# 步骤3：微调
# python -m dprt.train --src /data/kradar --cfg config/kradar_4modality.json --dst log/4modality_finetune --checkpoint log/lidar_only/best.pt

# 或者使用不同的学习率
config_4mod = {
    "train": {
        "optimizer": {
            "lr": 0.00001  # 更小的学习率用于微调
        }
    }
}

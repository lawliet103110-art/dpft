"""
修复LiDAR归一化的补丁

如果诊断发现归一化问题，可以尝试使用固定范围归一化
"""

# 在dataset.py中替换scale_lidar_data方法

def scale_lidar_data(self, sample: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    """Scales the lidar data to a range of 0 to 255 using fixed ranges

    Arguments:
        sample: Dictionary mapping the sample items to their data tensors.

    Returns:
        sample: Dictionary mapping the sample items to their scaled data tensors.
    """
    for k, v in sample.items():
        if k == 'lidar_top':
            # LiDAR BEV projection image has 6 channels
            # First 3 channels: intensity features (max, median, var)
            # Last 3 channels: range features (max, median, var)

            # 使用固定范围进行归一化（根据诊断结果调整）

            # Intensity通道归一化（假设原始范围0-100）
            intensity_channels = v[:, :, :3]
            intensity_min, intensity_max = 0.0, 100.0  # 根据实际数据调整
            intensity_scaled = torch.clip(
                (intensity_channels - intensity_min) / (intensity_max - intensity_min) * 255,
                0, 255
            )

            # Range通道归一化（假设原始范围0-50000）
            range_channels = v[:, :, 3:]
            range_min, range_max = 0.0, 50000.0  # 根据实际数据调整
            range_scaled = torch.clip(
                (range_channels - range_min) / (range_max - range_min) * 255,
                0, 255
            )

            sample[k] = torch.cat([intensity_scaled, range_scaled], dim=-1)

    return sample

"""Compare Radar vs LiDAR data format and statistics."""

import numpy as np
import torch
from dprt.datasets import init as init_dataset
from dprt.utils.config import load_config

def compare_radar_lidar(data_path, config_path):
    """Compare radar and lidar data side by side."""

    print("="*70)
    print("Radar vs LiDAR Data Comparison")
    print("="*70)

    # Load config
    config = load_config(config_path)

    # Initialize dataset
    dataset = init_dataset(
        dataset='kradar',
        src=data_path,
        split='train',
        config=config
    )

    print(f"\nDataset size: {len(dataset)} samples")

    # Get first sample
    sample, label = dataset[0]

    print("\n" + "="*70)
    print("SHAPE COMPARISON")
    print("="*70)

    if 'radar_bev' in sample:
        radar = sample['radar_bev']
        print(f"\nRadar BEV:")
        print(f"  Shape: {radar.shape}")
        print(f"  Dtype: {radar.dtype}")
        print(f"  Device: {radar.device}")

    if 'lidar_top' in sample:
        lidar = sample['lidar_top']
        print(f"\nLiDAR BEV:")
        print(f"  Shape: {lidar.shape}")
        print(f"  Dtype: {lidar.dtype}")
        print(f"  Device: {lidar.device}")

    print("\n" + "="*70)
    print("VALUE DISTRIBUTION")
    print("="*70)

    if 'radar_bev' in sample:
        radar = sample['radar_bev']
        print(f"\nRadar BEV:")
        print(f"  Min: {radar.min():.2f}")
        print(f"  Max: {radar.max():.2f}")
        print(f"  Mean: {radar.mean():.2f}")
        print(f"  Std: {radar.std():.2f}")
        print(f"  Zero ratio: {(radar == 0).float().mean()*100:.2f}%")

        # Per-channel stats
        for ch in range(radar.shape[-1]):
            ch_data = radar[:, :, ch]
            print(f"  Channel {ch}: mean={ch_data.mean():.2f}, "
                  f"max={ch_data.max():.2f}, "
                  f"non-zero={((ch_data != 0).float().mean()*100):.1f}%")

    if 'lidar_top' in sample:
        lidar = sample['lidar_top']
        print(f"\nLiDAR BEV:")
        print(f"  Min: {lidar.min():.2f}")
        print(f"  Max: {lidar.max():.2f}")
        print(f"  Mean: {lidar.mean():.2f}")
        print(f"  Std: {lidar.std():.2f}")
        print(f"  Zero ratio: {(lidar == 0).float().mean()*100:.2f}%")

        # Per-channel stats
        for ch in range(lidar.shape[-1]):
            ch_data = lidar[:, :, ch]
            print(f"  Channel {ch}: mean={ch_data.mean():.2f}, "
                  f"max={ch_data.max():.2f}, "
                  f"non-zero={((ch_data != 0).float().mean()*100):.1f}%")

    print("\n" + "="*70)
    print("PROJECTION MATRIX COMPARISON")
    print("="*70)

    if 'label_to_radar_bev_p' in sample:
        radar_p = sample['label_to_radar_bev_p']
        print(f"\nRadar BEV projection matrix:")
        print(radar_p)

    if 'label_to_lidar_top_p' in sample:
        lidar_p = sample['label_to_lidar_top_p']
        print(f"\nLiDAR BEV projection matrix:")
        print(lidar_p)

    print("\n" + "="*70)
    print("LABEL COVERAGE CHECK")
    print("="*70)

    if 'gt_center' in label:
        centers = label['gt_center']
        print(f"\nNumber of objects: {len(centers)}")

        if len(centers) > 0:
            print(f"Object centers:")
            print(f"  X range: [{centers[:, 0].min():.2f}, {centers[:, 0].max():.2f}]")
            print(f"  Y range: [{centers[:, 1].min():.2f}, {centers[:, 1].max():.2f}]")
            print(f"  Z range: [{centers[:, 2].min():.2f}, {centers[:, 2].max():.2f}]")

    # Check FOV config
    print("\nFOV from config:")
    print(f"  X: {config['data']['fov']['x']}")
    print(f"  Y: {config['data']['fov']['y']}")
    print(f"  Z: {config['data']['fov']['z']}")

    # Check lidar_info ranges
    from dprt.datasets.kradar.utils import lidar_info
    print("\nLiDAR projection ranges:")
    print(f"  X: {lidar_info.x_range_default}")
    print(f"  Y: {lidar_info.y_range_default}")
    print(f"  Z: ({lidar_info.z_min}, {lidar_info.z_max})")

    # Check if they match
    fov_x = config['data']['fov']['x']
    fov_y = config['data']['fov']['y']
    fov_z = config['data']['fov']['z']

    lidar_x = lidar_info.x_range_default
    lidar_y = lidar_info.y_range_default
    lidar_z = (lidar_info.z_min, lidar_info.z_max)

    print("\n⚠️  MISMATCH CHECK:")
    if tuple(fov_x) != lidar_x:
        print(f"  ❌ X range mismatch: FOV {fov_x} vs LiDAR {lidar_x}")
    else:
        print(f"  ✓ X range matches")

    if tuple(fov_y) != lidar_y:
        print(f"  ❌ Y range mismatch: FOV {fov_y} vs LiDAR {lidar_y}")
    else:
        print(f"  ✓ Y range matches")

    if tuple(fov_z) != lidar_z:
        print(f"  ❌ Z range mismatch: FOV {fov_z} vs LiDAR {lidar_z}")
    else:
        print(f"  ✓ Z range matches")

if __name__ == "__main__":
    import sys

    if len(sys.argv) < 3:
        print("Usage: python compare_radar_lidar.py <data_path> <config_path>")
        print("Example: python compare_radar_lidar.py /data/kradar/processed config/kradar_4modality.json")
        sys.exit(1)

    compare_radar_lidar(sys.argv[1], sys.argv[2])

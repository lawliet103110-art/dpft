"""Diagnose LiDAR BEV projection data quality."""

import numpy as np
import torch
from dprt.datasets import init as init_dataset
from dprt.utils.config import load_config

def diagnose_lidar_data(data_path, config_path):
    """Check LiDAR BEV data statistics."""

    print("="*60)
    print("LiDAR Data Quality Diagnosis")
    print("="*60)

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
    print("\nAnalyzing first 10 samples...")

    for i in range(min(10, len(dataset))):
        sample, label = dataset[i]

        if 'lidar_top' not in sample:
            print(f"Sample {i}: No lidar_top data")
            continue

        lidar_bev = sample['lidar_top']

        # Check shape
        print(f"\nSample {i}:")
        print(f"  Shape: {lidar_bev.shape}")

        # Check value range
        print(f"  Value range: [{lidar_bev.min():.2f}, {lidar_bev.max():.2f}]")

        # Check sparsity (percentage of zero values)
        zero_ratio = (lidar_bev == 0).float().mean().item()
        print(f"  Zero ratio: {zero_ratio*100:.2f}%")

        # Check per-channel statistics
        for ch in range(lidar_bev.shape[-1]):
            ch_data = lidar_bev[:, :, ch]
            non_zero = (ch_data != 0).float().mean().item()
            mean_val = ch_data[ch_data!=0].mean().item() if (ch_data!=0).any() else 0.0
            print(f"  Channel {ch}: non-zero={non_zero*100:.1f}%, "
                  f"mean={mean_val:.2f}, "
                  f"max={ch_data.max():.2f}")

    # Aggregate statistics
    print("\n" + "="*60)
    print("Aggregate Statistics (100 samples)")
    print("="*60)

    all_zero_ratios = []
    all_max_values = []
    all_mean_values = []

    for i in range(min(100, len(dataset))):
        sample, _ = dataset[i]
        if 'lidar_top' in sample:
            lidar_bev = sample['lidar_top']
            all_zero_ratios.append((lidar_bev == 0).float().mean().item())
            all_max_values.append(lidar_bev.max().item())
            non_zero_mask = lidar_bev != 0
            if non_zero_mask.any():
                all_mean_values.append(lidar_bev[non_zero_mask].mean().item())

    print(f"\nZero ratio: mean={np.mean(all_zero_ratios)*100:.2f}%, "
          f"std={np.std(all_zero_ratios)*100:.2f}%")
    print(f"Max value: mean={np.mean(all_max_values):.2f}, "
          f"std={np.std(all_max_values):.2f}")
    print(f"Non-zero mean: mean={np.mean(all_mean_values):.2f}, "
          f"std={np.std(all_mean_values):.2f}")

    # Compare with radar BEV
    print("\n" + "="*60)
    print("Comparison with Radar BEV")
    print("="*60)

    sample, _ = dataset[0]

    if 'radar_bev' in sample:
        radar_bev = sample['radar_bev']
        print(f"\nRadar BEV:")
        print(f"  Shape: {radar_bev.shape}")
        print(f"  Value range: [{radar_bev.min():.2f}, {radar_bev.max():.2f}]")
        print(f"  Zero ratio: {(radar_bev == 0).float().mean()*100:.2f}%")

    if 'lidar_top' in sample:
        lidar_bev = sample['lidar_top']
        print(f"\nLiDAR BEV:")
        print(f"  Shape: {lidar_bev.shape}")
        print(f"  Value range: [{lidar_bev.min():.2f}, {lidar_bev.max():.2f}]")
        print(f"  Zero ratio: {(lidar_bev == 0).float().mean()*100:.2f}%")

if __name__ == "__main__":
    import sys

    if len(sys.argv) < 3:
        print("Usage: python diagnose_lidar_bev.py <data_path> <config_path>")
        print("Example: python diagnose_lidar_bev.py /data/kradar/processed config/kradar_4modality.json")
        sys.exit(1)

    diagnose_lidar_data(sys.argv[1], sys.argv[2])

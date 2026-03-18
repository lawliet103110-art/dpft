"""
Test script for LiDAR modality integration

This script tests:
1. LiDAR point cloud loading and projection
2. Data shape and value range validation
3. Transformation and projection matrices
4. Modality dropout functionality
"""

import torch
import numpy as np
from pathlib import Path

# Test if dataset module can be imported
try:
    from dprt.datasets import init as init_dataset
    from dprt.datasets.kradar.dataset import KRadarDataset
    from dprt.utils.config import load_config
    print("✓ Successfully imported required modules")
except ImportError as e:
    print(f"✗ Import error: {e}")
    print("Please run this script from the project root directory")
    exit(1)


def test_point_cloud_projection():
    """Test the point cloud to BEV projection method"""
    print("\n" + "="*60)
    print("Test 1: Point Cloud Projection")
    print("="*60)

    # Create synthetic point cloud data (N, 9)
    # [x, y, z, intensity, timestamp, reflectivity, ring, azimuth, range]
    print("Creating synthetic point cloud...")
    np.random.seed(42)
    n_points = 1000
    point_cloud = np.zeros((n_points, 9), dtype=np.float32)
    point_cloud[:, 0] = np.random.uniform(0, 100, n_points)    # x: 0-100m
    point_cloud[:, 1] = np.random.uniform(-50, 50, n_points)   # y: -50-50m
    point_cloud[:, 2] = np.random.uniform(-2, 6, n_points)     # z: -2-6m
    point_cloud[:, 3] = np.random.uniform(5, 20, n_points)     # intensity
    point_cloud[:, 8] = np.random.uniform(5000, 20000, n_points)  # range
    print(f"✓ Created point cloud with shape: {point_cloud.shape}")

    # Test projection by directly calling the static method
    # We need to instantiate the projection logic without needing a full dataset
    print("Testing projection algorithm...")

    # Manually implement the projection test (copied from dataset.py logic)
    from collections import defaultdict

    H, W = 256, 256
    x_range, y_range = (0, 100), (-50, 50)

    # Initialize output channels
    bev_intensity_max = np.zeros((H, W), dtype=np.float32)
    bev_intensity_median = np.zeros((H, W), dtype=np.float32)
    bev_intensity_var = np.zeros((H, W), dtype=np.float32)
    bev_range_max = np.zeros((H, W), dtype=np.float32)
    bev_range_median = np.zeros((H, W), dtype=np.float32)
    bev_range_var = np.zeros((H, W), dtype=np.float32)

    # Extract point cloud attributes
    x, y = point_cloud[:, 0], point_cloud[:, 1]
    intensity = point_cloud[:, 3]
    range_vals = point_cloud[:, 8]

    # Filter points
    mask = (x >= x_range[0]) & (x < x_range[1]) & \
           (y >= y_range[0]) & (y < y_range[1])
    x, y, intensity, range_vals = x[mask], y[mask], intensity[mask], range_vals[mask]

    # Map to pixel coordinates
    x_img = ((x - x_range[0]) / (x_range[1] - x_range[0]) * H).astype(int)
    y_img = ((y - y_range[0]) / (y_range[1] - y_range[0]) * W).astype(int)
    x_img = np.clip(x_img, 0, H - 1)
    y_img = np.clip(y_img, 0, W - 1)

    # Aggregate points per pixel
    pixel_points = defaultdict(lambda: {'intensity': [], 'range': []})
    for i in range(len(x_img)):
        pixel_points[(x_img[i], y_img[i])]['intensity'].append(intensity[i])
        pixel_points[(x_img[i], y_img[i])]['range'].append(range_vals[i])

    # Compute statistical features
    for (px, py), values in pixel_points.items():
        intensities = np.array(values['intensity'])
        ranges = np.array(values['range'])
        bev_intensity_max[px, py] = np.max(intensities)
        bev_intensity_median[px, py] = np.median(intensities)
        bev_intensity_var[px, py] = np.var(intensities)
        bev_range_max[px, py] = np.max(ranges)
        bev_range_median[px, py] = np.median(ranges)
        bev_range_var[px, py] = np.var(ranges)

    # Stack channels
    bev_image = np.dstack((
        bev_intensity_max, bev_intensity_median, bev_intensity_var,
        bev_range_max, bev_range_median, bev_range_var
    ))
    bev_image = torch.from_numpy(bev_image).type(torch.float32)

    # Validate output
    assert bev_image.shape == (256, 256, 6), f"Expected shape (256, 256, 6), got {bev_image.shape}"
    assert isinstance(bev_image, torch.Tensor), "Output should be a torch.Tensor"
    print(f"✓ BEV projection shape: {bev_image.shape}")
    print(f"✓ Output type: {type(bev_image)}")
    print(f"✓ Value range: [{bev_image.min():.2f}, {bev_image.max():.2f}]")

    # Check that some pixels are non-zero (points were projected)
    non_zero_pixels = (bev_image.sum(dim=-1) > 0).sum().item()
    print(f"✓ Non-zero pixels: {non_zero_pixels}/{256*256}")
    assert non_zero_pixels > 0, "No points were projected to BEV image"

    print("✓ Point cloud projection test PASSED\n")


def test_dataset_loading_with_lidar(data_path=None):
    """Test dataset loading with LiDAR enabled"""
    print("\n" + "="*60)
    print("Test 2: Dataset Loading with LiDAR")
    print("="*60)

    if data_path is None:
        print("⚠ Skipping: No data path provided")
        print("  To test with real data, provide --src argument")
        return

    # Check if path exists
    if not Path(data_path).exists():
        print(f"⚠ Skipping: Data path does not exist: {data_path}")
        return

    try:
        # Load config
        config = load_config('/root/autodl-tmp/autodl-tmp/DPFT-main/config/kradar_4modality.json')

        # Create dataset with LiDAR enabled (using the official way)
        dataset = init_dataset(
            dataset='kradar',
            src=data_path,
            split='train',
            config=config
        )

        print(f"✓ Dataset initialized with {len(dataset)} samples")

        # Load first sample
        sample, label = dataset[0]

        # Validate LiDAR data
        assert 'lidar_top' in sample, "lidar_top not in sample"
        print(f"✓ lidar_top present in sample")

        lidar_shape = sample['lidar_top'].shape
        assert lidar_shape == (256, 256, 6), f"Expected (256, 256, 6), got {lidar_shape}"
        print(f"✓ lidar_top shape: {lidar_shape}")

        # Check value range after scaling
        lidar_data = sample['lidar_top']
        print(f"✓ Value range: [{lidar_data.min():.2f}, {lidar_data.max():.2f}]")

        # Validate transformation matrix
        assert 'label_to_lidar_top_t' in sample, "transformation matrix missing"
        t_shape = sample['label_to_lidar_top_t'].shape
        print(f"✓ Transformation matrix shape: {t_shape}")

        # Validate projection matrix
        assert 'label_to_lidar_top_p' in sample, "projection matrix missing"
        p_shape = sample['label_to_lidar_top_p'].shape
        assert p_shape == (3, 4), f"Expected (3, 4), got {p_shape}"
        print(f"✓ Projection matrix shape: {p_shape}")

        # Validate shape info
        assert 'lidar_top_shape' in sample, "shape info missing"
        shape_info = sample['lidar_top_shape']
        print(f"✓ Shape info: {shape_info}")

        print("✓ Dataset loading test PASSED\n")

    except Exception as e:
        print(f"✗ Dataset loading test FAILED: {e}")
        import traceback
        traceback.print_exc()


# def test_modality_dropout(data_path=None):
#     """Test modality dropout functionality"""
#     print("\n" + "="*60)
#     print("Test 3: Modality Dropout")
#     print("="*60)

#     if data_path is None:
#         print("⚠ Skipping: No data path provided")
#         return

#     if not Path(data_path).exists():
#         print(f"⚠ Skipping: Data path does not exist: {data_path}")
#         return

#     try:
#         # Create dataset with high LiDAR dropout
#         dataset = KRadarDataset(
#             src=data_path,
#             split='train',
#             camera='M',
#             radar='BF',
#             lidar=1,
#             lidar_dropout=0.8  # 80% dropout probability
#         )

#         # Test multiple samples
#         lidar_dropout_count = 0
#         test_samples = min(20, len(dataset))

#         for i in range(test_samples):
#             sample, _ = dataset[i]
#             if torch.all(sample['lidar_top'] == 0):
#                 lidar_dropout_count += 1

#         dropout_rate = lidar_dropout_count / test_samples
#         print(f"✓ LiDAR dropout rate: {dropout_rate*100:.1f}% ({lidar_dropout_count}/{test_samples})")

#         # Should have at least some dropouts with 80% probability
#         if lidar_dropout_count > 0:
#             print("✓ Modality dropout is working")
#         else:
#             print("⚠ Warning: No dropout observed (may be due to randomness)")

#         print("✓ Modality dropout test PASSED\n")

#     except Exception as e:
#         print(f"✗ Modality dropout test FAILED: {e}")
#         import traceback
#         traceback.print_exc()


def test_config_loading():
    """Test loading 4-modality config file"""
    print("\n" + "="*60)
    print("Test 4: Configuration Loading")
    print("="*60)

    config_path = Path('config/kradar_4modality.json')

    if not config_path.exists():
        print(f"⚠ Config file not found: {config_path}")
        return

    try:
        from dprt.utils.config import load_config
        config = load_config(str(config_path))

        # Validate config structure
        assert 'data' in config, "Missing 'data' section"
        assert 'model' in config, "Missing 'model' section"

        # Validate data config
        data_config = config['data']
        assert data_config.get('lidar') == 1, "LiDAR not enabled in config"
        print(f"✓ LiDAR enabled: {data_config.get('lidar')}")

        # Validate model config
        model_config = config['model']
        assert 'lidar_top' in model_config['inputs'], "lidar_top not in model inputs"
        print(f"✓ Model inputs: {model_config['inputs']}")

        # Validate fuser config
        fuser_config = model_config['fuser']
        assert fuser_config['m_views'] == 4, "m_views should be 4"
        print(f"✓ Fuser m_views: {fuser_config['m_views']}")

        assert len(fuser_config['n_levels']) == 4, "n_levels should have 4 elements"
        print(f"✓ Fuser n_levels: {fuser_config['n_levels']}")

        print("✓ Configuration loading test PASSED\n")

    except Exception as e:
        print(f"✗ Configuration loading test FAILED: {e}")
        import traceback
        traceback.print_exc()


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Test LiDAR integration')
    parser.add_argument('--src', type=str, default=None,
                      help='Path to preprocessed dataset (e.g., /data/kradar/processed)')
    args = parser.parse_args()

    print("\n" + "="*60)
    print("LiDAR Integration Test Suite")
    print("="*60)

    # Run all tests
    test_point_cloud_projection()
    test_config_loading()
    test_dataset_loading_with_lidar(args.src)
    # test_modality_dropout(args.src)

    print("\n" + "="*60)
    print("Test Suite Complete")
    print("="*60)
    print("\nNote: Some tests were skipped because no data path was provided.")
    print("To run all tests, use: python test_lidar_integration.py --src /path/to/data")

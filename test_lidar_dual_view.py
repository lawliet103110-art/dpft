"""Test script for LiDAR dual-view (BEV + Side view) integration.

This script tests:
1. LiDAR dual-view projection (BEV + Side)
2. Data shape and value range validation
3. Transformation and projection matrices
4. Model forward pass with dual-view inputs

Usage:
    python test_lidar_dual_view.py --src /path/to/dataset
"""

import torch
import numpy as np
from pathlib import Path
import argparse

# Test if dataset module can be imported
try:
    from dprt.datasets import init as init_dataset
    from dprt.datasets.kradar.dataset import KRadarDataset
    from dprt.utils.config import load_config
    from dprt.datasets.loader import load_listed
    print("✓ Successfully imported required modules")
except ImportError as e:
    print(f"✗ Import error: {e}")
    print("Please run this script from the project root directory")
    exit(1)


def test_dual_view_projection():
    """Test the dual-view (BEV + Side) projection methods"""
    print("\n" + "="*60)
    print("Test 1: Dual-View Point Cloud Projection")
    print("="*60)

    # Create synthetic point cloud data (N, 9)
    print("Creating synthetic point cloud...")
    np.random.seed(42)
    n_points = 1000
    point_cloud = np.zeros((n_points, 9), dtype=np.float32)
    point_cloud[:, 0] = np.random.uniform(0, 80, n_points)    # x: 0-80m
    point_cloud[:, 1] = np.random.uniform(-40, 40, n_points)  # y: -40-40m
    point_cloud[:, 2] = np.random.uniform(-5, 10, n_points)   # z: -5-10m
    point_cloud[:, 3] = np.random.uniform(5, 20, n_points)    # intensity
    point_cloud[:, 8] = np.random.uniform(5000, 20000, n_points)  # range
    print(f"✓ Created point cloud with shape: {point_cloud.shape}")

    # Create a dummy dataset instance to test projection methods
    print("\nTesting projection methods...")
    try:
        dataset = KRadarDataset(
            src="dummy",  # Not used for projection test
            lidar=1
        )
    except Exception as e:
        print(f"Note: Creating minimal dataset instance for projection test")
        # Just test the projection logic manually
        pass

    # Test BEV projection
    print("\n1. Testing BEV projection...")
    try:
        dataset = KRadarDataset(src="dummy", lidar=1)
        bev_image = dataset.project_lidar_to_bev(point_cloud)
        assert bev_image.shape == (128, 128, 6), f"Expected (128, 128, 6), got {bev_image.shape}"
        print(f"   ✓ BEV projection shape: {bev_image.shape}")
        print(f"   ✓ BEV value range: [{bev_image.min():.2f}, {bev_image.max():.2f}]")
    except Exception as e:
        print(f"   ✗ BEV projection failed: {e}")
        return False

    # Test Side view projection
    print("\n2. Testing Side view projection...")
    try:
        side_image = dataset.project_lidar_to_side(point_cloud)
        assert side_image.shape == (64, 128, 6), f"Expected (64, 128, 6), got {side_image.shape}"
        print(f"   ✓ Side projection shape: {side_image.shape}")
        print(f"   ✓ Side value range: [{side_image.min():.2f}, {side_image.max():.2f}]")
    except Exception as e:
        print(f"   ✗ Side projection failed: {e}")
        return False

    print("\n✓ Dual-view projection test PASSED\n")
    return True


def test_config_loading():
    """Test loading dual-view config file"""
    print("\n" + "="*60)
    print("Test 2: Configuration Loading")
    print("="*60)

    config_path = Path('config/kradar_lidar_dual_view.json')

    if not config_path.exists():
        print(f"✗ Config file not found: {config_path}")
        return False

    try:
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
        expected_inputs = ['camera_mono', 'radar_bev', 'radar_front', 'lidar_top', 'lidar_side']
        actual_inputs = model_config['inputs']

        for inp in expected_inputs:
            assert inp in actual_inputs, f"{inp} not in model inputs"
        print(f"✓ Model inputs: {actual_inputs}")

        # Validate fuser config
        fuser_config = model_config['fuser']
        assert fuser_config['m_views'] == 5, f"m_views should be 5, got {fuser_config['m_views']}"
        print(f"✓ Fuser m_views: {fuser_config['m_views']}")

        assert len(fuser_config['n_levels']) == 5, f"n_levels should have 5 elements, got {len(fuser_config['n_levels'])}"
        print(f"✓ Fuser n_levels: {fuser_config['n_levels']}")

        assert len(fuser_config['n_heads']) == 5, f"n_heads should have 5 elements"
        print(f"✓ Fuser n_heads: {fuser_config['n_heads']}")

        print("\n✓ Configuration loading test PASSED\n")
        return True

    except Exception as e:
        print(f"✗ Configuration loading test FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_dual_view_dataset(data_path=None):
    """Test if dataset correctly generates both BEV and side views."""

    print("\n" + "="*60)
    print("Test 3: Dataset Loading with Dual-View")
    print("="*60)

    if data_path is None:
        print("⚠ Skipping: No data path provided")
        print("  To test with real data, provide --src argument")
        return True

    # Check if path exists
    if not Path(data_path).exists():
        print(f"⚠ Skipping: Data path does not exist: {data_path}")
        return True

    # Load configuration
    config_path = "config/kradar_lidar_dual_view.json"
    print(f"Loading config from: {config_path}")

    try:
        config = load_config(config_path)
    except Exception as e:
        print(f"✗ Error loading config: {e}")
        return False

    # Initialize dataset
    print("Initializing dataset...")
    try:
        dataset = init_dataset(
            dataset='kradar',
            src=data_path,
            split='train',
            config=config
        )
        print(f"✓ Dataset initialized with {len(dataset)} samples")
    except Exception as e:
        print(f"✗ Error initializing dataset: {e}")
        import traceback
        traceback.print_exc()
        return False

    # Test a single sample
    print("\nTesting single sample...")
    try:
        sample, label = dataset[0]
        print(f"Sample keys: {sample.keys()}")

        # Check for LiDAR modalities
        required_keys = ['lidar_top', 'lidar_side',
                        'lidar_top_shape', 'lidar_side_shape',
                        'label_to_lidar_top_t', 'label_to_lidar_side_t',
                        'label_to_lidar_top_p', 'label_to_lidar_side_p']

        print("\nChecking LiDAR modalities:")
        all_present = True
        for key in required_keys:
            if key in sample:
                tensor = sample[key]
                print(f"  ✓ {key}: shape = {tensor.shape}, dtype = {tensor.dtype}")
            else:
                print(f"  ✗ {key}: MISSING")
                all_present = False

        if not all_present:
            return False

        # Check shapes
        print("\nValidating shapes:")
        lidar_top = sample['lidar_top']
        lidar_side = sample['lidar_side']

        print(f"  lidar_top shape: {lidar_top.shape}")
        assert lidar_top.shape == (128, 128, 6), f"Expected (128, 128, 6), got {lidar_top.shape}"
        print("  ✓ lidar_top shape correct")

        print(f"  lidar_side shape: {lidar_side.shape}")
        assert lidar_side.shape == (64, 128, 6), f"Expected (64, 128, 6), got {lidar_side.shape}"
        print("  ✓ lidar_side shape correct")

        # Check value ranges (should be 0-255 after scaling)
        print("\nChecking value ranges:")
        print(f"  lidar_top range: [{lidar_top.min():.2f}, {lidar_top.max():.2f}]")
        print(f"  lidar_side range: [{lidar_side.min():.2f}, {lidar_side.max():.2f}]")

        assert 0 <= lidar_top.min() and lidar_top.max() <= 255, "lidar_top values out of range!"
        assert 0 <= lidar_side.min() and lidar_side.max() <= 255, "lidar_side values out of range!"
        print("  ✓ Value ranges correct")

        # Check projection matrices
        print("\nChecking projection matrices:")
        p_top = sample['label_to_lidar_top_p']
        p_side = sample['label_to_lidar_side_p']

        print(f"  lidar_top projection shape: {p_top.shape}")
        print(f"  lidar_side projection shape: {p_side.shape}")

        assert p_top.shape == (3, 4), f"Expected (3, 4), got {p_top.shape}"
        assert p_side.shape == (3, 4), f"Expected (3, 4), got {p_side.shape}"
        print("  ✓ Projection matrices correct")

        print("\n✓ Dataset loading test PASSED\n")
        return True

    except Exception as e:
        print(f"\n✗ Error during testing: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_model_forward(data_path=None):
    """Test if model can forward with dual-view inputs."""

    print("\n" + "=" * 60)
    print("Test 4: Model Forward Pass with Dual-View")
    print("=" * 60)

    if data_path is None:
        print("⚠ Skipping: No data path provided")
        print("  To test model forward, provide --src argument")
        return True

    if not Path(data_path).exists():
        print(f"⚠ Skipping: Data path does not exist: {data_path}")
        return True

    config_path = "config/kradar_lidar_dual_view.json"
    print(f"Loading config from: {config_path}")

    try:
        config = load_config(config_path)
    except Exception as e:
        print(f"✗ Error loading config: {e}")
        return False

    try:
        from dprt.models.dprt import build_dprt

        print("Building model...")
        model = build_dprt(config)
        model.eval()
        print(f"✓ Model built successfully")
        print(f"✓ Model inputs: {model.inputs}")

        # Initialize dataset and dataloader
        print("\nLoading dataset...")
        dataset = init_dataset(
            dataset='kradar',
            src=data_path,
            split='train',
            config=config
        )

        dataloader = load_listed(dataset, config)
        batch_sample, batch_label = next(iter(dataloader))

        print("\nRunning forward pass...")
        with torch.no_grad():
            output = model(batch_sample)

        print(f"✓ Forward pass successful!")
        print(f"Output keys: {output.keys()}")

        for key, val in output.items():
            if isinstance(val, torch.Tensor):
                print(f"  - {key}: shape = {val.shape}")

        print("\n✓ Model forward pass test PASSED\n")
        return True

    except Exception as e:
        print(f"\n✗ Error during model testing: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Test LiDAR dual-view integration')
    parser.add_argument('--src', type=str, default=None,
                      help='Path to preprocessed dataset (e.g., /data/kradar/processed)')
    args = parser.parse_args()

    print("\n" + "="*60)
    print("LiDAR Dual-View Integration Test Suite")
    print("="*60)

    # Run all tests
    test1_ok = test_dual_view_projection()
    test2_ok = test_config_loading()
    test3_ok = test_dual_view_dataset(args.src)
    test4_ok = test_model_forward(args.src)

    print("\n" + "="*60)
    print("Test Suite Complete")
    print("="*60)

    if test1_ok and test2_ok and test3_ok and test4_ok:
        print("\n✓ ALL TESTS PASSED!")
        print("\nYou can now train with:")
        print("  python train.py --cfg config/kradar_lidar_dual_view.json")
    else:
        print("\n⚠ Some tests were skipped or failed")
        print("To run all tests with real data, use:")
        print("  python test_lidar_dual_view.py --src /path/to/dataset")

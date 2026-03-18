#!/bin/bash
# Quick Fix Commands for LiDAR mAP Drop Issue
# Run these on your server: /root/autodl-tmp/autodl-tmp/DPFT-main

echo "========================================="
echo "LiDAR Fix Commands - Run in Order"
echo "========================================="

# Step 1: Diagnose the problem
echo ""
echo "Step 1: Run Diagnostic"
echo "----------------------"
echo "python diagnose_lidar.py --src /root/autodl-tmp/autodl-tmp/data/kradar --samples 50"
echo ""
echo "Look for: empty LiDAR percentage, sparse LiDAR percentage, value ranges"
echo "Press Enter when ready to see Step 2..."
read

# Step 2: Check actual point cloud ranges
echo ""
echo "Step 2: Check Your Actual Point Cloud Ranges"
echo "---------------------------------------------"
cat << 'EOF'
cd /root/autodl-tmp/autodl-tmp/data/kradar/train
python3 -c "
import numpy as np
from pathlib import Path

print('Analyzing point cloud coordinate ranges...')
print('=' * 60)

for seq_dir in sorted(Path('.').iterdir())[:3]:
    if seq_dir.is_dir():
        for sample_dir in list(seq_dir.iterdir())[:2]:
            os1_file = sample_dir / 'os1.npy'
            if os1_file.exists():
                data = np.load(os1_file)
                print(f'\n{os1_file.relative_to(Path.cwd())}:')
                print(f'  x: [{data[:, 0].min():8.2f}, {data[:, 0].max():8.2f}]')
                print(f'  y: [{data[:, 1].min():8.2f}, {data[:, 1].max():8.2f}]')
                print(f'  z: [{data[:, 2].min():8.2f}, {data[:, 2].max():8.2f}]')
                print(f'  intensity: [{data[:, 3].min():8.2f}, {data[:, 3].max():8.2f}]')
                print(f'  range: [{data[:, 8].min():8.2f}, {data[:, 8].max():8.2f}]')

print('\n' + '=' * 60)
print('Compare these ranges with project_lidar_to_bev defaults:')
print('  Current code: x_range=(0, 100), y_range=(-50, 50)')
print('  If your x is negative, most points are filtered out!')
"
cd -
EOF
echo ""
echo "Press Enter when ready to see Step 3..."
read

# Step 3: Apply auto-fix
echo ""
echo "Step 3: Apply Automatic Fix"
echo "----------------------------"
echo "# First preview (doesn't modify files)"
echo "python auto_fix_lidar_projection.py --src /root/autodl-tmp/autodl-tmp/data/kradar --samples 20"
echo ""
echo "# Review the output, then apply the fix:"
echo "python auto_fix_lidar_projection.py --src /root/autodl-tmp/autodl-tmp/data/kradar --samples 20 --apply"
echo ""
echo "This will:"
echo "  - Analyze 20 point cloud files"
echo "  - Calculate appropriate x_range and y_range"
echo "  - Backup dataset.py → dataset.py.backup_projection"
echo "  - Patch dataset.py with correct projection parameters"
echo ""
echo "Press Enter when ready to see Step 4..."
read

# Step 4: Retrain
echo ""
echo "Step 4: Retrain the Model"
echo "-------------------------"
echo "python -m dprt.train \\"
echo "    --src /root/autodl-tmp/autodl-tmp/data/kradar \\"
echo "    --cfg config/kradar_4modality.json \\"
echo "    --dst log/4modality_fixed"
echo ""
echo "Monitor training and check if mAP improves!"
echo ""
echo "========================================="
echo "Alternative: Manual Fix"
echo "========================================="
echo ""
echo "If auto-fix doesn't work, manually edit:"
echo "  src/dprt/datasets/kradar/dataset.py"
echo ""
echo "Line 406-409 in project_lidar_to_bev() method:"
echo "  Change x_range and y_range to match YOUR data"
echo ""
echo "Example:"
echo "  If your x is [-25, 0], use: x_range=(-30, 5)"
echo "  If your y is [-15, 20], use: y_range=(-20, 25)"
echo ""
echo "Add 10-20% margin to ensure all points are included"
echo ""

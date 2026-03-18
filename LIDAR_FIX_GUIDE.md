# LiDAR mAP Drop Fix Guide

## Problem Summary

After adding LiDAR modality, mAP dropped by 7 points. This is likely caused by **projection range mismatch**.

**Root Cause**: Your point cloud data has x-coordinates around -17.7 to -11.2 meters (negative values), but the projection code assumes x-range of (0, 100) meters. This causes most points to be filtered out, resulting in empty/sparse LiDAR features that actually harm performance.

---

## Solution Steps

### Step 1: Run Diagnostic (on your server)

```bash
cd /root/autodl-tmp/autodl-tmp/DPFT-main
python diagnose_lidar.py --src /root/autodl-tmp/autodl-tmp/data/kradar --samples 50
```

**Expected Output to Look For:**

✅ **Healthy LiDAR data should show:**
```
非零像素比例:
  平均: 15-30%

LiDAR数值范围:
  最大值范围: [180, 255]
```

❌ **Problem indicators:**
```
空LiDAR数据（全零）: 40%+  ← BAD!
稀疏LiDAR（<5%非零）: 60%+  ← BAD!
LiDAR最大值仅为 3.45  ← BAD! (should be ~255)
```

---

### Step 2A: If Diagnostic Shows Projection Range Issues

Run the auto-fix tool to analyze your actual data ranges and patch the code:

```bash
# First, analyze only (doesn't modify anything)
python auto_fix_lidar_projection.py --src /root/autodl-tmp/autodl-tmp/data/kradar --samples 20

# Review the suggested ranges, then apply the fix
python auto_fix_lidar_projection.py --src /root/autodl-tmp/autodl-tmp/data/kradar --samples 20 --apply
```

**What this does:**
1. Analyzes 20 point cloud files to find actual coordinate ranges
2. Generates appropriate x_range and y_range parameters (with 10% margin)
3. Patches `src/dprt/datasets/kradar/dataset.py` with correct values
4. Updates normalization to use fixed ranges instead of adaptive

**Example output:**
```
实际数据范围:
  x         : [ -18.50,  -10.20]
  y         : [ -15.30,   16.80]

建议的投影范围:
  x_range: (-21.33, -7.37)
  y_range: (-18.51, 20.01)
```

---

### Step 2B: Manual Fix (if auto-fix doesn't work)

If you prefer to fix manually or auto-fix fails, check your actual point cloud ranges first:

```bash
cd /root/autodl-tmp/autodl-tmp/data/kradar/train
python3 -c "
import numpy as np
from pathlib import Path

# Load a few samples
for seq_dir in sorted(Path('.').iterdir())[:3]:
    if seq_dir.is_dir():
        for sample_dir in list(seq_dir.iterdir())[:2]:
            os1_file = sample_dir / 'os1.npy'
            if os1_file.exists():
                data = np.load(os1_file)
                print(f'{os1_file}:')
                print(f'  x: [{data[:, 0].min():.2f}, {data[:, 0].max():.2f}]')
                print(f'  y: [{data[:, 1].min():.2f}, {data[:, 1].max():.2f}]')
                print(f'  z: [{data[:, 2].min():.2f}, {data[:, 2].max():.2f}]')
                print(f'  intensity: [{data[:, 3].min():.2f}, {data[:, 3].max():.2f}]')
                print(f'  range: [{data[:, 8].min():.2f}, {data[:, 8].max():.2f}]')
                print()
"
```

Then manually edit `src/dprt/datasets/kradar/dataset.py`:

**Line 406-409** - Update default ranges in `project_lidar_to_bev`:
```python
def project_lidar_to_bev(self, point_cloud: np.ndarray,
                        img_size: Tuple[int, int] = (256, 256),
                        x_range: Tuple[float, float] = (-25, 0),      # ← CHANGE THIS
                        y_range: Tuple[float, float] = (-20, 20)) -> torch.Tensor:  # ← AND THIS
```

Use ranges that cover your data with ~10% margin.

---

### Step 3: Retrain the Model

After applying the fix:

```bash
# Clean start with fixed projection
python -m dprt.train \
    --src /root/autodl-tmp/autodl-tmp/data/kradar \
    --cfg config/kradar_4modality.json \
    --dst log/4modality_fixed
```

---

## Understanding the Problem

### Why Does Range Mismatch Cause Performance Drop?

1. **Point Filtering**: In `project_lidar_to_bev` (line 444-447):
   ```python
   mask = (x >= x_range[0]) & (x < x_range[1]) & \
          (y >= y_range[0]) & (y < y_range[1])
   ```
   If your x is -17 but x_range is (0, 100), the mask filters out ALL points.

2. **Empty Features**: Results in BEV image that's almost all zeros (sparse)

3. **Model Confusion**: The fusion network gets:
   - Camera: rich features ✓
   - Radar BEV: rich features ✓
   - Radar Front: rich features ✓
   - LiDAR: nearly empty! ✗

   The model tries to learn from this "noisy" LiDAR branch, which actually hurts performance.

### Why Adaptive Normalization Can Also Cause Issues

Original code (line 357-373) uses adaptive normalization:
```python
intensity_min = intensity_channels.min()
intensity_max = intensity_channels.max()
intensity_scaled = (intensity_channels - intensity_min) / (intensity_max - intensity_min) * 255
```

**Problem**: If intensity_max ≈ intensity_min (sparse data), you get division by near-zero or NaN values.

**Fix**: Use fixed ranges based on your actual data statistics:
```python
# Fixed ranges (determined from data analysis)
intensity_scaled = torch.clip(
    (intensity_channels - 0.0) / 100.0 * 255,
    0, 255
)
range_scaled = torch.clip(
    (range_channels - 0.0) / 50000.0 * 255,
    0, 255
)
```

---

## Verification

After retraining, check if LiDAR actually helps:

```bash
# Compare 3-modality baseline
python -m dprt.train \
    --src /root/autodl-tmp/autodl-tmp/data/kradar \
    --cfg config/kradar.json \
    --dst log/3modality_baseline

# vs 4-modality with fixed projection
# (already trained in Step 3)
```

**Expected result**: 4-modality should now show 2-5% mAP improvement over 3-modality baseline.

---

## Alternative: Progressive Training

If you still get poor results after fixing projection, try this training strategy:

```bash
# Step 1: Train 3 modalities first (camera + radar)
python -m dprt.train \
    --src /root/autodl-tmp/autodl-tmp/data/kradar \
    --cfg config/kradar.json \
    --dst log/3modality_pretrain

# Step 2: Add LiDAR and continue training
# Edit config/kradar_4modality.json to add:
# "train": {
#     "checkpoint": "log/3modality_pretrain/best.pt",
#     "optimizer": {
#         "lr": 0.00001  // Lower LR for fine-tuning
#     }
# }

python -m dprt.train \
    --src /root/autodl-tmp/autodl-tmp/data/kradar \
    --cfg config/kradar_4modality.json \
    --dst log/4modality_finetune
```

---

## Quick Checklist

- [ ] Run diagnostic: `python diagnose_lidar.py --src ... --samples 50`
- [ ] Check for empty/sparse LiDAR warnings
- [ ] Run auto-fix: `python auto_fix_lidar_projection.py --src ... --apply`
- [ ] Or manually check ranges and update dataset.py
- [ ] Retrain model with fixed projection
- [ ] Verify mAP improves over 3-modality baseline

---

## Files Modified by Auto-Fix

- `src/dprt/datasets/kradar/dataset.py`
  - Line 406-409: x_range, y_range in `project_lidar_to_bev`
  - Line 351-390: `scale_lidar_data` method (fixed normalization)

Backup will be saved as: `src/dprt/datasets/kradar/dataset.py.backup_projection`

---

## Need Help?

If after following these steps you still get poor results:

1. Share the diagnostic output
2. Share the point cloud range analysis
3. Check if batch_size was reduced (memory issue)
4. Verify training ran for enough epochs (4-modality may need longer)

The key is ensuring your LiDAR data is actually being used (non-zero, properly projected) rather than being mostly empty noise.

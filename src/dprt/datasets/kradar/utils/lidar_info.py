"""LiDAR sensor (OS1-128) data rasterization information.

BEV projection uses Cartesian (x, y) coordinates so that the projection
matrix P receives Cartesian queries (x, y, z) directly (label_to_lidar_top_t
is set to zeros to bypass the spherical conversion in get_reference_points).

Key design decisions:
- x_range_default / y_range_default must cover the full model query space.
  The querent spans azimuth ±50° at range up to 72 m, which in Cartesian
  gives max |y| = 72 * sin(50°) ≈ 55.2 m.  Using ±60 m adds a small margin.
- bev_resolution = 256 keeps spatial resolution ≈ 0.47 m/px in the y-axis,
  comparable to the radar BEV (~0.29 m/px in range).
- range values in the Ouster PCD file are in millimetres (raw OS1 output).
  max_range_norm = 72 000 mm covers the full 72 m detection horizon.
"""

# OS1-128 LiDAR sensor specifications
# Reference: https://ouster.com/products/scanning-lidar/os1-sensor/

# Horizontal field of view (azimuth): 360 degrees
azimuth_fov = (-180.0, 180.0)  # degrees, full 360°

# Vertical field of view (elevation): 33.2 degrees
# OS1-128: -16.6° to +16.6°
elevation_fov = (-16.6, 16.6)  # degrees

# Maximum range: 120 meters (typical for OS1-128)
max_range = 120.0  # meters

# BEV projection parameters
# X-axis: forward direction (meters), Y-axis: lateral direction (meters)
#
# y_range_default must be wide enough to contain every model query point.
# At r=72 m and azimuth=±50°: y = ±72*sin(50°) ≈ ±55.2 m → use ±60 m.
x_range_default = (0, 80)   # meters
y_range_default = (-60, 60)  # meters  (was -20,20; too narrow for ±50° azimuth)
z_range_default = (-5, 10)  # meters

# Intensity normalization (raw OS1-128 intensity values, 0–65535 scale)
min_intensity = 2.00
max_intensity = 299.00

# Range normalization (OS1 PCD 'range' field is in millimetres)
# Detection horizon is 72 m = 72 000 mm → use 0–72 000 for full coverage.
# Previous values (4680–16880 mm ≈ 4.7–16.9 m) saturated all far objects.
min_range = 0.0
max_range_norm = 72000.0

# Grid resolution for BEV projection
# 256 px over 120 m lateral range ≈ 0.47 m/px (previously 128 px over 40 m)
bev_resolution = 256  # pixels

# Side view projection parameters (X-Z plane)
# Width (X): forward distance, Height (Z): vertical height
side_width = 256  # pixels (X-axis resolution)
side_height = 64  # pixels (Z-axis resolution)

# Z-axis filtering for BEV projection
z_min = -5  # meters
z_max = 10  # meters

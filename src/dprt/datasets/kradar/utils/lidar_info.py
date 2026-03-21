# """LiDAR sensor (OS1-128) data rasterization information.

# Auto-generated from actual point cloud data analysis.
# Similar to radar_info.py structure.

# Configuration: 90%分位数范围（排除极值）- 数据驱动
# Point cloud coverage: 82.5%
# """

# # OS1-128 LiDAR sensor specifications
# # Reference: https://ouster.com/products/scanning-lidar/os1-sensor/

# # Horizontal field of view (azimuth): 360 degrees
# azimuth_fov = (-180.0, 180.0)  # degrees, full 360°

# # Vertical field of view (elevation): 33.2 degrees
# # OS1-128: -16.6° to +16.6°
# elevation_fov = (-16.6, 16.6)  # degrees

# # Maximum range: 120 meters (typical for OS1-128)
# max_range = 120.0  # meters

# # BEV projection parameters (auto-detected from data)
# # X-axis: forward/backward direction (meters)
# # Y-axis: left/right direction (meters)

# x_range_default = (-50, 100)  # meters
# y_range_default = (-40, 40)  # meters
# z_range_default = (-10, 20)  # meters

# # Intensity normalization range (from OS1-128 data statistics)
# # Based on 1-99 percentile to avoid outliers
# min_intensity = 2.00
# max_intensity = 299.00

# # Range normalization (raw range values from point cloud column 8)
# # Based on 1-99 percentile to avoid outliers
# min_range = 4680.00
# max_range_norm = 16880.04

# # Grid resolution for BEV projection
# bev_resolution = 256  # pixels (128x128 grid)

# # Side view projection parameters (X-Z plane)
# # Width (X): forward distance, Height (Z): vertical height
# side_width = 256  # pixels (X-axis resolution)
# side_height = 128  # pixels (Z-axis resolution, smaller to save computation)

# # Z-axis filtering for BEV projection
# # Only keep points within certain height range (based on 1-99 percentile)
# z_min = -2  # meters
# z_max = 6  # meters
"""LiDAR sensor (OS1-128) data rasterization information.

Auto-generated from actual point cloud data analysis.
Similar to radar_info.py structure.

Configuration: 90%分位数范围（排除极值）- 数据驱动
Point cloud coverage: 82.5%
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

# BEV projection parameters (auto-detected from data)
# X-axis: forward/backward direction (meters)
# Y-axis: left/right direction (meters)

# Selected range based on data analysis
# x_range_default = (-5, 100)  # meters
# y_range_default = (-20,20)  # meters
# z_range_default = (-5, 10)  # meters
x_range_default = (-10, 100)  # meters
y_range_default = (-40, 40)  # meters
z_range_default = (-5, 10)  # meters

# Intensity normalization range (from OS1-128 data statistics)
# Based on 1-99 percentile to avoid outliers
# min_intensity = 1.00
# max_intensity = 2994.00
# # min_intensity = 2.00
# # max_intensity = 299.00

# # Range normalization (raw range values from point cloud column 8)
# # Based on 1-99 percentile to avoid outliers
# min_range = 1242.00
# max_range_norm = 46236.00
# # min_range = 4680.00
# # max_range_norm = 16880.04

# min_intensity = 1.00
# max_intensity = 814.00
min_intensity = 2.00
max_intensity = 299.00

# Range normalization (raw range values from point cloud column 8)
# Based on 1-99 percentile to avoid outliers
# min_range = 2685.00
# max_range_norm = 35018.84
min_range = 4680.00
max_range_norm = 16880.04

# Grid resolution for BEV projection
bev_resolution = 128  # pixels (128x128 grid)

# Side view projection parameters (X-Z plane)
# Width (X): forward distance, Height (Z): vertical height
side_width = 128  # pixels (X-axis resolution)
side_height = 64  # pixels (Z-axis resolution, smaller to save computation)

# Z-axis filtering for BEV projection
# Only keep points within certain height range (based on 1-99 percentile)
z_min = -5  # meters
z_max = 10  # meters

# BEV channel normalization constants
z_range_norm = 15.0          # z_max - z_min = 10 - (-5) = 15 meters
max_count_per_pixel = 20.0   # maximum expected points per BEV pixel

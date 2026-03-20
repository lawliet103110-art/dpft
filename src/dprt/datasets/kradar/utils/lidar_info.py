"""LiDAR sensor (OS1-128) data rasterization information.

BEV projection uses Cartesian (x, y) coordinates so that the projection
matrix P receives Cartesian queries (x, y, z) directly (label_to_lidar_top_t
is set to zeros to bypass the spherical conversion in get_reference_points).

=== Why label_to_lidar_top_t must be zeros, not eye(4) ===

In IMPFusion.get_reference_points() (mpfusion.py):

    if transformation.any():             # branch A: spherical path
        ref = T @ query                  # apply transformation first
        r, phi, roh = cart2spher(...)    # then convert Cartesian → Spherical
    else:                                # branch B: Cartesian path (skips conversion)
        ref = query

    result = P @ ref                     # apply projection matrix

Radar modalities set T=eye(4), so any()=True → branch A fires → queries
are converted to spherical (r, phi, roh) before P.  This is correct for
radar because the radar projection matrix P is designed for spherical input.

LiDAR BEV uses a Cartesian projection matrix P (see _get_lidar_bev_projection).
P maps (x, y, z) → (u_pixel, v_pixel).  If T=eye(4) is used:
  - any()=True → branch A fires → queries become spherical (r, phi, roh)
  - P receives spherical input but was designed for Cartesian
  - Result: every query samples the WRONG pixel in the BEV image

Concrete example (query at r=72 m, phi=50°, i.e. x=46.28 m, y=55.16 m):
  With eye(4):  P receives (r=72, phi=50°, roh=0°)
    u = 2.133*phi + 128 = 234.7 px  →  u_norm = 0.917  (wrong pixel)
    v = 3.200*r   + 0   = 230.4 px  →  v_norm = 0.900  (wrong pixel)
  With zeros:   P receives (x=46.28, y=55.16, z=0)
    u = 2.133*y + 128   = 245.7 px  →  u_norm = 0.960  (correct pixel)
    v = 3.200*x + 0     = 148.1 px  →  v_norm = 0.579  (correct pixel)

Setting T=zeros(4,4) makes any()=False → branch B → queries stay Cartesian
→ P maps them to the correct BEV pixels.

=== Why max_range_norm = 72 000 (not 16 880) ===

The Ouster OS1 SDK stores the 'range' field in the PCD file as a uint32 in
millimetres (mm), matching the raw lidar hardware output.  Evidence:

  1. Observed PCD data (LIDAR_INTEGRATION_REPORT.md, line 238):
       example point: x=-17.7 m, y=3.2 m
       Euclidean distance = sqrt(17.7² + 3.2²) * 1000 ≈ 17 987 mm
       measured range column values: 12 583 – 19 686   ← consistent with mm

  2. The old 1–99 percentile values (min_range=4680, max_range=16880) represent
     4.68 m – 16.88 m.  If range were in metres, 16880 m would exceed the
     sensor's 120 m maximum — clearly impossible.  In mm it equals 16.88 m.

  3. Setting max_range_norm=16880 mm saturated every object beyond ~17 m,
     covering only 23% of the 72 m detection horizon.  All distant objects
     had an identical, meaningless range feature value of 255.

  Using max_range_norm = 72 000 mm maps the full 0–72 m detection range to
  the 0–255 feature value interval without saturation.
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

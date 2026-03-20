from __future__ import annotations  # noqa: F407

import os
import os.path as osp

from itertools import chain
from typing import Any, Dict, List, Tuple, Union

import torch
import numpy as np

from torch.utils.data import Dataset
from torchvision.io import read_image
from torchvision.transforms.functional import resize

from dprt.datasets.kradar.utils import radar_info
from dprt.datasets.kradar.utils import lidar_info


class KRadarDataset(Dataset):
    def __init__(self,
                 src: str,
                 version: str = '',
                 split: str = 'train',
                 camera: str = 'M',
                 camera_dropout: float = 0.0,
                 image_size: Union[int, Tuple[int, int]] = None,
                 radar: str = 'BF',
                 radar_dropout: float = 0.0,
                 lidar: int = 0,
                 lidar_dropout: float = 0.0,  # 新增
                 label: str = 'detection',
                 num_classes: int = 1,
                 sequential: bool = False,
                 scale: bool = True,
                 fov: Dict[str, Tuple[float, float]] = None,
                 dtype: str = 'float32',
                 **kwargs):
        """Dataset class for the K-Radar dataset.

        Arguments:
            src: Source path to the pre-processed
                dataset folder.
            version: Dataset version. One of either
                mini or None (full dataset).
            split: Dataset split to load. One of
                either train or test.
            camera: Camera modalities to use. One of
                either 'M' (mono camera), 'S' (stereo camera)
                or None.
            camera_dropout: Camera modality dropout probability 
                between 0 and 1.
            image_size: Image size to resize image data to.
                Either None (no resizing), int or tuple of two
                int specifying the height and width.
            radar: Radar modalities to use. Any combination
                of 'B' (BEV) and 'F' (Front) or None
            radar_dropout: Radar modality dropout probability
                between 0 and 1.
            lidar_dropout: Lidar modality dropout probability
                between 0 and 1.
            lidar: Lidar modality to use. One of either
                0 (no lidar), 1 (OS1) or 2 (OS2).
            label: Type of label data to use. One of either
                'detection' (3D bounding boxes), 'occupancy'
                (3D occupancy grid) or None.
            num_classes: Number of object classes used for
                one hot encoding.
            sequential: Whether to consume sequneces of
                samples or single samples.
            scale: Whether to scale the radar data to
                a value range of [0, 255] or not.
            fov: Field of view to limit the lables to. Can
                contain values for x, y, z and azimuth.
        """
        # Initialize parent dataset class
        super().__init__()

        # Check attribute values
        assert camera_dropout + radar_dropout + lidar_dropout<= 1.0
        # assert camera_dropout + radar_dropout <= 1.0

        # Initialize instance attributes
        self.src = src
        self.version = version
        self.split = split
        self.camera = camera
        self.camera_dropout = camera_dropout
        self.image_size = image_size
        self.radar = radar
        self.radar_dropout = radar_dropout
        self.lidar = lidar
        self.lidar_dropout = lidar_dropout
        self.label = label
        self.num_classes = num_classes
        self.sequential = sequential
        self.scale = scale
        self.fov = fov if fov is not None else {}
        self.dtype = dtype

        # Adjust split according to dataset version
        if self.version:
            self.split = f"{self.version}_{self.split}"

        # Define lottery pot
        if self.lidar > 0:
            # 4模态：支持LiDAR dropout (两个lidar视图作为一个整体)
            self.lottery = [
                {},
                {'camera_mono', 'camera_stereo'},
                {'radar_bev', 'radar_front'},
                {'lidar_top', 'lidar_side'}
            ]
            self.dropout = [
                1 - (self.camera_dropout + self.radar_dropout + self.lidar_dropout),
                self.camera_dropout,
                self.radar_dropout,
                self.lidar_dropout
            ]
        else:
            # 原有3模态逻辑
            self.lottery = [
                {},
                {'camera_mono', 'camera_stereo'},
                {'radar_bev', 'radar_front'}
            ]
            self.dropout = [
                1 - (self.camera_dropout + self.radar_dropout),
                self.camera_dropout,
                self.radar_dropout
            ]
        # # Initialize moality dropout attributes
        # # Define the lottery pot to draw from (None, camera, radar)
        # self.lottery = [
        #     {},
        #     {'camera_mono', 'camera_stereo'},
        #     {'radar_bev', 'radar_front'}
        # ]

        # # Define dropout probabilities (sum of probabilities must be <= 1)
        # self.dropout = [
        #     1 - (self.camera_dropout + self.radar_dropout),
        #     self.camera_dropout,
        #     self.radar_dropout
        # ]

        # Get dataset path
        self.dataset_paths = self.get_dataset_paths(self.src)

    def __len__(self):
        return len(self.dataset_paths)

    def __getitem__(self, index) -> Dict[str, torch.Tensor]:
        """Returns an item of the dataset given its index.
        Handles missing modalities by only processing available data.

        Whether or not the retured Tensors include a time
        dimension depends on whether or not sequential is
        true or false.

        Arguments:
            index: Index of the dataset item to return.

        Returns:
            item: Dataset item as dictionary of tensors.
        """
        # Map index to sequence number for sequential data
        if self.sequential:
            index = list(sorted(self.dataset_paths.keys()))[index]

        # Get item from dataset
        item = self._to_list(self.dataset_paths[index])

        # Load data from file paths
        for sample in item:
            sample = self.load_sample_data(sample)

        # Check if modalities exist before processing
        has_radar = any(mode in sample for mode in ['radar_bev', 'radar_front'])
        has_lidar = any(mode in sample for mode in ['lidar_top', 'lidar_side'])
        has_camera = any(mode in sample for mode in ['camera_mono', 'camera_stereo'])
        has_labels = 'label' in sample
        has_description = 'description' in sample

        # Scale radar data only if it exists
        if self.scale and has_radar:
            sample = self.scale_radar_data(sample)

        # Scale radar and lidar data
        if self.scale and has_lidar:
            sample = self.scale_lidar_data(sample)  # 新增

        # Apply modality dropout
        if has_radar or has_camera or has_lidar:
        # if has_radar or has_camera:
            sample = self.modality_dropout(sample)

        # Get task specific label
        label = {}
        if self.label == 'detection':
            if has_labels:
                label = self.get_detection_label(sample.pop('label'))
            else:
                # Create empty label if missing
                label = {
                    'gt_center': torch.empty((0, 3), dtype=getattr(torch, self.dtype)),
                    'gt_size': torch.empty((0, 3), dtype=getattr(torch, self.dtype)),
                    'gt_angle': torch.empty((0, 2), dtype=getattr(torch, self.dtype)),
                    'gt_class': torch.empty((0, self.num_classes), dtype=getattr(torch, self.dtype))
                }

        # Add description to label if it exists
        if has_description:
            label.update({'description': sample.pop('description')})
        else:
            # Add empty description if missing
            label.update({'description': torch.zeros((4,), dtype=getattr(torch, self.dtype))})

        # Only add transformations for existing modalities
        if any(mode in sample for mode in ['camera_mono', 'camera_stereo', 'radar_bev', 'radar_front', 'lidar_top', 'lidar_side']):
        # if any(mode in sample for mode in ['camera_mono', 'camera_stereo', 'radar_bev', 'radar_front']):
            sample = self._add_transformations(sample)

        # Only add projections for existing modalities
        if any(mode in sample for mode in ['camera_mono', 'camera_stereo', 'radar_bev', 'radar_front', 'lidar_top', 'lidar_side']):
        # if any(mode in sample for mode in ['camera_mono', 'camera_stereo', 'radar_bev', 'radar_front']):
            sample = self._add_projections(sample)

        # Only add shapes for existing modalities
        if any(mode in sample for mode in ['camera_mono', 'camera_stereo', 'radar_bev', 'radar_front', 'lidar_top', 'lidar_side']):
        # if any(mode in sample for mode in ['camera_mono', 'camera_stereo', 'radar_bev', 'radar_front']):
            sample = self._add_shape(sample)

        # Resize image only if required and image exists
        if self.image_size is not None and has_camera:
            sample = self.resize_image(sample)

        # Convert list of dicts to dict of stacked tensors
        if self.sequential:
            # Stack tensors along the time dimension (use padding for variable
            # size inputs, e.g. label)
            # item = {key: default_collate([d[key] for d in item]) for key in sample}
            raise NotImplementedError()
        else:
            # There is just a single sample for non sequential data
            item = sample

        return item, label

    @classmethod
    def from_config(cls, config: Dict, *args, **kwargs) -> KRadarDataset:  # noqa: F821
        return cls(*args, **dict(config['computing'] | config['data']), **kwargs)

    @staticmethod
    def _to_list(item: Any) -> List[Any]:
        if not isinstance(item, (list, tuple, set)):
            return [item]
        return item

    def _add_transformations(self, sample: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Adds the transformation matrices to the sample only for existing modalities.

        Arguments:
            sample: Dictionary mapping the sample
                items to their data tensors.

        Returns:
            sample: Dictionary mapping the sample
                items to their scaled data tensors.
        """
        # Only process modalities that exist in the sample
        if 'camera_mono' in sample and 'label_to_camera_mono' in sample:
            sample['label_to_camera_mono_t'] = torch.zeros_like(sample['label_to_camera_mono'])
        elif 'camera_mono' in sample and 'label_to_camera_mono' not in sample:
            # Create identity transformation if calibration is missing
            sample['label_to_camera_mono_t'] = torch.eye(4, dtype=getattr(torch, self.dtype))

        if 'camera_stereo' in sample and 'label_to_camera_stereo' in sample:
            sample['label_to_camera_stereo_t'] = torch.zeros_like(sample['label_to_camera_stereo'])
        elif 'camera_stereo' in sample and 'label_to_camera_stereo' not in sample:
            sample['label_to_camera_stereo_t'] = torch.eye(4, dtype=getattr(torch, self.dtype))

        if 'radar_bev' in sample and 'label_to_radar_bev' in sample:
            sample['label_to_radar_bev_t'] = sample.pop('label_to_radar_bev')
        elif 'radar_bev' in sample and 'label_to_radar_bev' not in sample:
            sample['label_to_radar_bev_t'] = torch.eye(4, dtype=getattr(torch, self.dtype))

        if 'radar_front' in sample and 'label_to_radar_front' in sample:
            sample['label_to_radar_front_t'] = sample.pop('label_to_radar_front')
        elif 'radar_front' in sample and 'label_to_radar_front' not in sample:
            sample['label_to_radar_front_t'] = torch.eye(4, dtype=getattr(torch, self.dtype))

        if self.lidar > 0 and 'label_to_lidar_top' in sample:
            sample['label_to_lidar_top_t'] = sample.pop('label_to_lidar_top')
        elif self.lidar > 0 and 'lidar_top' in sample:
            # Use zero matrix so get_reference_points skips the spherical
            # conversion and forwards Cartesian query coordinates directly to
            # the Cartesian BEV projection matrix P.  An identity matrix would
            # cause any() == True, triggering cart→spher conversion before P,
            # which maps (r, phi, roh) instead of (x, y, z) — a projection bug.
            sample['label_to_lidar_top_t'] = torch.zeros(
                (4, 4), dtype=getattr(torch, self.dtype))

        if self.lidar > 0 and 'lidar_side' in sample:
            # Side view also uses Cartesian projection; keep zeros for consistency.
            sample['label_to_lidar_side_t'] = torch.zeros(
                (4, 4), dtype=getattr(torch, self.dtype))

        return sample

    def _add_projections(self, sample: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Adds the projection matrices to the sample only for existing modalities.

        Arguments:
            sample: Dictionary mapping the sample
                items to their data tensors.

        Returns:
            sample: Dictionary mapping the sample
                items to their scaled data tensors.
        """
        # Only process modalities that exist in the sample
        if 'camera_mono' in sample and 'label_to_camera_mono' in sample:
            sample['label_to_camera_mono_p'] = sample.pop('label_to_camera_mono')
        elif 'camera_mono' in sample:
            sample['label_to_camera_mono_p'] = torch.eye(4, dtype=getattr(torch, self.dtype))

        if 'camera_stereo' in sample and 'label_to_camera_stereo' in sample:
            sample['label_to_camera_stereo_p'] = sample.pop('label_to_camera_stereo')
        elif 'camera_stereo' in sample:
            sample['label_to_camera_stereo_p'] = torch.eye(4, dtype=getattr(torch, self.dtype))

        if 'radar_bev' in sample:
            sample['label_to_radar_bev_p'] = self._get_radar_ra_projection()

        if 'radar_front' in sample:
            sample['label_to_radar_front_p'] = self._get_radar_ea_projection()
        if self.lidar > 0 and 'lidar_top' in sample:
            sample['label_to_lidar_top_p'] = self._get_lidar_bev_projection()
        if self.lidar > 0 and 'lidar_side' in sample:
            sample['label_to_lidar_side_p'] = self._get_lidar_side_projection()


        return sample

    def _add_shape(self, sample: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Adds the original input data shape to the sample only for existing modalities.

        Arguments:
            sample: Dictionary mapping the sample
                items to their data tensors.

        Returns:
            sample: Dictionary mapping the sample
                items to their scaled data tensors.
        """
        # Only process modalities that exist in the sample
        if 'camera_mono' in sample:
            sample['camera_mono_shape'] = torch.as_tensor(sample['camera_mono'].shape)

        if 'camera_stereo' in sample:
            sample['camera_stereo_shape'] = torch.as_tensor(sample['camera_stereo'].shape)

        if 'radar_bev' in sample:
            sample['radar_bev_shape'] = torch.as_tensor(sample['radar_bev'].shape)

        if 'radar_front' in sample:
            sample['radar_front_shape'] = torch.as_tensor(sample['radar_front'].shape)

        if 'lidar_top' in sample:
            sample['lidar_top_shape'] = torch.as_tensor(sample['lidar_top'].shape)

        if 'lidar_side' in sample:
            sample['lidar_side_shape'] = torch.as_tensor(sample['lidar_side'].shape)

        return sample

    def project_lidar_to_bev(self, point_cloud: np.ndarray,
                            img_size: Tuple[int, int] = None,
                            x_range: Tuple[float, float] = None,
                            y_range: Tuple[float, float] = None) -> torch.Tensor:
        """Project LiDAR point cloud to BEV image with 6 channels.

        Arguments:
            point_cloud: LiDAR data with shape (N, 9)
                [x, y, z, intensity, timestamp, reflectivity, ring, azimuth, range]
            img_size: Output image size (H, W)
            x_range: X-axis range in meters (min, max)
            y_range: Y-axis range in meters (min, max)

        Returns:
            bev_image: BEV projection with shape (H, W, 6) as torch.Tensor
        """
        if img_size is None:
            img_size = (lidar_info.bev_resolution, lidar_info.bev_resolution)
        if x_range is None:
            x_range = lidar_info.x_range_default
        if y_range is None:
            y_range = lidar_info.y_range_default

        H, W = img_size

        # Initialize output channels.
        # Height channels (bev_z_max / bev_z_min) are pre-filled with z_min so
        # that empty pixels normalize to exactly 0 after scale_lidar_data.
        bev_intensity_max = np.zeros((H, W), dtype=np.float32)
        bev_z_max = np.full((H, W), lidar_info.z_min, dtype=np.float32)
        bev_z_min = np.full((H, W), lidar_info.z_min, dtype=np.float32)
        bev_range_max = np.zeros((H, W), dtype=np.float32)
        bev_count = np.zeros((H, W), dtype=np.float32)
        bev_z_span = np.zeros((H, W), dtype=np.float32)

        # Extract point cloud coordinates and attributes
        x, y, z = point_cloud[:, 0], point_cloud[:, 1], point_cloud[:, 2]
        intensity = point_cloud[:, 3]
        range_vals = point_cloud[:, 8]

        # Filter points outside the range (including Z-axis filtering for BEV)
        mask = (x >= x_range[0]) & (x < x_range[1]) & \
               (y >= y_range[0]) & (y < y_range[1]) & \
               (z >= lidar_info.z_min) & (z < lidar_info.z_max)
        x, y, z, intensity, range_vals = x[mask], y[mask], z[mask], intensity[mask], range_vals[mask]

        # Map to pixel coordinates
        x_img = ((x - x_range[0]) / (x_range[1] - x_range[0]) * H).astype(int)
        y_img = ((y - y_range[0]) / (y_range[1] - y_range[0]) * W).astype(int)

        # Clip to image bounds
        x_img = np.clip(x_img, 0, H - 1)
        y_img = np.clip(y_img, 0, W - 1)

        # Vectorized aggregation using numpy (like radar processing)
        # Convert 2D pixel coords to 1D indices for binning
        pixel_indices = x_img * W + y_img

        # Get unique pixels and their point indices
        unique_pixels, inverse_indices = np.unique(pixel_indices, return_inverse=True)

        # Aggregate statistics for each unique pixel
        for i, pixel_idx in enumerate(unique_pixels):
            # Get all points belonging to this pixel
            px_mask = (inverse_indices == i)
            pixel_z = z[px_mask]
            pixel_intensities = intensity[px_mask]
            pixel_ranges = range_vals[px_mask]

            # Convert 1D pixel index back to 2D coords
            px = pixel_idx // W
            py = pixel_idx % W

            bev_intensity_max[px, py] = np.max(pixel_intensities)
            bev_z_max[px, py] = np.max(pixel_z)
            bev_z_min[px, py] = np.min(pixel_z)
            bev_range_max[px, py] = np.max(pixel_ranges)
            bev_count[px, py] = len(pixel_z)
            bev_z_span[px, py] = np.max(pixel_z) - np.min(pixel_z)

        # Stack channels (H, W, 6):
        #   0 intensity_max  – reflectivity of the brightest return
        #   1 z_max          – highest point in the pixel (key for vehicle height)
        #   2 z_min          – lowest point in the pixel
        #   3 range_max      – distance of the farthest return (mm)
        #   4 count          – number of LiDAR returns per pixel (density)
        #   5 z_span         – height extent (z_max − z_min) of the pixel column
        bev_image = np.dstack((
            bev_intensity_max, bev_z_max, bev_z_min,
            bev_range_max, bev_count, bev_z_span
        ))

        # Convert to torch tensor
        return torch.from_numpy(bev_image).type(getattr(torch, self.dtype))

    def project_lidar_to_side(self, point_cloud: np.ndarray,
                             img_size: Tuple[int, int] = None,
                             x_range: Tuple[float, float] = None,
                             y_range: Tuple[float, float] = None,
                             z_range: Tuple[float, float] = None) -> torch.Tensor:
        """Project LiDAR point cloud to Side View (X-Z plane) with 6 channels.

        Arguments:
            point_cloud: LiDAR data with shape (N, 9)
                [x, y, z, intensity, timestamp, reflectivity, ring, azimuth, range]
            img_size: Output image size (W, H) - width is X-axis, height is Z-axis
            x_range: X-axis range in meters (min, max) - forward distance
            y_range: Y-axis range in meters (min, max) - lateral filtering (FOV)
            z_range: Z-axis range in meters (min, max) - vertical height

        Returns:
            side_image: Side view projection with shape (H, W, 6) as torch.Tensor
        """
        if img_size is None:
            img_size = (lidar_info.side_width, lidar_info.side_height)
        if x_range is None:
            x_range = lidar_info.x_range_default
        if y_range is None:
            y_range = lidar_info.y_range_default
        if z_range is None:
            z_range = (lidar_info.z_min, lidar_info.z_max)

        W, H = img_size  # W: X-axis, H: Z-axis

        # Initialize output channels
        side_intensity_max = np.zeros((H, W), dtype=np.float32)
        side_intensity_median = np.zeros((H, W), dtype=np.float32)
        side_intensity_var = np.zeros((H, W), dtype=np.float32)
        side_range_max = np.zeros((H, W), dtype=np.float32)
        side_range_median = np.zeros((H, W), dtype=np.float32)
        side_range_var = np.zeros((H, W), dtype=np.float32)

        # Extract point cloud coordinates and attributes
        x, y, z = point_cloud[:, 0], point_cloud[:, 1], point_cloud[:, 2]
        intensity = point_cloud[:, 3]
        range_vals = point_cloud[:, 8]

        # Filter points outside the FOV range (X, Y, Z all filtered like BEV)
        mask = (x >= x_range[0]) & (x < x_range[1]) & \
               (y >= y_range[0]) & (y < y_range[1]) & \
               (z >= lidar_info.z_min) & (z < lidar_info.z_max)
        x, y, z, intensity, range_vals = x[mask], y[mask], z[mask], intensity[mask], range_vals[mask]

        # Map to pixel coordinates
        # X maps to width (columns), Z maps to height (rows, inverted)
        x_img = ((x - x_range[0]) / (x_range[1] - x_range[0]) * W).astype(int)
        z_img = ((z_range[1] - z) / (z_range[1] - z_range[0]) * H).astype(int)  # Invert Z

        # Clip to image bounds
        x_img = np.clip(x_img, 0, W - 1)
        z_img = np.clip(z_img, 0, H - 1)

        # Vectorized aggregation
        pixel_indices = z_img * W + x_img
        unique_pixels, inverse_indices = np.unique(pixel_indices, return_inverse=True)

        # Aggregate statistics for each unique pixel
        for i, pixel_idx in enumerate(unique_pixels):
            mask = (inverse_indices == i)
            pixel_intensities = intensity[mask]
            pixel_ranges = range_vals[mask]

            # Convert 1D pixel index back to 2D coords
            pz = pixel_idx // W
            px = pixel_idx % W

            # Compute statistics
            side_intensity_max[pz, px] = np.max(pixel_intensities)
            side_intensity_median[pz, px] = np.median(pixel_intensities)
            side_intensity_var[pz, px] = np.var(pixel_intensities)

            side_range_max[pz, px] = np.max(pixel_ranges)
            side_range_median[pz, px] = np.median(pixel_ranges)
            side_range_var[pz, px] = np.var(pixel_ranges)

        # Stack channels (H, W, 6)
        side_image = np.dstack((
            side_intensity_max, side_intensity_median, side_intensity_var,
            side_range_max, side_range_median, side_range_var
        ))

        # Convert to torch tensor
        return torch.from_numpy(side_image).type(getattr(torch, self.dtype))

    def _get_radar_ea_projection(self) -> torch.Tensor:
        """Returns a projection matrix for the elevation-azimuth projection.

        The projection matrix P is given that
        [u]
        [v] = P [r, phi, roh, 1]
        [1]

        with range (r), azimuth (phi) and elevation (roh) in spherical
        coordinates. So that, u and v represent the indices of the radar
        grid (raster).
        """
        return torch.Tensor([
            [0, -1, 0, (len(radar_info.azimuth_raster) - 1) / 2],
            [0, 0, 1, (len(radar_info.elevation_raster) - 1) / 2],
            [0, 0, 0, 1]
        ]).type(getattr(torch, self.dtype))

    def _get_radar_ra_projection(self) -> torch.Tensor:
        """Returns a projection matrix for the range-azimuth projection.

        The projection matrix P is given that
        [u]
        [v] = P [r, phi, roh, 1]
        [1]

        with range (r), azimuth (phi) and elevation (roh) in spherical
        coordinates. So that, u and v represent the indices of the radar
        grid (raster).
        """
        return torch.Tensor([
            [0, -1, 0, (len(radar_info.azimuth_raster) - 1) / 2],
            [len(radar_info.range_raster) / max(radar_info.range_raster), 0, 0, 0],
            [0, 0, 0, 1]
        ]).type(getattr(torch, self.dtype))

    def scale_radar_data(self, sample: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Scales the radar data to a range of 0 to 255

        Arguments:
            sample: Dictionary mapping the sample
                items to thier data tensors.

        Returns:
            sample: Dictionary mapping the sample
                items to thier scaled data tensors.
        """
        for k, v in sample.items():
            if k in {'radar_bev', 'radar_front'}:
                # Scale data to target value range
                sample[k] = \
                    (v - radar_info.min_power) \
                    / (radar_info.max_power - radar_info.min_power) \
                    * (255 - 0) + 0

                # Ensure target value range
                sample[k] = torch.clip(sample[k], 0, 255)

        return sample

    def _get_lidar_bev_projection(self) -> torch.Tensor:
        """Returns a projection matrix for the lidar BEV projection.

        Similar to radar BEV but adapted for LiDAR coordinate system.

        Returns:
            projection_matrix: 3x4 projection matrix
        """
        # Use parameters from lidar_info (similar to radar)
        img_width = lidar_info.bev_resolution
        img_height = lidar_info.bev_resolution

        # X and Y range from lidar_info
        x_min, x_max = lidar_info.x_range_default
        y_min, y_max = lidar_info.y_range_default
        x_range = x_max - x_min
        y_range = y_max - y_min

        # return torch.Tensor([
        #     [0, -img_width / y_range, 0, img_width * (-y_min / y_range)],
        #     [img_height / x_range, 0, 0, img_height * (-x_min / x_range)],
        #     [0, 0, 0, 1]
        # ]).type(getattr(torch, self.dtype))
        return torch.Tensor([
            [0, img_width / y_range, 0, -y_min * img_width / y_range],
            [img_height / x_range, 0, 0, -x_min * img_height / x_range],
            [0, 0, 0, 1]
        ]).type(getattr(torch, self.dtype))

    def _get_lidar_side_projection(self) -> torch.Tensor:
        """Returns a projection matrix for the lidar Side View projection (X-Z plane).

        Similar to radar EA (elevation-azimuth) but adapted for LiDAR side view.

        Returns:
            projection_matrix: 3x4 projection matrix
        """
        # Use parameters from lidar_info
        img_width = lidar_info.side_width
        img_height = lidar_info.side_height

        # X and Z range from lidar_info (use z_min/z_max to match filtering)
        x_min, x_max = lidar_info.x_range_default
        z_min, z_max = lidar_info.z_min, lidar_info.z_max
        x_range = x_max - x_min
        z_range = z_max - z_min

        # X maps to width (u), Z maps to height (v, inverted)
        return torch.Tensor([
            [img_width / x_range, 0, 0, img_width * (-x_min / x_range)],
            [0, 0, -img_height / z_range, img_height * (z_max / z_range)],
            [0, 0, 0, 1]
        ]).type(getattr(torch, self.dtype))


    def resize_image(self, sample: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Resizes the images in the sample.

        Resizes the images to a Tensor of shape (C, self.image_size[0], self.image_size[1])
        if image size if given as a tuple of interger values, otherwise it resizes it to a
        tensor with shape (C, self.image_size, self.image_size * width / height).

        Arguments:
            sample: Dictionary mapping the sample
                items to thier data tensors.

        Returns:
            sample: Dictionary mapping the sample
                items to thier data tensors.
        """
        if 'M' in self.camera:
            sample['camera_mono'] = \
                resize(sample['camera_mono'].movedim(-1, 0), self.image_size,antialias=True).movedim(0, -1)
        if 'S' in self.camera:
            sample['camera_stereo'] = \
                resize(sample['camera_stereo'].movedim(-1, 0), self.image_size,antialias=True).movedim(0, -1)

        return sample

    def get_detection_label(self, raw_label: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Get detection task label data.

        Splits the K-Radar dataset label, given as bounding box of
        [x, y, z, theta, l, w, h, category index, object id], into
        its individual components.

        Arguments:
            raw_label: Collection of sample data.

        Returns:
            label: Modified collecton of sample data.
        """
        # Initialize label
        label = {}

        # Split label data into its components
        label['gt_center'] = raw_label[:, (0, 1, 2)]
        label['gt_size'] = raw_label[:, (4, 5, 6)]

        # Encode angle by its sin and cos part
        label['gt_angle'] = torch.cat(
            (torch.sin(raw_label[:, (3, )]), torch.cos(raw_label[:, (3, )])),
            dim=-1
        )

        # One hot encode class labels (+1 for ignore class)
        label['gt_class'] = torch.nn.functional.one_hot(
            raw_label[:, 7].long() + 1,
            self.num_classes
        ).type(getattr(torch, self.dtype))

        # Get configured field of view
        x_min, x_max = self.fov.get('x', torch.tensor([-float('inf'), float('inf')]))
        y_min, y_max = self.fov.get('y', torch.tensor([-float('inf'), float('inf')]))
        z_min, z_max = self.fov.get('z', torch.tensor([-float('inf'), float('inf')]))
        a_min, a_max = self.fov.get('azimuth', torch.tensor([-float('inf'), float('inf')]))

        # Get azimuth angle of the center points
        azimuth = torch.rad2deg(torch.arctan2(label['gt_center'][:, 1], label['gt_center'][:, 0]))

        # Limit lables to configured field of view (FoV)
        x_mask = (x_min < label['gt_center'][:, 0]) & (label['gt_center'][:, 0] < x_max)
        y_mask = (y_min < label['gt_center'][:, 1]) & (label['gt_center'][:, 1] < y_max)
        z_mask = (z_min < label['gt_center'][:, 2]) & (label['gt_center'][:, 2] < z_max)
        a_mask = (a_min < azimuth) & (azimuth < a_max)

        fov_mask = x_mask & y_mask & z_mask & a_mask

        # Mask lables according to the field of view
        label = {k: v[fov_mask] for k, v in label.items()}

        return label

    def get_sample_path(self, src: str) -> Dict[str, str]:
        """Returns all data paths of a given dataset sample.
        Only includes paths for files that actually exist in the directory,
        allowing for modality absence to be detected by file absence.

        Arguments:
            src: Source path to the data files
                of a single dataset sample.

        Returns:
            sample_path: Dictionary mapping the sample
                items to filenames for existing files only.
        """
        # Initialize sample paths
        sample_path = {}

        # Check if the source directory exists
        if not osp.exists(src):
            print(f"Warning: Sample directory not found: {src}")
            return sample_path
            
        # List all files in the directory to check what's actually available
        available_files = set(os.listdir(src))

        # Get sensor data and calibration information only if files exist
        if 'M' in self.camera:
            if 'mono.jpg' in available_files:
                sample_path['camera_mono'] = osp.join(src, 'mono.jpg')
                # Also include calibration file if it exists
                if 'mono_info.npy' in available_files:
                    sample_path['label_to_camera_mono'] = osp.join(src, 'mono_info.npy')

        if 'S' in self.camera:
            if 'stereo.jpg' in available_files:
                sample_path['camera_stereo'] = osp.join(src, 'stereo.jpg')
                if 'stereo_info.npy' in available_files:
                    sample_path['label_to_camera_stereo'] = osp.join(src, 'stereo_info.npy')

        if 'B' in self.radar:
            if 'ra.npy' in available_files:
                sample_path['radar_bev'] = osp.join(src, 'ra.npy')
                if 'ra_info.npy' in available_files:
                    sample_path['label_to_radar_bev'] = osp.join(src, 'ra_info.npy')

        if 'F' in self.radar:
            if 'ea.npy' in available_files:
                sample_path['radar_front'] = osp.join(src, 'ea.npy')
                if 'ea_info.npy' in available_files:
                    sample_path['label_to_radar_front'] = osp.join(src, 'ea_info.npy')

        if self.lidar == 1 and 'os1.npy' in available_files:
            sample_path['lidar_top'] = osp.join(src, 'os1.npy')

        if self.lidar == 2 and 'os2.npy' in available_files:
            sample_path['lidar_top'] = osp.join(src, 'os2.npy')

        # Get annotation data if it exists
        if self.label == 'detection' and 'labels.npy' in available_files:
            sample_path['label'] = osp.join(src, 'labels.npy')

        # Get description data if it exists
        if 'description.npy' in available_files:
            sample_path['description'] = osp.join(src, 'description.npy')

        return sample_path

    def get_dataset_paths(
        self,
        src: str
    ) -> Union[Dict[str, List[Dict[str, str]]], List[Dict[str, str]]]:
        """Returns the paths of all dataset items.

        The return type is either a list of dictionaries (each representing
        a single sample) or a dictionary of lists (each representing a
        single sequence), where each list holds the dictionaries of the
        single samples.

            sequential: Dict[sequence number, List[sample dicts]]
            non sequential: List[sample dicts]

        Arguments:
            src: Source path to the pre-processed
                dataset folder.

        Returns:
            dataset_paths: File paths of all dataset
                items (either sequences or samples).
        """
        # Initialize dataset paths
        dataset_paths = {}

        # List all sequences in the dataset
        for sequence in os.listdir(osp.join(src, self.split)):
            # Set sequence path
            sequence_path = osp.join(src, self.split, sequence)

            # List all samples in the sequence
            samples = sorted(os.listdir(sequence_path))

            # Disolve all sample data paths
            dataset_paths[sequence] = [
                self.get_sample_path(osp.join(sequence_path, sample)) for sample in samples
            ]

        # Concatenate all sequences for non sequential data
        if not self.sequential:
            dataset_paths = list(chain.from_iterable(dataset_paths.values()))

        return dataset_paths

    def load_sample_data(self, sample_path: Dict[str, str]) -> Dict[str, torch.Tensor]:
        """Returns the actual sample data given their paths.
        Handles missing modalities by only including data for files that exist.

        Arguments:
            sample_path: Dictionary mapping the sample
                items to filenames.

        Returns:
            sample: Dictionary mapping the sample
                items to their data tensors for available modalities only.
        """
        # Initialize sample
        sample = {}

        # Load data for each path in sample_path (only includes existing files)
        for key, path in sample_path.items():
            if osp.splitext(path)[-1] in {'.png', '.jpg'}:
                # Load image
                img: torch.Tensor = read_image(path).type(getattr(torch, self.dtype))
                # Change to channel last format (C, H, W) -> (H, W, C)
                sample[key] = img.movedim(0, -1)
            elif osp.splitext(path)[-1] in {'.npy'}:
                # sample[key] = torch.from_numpy(np.load(path)).type(getattr(torch, self.dtype))
                data = np.load(path)

                # Special handling for LiDAR point cloud data
                if key == 'lidar_top' and data.ndim == 2 and data.shape[1] == 9:
                    # This is a point cloud (N, 9), project to BEV and Side view
                    sample['lidar_top'] = self.project_lidar_to_bev(data)
                    sample['lidar_side'] = self.project_lidar_to_side(data)
                else:
                    # Regular .npy file (image or other data)
                    sample[key] = torch.from_numpy(data).type(getattr(torch, self.dtype))

        return sample

    def scale_lidar_data(self, sample: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Scales the lidar data to a range of 0 to 255

        Arguments:
            sample: Dictionary mapping the sample
                items to thier data tensors.

        Returns:
            sample: Dictionary mapping the sample
                items to thier scaled data tensors.
        """
        for k, v in sample.items():
            if k == 'lidar_top':
                # Channel layout for lidar_top BEV (see project_lidar_to_bev):
                #   0  intensity_max  – normalize with OS1 intensity range
                #   1  z_max          – normalize with z height range
                #   2  z_min          – normalize with z height range
                #   3  range_max      – normalize with detection range in mm
                #   4  count          – normalize by max expected points per pixel
                #   5  z_span         – normalize with full z height span

                # Channel 0: intensity_max
                ch0 = (v[..., 0:1] - lidar_info.min_intensity) \
                    / (lidar_info.max_intensity - lidar_info.min_intensity) * 255
                ch0 = torch.clip(ch0, 0, 255)

                # Channels 1–2: z_max and z_min
                # Empty pixels are initialized to z_min, so they map to 0.
                ch12 = (v[..., 1:3] - lidar_info.z_min) / lidar_info.z_range_norm * 255
                ch12 = torch.clip(ch12, 0, 255)

                # Channel 3: range_max
                ch3 = (v[..., 3:4] - lidar_info.min_range) \
                    / (lidar_info.max_range_norm - lidar_info.min_range) * 255
                ch3 = torch.clip(ch3, 0, 255)

                # Channel 4: point count
                ch4 = v[..., 4:5] / lidar_info.max_count_per_pixel * 255
                ch4 = torch.clip(ch4, 0, 255)

                # Channel 5: z_span
                ch5 = v[..., 5:6] / lidar_info.z_range_norm * 255
                ch5 = torch.clip(ch5, 0, 255)

                sample[k] = torch.cat([ch0, ch12, ch3, ch4, ch5], dim=-1)

            elif k == 'lidar_side':
                # lidar_side uses the old intensity/range layout (not used by
                # the model, kept for consistency with project_lidar_to_side).
                intensity_channels = v[..., :3]
                intensity_scaled = \
                    (intensity_channels - lidar_info.min_intensity) \
                    / (lidar_info.max_intensity - lidar_info.min_intensity) \
                    * (255 - 0) + 0
                intensity_scaled = torch.clip(intensity_scaled, 0, 255)

                range_channels = v[..., 3:]
                range_scaled = \
                    (range_channels - lidar_info.min_range) \
                    / (lidar_info.max_range_norm - lidar_info.min_range) \
                    * (255 - 0) + 0
                range_scaled = torch.clip(range_scaled, 0, 255)

                sample[k] = torch.cat([intensity_scaled, range_scaled], dim=-1)

        return sample
    def modality_dropout(self, sample: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Applies modality dropout to the sample data.

        Randomly drops one input modality by setting all input
        values to zero. The drop ratio of each modality is given
        by their individual dropout propabilities.

        Note: It is ensured that not all modalities are dropped
        at the same time but at least one modality remains.

        Arguments:
            sample: Dictionary mapping the sample
                items to thier data tensors.

        Returns:
            sample: Dictionary mapping the sample items to
                thier data tensors with applied dropout.
        """
        # Draw of lots (select a modality based on their probabilities)
        # drawing = self.lottery[np.random.choice(3, replace=True, p=self.dropout)]
        n_modalities = 4 if self.lidar > 0 else 3
        drawing = self.lottery[np.random.choice(n_modalities, replace=True, p=self.dropout)]

        # Apply dropout (replace selected input modality with zeros)
        for modality in drawing:
            if modality in sample:
                sample[modality] = torch.zeros_like(sample[modality])

        return sample


def initialize_kradar(*args, **kwargs):
    return KRadarDataset.from_config(*args, **kwargs)


# from __future__ import annotations  # noqa: F407

# import os
# import os.path as osp

# from itertools import chain
# from typing import Any, Dict, List, Tuple, Union

# import torch
# import numpy as np

# from torch.utils.data import Dataset
# from torchvision.io import read_image
# from torchvision.transforms.functional import resize

# from dprt.datasets.kradar.utils import radar_info


# class KRadarDataset(Dataset):
#     def __init__(self,
#                  src: str,
#                  version: str = '',
#                  split: str = 'train',
#                  camera: str = 'M',
#                  camera_dropout: float = 0.0,
#                  image_size: Union[int, Tuple[int, int]] = None,
#                  radar: str = 'BF',
#                  radar_dropout: float = 0.0,
#                  lidar: int = 0,
#                  lidar_dropout: float = 0.0,  # 新增
#                  label: str = 'detection',
#                  num_classes: int = 1,
#                  sequential: bool = False,
#                  scale: bool = True,
#                  fov: Dict[str, Tuple[float, float]] = None,
#                  dtype: str = 'float32',
#                  **kwargs):
#         """Dataset class for the K-Radar dataset.

#         Arguments:
#             src: Source path to the pre-processed
#                 dataset folder.
#             version: Dataset version. One of either
#                 mini or None (full dataset).
#             split: Dataset split to load. One of
#                 either train or test.
#             camera: Camera modalities to use. One of
#                 either 'M' (mono camera), 'S' (stereo camera)
#                 or None.
#             camera_dropout: Camera modality dropout probability 
#                 between 0 and 1.
#             image_size: Image size to resize image data to.
#                 Either None (no resizing), int or tuple of two
#                 int specifying the height and width.
#             radar: Radar modalities to use. Any combination
#                 of 'B' (BEV) and 'F' (Front) or None
#             radar_dropout: Radar modality dropout probability
#                 between 0 and 1.
#             lidar_dropout: Lidar modality dropout probability
#                 between 0 and 1.
#             lidar: Lidar modality to use. One of either
#                 0 (no lidar), 1 (OS1) or 2 (OS2).
#             label: Type of label data to use. One of either
#                 'detection' (3D bounding boxes), 'occupancy'
#                 (3D occupancy grid) or None.
#             num_classes: Number of object classes used for
#                 one hot encoding.
#             sequential: Whether to consume sequneces of
#                 samples or single samples.
#             scale: Whether to scale the radar data to
#                 a value range of [0, 255] or not.
#             fov: Field of view to limit the lables to. Can
#                 contain values for x, y, z and azimuth.
#         """
#         # Initialize parent dataset class
#         super().__init__()

#         # Check attribute values
#         assert camera_dropout + radar_dropout + lidar_dropout<= 1.0
#         # assert camera_dropout + radar_dropout <= 1.0

#         # Initialize instance attributes
#         self.src = src
#         self.version = version
#         self.split = split
#         self.camera = camera
#         self.camera_dropout = camera_dropout
#         self.image_size = image_size
#         self.radar = radar
#         self.radar_dropout = radar_dropout
#         self.lidar = lidar
#         self.lidar_dropout = lidar_dropout
#         self.label = label
#         self.num_classes = num_classes
#         self.sequential = sequential
#         self.scale = scale
#         self.fov = fov if fov is not None else {}
#         self.dtype = dtype

#         # Adjust split according to dataset version
#         if self.version:
#             self.split = f"{self.version}_{self.split}"

#         # Define lottery pot
#         if self.lidar > 0:
#             # 4模态：支持LiDAR dropout
#             self.lottery = [
#                 {},
#                 {'camera_mono', 'camera_stereo'},
#                 {'radar_bev', 'radar_front'},
#                 {'lidar_top'}
#             ]
#             self.dropout = [
#                 1 - (self.camera_dropout + self.radar_dropout + self.lidar_dropout),
#                 self.camera_dropout,
#                 self.radar_dropout,
#                 self.lidar_dropout
#             ]
#         else:
#             # 原有3模态逻辑
#             self.lottery = [
#                 {},
#                 {'camera_mono', 'camera_stereo'},
#                 {'radar_bev', 'radar_front'}
#             ]
#             self.dropout = [
#                 1 - (self.camera_dropout + self.radar_dropout),
#                 self.camera_dropout,
#                 self.radar_dropout
#             ]
#         # # Initialize moality dropout attributes
#         # # Define the lottery pot to draw from (None, camera, radar)
#         # self.lottery = [
#         #     {},
#         #     {'camera_mono', 'camera_stereo'},
#         #     {'radar_bev', 'radar_front'}
#         # ]

#         # # Define dropout probabilities (sum of probabilities must be <= 1)
#         # self.dropout = [
#         #     1 - (self.camera_dropout + self.radar_dropout),
#         #     self.camera_dropout,
#         #     self.radar_dropout
#         # ]

#         # Get dataset path
#         self.dataset_paths = self.get_dataset_paths(self.src)

#     def __len__(self):
#         return len(self.dataset_paths)

#     def __getitem__(self, index) -> Dict[str, torch.Tensor]:
#         """Returns an item of the dataset given its index.
#         Handles missing modalities by only processing available data.

#         Whether or not the retured Tensors include a time
#         dimension depends on whether or not sequential is
#         true or false.

#         Arguments:
#             index: Index of the dataset item to return.

#         Returns:
#             item: Dataset item as dictionary of tensors.
#         """
#         # Map index to sequence number for sequential data
#         if self.sequential:
#             index = list(sorted(self.dataset_paths.keys()))[index]

#         # Get item from dataset
#         item = self._to_list(self.dataset_paths[index])

#         # Load data from file paths
#         for sample in item:
#             sample = self.load_sample_data(sample)

#         # Check if modalities exist before processing
#         has_radar = any(mode in sample for mode in ['radar_bev', 'radar_front'])
#         has_lidar = any(mode in sample for mode in ['lidar_top'])
#         has_camera = any(mode in sample for mode in ['camera_mono', 'camera_stereo'])
#         has_labels = 'label' in sample
#         has_description = 'description' in sample

#         # Scale radar data only if it exists
#         if self.scale and has_radar:
#             sample = self.scale_radar_data(sample)

#         # Scale radar and lidar data
#         if self.scale and has_lidar:
#             sample = self.scale_lidar_data(sample)  # 新增

#         # Apply modality dropout
#         if has_radar or has_camera or has_lidar:
#         # if has_radar or has_camera:
#             sample = self.modality_dropout(sample)

#         # Get task specific label
#         label = {}
#         if self.label == 'detection':
#             if has_labels:
#                 label = self.get_detection_label(sample.pop('label'))
#             else:
#                 # Create empty label if missing
#                 label = {
#                     'gt_center': torch.empty((0, 3), dtype=getattr(torch, self.dtype)),
#                     'gt_size': torch.empty((0, 3), dtype=getattr(torch, self.dtype)),
#                     'gt_angle': torch.empty((0, 2), dtype=getattr(torch, self.dtype)),
#                     'gt_class': torch.empty((0, self.num_classes), dtype=getattr(torch, self.dtype))
#                 }

#         # Add description to label if it exists
#         if has_description:
#             label.update({'description': sample.pop('description')})
#         else:
#             # Add empty description if missing
#             label.update({'description': torch.zeros((4,), dtype=getattr(torch, self.dtype))})

#         # Only add transformations for existing modalities
#         if any(mode in sample for mode in ['camera_mono', 'camera_stereo', 'radar_bev', 'radar_front','lidar_top']):
#         # if any(mode in sample for mode in ['camera_mono', 'camera_stereo', 'radar_bev', 'radar_front']):
#             sample = self._add_transformations(sample)

#         # Only add projections for existing modalities
#         if any(mode in sample for mode in ['camera_mono', 'camera_stereo', 'radar_bev', 'radar_front','lidar_top']):
#         # if any(mode in sample for mode in ['camera_mono', 'camera_stereo', 'radar_bev', 'radar_front']):
#             sample = self._add_projections(sample)

#         # Only add shapes for existing modalities
#         if any(mode in sample for mode in ['camera_mono', 'camera_stereo', 'radar_bev', 'radar_front','lidar_top']):
#         # if any(mode in sample for mode in ['camera_mono', 'camera_stereo', 'radar_bev', 'radar_front']):
#             sample = self._add_shape(sample)

#         # Resize image only if required and image exists
#         if self.image_size is not None and has_camera:
#             sample = self.resize_image(sample)

#         # Convert list of dicts to dict of stacked tensors
#         if self.sequential:
#             # Stack tensors along the time dimension (use padding for variable
#             # size inputs, e.g. label)
#             # item = {key: default_collate([d[key] for d in item]) for key in sample}
#             raise NotImplementedError()
#         else:
#             # There is just a single sample for non sequential data
#             item = sample

#         return item, label

#     @classmethod
#     def from_config(cls, config: Dict, *args, **kwargs) -> KRadarDataset:  # noqa: F821
#         return cls(*args, **dict(config['computing'] | config['data']), **kwargs)

#     @staticmethod
#     def _to_list(item: Any) -> List[Any]:
#         if not isinstance(item, (list, tuple, set)):
#             return [item]
#         return item

#     def _add_transformations(self, sample: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
#         """Adds the transformation matrices to the sample only for existing modalities.

#         Arguments:
#             sample: Dictionary mapping the sample
#                 items to their data tensors.

#         Returns:
#             sample: Dictionary mapping the sample
#                 items to their scaled data tensors.
#         """
#         # Only process modalities that exist in the sample
#         if 'camera_mono' in sample and 'label_to_camera_mono' in sample:
#             sample['label_to_camera_mono_t'] = torch.zeros_like(sample['label_to_camera_mono'])
#         elif 'camera_mono' in sample and 'label_to_camera_mono' not in sample:
#             # Create identity transformation if calibration is missing
#             sample['label_to_camera_mono_t'] = torch.eye(4, dtype=getattr(torch, self.dtype))

#         if 'camera_stereo' in sample and 'label_to_camera_stereo' in sample:
#             sample['label_to_camera_stereo_t'] = torch.zeros_like(sample['label_to_camera_stereo'])
#         elif 'camera_stereo' in sample and 'label_to_camera_stereo' not in sample:
#             sample['label_to_camera_stereo_t'] = torch.eye(4, dtype=getattr(torch, self.dtype))

#         if 'radar_bev' in sample and 'label_to_radar_bev' in sample:
#             sample['label_to_radar_bev_t'] = sample.pop('label_to_radar_bev')
#         elif 'radar_bev' in sample and 'label_to_radar_bev' not in sample:
#             sample['label_to_radar_bev_t'] = torch.eye(4, dtype=getattr(torch, self.dtype))

#         if 'radar_front' in sample and 'label_to_radar_front' in sample:
#             sample['label_to_radar_front_t'] = sample.pop('label_to_radar_front')
#         elif 'radar_front' in sample and 'label_to_radar_front' not in sample:
#             sample['label_to_radar_front_t'] = torch.eye(4, dtype=getattr(torch, self.dtype))

#         if self.lidar > 0 and 'label_to_lidar_top' in sample:
#             sample['label_to_lidar_top_t'] = sample.pop('label_to_lidar_top')
#         elif self.lidar > 0:
#             # Create identity transformation matrix if not loaded from file
#             sample['label_to_lidar_top_t'] = torch.eye(4).type(getattr(torch, self.dtype))

#         return sample

#     def _add_projections(self, sample: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
#         """Adds the projection matrices to the sample only for existing modalities.

#         Arguments:
#             sample: Dictionary mapping the sample
#                 items to their data tensors.

#         Returns:
#             sample: Dictionary mapping the sample
#                 items to their scaled data tensors.
#         """
#         # Only process modalities that exist in the sample
#         if 'camera_mono' in sample and 'label_to_camera_mono' in sample:
#             sample['label_to_camera_mono_p'] = sample.pop('label_to_camera_mono')
#         elif 'camera_mono' in sample:
#             sample['label_to_camera_mono_p'] = torch.eye(4, dtype=getattr(torch, self.dtype))

#         if 'camera_stereo' in sample and 'label_to_camera_stereo' in sample:
#             sample['label_to_camera_stereo_p'] = sample.pop('label_to_camera_stereo')
#         elif 'camera_stereo' in sample:
#             sample['label_to_camera_stereo_p'] = torch.eye(4, dtype=getattr(torch, self.dtype))

#         if 'radar_bev' in sample:
#             sample['label_to_radar_bev_p'] = self._get_radar_ra_projection()

#         if 'radar_front' in sample:
#             sample['label_to_radar_front_p'] = self._get_radar_ea_projection()
#         if self.lidar > 0:
#             sample['label_to_lidar_top_p'] = self._get_lidar_bev_projection()


#         return sample

#     def _add_shape(self, sample: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
#         """Adds the original input data shape to the sample only for existing modalities.

#         Arguments:
#             sample: Dictionary mapping the sample
#                 items to their data tensors.

#         Returns:
#             sample: Dictionary mapping the sample
#                 items to their scaled data tensors.
#         """
#         # Only process modalities that exist in the sample
#         if 'camera_mono' in sample:
#             sample['camera_mono_shape'] = torch.as_tensor(sample['camera_mono'].shape)

#         if 'camera_stereo' in sample:
#             sample['camera_stereo_shape'] = torch.as_tensor(sample['camera_stereo'].shape)

#         if 'radar_bev' in sample:
#             sample['radar_bev_shape'] = torch.as_tensor(sample['radar_bev'].shape)

#         if 'radar_front' in sample:
#             sample['radar_front_shape'] = torch.as_tensor(sample['radar_front'].shape)

#         if 'lidar_top' in sample:
#             sample['lidar_top_shape'] = torch.as_tensor(sample['lidar_top'].shape)

#         return sample

#     def project_lidar_to_bev(self, point_cloud: np.ndarray,
#                             img_size: Tuple[int, int] = (256, 256),
#                             x_range: Tuple[float, float] = (0, 100),
#                             y_range: Tuple[float, float] = (-50, 50)) -> torch.Tensor:
#         """Project LiDAR point cloud to BEV image with 6 channels.

#         Arguments:
#             point_cloud: LiDAR data with shape (N, 9)
#                 [x, y, z, intensity, timestamp, reflectivity, ring, azimuth, range]
#             img_size: Output image size (H, W)
#             x_range: X-axis range in meters (min, max)
#             y_range: Y-axis range in meters (min, max)

#         Returns:
#             bev_image: BEV projection with shape (H, W, 6) as torch.Tensor
#         """
#         H, W = img_size

#         # Initialize output channels
#         bev_intensity_max = np.zeros((H, W), dtype=np.float32)
#         bev_intensity_median = np.zeros((H, W), dtype=np.float32)
#         bev_intensity_var = np.zeros((H, W), dtype=np.float32)
#         bev_range_max = np.zeros((H, W), dtype=np.float32)
#         bev_range_median = np.zeros((H, W), dtype=np.float32)
#         bev_range_var = np.zeros((H, W), dtype=np.float32)

#         # Extract point cloud coordinates and attributes
#         x, y, z = point_cloud[:, 0], point_cloud[:, 1], point_cloud[:, 2]
#         intensity = point_cloud[:, 3]
#         range_vals = point_cloud[:, 8]

#         # Filter points outside the range
#         mask = (x >= x_range[0]) & (x < x_range[1]) & \
#                (y >= y_range[0]) & (y < y_range[1])
#         x, y, intensity, range_vals = x[mask], y[mask], intensity[mask], range_vals[mask]

#         # Map to pixel coordinates
#         x_img = ((x - x_range[0]) / (x_range[1] - x_range[0]) * H).astype(int)
#         y_img = ((y - y_range[0]) / (y_range[1] - y_range[0]) * W).astype(int)

#         # Clip to image bounds
#         x_img = np.clip(x_img, 0, H - 1)
#         y_img = np.clip(y_img, 0, W - 1)

#         # Aggregate points per pixel using dictionaries
#         from collections import defaultdict
#         pixel_points = defaultdict(lambda: {'intensity': [], 'range': []})

#         for i in range(len(x_img)):
#             pixel_points[(x_img[i], y_img[i])]['intensity'].append(intensity[i])
#             pixel_points[(x_img[i], y_img[i])]['range'].append(range_vals[i])

#         # Compute statistical features
#         for (px, py), values in pixel_points.items():
#             intensities = np.array(values['intensity'])
#             ranges = np.array(values['range'])

#             bev_intensity_max[px, py] = np.max(intensities)
#             bev_intensity_median[px, py] = np.median(intensities)
#             bev_intensity_var[px, py] = np.var(intensities)

#             bev_range_max[px, py] = np.max(ranges)
#             bev_range_median[px, py] = np.median(ranges)
#             bev_range_var[px, py] = np.var(ranges)

#         # Stack channels (H, W, 6)
#         bev_image = np.dstack((
#             bev_intensity_max, bev_intensity_median, bev_intensity_var,
#             bev_range_max, bev_range_median, bev_range_var
#         ))

#         # Convert to torch tensor
#         return torch.from_numpy(bev_image).type(getattr(torch, self.dtype))
#     def _get_radar_ea_projection(self) -> torch.Tensor:
#         """Returns a projection matrix for the elevation-azimuth projection.

#         The projection matrix P is given that
#         [u]
#         [v] = P [r, phi, roh, 1]
#         [1]

#         with range (r), azimuth (phi) and elevation (roh) in spherical
#         coordinates. So that, u and v represent the indices of the radar
#         grid (raster).
#         """
#         return torch.Tensor([
#             [0, -1, 0, (len(radar_info.azimuth_raster) - 1) / 2],
#             [0, 0, 1, (len(radar_info.elevation_raster) - 1) / 2],
#             [0, 0, 0, 1]
#         ]).type(getattr(torch, self.dtype))

#     def _get_radar_ra_projection(self) -> torch.Tensor:
#         """Returns a projection matrix for the range-azimuth projection.

#         The projection matrix P is given that
#         [u]
#         [v] = P [r, phi, roh, 1]
#         [1]

#         with range (r), azimuth (phi) and elevation (roh) in spherical
#         coordinates. So that, u and v represent the indices of the radar
#         grid (raster).
#         """
#         return torch.Tensor([
#             [0, -1, 0, (len(radar_info.azimuth_raster) - 1) / 2],
#             [len(radar_info.range_raster) / max(radar_info.range_raster), 0, 0, 0],
#             [0, 0, 0, 1]
#         ]).type(getattr(torch, self.dtype))

#     def scale_radar_data(self, sample: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
#         """Scales the radar data to a range of 0 to 255

#         Arguments:
#             sample: Dictionary mapping the sample
#                 items to thier data tensors.

#         Returns:
#             sample: Dictionary mapping the sample
#                 items to thier scaled data tensors.
#         """
#         for k, v in sample.items():
#             if k in {'radar_bev', 'radar_front'}:
#                 # Scale data to target value range
#                 sample[k] = \
#                     (v - radar_info.min_power) \
#                     / (radar_info.max_power - radar_info.min_power) \
#                     * (255 - 0) + 0

#                 # Ensure target value range
#                 sample[k] = torch.clip(sample[k], 0, 255)

#         return sample

#     def _get_lidar_bev_projection(self) -> torch.Tensor:
#         """Returns a projection matrix for the lidar BEV projection.

#         Similar to radar BEV but adapted for LiDAR coordinate system.

#         Returns:
#             projection_matrix: 3x4 projection matrix
#         """
#         # LiDAR projection image parameters
#         img_width = 256
#         img_height = 256
#         x_range = 100.0  # Forward 0-100 meters
#         y_range = 100.0  # Left-right ±50 meters (total 100 meters)

#         return torch.Tensor([
#             [0, -img_width / y_range, 0, img_width / 2],
#             [img_height / x_range, 0, 0, 0],
#             [0, 0, 0, 1]
#         ]).type(getattr(torch, self.dtype))
    

#     def resize_image(self, sample: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
#         """Resizes the images in the sample.

#         Resizes the images to a Tensor of shape (C, self.image_size[0], self.image_size[1])
#         if image size if given as a tuple of interger values, otherwise it resizes it to a
#         tensor with shape (C, self.image_size, self.image_size * width / height).

#         Arguments:
#             sample: Dictionary mapping the sample
#                 items to thier data tensors.

#         Returns:
#             sample: Dictionary mapping the sample
#                 items to thier data tensors.
#         """
#         if 'M' in self.camera:
#             sample['camera_mono'] = \
#                 resize(sample['camera_mono'].movedim(-1, 0), self.image_size,antialias=True).movedim(0, -1)
#         if 'S' in self.camera:
#             sample['camera_stereo'] = \
#                 resize(sample['camera_stereo'].movedim(-1, 0), self.image_size,antialias=True).movedim(0, -1)

#         return sample

#     def get_detection_label(self, raw_label: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
#         """Get detection task label data.

#         Splits the K-Radar dataset label, given as bounding box of
#         [x, y, z, theta, l, w, h, category index, object id], into
#         its individual components.

#         Arguments:
#             raw_label: Collection of sample data.

#         Returns:
#             label: Modified collecton of sample data.
#         """
#         # Initialize label
#         label = {}

#         # Split label data into its components
#         label['gt_center'] = raw_label[:, (0, 1, 2)]
#         label['gt_size'] = raw_label[:, (4, 5, 6)]

#         # Encode angle by its sin and cos part
#         label['gt_angle'] = torch.cat(
#             (torch.sin(raw_label[:, (3, )]), torch.cos(raw_label[:, (3, )])),
#             dim=-1
#         )

#         # One hot encode class labels (+1 for ignore class)
#         label['gt_class'] = torch.nn.functional.one_hot(
#             raw_label[:, 7].long() + 1,
#             self.num_classes
#         ).type(getattr(torch, self.dtype))

#         # Get configured field of view
#         x_min, x_max = self.fov.get('x', torch.tensor([-float('inf'), float('inf')]))
#         y_min, y_max = self.fov.get('y', torch.tensor([-float('inf'), float('inf')]))
#         z_min, z_max = self.fov.get('z', torch.tensor([-float('inf'), float('inf')]))
#         a_min, a_max = self.fov.get('azimuth', torch.tensor([-float('inf'), float('inf')]))

#         # Get azimuth angle of the center points
#         azimuth = torch.rad2deg(torch.arctan2(label['gt_center'][:, 1], label['gt_center'][:, 0]))

#         # Limit lables to configured field of view (FoV)
#         x_mask = (x_min < label['gt_center'][:, 0]) & (label['gt_center'][:, 0] < x_max)
#         y_mask = (y_min < label['gt_center'][:, 1]) & (label['gt_center'][:, 1] < y_max)
#         z_mask = (z_min < label['gt_center'][:, 2]) & (label['gt_center'][:, 2] < z_max)
#         a_mask = (a_min < azimuth) & (azimuth < a_max)

#         fov_mask = x_mask & y_mask & z_mask & a_mask

#         # Mask lables according to the field of view
#         label = {k: v[fov_mask] for k, v in label.items()}

#         return label

#     def get_sample_path(self, src: str) -> Dict[str, str]:
#         """Returns all data paths of a given dataset sample.
#         Only includes paths for files that actually exist in the directory,
#         allowing for modality absence to be detected by file absence.

#         Arguments:
#             src: Source path to the data files
#                 of a single dataset sample.

#         Returns:
#             sample_path: Dictionary mapping the sample
#                 items to filenames for existing files only.
#         """
#         # Initialize sample paths
#         sample_path = {}

#         # Check if the source directory exists
#         if not osp.exists(src):
#             print(f"Warning: Sample directory not found: {src}")
#             return sample_path
            
#         # List all files in the directory to check what's actually available
#         available_files = set(os.listdir(src))

#         # Get sensor data and calibration information only if files exist
#         if 'M' in self.camera:
#             if 'mono.jpg' in available_files:
#                 sample_path['camera_mono'] = osp.join(src, 'mono.jpg')
#                 # Also include calibration file if it exists
#                 if 'mono_info.npy' in available_files:
#                     sample_path['label_to_camera_mono'] = osp.join(src, 'mono_info.npy')

#         if 'S' in self.camera:
#             if 'stereo.jpg' in available_files:
#                 sample_path['camera_stereo'] = osp.join(src, 'stereo.jpg')
#                 if 'stereo_info.npy' in available_files:
#                     sample_path['label_to_camera_stereo'] = osp.join(src, 'stereo_info.npy')

#         if 'B' in self.radar:
#             if 'ra.npy' in available_files:
#                 sample_path['radar_bev'] = osp.join(src, 'ra.npy')
#                 if 'ra_info.npy' in available_files:
#                     sample_path['label_to_radar_bev'] = osp.join(src, 'ra_info.npy')

#         if 'F' in self.radar:
#             if 'ea.npy' in available_files:
#                 sample_path['radar_front'] = osp.join(src, 'ea.npy')
#                 if 'ea_info.npy' in available_files:
#                     sample_path['label_to_radar_front'] = osp.join(src, 'ea_info.npy')

#         if self.lidar == 1 and 'os1.npy' in available_files:
#             sample_path['lidar_top'] = osp.join(src, 'os1.npy')

#         if self.lidar == 2 and 'os2.npy' in available_files:
#             sample_path['lidar_top'] = osp.join(src, 'os2.npy')

#         # Get annotation data if it exists
#         if self.label == 'detection' and 'labels.npy' in available_files:
#             sample_path['label'] = osp.join(src, 'labels.npy')

#         # Get description data if it exists
#         if 'description.npy' in available_files:
#             sample_path['description'] = osp.join(src, 'description.npy')

#         return sample_path

#     def get_dataset_paths(
#         self,
#         src: str
#     ) -> Union[Dict[str, List[Dict[str, str]]], List[Dict[str, str]]]:
#         """Returns the paths of all dataset items.

#         The return type is either a list of dictionaries (each representing
#         a single sample) or a dictionary of lists (each representing a
#         single sequence), where each list holds the dictionaries of the
#         single samples.

#             sequential: Dict[sequence number, List[sample dicts]]
#             non sequential: List[sample dicts]

#         Arguments:
#             src: Source path to the pre-processed
#                 dataset folder.

#         Returns:
#             dataset_paths: File paths of all dataset
#                 items (either sequences or samples).
#         """
#         # Initialize dataset paths
#         dataset_paths = {}

#         # List all sequences in the dataset
#         for sequence in os.listdir(osp.join(src, self.split)):
#             # Set sequence path
#             sequence_path = osp.join(src, self.split, sequence)

#             # List all samples in the sequence
#             samples = sorted(os.listdir(sequence_path))

#             # Disolve all sample data paths
#             dataset_paths[sequence] = [
#                 self.get_sample_path(osp.join(sequence_path, sample)) for sample in samples
#             ]

#         # Concatenate all sequences for non sequential data
#         if not self.sequential:
#             dataset_paths = list(chain.from_iterable(dataset_paths.values()))

#         return dataset_paths

#     def load_sample_data(self, sample_path: Dict[str, str]) -> Dict[str, torch.Tensor]:
#         """Returns the actual sample data given their paths.
#         Handles missing modalities by only including data for files that exist.

#         Arguments:
#             sample_path: Dictionary mapping the sample
#                 items to filenames.

#         Returns:
#             sample: Dictionary mapping the sample
#                 items to their data tensors for available modalities only.
#         """
#         # Initialize sample
#         sample = {}

#         # Load data for each path in sample_path (only includes existing files)
#         for key, path in sample_path.items():
#             if osp.splitext(path)[-1] in {'.png', '.jpg'}:
#                 # Load image
#                 img: torch.Tensor = read_image(path).type(getattr(torch, self.dtype))
#                 # Change to channel last format (C, H, W) -> (H, W, C)
#                 sample[key] = img.movedim(0, -1)
#             elif osp.splitext(path)[-1] in {'.npy'}:
#                 # sample[key] = torch.from_numpy(np.load(path)).type(getattr(torch, self.dtype))
#                 data = np.load(path)

#                 # Special handling for LiDAR point cloud data
#                 if key == 'lidar_top' and data.ndim == 2 and data.shape[1] == 9:
#                     # This is a point cloud (N, 9), project to BEV image
#                     sample[key] = self.project_lidar_to_bev(data)
#                 else:
#                     # Regular .npy file (image or other data)
#                     sample[key] = torch.from_numpy(data).type(getattr(torch, self.dtype))

#         return sample

#     def scale_lidar_data(self, sample: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
#         """Scales the lidar data to a range of 0 to 255

#         Arguments:
#             sample: Dictionary mapping the sample
#                 items to thier data tensors.

#         Returns:
#             sample: Dictionary mapping the sample
#                 items to thier scaled data tensors.
#         """
#         for k, v in sample.items():
#             if k == 'lidar_top':
#                 # LiDAR BEV projection image has 6 channels
#                 # First 3 channels: intensity features (max, median, var)
#                 # Last 3 channels: range features (max, median, var)

#                 # Intensity channels normalization
#                 intensity_channels = v[:, :, :3]
#                 # Normalize based on actual range (adaptive normalization)
#                 intensity_min = intensity_channels.min()
#                 intensity_max = intensity_channels.max()
#                 if intensity_max > intensity_min:
#                     intensity_scaled = (intensity_channels - intensity_min) / (intensity_max - intensity_min) * 255
#                 else:
#                     intensity_scaled = intensity_channels

#                 # Range channels normalization
#                 range_channels = v[:, :, 3:]
#                 # Normalize based on actual range (adaptive normalization)
#                 range_min = range_channels.min()
#                 range_max = range_channels.max()
#                 if range_max > range_min:
#                     range_scaled = (range_channels - range_min) / (range_max - range_min) * 255
#                 else:
#                     range_scaled = range_channels

#                 sample[k] = torch.cat([intensity_scaled, range_scaled], dim=-1)

#         return sample
#     def modality_dropout(self, sample: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
#         """Applies modality dropout to the sample data.

#         Randomly drops one input modality by setting all input
#         values to zero. The drop ratio of each modality is given
#         by their individual dropout propabilities.

#         Note: It is ensured that not all modalities are dropped
#         at the same time but at least one modality remains.

#         Arguments:
#             sample: Dictionary mapping the sample
#                 items to thier data tensors.

#         Returns:
#             sample: Dictionary mapping the sample items to
#                 thier data tensors with applied dropout.
#         """
#         # Draw of lots (select a modality based on their probabilities)
#         # drawing = self.lottery[np.random.choice(3, replace=True, p=self.dropout)]
#         n_modalities = 4 if self.lidar > 0 else 3
#         drawing = self.lottery[np.random.choice(n_modalities, replace=True, p=self.dropout)]

#         # Apply dropout (replace selected input modality with zeros)
#         for modality in drawing:
#             if modality in sample:
#                 sample[modality] = torch.zeros_like(sample[modality])

#         return sample


# def initialize_kradar(*args, **kwargs):
#     return KRadarDataset.from_config(*args, **kwargs)

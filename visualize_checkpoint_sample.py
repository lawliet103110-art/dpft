"""
根据训练得到的 .pt 模型文件，对单个样本进行多视图可视化（BEV、XZ、原图叠加、雷达投影）。

输出会默认生成 per-sample 文件夹，例如：
  visualizations/<split>/<sequence>/<sample>/bev_xy.png

示例:
  实际数据示例: /root/autodl-tmp/autodl-tmp/data/kradar/test/1/00182_00150/mono.jpg
  python /root/autodl-tmp/autodl-tmp/DPFT-main/visualize_checkpoint_sample.py \
    --src /root/autodl-tmp/autodl-tmp/data/kradar \
    --cfg /root/autodl-tmp/autodl-tmp/DPFT-main/config/kradar.json \
    --checkpoint /root/autodl-tmp/autodl-tmp/DPFT-main/result/20251126-235801-585/checkpoints/20251126-235801-585_checkpoint_0199.pt \
    --index 0 \
    --output /root/autodl-tmp/autodl-tmp/DPFT-main/resultjsample_outputs
"""

import argparse
import os
import sys
from typing import Dict, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from dprt.datasets import init as init_dataset
from dprt.datasets.kradar.utils import radar_info
from dprt.models import load as load_model
from dprt.utils.config import load_config
from dprt.utils.geometry import get_box_corners
from dprt.utils.visu import visu_2d_radar_grid


def _select_device(device: Optional[str], config_device: str) -> torch.device:
    device = device or config_device
    if device.startswith('cuda') and not torch.cuda.is_available():
        print("⚠️ CUDA 不可用，已切换为 CPU")
        return torch.device('cpu')
    return torch.device(device)


def _add_batch_dim(sample: dict) -> dict:
    return {k: v.unsqueeze(0) if torch.is_tensor(v) else v for k, v in sample.items()}


def _to_device(sample: dict, device: torch.device) -> dict:
    return {k: v.to(device) if torch.is_tensor(v) else v for k, v in sample.items()}


def _decode_predictions(
    output: dict,
    conf_thr: float,
    use_softmax: bool,
    fov: Tuple[Tuple[float, float], Tuple[float, float], Tuple[float, float]],
) -> np.ndarray:
    """返回预测框，格式: [x, y, z, theta, l, w, h, class]."""
    output = {k: v[0].detach().cpu() for k, v in output.items()}
    class_logits = output['class']

    if use_softmax:
        class_scores = torch.softmax(class_logits, dim=-1)
    else:
        class_scores = class_logits

    confidence, categories = torch.max(class_scores, dim=-1)
    angle = torch.atan2(output['angle'][..., 0], output['angle'][..., 1])
    categories = categories - 1

    x_min, x_max = fov[0]
    y_min, y_max = fov[1]
    z_min, z_max = fov[2]

    x_mask = (x_min < output['center'][:, 0]) & (output['center'][:, 0] < x_max)
    y_mask = (y_min < output['center'][:, 1]) & (output['center'][:, 1] < y_max)
    z_mask = (z_min < output['center'][:, 2]) & (output['center'][:, 2] < z_max)
    cls_mask = categories >= 0
    conf_mask = confidence >= conf_thr
    mask = x_mask & y_mask & z_mask & cls_mask & conf_mask

    if not torch.any(mask):
        return np.empty((0, 8), dtype=np.float32)

    centers = output['center'][mask]
    sizes = output['size'][mask]
    classes = categories[mask].unsqueeze(-1)
    angles = angle[mask].unsqueeze(-1)

    boxes = torch.cat([centers, angles, sizes, classes], dim=-1)
    return boxes.numpy()


def _decode_ground_truth(label: dict) -> np.ndarray:
    """返回 GT 框，格式: [x, y, z, theta, l, w, h, class]."""
    if 'gt_center' not in label:
        return np.empty((0, 8), dtype=np.float32)

    centers = label['gt_center']
    sizes = label['gt_size']
    angles = torch.atan2(label['gt_angle'][..., 0], label['gt_angle'][..., 1]).unsqueeze(-1)
    classes = torch.argmax(label['gt_class'], dim=-1).unsqueeze(-1) - 1

    mask = classes.squeeze(-1) >= 0
    if not torch.any(mask):
        return np.empty((0, 8), dtype=np.float32)

    boxes = torch.cat([centers[mask], angles[mask], sizes[mask], classes[mask]], dim=-1)
    return boxes.detach().cpu().numpy()


def _draw_boxes(ax, boxes: np.ndarray, color: str, label: str) -> None:
    if boxes.size == 0:
        return
    corners = get_box_corners(boxes[:, :7])
    for i, box_corners in enumerate(corners):
        pts = box_corners[:4, :2]
        pts = np.vstack([pts, pts[0]])
        ax.plot(pts[:, 0], pts[:, 1], color=color, linewidth=1.5,
                label=label if i == 0 else None)


def _draw_boxes_xz(ax, boxes: np.ndarray, color: str, label: str) -> None:
    if boxes.size == 0:
        return
    corners = get_box_corners(boxes[:, :7])
    for i, box_corners in enumerate(corners):
        x_vals = box_corners[:, 0]
        z_vals = box_corners[:, 2]
        x_min, x_max = x_vals.min(), x_vals.max()
        z_min, z_max = z_vals.min(), z_vals.max()
        pts = np.array([
            [x_min, z_min],
            [x_max, z_min],
            [x_max, z_max],
            [x_min, z_max],
            [x_min, z_min],
        ])
        ax.plot(pts[:, 0], pts[:, 1], color=color, linewidth=1.5,
                label=label if i == 0 else None)


def _get_sample_path(dataset, index: int) -> Dict[str, str]:
    dataset_paths = getattr(dataset, 'dataset_paths', None)
    if dataset_paths is None:
        return {}
    if isinstance(dataset_paths, list):
        return dataset_paths[index]
    if isinstance(dataset_paths, dict):
        ordered_sequences = []
        for sequence_id in sorted(dataset_paths.keys()):
            ordered_sequences.extend(dataset_paths[sequence_id])
        return ordered_sequences[index] if ordered_sequences else {}
    return {}


def _resolve_sample_dir(sample_path: Dict[str, str]) -> Optional[str]:
    priority = [
        'camera_mono', 'camera_stereo', 'radar_bev', 'radar_front',
        'lidar_top', 'lidar_side', 'label', 'description',
    ]
    for key in priority:
        path = sample_path.get(key)
        if path:
            return os.path.dirname(path)
    return None


def _build_output_dir(args, sample_dir: Optional[str], split: str, index: int) -> str:
    output_root = args.output_dir or args.output or 'visualizations'
    if args.output and os.path.splitext(args.output)[1]:
        output_root = os.path.dirname(args.output) or '.'
    if sample_dir:
        sample_name = os.path.basename(sample_dir)
        sequence = os.path.basename(os.path.dirname(sample_dir))
        return os.path.join(output_root, split, sequence, sample_name)
    return os.path.join(output_root, split, f"index_{index:06d}")


def _write_sample_info(output_dir: str, split: str, index: int,
                       sample_dir: Optional[str], sample_path: Dict[str, str]) -> None:
    info_path = os.path.join(output_dir, 'sample_info.txt')
    lines = [
        f"split: {split}",
        f"index: {index}",
        f"sample_dir: {sample_dir or 'unknown'}",
        "",
        "sample_files:",
    ]
    for key in sorted(sample_path.keys()):
        lines.append(f"  {key}: {sample_path[key]}")
    with open(info_path, 'w', encoding='utf-8') as f:
        f.write("\n".join(lines))
    print(f"📄 已保存样本信息: {info_path}")


def _project_points(points: np.ndarray, projection: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    projection = projection[:3, :] if projection.shape == (4, 4) else projection
    points_h = np.hstack([points, np.ones((points.shape[0], 1))])
    projected = (projection @ points_h.T).T
    depth = projected[:, 2]
    valid = depth > 1e-6
    projected = projected[valid]
    pixels = np.column_stack([projected[:, 0] / projected[:, 2],
                              projected[:, 1] / projected[:, 2]])
    return pixels, valid


def _draw_projected_boxes(ax, boxes: np.ndarray, projection: np.ndarray,
                          color: str, label: str) -> None:
    if boxes.size == 0:
        return
    corners = get_box_corners(boxes[:, :7])
    flat = corners.reshape(-1, 3)
    pixels, valid = _project_points(flat, projection)
    projected = np.full((flat.shape[0], 2), np.nan)
    projected[valid] = pixels
    projected = projected.reshape(corners.shape[0], corners.shape[1], 2)
    valid = valid.reshape(corners.shape[0], corners.shape[1])
    edges = [
        (0, 1), (1, 2), (2, 3), (3, 0),
        (4, 5), (5, 6), (6, 7), (7, 4),
        (0, 4), (1, 5), (2, 6), (3, 7),
    ]
    for i in range(projected.shape[0]):
        for edge in edges:
            if not (valid[i, edge[0]] and valid[i, edge[1]]):
                continue
            pts = projected[i, list(edge)]
            ax.plot(pts[:, 0], pts[:, 1], color=color, linewidth=1.0,
                    label=label if edge == edges[0] and i == 0 else None)


def _save_bev(pred_boxes: np.ndarray, gt_boxes: np.ndarray,
              fov: Tuple[Tuple[float, float], Tuple[float, float], Tuple[float, float]],
              output_dir: str, show_gt: bool, show: bool) -> str:
    output_path = os.path.join(output_dir, 'bev_xy.png')
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.set_title('BEV (XY) Visualization')
    ax.set_xlabel('x (m)')
    ax.set_ylabel('y (m)')
    ax.set_xlim(fov[0])
    ax.set_ylim(fov[1])
    ax.set_aspect('equal', adjustable='box')

    if show_gt:
        _draw_boxes(ax, gt_boxes, color='green', label='Ground Truth')
    _draw_boxes(ax, pred_boxes, color='red', label='Predictions')
    if (show_gt and gt_boxes.size > 0) or pred_boxes.size > 0:
        ax.legend(loc='upper right')

    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    if show:
        plt.show()
    else:
        plt.close(fig)
    return output_path


def _save_xz(pred_boxes: np.ndarray, gt_boxes: np.ndarray,
             fov: Tuple[Tuple[float, float], Tuple[float, float], Tuple[float, float]],
             output_dir: str, show_gt: bool) -> str:
    output_path = os.path.join(output_dir, 'xz_view.png')
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.set_title('XZ Visualization')
    ax.set_xlabel('x (m)')
    ax.set_ylabel('z (m)')
    ax.set_xlim(fov[0])
    ax.set_ylim(fov[2])
    ax.set_aspect('auto')

    if show_gt:
        _draw_boxes_xz(ax, gt_boxes, color='green', label='Ground Truth')
    _draw_boxes_xz(ax, pred_boxes, color='red', label='Predictions')
    if (show_gt and gt_boxes.size > 0) or pred_boxes.size > 0:
        ax.legend(loc='upper right')

    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)
    return output_path


def _save_camera_overlay(sample: dict, pred_boxes: np.ndarray, gt_boxes: np.ndarray,
                         output_dir: str, show_gt: bool) -> Optional[str]:
    if 'camera_mono' not in sample:
        return None
    projection = sample.get('label_to_camera_mono_p')
    if projection is None:
        return None
    projection = projection.detach().cpu().numpy() if torch.is_tensor(projection) else projection
    img = sample['camera_mono'].detach().cpu().numpy()
    if img.dtype != np.uint8:
        img = np.clip(img, 0, 255).astype(np.uint8)
    output_path = os.path.join(output_dir, 'camera_mono_overlay.png')
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.imshow(img)
    ax.set_title('Camera Mono Overlay')
    ax.axis('off')

    if show_gt:
        _draw_projected_boxes(ax, gt_boxes, projection, color='green', label='Ground Truth')
    _draw_projected_boxes(ax, pred_boxes, projection, color='red', label='Predictions')
    if (show_gt and gt_boxes.size > 0) or pred_boxes.size > 0:
        ax.legend(loc='upper right')

    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)
    return output_path


def _save_radar_bev(sample: dict, pred_boxes: np.ndarray, gt_boxes: np.ndarray,
                    output_dir: str, show_gt: bool) -> Optional[str]:
    if 'radar_bev' not in sample:
        return None
    radar = sample['radar_bev'].detach().cpu().numpy()
    if radar.ndim == 3:
        radar = radar[:, :, 0]
    radar = np.clip(radar, 1e-6, None)
    output_path = os.path.join(output_dir, 'radar_bev_overlay.png')
    fig, ax = plt.subplots(figsize=(10, 4))
    visu_2d_radar_grid(
        ax=ax,
        grid=radar,
        raster=[np.array(radar_info.range_raster), np.array(radar_info.azimuth_raster)],
        cart=True,
        dims='ra',
        r_max=max(radar_info.range_raster),
        cm='viridis',
        flip=False
    )
    if show_gt:
        _draw_boxes(ax, gt_boxes, color='green', label='Ground Truth')
    _draw_boxes(ax, pred_boxes, color='red', label='Predictions')
    if (show_gt and gt_boxes.size > 0) or pred_boxes.size > 0:
        ax.legend(loc='upper right')
    ax.set_title('Radar BEV Overlay')
    ax.set_xlabel('x (m)')
    ax.set_ylabel('y (m)')
    ax.set_aspect('equal', adjustable='box')

    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)
    return output_path


def visualize_sample(
    model: torch.nn.Module,
    sample: dict,
    raw_sample: dict,
    label: dict,
    conf_thr: float,
    use_softmax: bool,
    output_dir: str,
    show_gt: bool,
    show: bool,
    fov: Tuple[Tuple[float, float], Tuple[float, float], Tuple[float, float]],
) -> None:
    model.eval()
    with torch.no_grad():
        output = model(sample)

    pred_boxes = _decode_predictions(output, conf_thr, use_softmax, fov)
    gt_boxes = _decode_ground_truth(label) if show_gt else np.empty((0, 8), dtype=np.float32)

    os.makedirs(output_dir, exist_ok=True)

    outputs = []
    outputs.append(_save_bev(pred_boxes, gt_boxes, fov, output_dir, show_gt, show))
    outputs.append(_save_xz(pred_boxes, gt_boxes, fov, output_dir, show_gt))

    camera_path = _save_camera_overlay(raw_sample, pred_boxes, gt_boxes, output_dir, show_gt)
    if camera_path:
        outputs.append(camera_path)
    else:
        print("⚠️ 未检测到 camera_mono 或标定信息，跳过原图叠加输出")

    radar_path = _save_radar_bev(raw_sample, pred_boxes, gt_boxes, output_dir, show_gt)
    if radar_path:
        outputs.append(radar_path)
    else:
        print("⚠️ 未检测到 radar_bev，跳过雷达投影输出")

    print("✅ 已保存可视化结果:")
    for path in outputs:
        print(f"  - {path}")
    print(f"预测框数量: {len(pred_boxes)}")
    if show_gt:
        print(f"真实框数量: {len(gt_boxes)}")


def _get_fov(args, config) -> Tuple[Tuple[float, float], Tuple[float, float], Tuple[float, float]]:
    config_fov = config.get('data', {}).get('fov', {})
    x = args.fov_x or config_fov.get('x', [0.0, 72.0])
    y = args.fov_y or config_fov.get('y', [-6.4, 6.4])
    z = args.fov_z or config_fov.get('z', [-2.0, 6.0])
    return (tuple(map(float, x)), tuple(map(float, y)), tuple(map(float, z)))


def main() -> None:
    parser = argparse.ArgumentParser(description='Single-sample visualization from checkpoint')
    parser.add_argument('--src', type=str, default='/data/kradar/processed',
                        help='Path to processed dataset root.')
    parser.add_argument('--cfg', type=str, default='config/kradar.json',
                        help='Path to config file.')
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='Path to model checkpoint (.pt).')
    parser.add_argument('--split', type=str, default='test',
                        help='Dataset split to use (train/val/test).')
    parser.add_argument('--index', type=int, default=0,
                        help='Sample index in the split.')
    parser.add_argument('--conf', type=float, default=0.5,
                        help='Confidence threshold for predictions.')
    parser.add_argument('--output', type=str, default=None,
                        help='Output root directory (or legacy PNG path).')
    parser.add_argument('--output-dir', type=str, default=None,
                        help='Output root directory for per-sample folders.')
    parser.add_argument('--device', type=str, default=None,
                        help='Override device (e.g. cpu, cuda:0).')
    parser.add_argument('--no-gt', action='store_true',
                        help='Disable ground-truth overlay.')
    parser.add_argument('--use-softmax', action='store_true',
                        help='Apply softmax to class scores before thresholding.')
    parser.add_argument('--show', action='store_true',
                        help='Show the matplotlib window.')
    parser.add_argument('--fov-x', type=float, nargs=2, default=None,
                        help='Override x-axis FOV (min max).')
    parser.add_argument('--fov-y', type=float, nargs=2, default=None,
                        help='Override y-axis FOV (min max).')
    parser.add_argument('--fov-z', type=float, nargs=2, default=None,
                        help='Override z-axis FOV (min max).')
    args = parser.parse_args()

    config = load_config(args.cfg)
    device = _select_device(args.device, config['computing']['device'])

    dataset = init_dataset(dataset=config['dataset'], src=args.src, split=args.split, config=config)
    if args.index < 0 or args.index >= len(dataset):
        raise IndexError(f"Index {args.index} is out of range (0-{len(dataset) - 1}).")

    sample, label = dataset[args.index]
    sample_path = _get_sample_path(dataset, args.index)
    sample_dir = _resolve_sample_dir(sample_path)
    output_dir = _build_output_dir(args, sample_dir, args.split, args.index)

    os.makedirs(output_dir, exist_ok=True)
    print(f"📌 当前使用 split: {args.split}, index: {args.index}")
    if sample_dir:
        print(f"📂 样本目录: {sample_dir}")
    camera_path = sample_path.get('camera_mono')
    if camera_path:
        print(f"🖼️ 标注原图: {camera_path}")
    _write_sample_info(output_dir, args.split, args.index, sample_dir, sample_path)

    batch = _to_device(_add_batch_dim(sample), device)

    model, _, _ = load_model(args.checkpoint)
    model.to(device)

    fov = _get_fov(args, config)
    visualize_sample(
        model=model,
        sample=batch,
        raw_sample=sample,
        label=label,
        conf_thr=args.conf,
        use_softmax=args.use_softmax,
        output_dir=output_dir,
        show_gt=not args.no_gt,
        show=args.show,
        fov=fov,
    )


if __name__ == '__main__':
    main()

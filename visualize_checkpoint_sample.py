"""
根据训练得到的 .pt 模型文件，对单个样本进行可视化（BEV 俯视图）。

示例:
  实际数据示例: /root/autodl-tmp/autodl-tmp/data/kradar/test/1/00182_00150/mono.jpg
  python /root/autodl-tmp/autodl-tmp/DPFT-main/visualize_checkpoint_sample.py \
    --src /root/autodl-tmp/autodl-tmp/data/kradar \
    --cfg /root/autodl-tmp/autodl-tmp/DPFT-main/config/kradar.json \
    --checkpoint /root/autodl-tmp/autodl-tmp/DPFT-main/result/20251126-235801-585/checkpoints/20251126-235801-585_checkpoint_0199.pt \
    --index 0 \
    --output /root/autodl-tmp/autodl-tmp/DPFT-main/resultjsample_000000.png
"""

import argparse
import os
import sys
from typing import Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from dprt.datasets import init as init_dataset
from dprt.models import load as load_model
from dprt.utils.config import load_config
from dprt.utils.geometry import get_box_corners


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


def visualize_sample(
    model: torch.nn.Module,
    sample: dict,
    label: dict,
    conf_thr: float,
    use_softmax: bool,
    output_path: str,
    show_gt: bool,
    show: bool,
    fov: Tuple[Tuple[float, float], Tuple[float, float], Tuple[float, float]],
) -> None:
    model.eval()
    with torch.no_grad():
        output = model(sample)

    pred_boxes = _decode_predictions(output, conf_thr, use_softmax, fov)
    gt_boxes = _decode_ground_truth(label) if show_gt else np.empty((0, 8), dtype=np.float32)

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.set_title('BEV Visualization (Prediction vs GT)')
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

    os.makedirs(os.path.dirname(output_path) or '.', exist_ok=True)
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    print(f"✅ 已保存可视化结果: {output_path}")
    print(f"预测框数量: {len(pred_boxes)}")
    if show_gt:
        print(f"真实框数量: {len(gt_boxes)}")

    if show:
        plt.show()
    else:
        plt.close(fig)


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
                        help='Dataset split to use (train/test).')
    parser.add_argument('--index', type=int, default=0,
                        help='Sample index in the split.')
    parser.add_argument('--conf', type=float, default=0.5,
                        help='Confidence threshold for predictions.')
    parser.add_argument('--output', type=str, default=None,
                        help='Output PNG file path.')
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
    batch = _to_device(_add_batch_dim(sample), device)

    model, _, _ = load_model(args.checkpoint)
    model.to(device)

    output_path = args.output or os.path.join(
        'visualizations', f"{args.split}_sample_{args.index:06d}.png"
    )

    fov = _get_fov(args, config)
    visualize_sample(
        model=model,
        sample=batch,
        label=label,
        conf_thr=args.conf,
        use_softmax=args.use_softmax,
        output_path=output_path,
        show_gt=not args.no_gt,
        show=args.show,
        fov=fov,
    )


if __name__ == '__main__':
    main()

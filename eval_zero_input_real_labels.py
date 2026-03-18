"""
使用全零输入但保留真实标签来评估 DPRT 模型

用途：测试模型在极端输入（全零）下的漏检率
预期结果：mAP 应该接近 0（因为全零输入无法检测到任何物体）

使用方法:
    python eval_zero_input_real_labels.py \
        --src /path/to/kradar/processed \
        --checkpoint /path/to/model.pth \
        --cfg config/kradar_camera_mono.json
"""

import argparse
from typing import Dict, Tuple

import torch
from torch.utils.data import Dataset

from dprt.datasets import init, load
from dprt.evaluation import evaluate
from dprt.utils.config import load_config
from dprt.utils.misc import set_seed


class ZeroInputRealLabelsDataset(Dataset):
    """
    全零输入 + 真实标签的包装数据集

    包装真实的 KRadarDataset，将输入数据替换为全零，但保留真实标签
    """

    def __init__(self, real_dataset):
        """
        Arguments:
            real_dataset: 真实的 KRadarDataset 实例
        """
        self.real_dataset = real_dataset
        self.dtype = real_dataset.dtype

    def __len__(self):
        return len(self.real_dataset)

    def __getitem__(self, index) -> Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]]:
        """
        获取一个样本：全零输入 + 真实标签

        Returns:
            data: 全零输入数据（保留元数据）
            label: 真实标签数据
        """
        # 获取真实数据和标签
        real_data, real_label = self.real_dataset[index]

        # 创建全零输入，但保留所有元数据
        zero_data = {}

        for key, value in real_data.items():
            if isinstance(value, torch.Tensor):
                # 判断是否是输入数据（需要置零）还是元数据（保留）
                if self._is_input_data(key):
                    # 将输入数据置零
                    zero_data[key] = torch.zeros_like(value)
                else:
                    # 保留元数据（投影矩阵、变换矩阵、形状等）
                    zero_data[key] = value.clone()
            else:
                # 非 tensor 数据直接复制
                zero_data[key] = value

        # 返回全零输入和真实标签
        return zero_data, real_label

    def _is_input_data(self, key: str) -> bool:
        """
        判断是否是输入数据（需要置零）

        Arguments:
            key: 数据字典的键名

        Returns:
            True 如果是输入数据，False 如果是元数据
        """
        # 输入数据的关键词
        input_keywords = [
            'camera_mono',
            'camera_stereo',
            'radar_bev',
            'radar_front',
            'lidar'
        ]

        # 元数据的关键词（不应该置零）
        metadata_keywords = [
            '_p',  # projection matrices
            '_t',  # transformation matrices
            '_shape',  # shape information
            '_intrinsics',
            '_extrinsics'
        ]

        # 如果包含元数据关键词，则不置零
        for meta_kw in metadata_keywords:
            if meta_kw in key:
                return False

        # 如果匹配输入数据关键词，则置零
        for input_kw in input_keywords:
            if input_kw == key:  # 精确匹配
                return True

        # 默认不置零（保守策略）
        return False


def main(src: str, cfg: str, checkpoint: str, dst: str):
    """
    使用全零输入和真实标签评估模型

    Arguments:
        src: 真实数据集路径
        cfg: 配置文件路径
        checkpoint: 模型checkpoint路径
        dst: 保存评估结果的目录
    """
    print("="*70)
    print("全零输入 + 真实标签评估")
    print("="*70)

    # 加载配置
    print(f"\n📁 加载配置: {cfg}")
    config = load_config(cfg)

    # 设置随机种子
    set_seed(config['computing']['seed'])
    print(f"🎲 设置随机种子: {config['computing']['seed']}")

    # 初始化真实数据集
    print(f"\n📦 加载真实数据集: {src}")
    real_dataset = init(dataset=config['dataset'], src=src, split='test', config=config)
    print(f"   数据集大小: {len(real_dataset)} 个样本")

    # 创建全零输入 + 真实标签的包装数据集
    print("\n🔄 创建全零输入包装器...")
    zero_input_dataset = ZeroInputRealLabelsDataset(real_dataset)

    # 加载数据
    print("📚 创建数据加载器...")
    test_loader = load(zero_input_dataset, config=config)
    print(f"   Batch size: {config['train']['batch_size']}")
    print(f"   批次数: {len(test_loader)}")

    # 评估模型
    print(f"\n🚀 开始评估...")
    print(f"   Checkpoint: {checkpoint}")
    print(f"   输出目录: {dst}")
    print("\n" + "="*70)

    evaluator = evaluate(config)
    evaluator(checkpoint, test_loader, dst)

    print("\n" + "="*70)
    print("✅ 评估完成!")
    print("="*70)
    print("\n预期结果：")
    print("  - mAP 应该接近 0（全零输入无法检测到物体）")
    print("  - 这测试了模型在极端输入下的鲁棒性")
    print(f"\n结果已保存到: {dst}")
    print("="*70)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='DPRT evaluation with zero input and real labels',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument(
        '--src',
        type=str,
        required=True,
        help='Path to the processed dataset folder (for labels)'
    )

    parser.add_argument(
        '--checkpoint',
        type=str,
        required=True,
        help='Path to the model checkpoint (.pth file)'
    )

    parser.add_argument(
        '--cfg',
        type=str,
        default='config/kradar_camera_mono.json',
        help='Path to the configuration file'
    )

    parser.add_argument(
        '--dst',
        type=str,
        default='./log',
        help='Path to save the evaluation results'
    )

    args = parser.parse_args()

    main(
        src=args.src,
        cfg=args.cfg,
        checkpoint=args.checkpoint,
        dst=args.dst
    )

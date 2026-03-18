#!/usr/bin/env python
"""
知识蒸馏训练示例脚本
Example script for knowledge distillation training

使用方法:
python distillation_train_example.py --teacher path/to/teacher.pt --config config/kradar.json

Requirements:
- 已训练好的教师模型checkpoint
- 标准的训练配置文件
"""

import argparse
import torch
from pathlib import Path

from dprt.utils.config import load_config, save_config
from dprt.models.dprt import build_dprt
from dprt.datasets.kradar import init_dataset, load_dataset
from dprt.training.loss import DistillationLoss, KDLoss, Loss
from dprt.training.trainer import KnowledgeDistillationTrainer, build_trainer
from dprt.training.optimizer import build_optimizer
from dprt.training.scheduler import build_scheduler
from dprt.evaluation.metric import build_metric


def create_distillation_config(base_config, teacher_checkpoint, temperature=4.0, alpha=0.5):
    """
    从基础配置创建蒸馏训练配置

    Args:
        base_config: 基础配置字典
        teacher_checkpoint: 教师模型checkpoint路径
        temperature: 蒸馏温度参数
        alpha: 蒸馏损失权重

    Returns:
        添加了蒸馏配置的新配置字典
    """
    # 深拷贝配置
    import copy
    distill_config = copy.deepcopy(base_config)

    # 添加蒸馏配置
    distill_config['train']['distillation'] = {
        'teacher_checkpoint': str(teacher_checkpoint),
        'freeze_teacher': True,
        'temperature': temperature,
        'alpha': alpha
    }

    return distill_config


def create_student_model_config(teacher_config, compression_level='medium'):
    """
    基于教师模型配置创建学生模型配置（更轻量）

    Args:
        teacher_config: 教师模型配置
        compression_level: 压缩级别 ('light', 'medium', 'heavy')

    Returns:
        学生模型配置
    """
    import copy
    student_config = copy.deepcopy(teacher_config)

    # 定义压缩策略
    compression_map = {
        'light': {
            'ResNet101': 'ResNet101',  # 保持不变
            'ResNet50': 'ResNet50',
            'n_iterations': lambda x: x,
            'num_queries': lambda x: x,
        },
        'medium': {
            'ResNet101': 'ResNet50',   # 降低一级
            'ResNet50': 'ResNet34',
            'n_iterations': lambda x: max(2, x - 2),
            'num_queries': lambda x: int(x * 0.7),
        },
        'heavy': {
            'ResNet101': 'ResNet34',   # 降低两级
            'ResNet50': 'ResNet18',
            'n_iterations': lambda x: max(1, x - 3),
            'num_queries': lambda x: int(x * 0.5),
        }
    }

    strategy = compression_map.get(compression_level, compression_map['medium'])

    # 修改backbone
    for key, backbone in student_config['model']['backbones'].items():
        if backbone['name'] in strategy:
            student_config['model']['backbones'][key]['name'] = strategy[backbone['name']]

    # 修改fuser iterations
    if 'n_iterations' in student_config['model']['fuser']:
        old_iters = student_config['model']['fuser']['n_iterations']
        student_config['model']['fuser']['n_iterations'] = strategy['n_iterations'](old_iters)

    # 修改queries数量
    if 'num_queries' in student_config['model']['querent']:
        old_queries = student_config['model']['querent']['num_queries']
        student_config['model']['querent']['num_queries'] = strategy['num_queries'](old_queries)

    return student_config


def main():
    parser = argparse.ArgumentParser(description='Knowledge Distillation Training for DPFT')
    parser.add_argument('--teacher', type=str, required=True,
                        help='Path to teacher model checkpoint (.pt file)')
    parser.add_argument('--config', type=str, required=True,
                        help='Path to base training config (.json file)')
    parser.add_argument('--src', type=str, required=True,
                        help='Path to dataset directory')
    parser.add_argument('--dst', type=str, default='outputs/distillation',
                        help='Output directory for checkpoints and logs')
    parser.add_argument('--temperature', type=float, default=4.0,
                        help='Distillation temperature (default: 4.0)')
    parser.add_argument('--alpha', type=float, default=0.5,
                        help='Distillation loss weight (default: 0.5)')
    parser.add_argument('--compression', type=str, default='medium',
                        choices=['light', 'medium', 'heavy'],
                        help='Student model compression level (default: medium)')
    parser.add_argument('--save-config', type=str, default=None,
                        help='Save generated distillation config to file')

    args = parser.parse_args()

    # 验证教师模型存在
    teacher_path = Path(args.teacher)
    if not teacher_path.exists():
        raise FileNotFoundError(f"Teacher model not found: {args.teacher}")

    print("=" * 60)
    print("Knowledge Distillation Training Setup")
    print("=" * 60)

    # 1. 加载基础配置
    print(f"\n[1/6] Loading base config from: {args.config}")
    config = load_config(args.config)

    # 2. 创建蒸馏配置
    print(f"[2/6] Creating distillation config...")
    print(f"  - Teacher checkpoint: {args.teacher}")
    print(f"  - Temperature: {args.temperature}")
    print(f"  - Alpha (distillation weight): {args.alpha}")

    distill_config = create_distillation_config(
        config,
        teacher_checkpoint=args.teacher,
        temperature=args.temperature,
        alpha=args.alpha
    )

    # 3. 创建学生模型配置
    print(f"[3/6] Creating student model config (compression: {args.compression})...")
    distill_config['model'] = create_student_model_config(
        distill_config,
        compression_level=args.compression
    )

    # 打印模型对比
    print("\n  Model Architecture Comparison:")
    print("  " + "-" * 56)
    print(f"  {'Component':<20} {'Teacher':<15} {'Student':<15}")
    print("  " + "-" * 56)

    # 比较backbone
    for key in config['model']['backbones'].keys():
        teacher_backbone = config['model']['backbones'][key]['name']
        student_backbone = distill_config['model']['backbones'][key]['name']
        print(f"  {key:<20} {teacher_backbone:<15} {student_backbone:<15}")

    # 比较其他参数
    if 'fuser' in config['model']:
        teacher_iters = config['model']['fuser'].get('n_iterations', 'N/A')
        student_iters = distill_config['model']['fuser'].get('n_iterations', 'N/A')
        print(f"  {'Fuser iterations':<20} {str(teacher_iters):<15} {str(student_iters):<15}")

    if 'querent' in config['model']:
        teacher_queries = config['model']['querent'].get('num_queries', 'N/A')
        student_queries = distill_config['model']['querent'].get('num_queries', 'N/A')
        print(f"  {'Num queries':<20} {str(teacher_queries):<15} {str(student_queries):<15}")

    print("  " + "-" * 56)

    # 保存配置（如果指定）
    if args.save_config:
        save_config(distill_config, args.save_config)
        print(f"\n  Config saved to: {args.save_config}")

    # 4. 初始化数据集
    print(f"\n[4/6] Initializing datasets from: {args.src}")
    train_dataset = init_dataset(distill_config, args.src, split='train')
    val_dataset = init_dataset(distill_config, args.src, split='val')

    train_loader = load_dataset(train_dataset, distill_config, split='train')
    val_loader = load_dataset(val_dataset, distill_config, split='val')

    print(f"  - Training samples: {len(train_dataset)}")
    print(f"  - Validation samples: {len(val_dataset)}")
    print(f"  - Batch size: {distill_config['train']['batch_size']}")

    # 5. 构建学生模型
    print(f"\n[5/6] Building student model...")
    student_model = build_dprt(distill_config['model'])

    # 计算参数量
    total_params = sum(p.numel() for p in student_model.parameters())
    trainable_params = sum(p.numel() for p in student_model.parameters() if p.requires_grad)
    print(f"  - Total parameters: {total_params:,}")
    print(f"  - Trainable parameters: {trainable_params:,}")

    # 6. 构建蒸馏训练器
    print(f"\n[6/6] Building distillation trainer...")
    trainer = build_trainer(distill_config)

    print(f"  - Training epochs: {distill_config['train']['epochs']}")
    print(f"  - Optimizer: {distill_config['train']['optimizer']['name']}")
    print(f"  - Learning rate: {distill_config['train']['optimizer']['lr']}")
    print(f"  - Device: {distill_config['computing']['device']}")

    # 开始训练
    print("\n" + "=" * 60)
    print("Starting Distillation Training")
    print("=" * 60 + "\n")

    trainer.train(
        model=student_model,
        data_loader=train_loader,
        val_loader=val_loader,
        dst=args.dst
    )

    print("\n" + "=" * 60)
    print("Training Completed!")
    print("=" * 60)
    print(f"\nCheckpoints saved to: {args.dst}")


if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nTraining interrupted by user.")
    except Exception as e:
        print(f"\n\nError: {e}")
        raise

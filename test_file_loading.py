"""
测试文件缺失时的行为
"""
import os
from pathlib import Path
from dprt.datasets import init
from dprt.utils.config import load_config

# 加载配置
config = load_config('config/kradar_radar.json')

# 修改为你的数据路径
data_path = Path('你的数据路径/processed/test')  # 需要修改！

# 找到第一个样本目录
for seq_dir in data_path.iterdir():
    if seq_dir.is_dir():
        for sample_dir in seq_dir.iterdir():
            if sample_dir.is_dir():
                print(f"找到样本目录: {sample_dir}")

                # 检查文件存在性
                ra_file = sample_dir / "ra.npy"
                ea_file = sample_dir / "ea.npy"
                ra_unable = sample_dir / "ra_unable.npy"
                ea_unable = sample_dir / "ea_unable.npy"

                print(f"\n文件状态：")
                print(f"  ra.npy 存在: {ra_file.exists()}")
                print(f"  ea.npy 存在: {ea_file.exists()}")
                print(f"  ra_unable.npy 存在: {ra_unable.exists()}")
                print(f"  ea_unable.npy 存在: {ea_unable.exists()}")

                # 如果文件被重命名，尝试加载数据集会发生什么？
                if not ra_file.exists():
                    print(f"\n⚠️  雷达文件已被重命名！尝试加载数据集...")
                    try:
                        dataset = init(
                            dataset='kradar',
                            src=str(data_path.parent),
                            split='test',
                            config=config
                        )
                        data, label = dataset[0]
                        print("✓ 数据集加载成功（这很奇怪！）")
                    except Exception as e:
                        print(f"✗ 数据集加载失败（符合预期）: {type(e).__name__}: {e}")

                break
        break

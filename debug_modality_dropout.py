"""
调试脚本：验证模态dropout是否生效
"""
import torch
from dprt.datasets import init
from dprt.datasets import load
from dprt.utils.config import load_config

# 加载配置
config = load_config('config/kradar_radar.json')

# 创建两个数据集
# 1. 正常数据集
normal_dataset = init(
    dataset='kradar',
    src='你的数据路径/processed',  # 需要替换
    split='test',
    config=config,
    radar_dropout=0.0  # 不dropout
)

# 2. dropout数据集
dropout_dataset = init(
    dataset='kradar',
    src='你的数据路径/processed',  # 需要替换
    split='test',
    config=config,
    radar_dropout=1.0  # 完全dropout雷达
)

# 获取第一个样本
normal_data, normal_label = normal_dataset[0]
dropout_data, dropout_label = dropout_dataset[0]

# 检查雷达数据
print("=" * 50)
print("正常数据集 - radar_bev 统计:")
if 'radar_bev' in normal_data:
    print(f"  Mean: {normal_data['radar_bev'].mean():.4f}")
    print(f"  Std: {normal_data['radar_bev'].std():.4f}")
    print(f"  All zeros: {torch.all(normal_data['radar_bev'] == 0)}")
else:
    print("  键不存在!")

print("\nDropout数据集 - radar_bev 统计:")
if 'radar_bev' in dropout_data:
    print(f"  Mean: {dropout_data['radar_bev'].mean():.4f}")
    print(f"  Std: {dropout_data['radar_bev'].std():.4f}")
    print(f"  All zeros: {torch.all(dropout_data['radar_bev'] == 0)}")
else:
    print("  键不存在!")

print("\n正常数据集 - radar_front 统计:")
if 'radar_front' in normal_data:
    print(f"  Mean: {normal_data['radar_front'].mean():.4f}")
    print(f"  Std: {normal_data['radar_front'].std():.4f}")
    print(f"  All zeros: {torch.all(normal_data['radar_front'] == 0)}")
else:
    print("  键不存在!")

print("\nDropout数据集 - radar_front 统计:")
if 'radar_front' in dropout_data:
    print(f"  Mean: {dropout_data['radar_front'].mean():.4f}")
    print(f"  Std: {dropout_data['radar_front'].std():.4f}")
    print(f"  All zeros: {torch.all(dropout_data['radar_front'] == 0)}")
else:
    print("  键不存在!")

print("=" * 50)

# 验证 querent 输出
from dprt.models import build_dprt

model = build_dprt.from_config(config)
model.eval()

# 正常输入的查询点
query_normal = model.querent(normal_data)
print("\n正常数据的查询点:")
print(f"  Shape: {query_normal['center'].shape}")
print(f"  First 3: {query_normal['center'][0, :3]}")

# dropout输入的查询点
query_dropout = model.querent(dropout_data)
print("\nDropout数据的查询点:")
print(f"  Shape: {query_dropout['center'].shape}")
print(f"  First 3: {query_dropout['center'][0, :3]}")

print(f"\n查询点是否完全相同: {torch.equal(query_normal['center'], query_dropout['center'])}")
print("=" * 50)

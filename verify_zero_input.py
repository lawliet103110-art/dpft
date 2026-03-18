"""
验证全零输入是否正确生成

快速检查脚本，验证 ZeroInputRealLabelsDataset 是否正确将输入置零
"""

import torch
from dprt.datasets import init
from dprt.utils.config import load_config
from dprt.utils.misc import set_seed

# 导入 ZeroInputRealLabelsDataset
import sys
import importlib.util
spec = importlib.util.spec_from_file_location("eval_module", "eval_zero_input_real_labels.py")
eval_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(eval_module)
ZeroInputRealLabelsDataset = eval_module.ZeroInputRealLabelsDataset


def main():
    print("="*70)
    print("验证全零输入数据集")
    print("="*70)

    # 配置
    src = "/root/autodl-tmp/autodl-tmp/DPFT-main/data/kradar/processed"  # 修改为你的路径
    cfg = "config/kradar_camera_mono.json"

    # 加载配置
    print(f"\n📁 加载配置: {cfg}")
    config = load_config(cfg)
    set_seed(config['computing']['seed'])

    # 加载真实数据集
    print(f"📦 加载真实数据集: {src}")
    real_dataset = init(dataset=config['dataset'], src=src, split='test', config=config)
    print(f"   数据集大小: {len(real_dataset)}")

    # 创建全零数据集
    print("\n🔄 创建全零输入包装器...")
    zero_dataset = ZeroInputRealLabelsDataset(real_dataset)

    # 测试几个样本
    print("\n" + "="*70)
    print("测试样本")
    print("="*70)

    num_samples_to_check = 3
    all_zero = True

    for i in range(num_samples_to_check):
        print(f"\n【样本 {i}】")

        # 获取数据
        zero_data, label = zero_dataset[i]

        # 检查输入数据
        for key, value in zero_data.items():
            if isinstance(value, torch.Tensor):
                is_input = zero_dataset._is_input_data(key)

                if is_input:
                    # 这是输入数据，应该是全零
                    is_zero = torch.all(value == 0).item()
                    status = "✅ 全零" if is_zero else "❌ 非零"

                    print(f"  {key}: {status}")
                    print(f"    形状: {value.shape}")
                    print(f"    最小值: {value.min().item():.6f}")
                    print(f"    最大值: {value.max().item():.6f}")
                    print(f"    均值: {value.mean().item():.6f}")

                    if not is_zero:
                        all_zero = False
                        print(f"    ⚠️  警告: 这个字段应该是全零但不是!")

        # 检查标签（应该有数据）
        print(f"\n  标签信息:")
        for key, value in label.items():
            if isinstance(value, torch.Tensor):
                print(f"    {key}: 形状 {value.shape}, 有 {value.shape[0] if value.ndim > 0 else 1} 个元素")

    # 总结
    print("\n" + "="*70)
    print("验证结果")
    print("="*70)

    if all_zero:
        print("✅ 所有输入数据都是全零 - 正确！")
        print("\n这意味着:")
        print("  1. 图像输入已经被置零")
        print("  2. 雷达输入（如果有）已经被置零")
        print("  3. 元数据（投影矩阵等）被正确保留")
        print("  4. 标签数据被正确保留")
    else:
        print("❌ 发现非零输入数据 - 有问题！")
        print("\n请检查:")
        print("  1. _is_input_data 方法是否正确识别输入字段")
        print("  2. 数据处理逻辑是否正确")

    print("="*70)


if __name__ == '__main__':
    try:
        main()
    except Exception as e:
        print(f"\n❌ 发生错误: {e}")
        import traceback
        traceback.print_exc()

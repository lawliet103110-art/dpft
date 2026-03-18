"""
检查模型预测结果的脚本

用于验证模型是否真的没有预测出任何框
"""

import torch
from dprt.models import load as load_model
from dprt.utils.config import load_config
from dprt.utils.misc import set_seed

# 配置
checkpoint_path = "/path/to/your/checkpoint.pt"
config_path = "config/kradar_camera_mono.json"

# 加载配置
config = load_config(config_path)
set_seed(config['computing']['seed'])

# 加载模型
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model, epoch, timestamp = load_model(checkpoint_path)
model.to(device)
model.eval()

# 创建全零输入
image_size = config['data']['image_size']
zero_input = {
    'camera_mono': torch.zeros(1, image_size, image_size, 3).to(device)
}

# 推理
with torch.no_grad():
    output = model(zero_input)

# 打印输出
print("="*70)
print("模型输出分析")
print("="*70)

for key, value in output.items():
    if isinstance(value, torch.Tensor):
        print(f"\n{key}:")
        print(f"  形状: {value.shape}")
        print(f"  最小值: {value.min().item():.6f}")
        print(f"  最大值: {value.max().item():.6f}")
        print(f"  均值: {value.mean().item():.6f}")

        if key == 'class':
            # 检查类别预测
            probs = torch.softmax(value, dim=-1)
            max_probs, max_classes = torch.max(probs, dim=-1)
            print(f"  最大概率: {max_probs.max().item():.6f}")
            print(f"  最大概率对应的类别: {max_classes[max_probs.argmax()]}")

            # 统计每个类别的最大概率
            for c in range(probs.shape[-1]):
                class_max = probs[..., c].max().item()
                print(f"  类别 {c} 最大概率: {class_max:.6f}")

print("\n" + "="*70)
print("结论：")
print("="*70)

# 分析
if 'class' in output:
    probs = torch.softmax(output['class'], dim=-1)
    max_prob = probs.max().item()

    if max_prob < 0.5:  # 假设置信度阈值是 0.5
        print("✅ 模型没有预测出高置信度的框")
        print(f"   最大置信度: {max_prob:.6f} < 0.5")
        print("   这就是为什么 mAP = 1.0 的原因！")
        print("   （空真值 + 空预测 = 完美匹配）")
    else:
        print("⚠️  模型预测出了高置信度的框")
        print(f"   最大置信度: {max_prob:.6f} >= 0.5")
        print("   这种情况下 mAP 不应该是 1.0")
        print("   可能存在其他问题")

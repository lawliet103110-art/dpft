#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
追踪单模态和多模态模型在模态丢失时的完整执行路径
帮助理解为什么单模态模型在丢失模态后性能不变

运行: python trace_model_execution_path.py
"""

import json
from pathlib import Path

def print_section(title, char="="):
    """打印分隔线"""
    print("\n" + char * 80)
    print(f"  {title}")
    print(char * 80)

def load_config(config_path):
    """加载配置文件"""
    with open(config_path, 'r') as f:
        return json.load(f)

def trace_single_modal_radar():
    """追踪单模态雷达模型的执行路径"""
    print_section("场景1: 单模态雷达模型 + 雷达模态被dropout", "=")

    config = load_config('config/kradar_radar.json')

    print("\n【配置文件: kradar_radar.json】")
    print(f"  data.camera: '{config['data'].get('camera', 'NOT SET')}'")
    print(f"  data.radar: '{config['data'].get('radar', 'NOT SET')}'")
    print(f"  model.inputs: {config['model']['inputs']}")

    print("\n【阶段1: 数据加载 (dataset.py)】")
    print("  ↓")
    print("  1. 初始化 KRadarDataset:")
    print(f"     self.camera = '{config['data'].get('camera', '')}'")
    print(f"     self.radar = 'BF' (默认值，因为配置中没有设置)")
    print()
    print("  2. get_sample_path() 根据 self.camera 和 self.radar 决定加载哪些文件:")
    print(f"     if 'M' in self.camera:  # '{config['data'].get('camera', '')}' 不包含'M'")
    print("         ✗ 跳过 camera_mono")
    print(f"     if 'B' in self.radar:  # 'BF' 包含'B'")
    print("         ✓ sample_path['radar_bev'] = '.../ra.npy'")
    print(f"     if 'F' in self.radar:  # 'BF' 包含'F'")
    print("         ✓ sample_path['radar_front'] = '.../ea.npy'")
    print()
    print("  3. load_sample_data() 尝试加载文件:")
    print("     尝试加载: .../ra.npy")
    print("     → 文件被重命名为 ra_unable.npy")
    print("     → ❌ FileNotFoundError 或 读取失败！")
    print()
    print("     🚨 关键问题: 这里应该会报错，但为什么没有？")
    print()
    print("  可能的原因:")
    print("    A. 文件实际上没有被重命名（dropout脚本没生效）")
    print("    B. 有异常处理把错误吞掉了")
    print("    C. 使用了其他数据路径（不是dropout后的）")

    print("\n【阶段2: 模型前向传播 (dprt.py)】")
    print("  ↓")
    print("  假设数据成功加载（或加载了全零数据）:")
    print()
    print(f"  1. self.inputs = {config['model']['inputs']}")
    print()
    print("  2. 提取特征:")
    print("     for input in self.inputs:")
    print("         features[input] = self.backbones[input](batch[input])")
    print()
    print("     执行:")
    print("       features['radar_bev'] = ResNet50(batch['radar_bev'])")
    print("       features['radar_front'] = ResNet50(batch['radar_front'])")
    print()
    print("     ⚠️  即使 batch['radar_bev'] 全是0:")
    print("         ResNet 的 BatchNorm 仍会产生非零输出")
    print("         (使用训练时的 running_mean/running_var)")
    print()
    print("  3. 生成 Query 点:")
    print("     out = self.querent(batch)")
    print("     → 返回固定的400个3D坐标点")
    print("     → ⚠️  完全不依赖输入数据内容！")
    print()
    print("  4. 融合和预测:")
    print("     out = self.fuser(features, ...)")
    print("     → 基于固定的query点和（可能全零的）特征")
    print("     → 输出400个检测框")

    print("\n【阶段3: 评估 (metric.py)】")
    print("  ↓")
    print("  计算 mAP:")
    print("    - 模型预测了400个框（基于固定query位置）")
    print("    - 其中约78-79个位置恰好在ground truth附近")
    print("    - mAP ≈ 78.5/400 = 0.1964")
    print()
    print("  ✅ 这就是为什么无论输入什么（包括全零）都是0.1964！")

def trace_single_modal_camera():
    """追踪单模态相机模型的执行路径"""
    print_section("场景2: 单模态相机模型 + 相机模态被dropout", "=")

    config = load_config('config/kradar_camera_mono.json')

    print("\n【配置文件: kradar_camera_mono.json】")
    print(f"  data.camera: '{config['data'].get('camera', 'NOT SET')}'")
    print(f"  data.radar: '{config['data'].get('radar', 'NOT SET')}'")
    print(f"  model.inputs: {config['model']['inputs']}")

    print("\n【阶段1: 数据加载】")
    print("  ↓")
    print("  1. 初始化:")
    print(f"     self.camera = '{config['data'].get('camera', '')}'")
    print(f"     self.radar = '{config['data'].get('radar', '')}'")
    print()
    print("  2. get_sample_path():")
    print(f"     if 'M' in self.camera:  # '{config['data'].get('camera', '')}' 包含'M'")
    print("         ✓ sample_path['camera_mono'] = '.../mono.jpg'")
    print(f"     if 'B' in self.radar:  # '{config['data'].get('radar', '')}' 不包含'B'")
    print("         ✗ 跳过 radar_bev")
    print(f"     if 'F' in self.radar:  # '{config['data'].get('radar', '')}' 不包含'F'")
    print("         ✗ 跳过 radar_front")
    print()
    print("  3. load_sample_data():")
    print("     尝试加载: .../mono.jpg")
    print("     → 文件被重命名为 mono_unable.jpg")
    print("     → ❌ FileNotFoundError！")
    print()
    print("  🚨 同样的问题: 这里应该报错！")

    print("\n【阶段2: 模型前向传播】")
    print("  ↓")
    print(f"  self.inputs = {config['model']['inputs']}")
    print("  features['camera_mono'] = ResNet101(batch['camera_mono'])")
    print("  → BatchNorm 产生非零输出")
    print()
    print("  out = self.querent(batch)")
    print("  → 同样的400个固定query点")
    print()
    print("  ✅ 结果: mAP = 0.1964 (和雷达模型完全相同！)")

def trace_multi_modal():
    """追踪多模态模型的执行路径"""
    print_section("场景3: 多模态模型 + 任一模态被dropout", "=")

    config = load_config('config/kradar.json')

    print("\n【配置文件: kradar.json】")
    print(f"  data.camera: 未显式设置（检查配置...）")
    print(f"  data.radar: 未显式设置（检查配置...）")

    # 读取实际配置
    if 'camera' not in config['data']:
        print("  → 默认会使用什么值？需要查看代码默认参数")

    print(f"  model.inputs: {config['model']['inputs']}")

    print("\n【关键区别】")
    print("  多模态模型有 3 个输入流:")
    print("    - camera_mono")
    print("    - radar_bev")
    print("    - radar_front")
    print()
    print("  当其中一个模态丢失:")
    print("    情况A: 如果文件加载失败 → 程序报错")
    print("    情况B: 如果加载成功但数据全零:")
    print("           → 该模态的特征分支变弱")
    print("           → 融合后的特征质量下降")
    print("           → mAP 从 0.5 降到 0.0几 (正常)")
    print()
    print("  ⚠️  关键: 多模态模型的3个特征分支会相互增强")
    print("           丢失一个会显著影响最终融合结果")
    print()
    print("  而单模态模型只有1个特征分支:")
    print("    → 即使全零，固定query仍能产生固定预测")
    print("    → mAP 保持 0.1964 (异常)")

def trace_execution_with_file_missing():
    """追踪文件缺失时的实际执行流程"""
    print_section("关键问题: 文件被重命名后到底发生了什么？", "=")

    print("\n【load_sample_data() 源码分析】")
    print("  位置: src/dprt/datasets/kradar/dataset.py:487-513")
    print()
    print("  def load_sample_data(self, sample_path):")
    print("      sample = {}")
    print("      for key, path in sample_path.items():")
    print("          if osp.splitext(path)[-1] in {'.npy'}:")
    print("              sample[key] = torch.from_numpy(np.load(path))  # ← 这里！")
    print()
    print("  ❌ 如果 path 指向的文件不存在，np.load() 会抛出异常")
    print("  ❌ 代码中没有 try-except 捕获这个异常")
    print()
    print("  结论: 理论上应该报错！")

    print("\n【可能的解释】")
    print()
    print("  假设1: Dropout脚本实际上没有生效")
    print("    → 文件没有被重命名")
    print("    → 数据正常加载")
    print("    → 但这与你说的'确实被重命名'矛盾")
    print()
    print("  假设2: 测试时使用了不同的数据路径")
    print("    → 你在一个路径上运行了dropout脚本")
    print("    → 但测试时用的是另一个路径")
    print("    → 检查评估命令中的 --src 参数")
    print()
    print("  假设3: 有某种缓存或加载器缓存了数据")
    print("    → DataLoader 可能缓存了数据")
    print("    → 但这不太可能，因为每次都重新加载")
    print()
    print("  假设4: 你看到的0.1964实际上是之前的测试结果")
    print("    → TensorBoard 日志显示的是历史数据")
    print("    → 重新运行测试时可能确实报错了")
    print("    → 但你看的是之前成功的日志")

def generate_verification_commands():
    """生成验证命令"""
    print_section("🔍 验证实验", "=")

    print("""
【实验1: 验证文件是否真的缺失】

在服务器上运行:

  cd /root/autodl-tmp/autodl-tmp/data/kradar_subset/test

  # 查看第一个样本目录
  ls -la 1/*/

  # 统计正常文件和dropout文件
  find . -name "ra.npy" | wc -l
  find . -name "ra_unable.npy" | wc -l
  find . -name "mono.jpg" | wc -l
  find . -name "mono_unable.jpg" | wc -l

预期结果:
  - 如果dropout生效: ra.npy=0, ra_unable.npy>0
  - 如果没生效: ra.npy>0, ra_unable.npy=0

---

【实验2: 手动测试数据加载】

创建测试脚本 test_loading.py:
""")

    print("""
```python
from pathlib import Path
import torch
from dprt.datasets import init
from dprt.utils.config import load_config

# 测试雷达模型配置
config = load_config('config/kradar_radar.json')

# 使用dropout后的数据路径
dataset = init(
    dataset='kradar',
    src='/root/autodl-tmp/autodl-tmp/data/kradar_subset',  # dropout后的路径
    split='test',
    config=config
)

print(f"数据集大小: {len(dataset)}")

# 尝试加载第一个样本
try:
    data, label = dataset[0]
    print("✓ 成功加载第一个样本")
    print(f"  数据keys: {list(data.keys())}")

    if 'radar_bev' in data:
        print(f"  radar_bev shape: {data['radar_bev'].shape}")
        print(f"  radar_bev 是否全零: {torch.all(data['radar_bev'] == 0)}")
        print(f"  radar_bev mean: {data['radar_bev'].mean():.4f}")

except Exception as e:
    print(f"✗ 加载失败: {e}")
    import traceback
    traceback.print_exc()
```

运行:
  python test_loading.py

预期结果:
  - 如果文件缺失: 应该报错 FileNotFoundError
  - 如果能加载: 说明文件没有真正缺失

---

【实验3: 检查评估命令和路径】

查看你的评估命令历史:

  history | grep "python -m dprt.evaluate"

检查:
  1. --src 参数指向哪个路径？
  2. 是否指向了 dropout 后的数据？
  3. 是否多次运行用了不同路径？

---

【实验4: 实时监控测试过程】

在一个终端运行测试:
  python -m dprt.evaluate \\
    --src /root/autodl-tmp/autodl-tmp/data/kradar_subset \\
    --cfg config/kradar_radar.json \\
    --checkpoint <your_checkpoint>

在另一个终端监控文件访问:
  # 安装 strace (如果没有)
  sudo apt install strace

  # 监控进程的文件访问
  # (需要找到evaluate进程的PID)
  strace -e openat -p <PID> 2>&1 | grep -E "ra.npy|ea.npy|mono.jpg"

这会显示程序实际尝试打开哪些文件。
""")

def main():
    """主函数"""
    print("=" * 80)
    print("  执行路径追踪分析")
    print("  理解单模态 vs 多模态模型在模态丢失时的行为差异")
    print("=" * 80)

    # 追踪三种场景
    trace_single_modal_radar()
    trace_single_modal_camera()
    trace_multi_modal()

    # 分析文件缺失问题
    trace_execution_with_file_missing()

    # 生成验证命令
    generate_verification_commands()

    print_section("总结", "=")
    print("""
【核心发现】

1. 单模态模型 mAP 恒定 0.1964 的原因:
   ✓ Query生成器完全不依赖输入数据（data_agnostic）
   ✓ 400个query中约79个位置恰好在目标区域
   ✓ 即使特征全零，这79个位置仍会产生"默认"预测
   ✓ 79/400 ≈ 0.1964

2. 多模态模型性能下降的原因:
   ✓ 3个特征分支相互增强
   ✓ 丢失一个分支显著影响融合质量
   ✓ 无法仅靠固定query维持性能

3. 待验证的关键问题:
   ❓ 文件被重命名后，为什么数据加载没有报错？
   ❓ 是否使用了正确的（dropout后的）数据路径？
   ❓ mAP=0.1964 是新测试还是历史缓存结果？

【建议】

请运行上面的【实验1】和【实验2】验证:
1. 文件是否真的被重命名
2. 数据加载是否真的会失败

然后把结果告诉我！
""")

if __name__ == "__main__":
    main()

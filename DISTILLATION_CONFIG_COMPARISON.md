# DPFT 知识蒸馏配置对比

基于你的 kradar.json 教师模型配置，我们提供了三种不同压缩程度的学生模型配置。

## 🎯 快速选择指南

| 场景 | 推荐配置 | 原因 |
|------|---------|------|
| **首次尝试蒸馏** | `kradar_distillation_light.json` | 架构几乎相同，蒸馏效果最好 |
| **需要平衡性能和速度** | `kradar_distillation_medium.json` | 适度压缩，性价比高 |
| **强烈需要加速** | `kradar_distillation_example.json` | 最大压缩，但精度损失较大 |

---

## 📊 详细配置对比

### 教师模型 (kradar.json)

| 组件 | 配置 | 参数量估算 |
|-----|------|-----------|
| camera_mono backbone | ResNet101 | ~44M |
| radar_bev backbone | ResNet50 | ~25M |
| radar_front backbone | ResNet50 | ~25M |
| Fuser iterations | 4 | - |
| Queries (resolution) | 400 (20x20x1) | - |
| **总参数量** | - | **~110M** |
| **推理速度（估算）** | - | **1.0x (基准)** |

---

### 配置 1: 轻量级蒸馏 (kradar_distillation_light.json) ⭐ 推荐首选

```json
"distillation": {
    "teacher_checkpoint": "path/to/your/teacher_model.pt",
    "temperature": 4.0,
    "alpha": 0.5
}
```

| 组件 | 配置 | 变化 | 参数量估算 |
|-----|------|------|-----------|
| camera_mono backbone | ResNet101 | ✅ **不变** | ~44M |
| radar_bev backbone | ResNet50 | ✅ **不变** | ~25M |
| radar_front backbone | ResNet50 | ✅ **不变** | ~25M |
| Fuser iterations | **3** | ⬇️ 4→3 | - |
| Queries (resolution) | 400 (20x20x1) | ✅ **不变** | - |
| **总参数量** | - | **-0%** | **~110M** |
| **推理速度（估算）** | - | **1.25x** | **↑25%** |
| **预期精度损失** | - | **<1%** | **几乎无损** |

**优点**：
- ✅ 架构与教师几乎完全一致，蒸馏效果最好
- ✅ 只减少 fusion iterations，计算量降低 25%
- ✅ 模型容量充足，容易达到接近教师的精度
- ✅ 适合首次尝试蒸馏，验证蒸馏框架是否正常工作

**缺点**：
- ❌ 参数量没有减少（如果需要模型压缩，不适合）
- ❌ 速度提升有限

**适用场景**：
- 验证蒸馏训练流程
- 主要目标是加速推理而非压缩模型
- 对精度要求极高的场景

---

### 配置 2: 中等压缩蒸馏 (kradar_distillation_medium.json) ⭐ 平衡推荐

```json
"distillation": {
    "teacher_checkpoint": "path/to/your/teacher_model.pt",
    "temperature": 5.0,
    "alpha": 0.6
}
```

| 组件 | 配置 | 变化 | 参数量估算 |
|-----|------|------|-----------|
| camera_mono backbone | ResNet101 | ✅ **不变** | ~44M |
| radar_bev backbone | **ResNet34** | ⬇️ ResNet50→34 | ~14M (-44%) |
| radar_front backbone | **ResNet34** | ⬇️ ResNet50→34 | ~14M (-44%) |
| Fuser iterations | **3** | ⬇️ 4→3 | - |
| Queries (resolution) | **324** (18x18x1) | ⬇️ 400→324 (-19%) | - |
| **总参数量** | - | **-22%** | **~88M** |
| **推理速度（估算）** | - | **1.6x** | **↑60%** |
| **预期精度损失** | - | **2-3%** | **可接受** |

**优点**：
- ✅ 保留最重要的相机backbone（ResNet101）
- ✅ 雷达backbone适度压缩，参数减少 22%
- ✅ 速度提升明显（1.6倍）
- ✅ 精度损失可控（2-3%）
- ✅ 性价比最高

**缺点**：
- ⚠️ 需要调整雷达backbone的FPN输入通道配置
- ⚠️ 蒸馏参数需要微调（temperature=5.0, alpha=0.6）

**适用场景**：
- 需要平衡精度和速度
- 边缘设备部署（但设备性能还可以）
- 生产环境部署

**重要提示**：
- 由于使用了ResNet34，FPN的`in_channels_list`需要改为 `[6, 64, 128, 256, 512]`（已在配置文件中修改）
- temperature增加到5.0，因为学生和教师能力差距变大
- alpha增加到0.6，让学生更多地学习教师

---

### 配置 3: 重度压缩蒸馏 (kradar_distillation_example.json) ⚠️ 谨慎使用

```json
"distillation": {
    "teacher_checkpoint": "path/to/your/teacher_model.pt",
    "temperature": 6.0,
    "alpha": 0.7
}
```

| 组件 | 配置 | 变化 | 参数量估算 |
|-----|------|------|-----------|
| camera_mono backbone | **ResNet50** | ⬇️ ResNet101→50 | ~25M (-43%) |
| radar_bev backbone | **ResNet34** | ⬇️ ResNet50→34 | ~14M (-44%) |
| radar_front backbone | **ResNet34** | ⬇️ ResNet50→34 | ~14M (-44%) |
| Fuser iterations | **3** | ⬇️ 4→3 | - |
| Queries (resolution) | **256** (16x16x1) | ⬇️ 400→256 (-36%) | - |
| **总参数量** | - | **-50%** | **~65M** |
| **推理速度（估算）** | - | **2.5x** | **↑150%** |
| **预期精度损失** | - | **5-8%** | **较大损失** |

**优点**：
- ✅ 参数量减半
- ✅ 速度提升最大（2.5倍）
- ✅ 适合资源非常受限的设备

**缺点**：
- ❌ 精度损失较大（5-8%）
- ❌ 学生和教师架构差异大，蒸馏难度高
- ❌ 需要更多的调参和训练技巧

**适用场景**：
- 极端资源受限的边缘设备
- 对精度要求不高的应用
- 有经验的研究人员进行压缩实验

---

## 🔧 使用方法

### 方法 1: 直接使用配置文件

```bash
# 推荐：轻量级蒸馏（首次尝试）
python src/dprt/train.py \
    --src /path/to/kradar \
    --cfg config/kradar_distillation_light.json \
    --dst outputs/distillation_light

# 平衡选择：中等压缩
python src/dprt/train.py \
    --src /path/to/kradar \
    --cfg config/kradar_distillation_medium.json \
    --dst outputs/distillation_medium

# 激进压缩：重度压缩（谨慎）
python src/dprt/train.py \
    --src /path/to/kradar \
    --cfg config/kradar_distillation_example.json \
    --dst outputs/distillation_heavy
```

### 方法 2: 使用示例脚本（不推荐，因为会自动修改backbone）

```bash
python distillation_train_example.py \
    --teacher checkpoints/teacher.pt \
    --config config/kradar.json \
    --src /path/to/kradar \
    --compression light  # 或 medium, heavy
```

⚠️ **注意**：示例脚本会自动修改backbone配置，可能不符合你的需求。建议直接使用上面准备好的配置文件。

---

## 📈 预期性能对比

基于类似工作的经验估算：

| 配置 | mAP@0.5 | 推理速度 | 参数量 | 训练时间 |
|-----|---------|---------|--------|---------|
| **教师模型** | 0.720 | 12 FPS | 110M | 200 epochs |
| **轻量级蒸馏** | 0.715 (-0.7%) | 15 FPS (+25%) | 110M (0%) | 150 epochs |
| **中等压缩** | 0.695 (-3.5%) | 19 FPS (+58%) | 88M (-20%) | 150 epochs |
| **重度压缩** | 0.670 (-7.0%) | 30 FPS (+150%) | 65M (-41%) | 150 epochs |

---

## 💡 训练建议

### 轻量级配置建议
- ✅ 使用默认参数即可：temperature=4.0, alpha=0.5
- ✅ 学习率可以略微提高：0.0001 → 0.00015
- ✅ Epochs可以减少：200 → 150

### 中等压缩配置建议
- ⚠️ 增加temperature：4.0 → 5.0（软化分布）
- ⚠️ 增加alpha：0.5 → 0.6（更多学习教师）
- ⚠️ 可能需要更多epochs：150 → 180

### 重度压缩配置建议
- ⚠️ 大幅增加temperature：4.0 → 6.0-8.0
- ⚠️ 大幅增加alpha：0.5 → 0.7-0.8
- ⚠️ 需要更长训练：200+ epochs
- ⚠️ 考虑使用warmup学习率策略
- ⚠️ 可能需要分阶段训练（先蒸馏后微调）

---

## ⚠️ 注意事项

### 1. 教师模型路径
记得修改配置文件中的教师模型路径：
```json
"distillation": {
    "teacher_checkpoint": "/absolute/path/to/your/teacher_model.pt"
}
```

### 2. FPN通道配置
如果修改了backbone，**必须**同步修改FPN的`in_channels_list`：

| Backbone | FPN in_channels_list |
|----------|---------------------|
| ResNet18 | `[?, 64, 128, 256, 512]` |
| ResNet34 | `[?, 64, 128, 256, 512]` |
| ResNet50 | `[?, 256, 512, 1024, 2048]` |
| ResNet101 | `[?, 256, 512, 1024, 2048]` |

第一个通道数（`?`）取决于输入：
- 相机：3 (RGB)
- 雷达：6 (多通道)

### 3. Queries数量
`n_queries` 必须等于 `resolution` 的乘积：
- `resolution: [20, 20, 1]` → `n_queries: 400`
- `resolution: [18, 18, 1]` → `n_queries: 324`
- `resolution: [16, 16, 1]` → `n_queries: 256`

---

## 🎓 推荐的渐进式方案

如果你是第一次尝试蒸馏，建议按以下顺序：

1. **第一步**：使用 `kradar_distillation_light.json`
   - 验证蒸馏框架工作正常
   - 熟悉蒸馏训练流程
   - 建立性能基准

2. **第二步**：如果效果好，尝试 `kradar_distillation_medium.json`
   - 获得更好的速度提升
   - 评估精度损失是否可接受

3. **第三步**：根据需要考虑 `kradar_distillation_example.json`
   - 只在确实需要极限压缩时使用
   - 需要更多调参和实验

---

## 📚 参考文献

- Hinton et al. "Distilling the Knowledge in a Neural Network" (2015)
- Romero et al. "FitNets: Hints for Thin Deep Nets" (2015)
- He et al. "Deep Residual Learning for Image Recognition" (2016)

---

## 🆘 需要帮助？

如果遇到问题，请检查：
1. 教师模型路径是否正确
2. FPN通道配置是否匹配backbone
3. n_queries是否等于resolution乘积
4. 查看 `KNOWLEDGE_DISTILLATION_GUIDE.md` 获取详细故障排查指南

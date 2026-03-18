"""
检查模型是否正确训练
"""
import torch

# 替换为你的checkpoint路径
radar_checkpoint = "path/to/radar_model.pt"
camera_checkpoint = "path/to/camera_model.pt"

print("=" * 60)
print("检查雷达模型 checkpoint")
print("=" * 60)

try:
    radar_ckpt = torch.load(radar_checkpoint, map_location='cpu')
    print(f"✓ 文件加载成功")
    print(f"  Keys: {list(radar_ckpt.keys())}")

    if 'epoch' in radar_ckpt:
        print(f"  训练轮数: {radar_ckpt['epoch']}")

    if 'model_state_dict' in radar_ckpt:
        # 检查模型权重
        state_dict = radar_ckpt['model_state_dict']
        first_param_name = list(state_dict.keys())[0]
        first_param = state_dict[first_param_name]
        print(f"  第一个参数: {first_param_name}")
        print(f"    形状: {first_param.shape}")
        print(f"    均值: {first_param.float().mean():.6f}")
        print(f"    标准差: {first_param.float().std():.6f}")
        print(f"    最大值: {first_param.float().max():.6f}")
        print(f"    最小值: {first_param.float().min():.6f}")

        # 检查是否全为0或全为同一值
        if torch.all(first_param == 0):
            print("  ⚠️  警告：参数全为0！")
        elif torch.all(first_param == first_param.flatten()[0]):
            print("  ⚠️  警告：参数全为同一值！")

    if 'optimizer_state_dict' in radar_ckpt:
        print(f"  ✓ 包含optimizer状态")

except FileNotFoundError:
    print(f"✗ 文件不存在: {radar_checkpoint}")
except Exception as e:
    print(f"✗ 加载失败: {e}")

print("\n" + "=" * 60)
print("检查相机模型 checkpoint")
print("=" * 60)

try:
    camera_ckpt = torch.load(camera_checkpoint, map_location='cpu')
    print(f"✓ 文件加载成功")
    print(f"  Keys: {list(camera_ckpt.keys())}")

    if 'epoch' in camera_ckpt:
        print(f"  训练轮数: {camera_ckpt['epoch']}")

    if 'model_state_dict' in camera_ckpt:
        state_dict = camera_ckpt['model_state_dict']
        first_param_name = list(state_dict.keys())[0]
        first_param = state_dict[first_param_name]
        print(f"  第一个参数: {first_param_name}")
        print(f"    形状: {first_param.shape}")
        print(f"    均值: {first_param.float().mean():.6f}")
        print(f"    标准差: {first_param.float().std():.6f}")

    # 比较两个模型是否相同
    if 'model_state_dict' in radar_ckpt and 'model_state_dict' in camera_ckpt:
        radar_params = radar_ckpt['model_state_dict']
        camera_params = camera_ckpt['model_state_dict']

        print("\n" + "=" * 60)
        print("比较两个模型")
        print("=" * 60)

        # 检查键是否相同
        radar_keys = set(radar_params.keys())
        camera_keys = set(camera_params.keys())

        common_keys = radar_keys & camera_keys
        print(f"  共同的参数数量: {len(common_keys)}")

        if common_keys:
            # 检查共同参数是否相同
            same_count = 0
            for key in list(common_keys)[:10]:  # 只检查前10个
                if torch.equal(radar_params[key], camera_params[key]):
                    same_count += 1

            print(f"  前10个共同参数中相同的: {same_count}/10")

            if same_count == 10:
                print("  ⚠️  警告：两个模型的参数完全相同！")
                print("  可能你用的是同一个checkpoint！")

except FileNotFoundError:
    print(f"✗ 文件不存在: {camera_checkpoint}")
except Exception as e:
    print(f"✗ 加载失败: {e}")

print("\n" + "=" * 60)

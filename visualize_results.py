"""
可视化K-Radar模型预测和真实标签的对比脚本
使用 visu.py 来可视化模型输出效果
"""

import os
import numpy as np
import sys

# 添加项目路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from dprt.utils.visu import visu_lidar_data, visu_2d_radar_data


# K-Radar 类别映射
CATEGORIES = {
    'sed': 0,  # Sedan
    'bus': 1,  # Bus or Truck
    'mot': 2,  # Motorcycle
    'bic': 3,  # Bicycle
    'big': 4,  # Bicycle Group
    'ped': 5,  # Pedestrian
    'peg': 6,  # Pedestrian Group
    'bg': 7,   # Background
    'dummy': -1  # 占位符
}


def load_boxes_from_txt(txt_path):
    """
    从txt文件加载边界框

    txt格式: name truncated occluded alpha bbox1 bbox2 bbox3 bbox4 h w l y z x theta
    返回格式: [x, y, z, theta, l, w, h, class]
    """
    boxes = []

    if not os.path.exists(txt_path):
        print(f"警告: 文件不存在 {txt_path}")
        return np.array([])

    with open(txt_path, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if not parts or parts[0] == 'dummy':
                continue

            # 解析数据
            cls_name = parts[0]  # 类别名称
            h, w, l = float(parts[8]), float(parts[9]), float(parts[10])  # 高宽长
            y, z, x = float(parts[11]), float(parts[12]), float(parts[13])  # y,z,x坐标
            theta = float(parts[14])  # 朝向角

            # 转换为 visu.py 期望的格式: [x, y, z, theta, l, w, h, class]
            cls_id = CATEGORIES.get(cls_name, -1)
            box = [x, y, z, theta, l, w, h, cls_id]
            boxes.append(box)

    return np.array(boxes) if boxes else np.empty((0, 8))


def load_radar_or_lidar_data(data_root, sample_id):
    """
    加载雷达或激光雷达数据

    注意: 由于你只有预测和真实标签，没有原始雷达/激光雷达数据，
    这个函数会返回空数组。如果你有原始数据，可以在这里加载。
    """
    # 这里你需要提供原始的雷达或激光雷达数据路径
    # 例如: data_root/radar_bev/{sample_id}.npy

    # 如果没有原始数据，返回空点云（只显示边界框）
    return np.empty((0, 4))  # 空点云 (N, 4) - x, y, z, intensity


def visualize_sample_3d(evaluatelog_root, sample_id, conf_threshold='0.9',
                        scene_type='all', data_root=None):
    """
    3D可视化单个样本的预测和真实标签

    参数:
        evaluatelog_root: Evaluatelog 文件夹路径
        sample_id: 样本ID (例如: '000000')
        conf_threshold: 置信度阈值 (例如: '0.9')
        scene_type: 场景类型 (例如: 'all', 'highway', 'rain')
        data_root: 原始数据根目录（如果有的话）
    """
    # 构建文件路径
    base_path = os.path.join(evaluatelog_root, 'exports', 'kradar',
                              conf_threshold, scene_type)

    pred_path = os.path.join(base_path, 'preds', f'{sample_id}.txt')
    gt_path = os.path.join(base_path, 'gts', f'{sample_id}.txt')

    print(f"\n{'='*60}")
    print(f"可视化样本 {sample_id} (置信度阈值: {conf_threshold})")
    print(f"{'='*60}\n")

    # 加载预测和真实标签
    pred_boxes = load_boxes_from_txt(pred_path)
    gt_boxes = load_boxes_from_txt(gt_path)

    print(f"预测框数量: {len(pred_boxes)}")
    print(f"真实框数量: {len(gt_boxes)}")

    # 加载点云数据（如果有）
    if data_root:
        pc = load_radar_or_lidar_data(data_root, sample_id)
    else:
        # 创建空点云用于显示边界框的参考
        pc = np.array([[0, 0, 0, 1]])  # 原点

    print("\n首先显示真实标签 (Ground Truth)...")
    print("关闭窗口后将显示预测结果")

    # 可视化真实标签
    if len(gt_boxes) > 0:
        visu_lidar_data(pc, boxes=gt_boxes, xlim=(0, 72), ylim=(-6.4, 6.4))
    else:
        print("警告: 没有真实标签")

    print("\n现在显示预测结果 (Predictions)...")

    # 可视化预测结果
    if len(pred_boxes) > 0:
        visu_lidar_data(pc, boxes=pred_boxes, xlim=(0, 72), ylim=(-6.4, 6.4))
    else:
        print("警告: 没有预测结果")


def compare_predictions(evaluatelog_root, conf_threshold='0.9', scene_type='all'):
    """
    比较所有样本的预测和真实标签统计
    """
    base_path = os.path.join(evaluatelog_root, 'exports', 'kradar',
                              conf_threshold, scene_type)

    val_file = os.path.join(base_path, 'val.txt')

    if not os.path.exists(val_file):
        print(f"错误: val.txt 文件不存在于 {base_path}")
        return

    # 读取所有样本ID
    with open(val_file, 'r') as f:
        sample_ids = [line.strip() for line in f if line.strip()]

    print(f"\n{'='*60}")
    print(f"统计分析 (置信度阈值: {conf_threshold}, 场景: {scene_type})")
    print(f"{'='*60}\n")
    print(f"总样本数: {len(sample_ids)}")

    # 统计预测和真实标签
    total_preds = 0
    total_gts = 0

    for sample_id in sample_ids[:10]:  # 只显示前10个样本的详细信息
        pred_path = os.path.join(base_path, 'preds', f'{sample_id}.txt')
        gt_path = os.path.join(base_path, 'gts', f'{sample_id}.txt')

        pred_boxes = load_boxes_from_txt(pred_path)
        gt_boxes = load_boxes_from_txt(gt_path)

        total_preds += len(pred_boxes)
        total_gts += len(gt_boxes)

        print(f"样本 {sample_id}: 预测={len(pred_boxes):2d}, 真实={len(gt_boxes):2d}")

    if len(sample_ids) > 10:
        print(f"... (还有 {len(sample_ids)-10} 个样本)")

    print(f"\n总预测框数: {total_preds}")
    print(f"总真实框数: {total_gts}")


def main():
    """主函数"""
    # 配置路径
    EVALUATELOG_ROOT = r"D:\DPFT\Evaluatelog"

    # 可选: 如果你有原始雷达/激光雷达数据，设置这个路径
    DATA_ROOT = None  # 例如: r"D:\DPFT\data\kradar"

    print("=" * 60)
    print("K-Radar 模型输出可视化工具")
    print("=" * 60)

    # 1. 显示统计信息
    compare_predictions(EVALUATELOG_ROOT, conf_threshold='0.9', scene_type='all')

    # 2. 可视化具体样本
    print("\n" + "=" * 60)
    print("3D可视化")
    print("=" * 60)

    # 可视化第一个样本
    visualize_sample_3d(
        evaluatelog_root=EVALUATELOG_ROOT,
        sample_id='000000',
        conf_threshold='0.9',
        scene_type='all',
        data_root=DATA_ROOT
    )

    # 如果你想可视化更多样本，可以取消下面的注释
    # visualize_sample_3d(EVALUATELOG_ROOT, '000001', '0.9', 'all', DATA_ROOT)

    print("\n可视化完成！")
    print("\n使用说明:")
    print("1. 修改 main() 函数中的 sample_id 来可视化不同的样本")
    print("2. 修改 conf_threshold 来查看不同置信度阈值的结果")
    print("3. 如果有原始数据，设置 DATA_ROOT 路径来显示点云")


if __name__ == '__main__':
    main()

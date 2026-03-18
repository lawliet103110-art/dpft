"""
快速可视化脚本 - 直接查看单个样本
使用方法: python quick_visualize.py [sample_id]
例如: python quick_visualize.py 000000
"""

import os
import sys
import numpy as np

# 添加项目路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from dprt.utils.visu import visu_lidar_data


# 类别映射
CATEGORIES = {
    'sed': 0, 'bus': 1, 'mot': 2, 'bic': 3,
    'big': 4, 'ped': 5, 'peg': 6, 'bg': 7
}


def parse_box_line(line):
    """解析一行边界框数据"""
    parts = line.strip().split()
    if not parts or parts[0] == 'dummy':
        return None

    cls_name = parts[0]
    h, w, l = float(parts[8]), float(parts[9]), float(parts[10])
    y, z, x = float(parts[11]), float(parts[12]), float(parts[13])
    theta = float(parts[14])
    cls_id = CATEGORIES.get(cls_name, -1)

    # 格式: [x, y, z, theta, l, w, h, class]
    return [x, y, z, theta, l, w, h, cls_id]


def load_boxes(txt_path):
    """快速加载边界框"""
    boxes = []
    if os.path.exists(txt_path):
        with open(txt_path, 'r') as f:
            for line in f:
                box = parse_box_line(line)
                if box:
                    boxes.append(box)
    return np.array(boxes) if boxes else np.empty((0, 8))


def visualize(sample_id='000000', conf='0.9'):
    """快速可视化"""
    base = rf"D:\DPFT\Evaluatelog\exports\kradar\{conf}\all"

    print(f"\n{'='*50}")
    print(f"可视化样本 {sample_id} (置信度: {conf})")
    print(f"{'='*50}\n")

    # 加载数据
    gt_boxes = load_boxes(os.path.join(base, 'gts', f'{sample_id}.txt'))
    pred_boxes = load_boxes(os.path.join(base, 'preds', f'{sample_id}.txt'))

    print(f"真实标签: {len(gt_boxes)} 个目标")
    print(f"预测结果: {len(pred_boxes)} 个目标\n")

    # 显示边界框详情
    if len(gt_boxes) > 0:
        print("真实标签详情:")
        for i, box in enumerate(gt_boxes):
            cls_names = {v: k for k, v in CATEGORIES.items()}
            print(f"  {i+1}. {cls_names.get(int(box[7]), 'unknown'):3s} "
                  f"位置:({box[0]:5.2f}, {box[1]:5.2f}, {box[2]:5.2f}) "
                  f"尺寸:({box[4]:4.2f}×{box[5]:4.2f}×{box[6]:4.2f})")

    if len(pred_boxes) > 0:
        print("\n预测结果详情:")
        for i, box in enumerate(pred_boxes):
            cls_names = {v: k for k, v in CATEGORIES.items()}
            print(f"  {i+1}. {cls_names.get(int(box[7]), 'unknown'):3s} "
                  f"位置:({box[0]:5.2f}, {box[1]:5.2f}, {box[2]:5.2f}) "
                  f"尺寸:({box[4]:4.2f}×{box[5]:4.2f}×{box[6]:4.2f})")

    # 创建空点云（只显示边界框）
    pc = np.array([[0, 0, 0, 1]])

    # K-Radar 检测范围
    xlim = (0, 72)      # 前向 0-72米
    ylim = (-6.4, 6.4)  # 横向 ±6.4米

    # 显示真实标签
    if len(gt_boxes) > 0:
        print("\n[1/2] 显示真实标签 (Ground Truth)")
        print("      关闭窗口后将显示预测结果...")
        visu_lidar_data(pc, boxes=gt_boxes, xlim=xlim, ylim=ylim)

    # 显示预测结果
    if len(pred_boxes) > 0:
        print("\n[2/2] 显示预测结果 (Predictions)")
        visu_lidar_data(pc, boxes=pred_boxes, xlim=xlim, ylim=ylim)

    print("\n完成！\n")


if __name__ == '__main__':
    # 从命令行参数获取样本ID
    sample_id = sys.argv[1] if len(sys.argv) > 1 else '000000'
    conf = sys.argv[2] if len(sys.argv) > 2 else '0.9'

    try:
        visualize(sample_id, conf)
    except Exception as e:
        print(f"\n错误: {e}\n")
        print("使用方法:")
        print("  python quick_visualize.py [sample_id] [conf_threshold]")
        print("\n示例:")
        print("  python quick_visualize.py 000000 0.9")
        print("  python quick_visualize.py 000001")

from typing import Any, Dict, List, Tuple

import torch

from torch.utils.data import DataLoader, Dataset, default_collate

from dprt.utils.misc import as_list


def listed_collating(
        data: List[Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]]]
) -> Tuple[Dict[str, torch.Tensor], List[Dict[str, torch.Tensor]]]:
    """
    Attributes:
        data: List to data tuples consiting of input and target values.

    Returns:
        batch: Batched data consiting of a tuple of batched inputs and
            a list of targets.
    """
    # Split data into inputs and targets (list of tuples to tuple of lists)
    inputs, targets = zip(*data)

    # Ensure list data type
    inputs = as_list(inputs)
    targets = as_list(targets)
    
    # Get all possible keys from all samples to handle missing modalities
    all_keys = set()
    for sample in inputs:
        all_keys.update(sample.keys())
    
    # 关键修复：确保即使在一个批次中所有样本都没有模态数据，也不会产生空字典
    # 这种情况在全集中更可能出现，而在子集中可能没有遇到
    if not all_keys:
        # 如果所有样本都是空字典，创建一个默认的输入键
        # 根据配置文件，模型期望的主要输入包括：camera_mono, radar_bev, radar_front
        # 为每个模态创建具有默认形状的零张量
        default_keys_shapes = {
            'camera_mono': (1, 512, 512, 3),  # 假设的图像形状 (H, W, C)
            'radar_bev': (1, 256, 256, 6),     # 假设的雷达BEV形状
            'radar_front': (1, 256, 256, 6),   # 假设的雷达前视图形状
            'lidar_top': (1, 256, 256, 6)      # LiDAR BEV投影形状
        }
        
        # 为每个样本添加所有默认键和零张量
        for i in range(len(inputs)):
            for key, shape in default_keys_shapes.items():
                inputs[i][key] = torch.zeros(shape, dtype=torch.float32)
        all_keys = set(default_keys_shapes.keys())
    
    # Ensure each sample has all keys, creating zero tensors for missing modalities
    for i, sample in enumerate(inputs):
        for key in all_keys:
            if key not in sample:
                # Find a sample that has this key to get the correct shape and type
                found_shape = False
                for other_sample in inputs:
                    if key in other_sample:
                        # Create a zero tensor with the same shape and type
                        sample[key] = torch.zeros_like(other_sample[key])
                        found_shape = True
                        break
                # 如果找不到匹配的键，使用默认形状
                if not found_shape:
                    # 为常见的模态设置默认形状
                    if key == 'camera_mono':
                        sample[key] = torch.zeros((1, 512, 512, 3), dtype=torch.float32)
                    elif key in ['radar_bev', 'radar_front', 'lidar_top']:
                        sample[key] = torch.zeros((1, 256, 256, 6), dtype=torch.float32)
                    else:
                        sample[key] = torch.zeros((1,), dtype=torch.float32)
    
    # Convert tensors to batch of tensors
    inputs = default_collate(inputs)

    # Combine inputs and outputs
    batch = (inputs, targets)

    return batch


def load_listed(dataset: Dataset, config: Dict[str, Any]) -> DataLoader:
    return DataLoader(
        dataset=dataset,
        batch_size=config['train']['batch_size'],
        shuffle=config['train']['shuffle'],
        num_workers=config['computing']['workers'],
        collate_fn=listed_collating
    )


# from typing import Any, Dict, List, Tuple

# import torch

# from torch.utils.data import DataLoader, Dataset, default_collate

# from dprt.utils.misc import as_list


# def listed_collating(
#         data: List[Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]]]
# ) -> Tuple[Dict[str, torch.Tensor], List[Dict[str, torch.Tensor]]]:
#     """
#     Attributes:
#         data: List to data tuples consiting of input and target values.

#     Returns:
#         batch: Batched data consiting of a tuple of batched inputs and
#             a list of targets.
#     """
#     # Split data into inputs and targets (list of tuples to tuple of lists)
#     inputs, targets = zip(*data)

#     # Ensure list data type
#     inputs = as_list(inputs)
#     targets = as_list(targets)

#     # Convert tensors to batch of tensors
#     inputs = default_collate(inputs)

#     # Combine inputs and outputs
#     batch = (inputs, targets)

#     return batch


# def load_listed(dataset: Dataset, config: Dict[str, Any]) -> DataLoader:
#     return DataLoader(
#         dataset=dataset,
#         batch_size=config['train']['batch_size'],
#         shuffle=config['train']['shuffle'],
#         num_workers=config['computing']['workers'],
#         collate_fn=listed_collating
#     )

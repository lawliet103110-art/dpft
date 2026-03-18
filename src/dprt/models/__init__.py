import os

from typing import Tuple

import torch

from dprt.models.dprt import build_dprt


def build(model: str, *args, **kwargs):
    if model == 'dprt':
        return build_dprt(*args, **kwargs)


def load(checkpoint: str, *args, **kwargs) -> Tuple[torch.nn.Module, int, str]:
    filename = os.path.splitext(os.path.basename(checkpoint))[0]

    # Parse filename - support two formats:
    # 1. timestamp_checkpoint_epoch.pt
    # 2. timestamp_best_mAP_xxx_epoch_yyyy.pt
    parts = filename.split('_')

    # Extract timestamp (first part)
    timestamp = parts[0]

    # Extract epoch number (last part after 'epoch')
    if 'epoch' in parts:
        epoch_idx = parts.index('epoch')
        epoch = int(parts[epoch_idx + 1])
    else:
        # Fallback: assume last part is epoch number
        epoch = int(parts[-1])

    return torch.load(checkpoint), epoch, timestamp

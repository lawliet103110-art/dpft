from __future__ import annotations  # noqa: F407

from typing import Any, Dict, List, Tuple

import torch
import torch.nn.functional as F

from torch import nn
from torch.utils.data import default_collate

from dprt.training.assigner import build_anassigner
from dprt.utils.bbox import get_box_corners
from dprt.utils.data import decollate_batch
from dprt.utils.iou import giou3d


def focal_loss(inputs: torch.Tensor, targets: torch.Tensor,
               alpha: float = 0.75, gamma: float = 2.0,
               reduction: str = "none") -> torch.Tensor:
    """Focal loss function.

    Reference: https://arxiv.org/abs/1708.02002

    Arguments:
        inputs: A float tensor of arbitrary shape (B, ...).
            The predictions for each example.
        targets: A float tensor with the same shape as inputs.
            Stores the binary classification label for each element
            in inputs (0 for the negative class and 1 for the positive class).
        alpha: Weighting factor in range (0, 1) to balance
            positive vs negative examples.
        gamma: Exponent of the modulating factor (1 - p_t) to
            balance easy vs hard examples.
        reduction: Reduction mode for the per class loss values.
            One of either none, sum or average.

    Retruns:
        loss: Loss tensor with the reduction option applied.
    """
    # Get cross entropy loss
    ce_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction="none")

    # Get focal loss
    p_t = inputs * targets + (1 - inputs) * (1 - targets)
    loss = ce_loss * ((1 - p_t) ** gamma)

    # Balance loss
    if alpha >= 0:
        alpha_t = alpha * targets + (1 - alpha) * (1 - targets)
        loss = alpha_t * loss

    # Reduce loss (if required)
    if reduction == "none":
        pass
    elif reduction == "mean":
        loss = loss.mean()
    elif reduction == "sum":
        loss = loss.sum()

    return loss


class FocalLoss(nn.modules.loss._Loss):
    def __init__(self,
                 alpha: float = 0.75,
                 gamma: float = 2.0,
                 reduction: str = 'sum',
                 **kwargs):
        """Focal loss function.

        Reference: https://arxiv.org/abs/1708.02002

        Arguments:
            alpha: Weighting factor in range (0, 1) to balance
                positive vs negative examples.
            gamma: Exponent of the modulating factor (1 - p_t) to
                balance easy vs hard examples.
            reduction: Reduction mode for the per class loss values.
                One of either none, sum or average.
        """
        super().__init__(**kwargs)

        # Initialize instance attributes
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

        # Check input arguments
        if self.reduction not in {'none', 'mean', 'sum'}:
            raise ValueError(
                    f"Invalid Value for arg 'reduction': '{self.reduction}"
                    f"\n Supported reduction modes: 'none', 'mean', 'sum'"
                )

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """Returns the focal loss value for the given input.

        Arguments:
            inputs: A float tensor of arbitrary shape (B, ...).
                The predictions for each example.
            targets: A float tensor with the same shape as inputs.
                Stores the binary classification label for each element
                in inputs (0 for the negative class and 1 for the positive class).

        Retruns:
            loss: Loss tensor with the reduction option applied.
        """
        return focal_loss(inputs, targets, self.alpha, self.gamma, self.reduction)


class GIoULoss(nn.modules.loss._Loss):
    def __init__(self,
                 reduction: str = 'sum',
                 **kwargs):
        """Generalized intersection over union loss.

        Reference: https://giou.stanford.edu/

        Arguments:
            reduction: Reduction mode for the per class loss values.
                One of either none, sum or average.
        """
        super().__init__(**kwargs)

        # Initialize instance attributes
        self.reduction = reduction

        # Check input arguments
        if self.reduction not in {'none', 'mean', 'sum'}:
            raise ValueError(
                    f"Invalid Value for arg 'reduction': '{self.reduction}"
                    f"\n Supported reduction modes: 'none', 'mean', 'sum'"
                )

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """Returns the generalized intersection over union loss.

        Arguments:
            inputs: A float tensor of shape (B, N, 8). The inputs
                represent bounding boxes with elements
                (x, y, z, l, w, h, sin a, cos a).
            targets: A float tensor with the same shape as inputs.

        Retruns:
            loss: Loss tensor with the reduction option applied.
        """
        # Get input shape
        B = inputs.shape[0]
        N = inputs.shape[1]

        # Get bounding box angles
        angle = torch.atan2(inputs[..., 6], inputs[..., 7])
        gt_angle = torch.atan2(targets[..., 6], targets[..., 7])

        # Get box corners
        corners = get_box_corners(inputs[..., :3], inputs[..., 3:6], angle)
        gt_corners = get_box_corners(targets[..., :3], targets[..., 3:6], gt_angle)

        # Get giou (giou is between -1 and 1) [0, 2]
        loss = 1 - torch.diagonal(giou3d(corners, gt_corners))

        # Reshape and scale to [0, 1]
        loss = loss.reshape(B, N) / 2

        # Reduce loss (if required)
        if self.reduction == "none":
            pass
        elif self.reduction == "mean":
            loss = loss.mean()
        elif self.reduction == "sum":
            loss = loss.sum()

        return loss


class SetCriterion(nn.Module):
    def __init__(self):
        """Set-to-Set loss

        Inspired by: Deformable-DETR

        losses: Dictionary of loss functions. Maps a
            loss name to a loss function.
        loss_inputs: Dictionary of loss input values.
                Maps a loss name to input (value) names.
        """
        super().__init__()

        self.losses = {
            "total_class": "total_focal_loss",
            "object_class": "object_focal_loss",
            "center": "l1_loss",
            "size": "l1_loss",
            "angle": "l1_loss"
        }

        self.loss_inputs = {
            "total_class": ["class"],
            "object_class": ["class"],
            "center": ["center"],
            "size": ["size"],
            "angle": ["angle"]
        }

    @staticmethod
    def _batched_index_select(batch: torch.Tensor, dim: int, inds: torch.Tensor) -> torch.Tensor:
        """Returns elements of a batched tensor given their indices.

        Arguments:
            batch: The batched input tensor of shape (B, N, M)
            dim: The dimension in which we index.
            inds: The 2D tensor containing the indices to
                index with shape (B, N)
        """
        inds = inds.unsqueeze(2).expand(inds.size(0), inds.size(1), batch.size(2))
        out = batch.gather(dim, inds)
        return out

    @staticmethod
    def _dstack_dict(dictionary: Dict[str, torch.Tensor], keys: List[str]) -> torch.Tensor:
        """Retruns a stacked tensor from a dict of tensors.

        Arguments:
            dictionary: Dictionary of tensors with shape (B, N, C1),
                where C1 can vary across the tensors.
            keys: Dictionary keys of the tensors to stack.

        Returns:
            Stacked tensor with shape (B, N, C2), where
                C2 = sum(C1).
        """
        return torch.dstack([dictionary[k] for k in keys])

    def object_focal_loss(self, inputs: torch.Tensor, targets: torch.Tensor,
                          indices: Tuple[torch.Tensor, torch.Tensor]) -> torch.Tensor:
        """Returns an aggregated focal loss value for all associated predictions.

        Calculates a focal loss value between the associated predictions
        and ground truth.

        Arguments:
            inputs: Tensor of model predictions with shape (B, N, C1)
            targets: Tensor of ground truth values with shape (B, M, C1)
            indices: Tuple of indices for the association between prediction
                and ground truth values with shape ((B, M), (B, M))

        Returns:
            loss: Loss value with shape (B, )
        """
        # Split indices into prediction and target indices
        i, j = indices

        # Get number of queries
        N = inputs.shape[1]

        # Get number of ground truth objects
        M = j.numel()

        # Calculate L1 loss between the matched inputs and targets
        loss = focal_loss(
            self._batched_index_select(inputs, dim=1, inds=i),
            self._batched_index_select(targets, dim=1, inds=j),
            reduction='none'
        )

        # Average loss
        loss = (loss.mean(1).sum() / M) * N

        return loss

    def total_focal_loss(self, inputs: torch.Tensor, targets: torch.Tensor,
                         indices: Tuple[torch.Tensor, torch.Tensor]) -> torch.Tensor:
        """Returns an aggregated focal loss value for all predictions.

        Calculates a focal loss value for all predictions by matching all
        unassociated predictions to the background class.

        Note: It is assumed that the first ("zerost") class is the None class!

        Arguments:
            inputs: Tensor of model predictions with shape (B, N, C1)
            targets: Tensor of ground truth values with shape (B, M, C1)
            indices: Tuple of indices for the association between prediction
                and ground truth values with shape ((B, M), (B, M))

        Returns:
            loss: Loss value with shape (B, )
        """
        # Get input shape and device
        B, N, C = inputs.shape
        device = inputs.device
        dtype = inputs.dtype

        # Split indices into prediction and target indices
        i, j = indices

        # Get number of ground truth objects
        M = j.numel()

        # Initialize target class probabilities (one hot encoded)
        target_one_hot: torch.Tensor = F.one_hot(
            torch.zeros((B, N), dtype=torch.int64, device=device),
            num_classes=C
        )
        target_one_hot = target_one_hot.type(dtype)

        # Assign ground truth labels to the indices of the assigned predictions
        index = i.unsqueeze(2).expand(i.size(0), i.size(1), C)
        target_one_hot.scatter_(dim=1, index=index, src=targets)

        # Get focal loss value
        loss = focal_loss(inputs, target_one_hot, reduction='none')

        # Average loss
        loss = (loss.mean(1).sum() / M) * N

        return loss

    def l1_loss(self, inputs: torch.Tensor, targets: torch.Tensor,
                indices: Tuple[torch.Tensor, torch.Tensor]) -> torch.Tensor:
        """Returns the mean L1 loss for the given inputs.

        Arguments:
            inputs: Tensor of model predictions with shape (B, N, C1)
            targets: Tensor of ground truth values with shape (B, M, C1)
            indices: Tuple of indices for the association between prediction
                and ground truth values with shape ((B, M), (B, M))

        Returns:
            loss: Loss value with shape (B, )
        """
        # Split indices into prediction and target indices
        i, j = indices

        # Calculate L1 loss between the matched inputs and targets
        loss = F.l1_loss(
            self._batched_index_select(inputs, dim=1, inds=i),
            self._batched_index_select(targets, dim=1, inds=j),
            reduction='mean'
        )

        return loss

    def forward(
        self,
        inputs: Dict[str, torch.Tensor],
        targets: Dict[str, torch.Tensor],
        indices: Tuple[torch.Tensor, torch.Tensor]
    ) -> Dict[str, torch.Tensor]:
        """Returns the set-to-set loss values.

        Arguments:
            inputs: Dictionary of model predictions. Each value of the
                dictionary is a tensor with shape (B, N, C).
            targets: Dictionary of ground truth values. Each value of the
                dictionary is a tensor with shape (B, M, C).
            indices: Tuple of indices for the association between prediction
                and ground truth values with shape ((B, M), (B, M))

        Returns:
            losses: Dictionary of loss values with shape (B, ).
        """
        # Get losses
        losses = {
            name: getattr(self, func)(
                self._dstack_dict(inputs, self.loss_inputs[name]),
                self._dstack_dict(targets, [f"gt_{n}" for n in self.loss_inputs[name]]),
                indices
            )
            for name, func in self.losses.items()
        }

        return losses


class Loss(nn.modules.loss._Loss):
    def __init__(self,
                 anassigner: nn.Module = None,
                 criterion: nn.Module = None,
                 losses: Dict[str, nn.modules.loss._Loss] = None,
                 loss_inputs: Dict[str, List[str]] = None,
                 loss_weights: Dict[str, float] = None,
                 reduction: str = 'mean',
                 **kwargs):
        """Loss module.

        Arguments:
            anassigner: Anassigner to match model predictions
                with ground truth values.
            losses: Dictionary of loss functions. Maps a
                loss name to a loss function.
            loss_inputs: Dictionary of loss input values.
                Maps a loss name to input (value) names.
            loss_weights: Dictionary of loss weights. Maps a
                loss name to a loss weight.
            reduction: Reduction mode for the per batch loss values.
                One of either none, sum or mean.
        """
        # Initialize base class
        super().__init__(**kwargs)

        # Check input arguments
        if anassigner is not None:
            assert criterion is not None

        if reduction not in {'none', 'mean', 'sum'}:
            raise ValueError(
                    f"Invalid Value for arg 'reduction': '{self.reduction}"
                    f"\n Supported reduction modes: 'none', 'mean', 'sum'"
                )

        # Initialize loss instance
        self.losses = losses if losses is not None else {}
        self.loss_inputs = loss_inputs if loss_inputs is not None else {}
        self.loss_weights = loss_weights if loss_weights is not None else {}
        self.anassigner = anassigner
        self.criterion = criterion
        self.reduction = reduction

        # Get reduction function
        if self.reduction != 'none':
            self.reduction_fn = getattr(torch, self.reduction)

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> Loss:  # noqa: F821
        anassigner = None
        criterion = None
        losses = None
        loss_inputs = config.get('loss_inputs')
        loss_weights = config.get('loss_weights')
        reduction = config.get('reduction', 'mean')

        if 'anassigner' in config:
            anassigner = build_anassigner(config['anassigner'], config)

        if 'criterion' in config:
            criterion = _get_loss(config['criterion'])

        if 'losses' in config:
            losses = {k: _get_loss(v) for k, v in config['losses'].items()}

        return cls(
            anassigner=anassigner,
            criterion=criterion,
            losses=losses,
            loss_inputs=loss_inputs,
            loss_weights=loss_weights,
            reduction=reduction
        )

    @staticmethod
    def _get_input_device(input):
        if isinstance(input, torch.Tensor):
            device = input.device
        elif isinstance(input, dict):
            device = input[list(input.keys())[0]].device
        elif isinstance(input, (list, tuple, set)):
            device = input[0].device
        return device

    @staticmethod
    def _get_input_dtype(input):
        if isinstance(input, torch.Tensor):
            dtype = input.dtype
        elif isinstance(input, dict):
            dtype = input[list(input.keys())[0]].dtype
        elif isinstance(input, (list, tuple, set)):
            dtype = input[0].dtype
        return dtype

    @staticmethod
    def _dstack_dict(dictionary: Dict[str, torch.Tensor], keys: List[str]) -> torch.Tensor:
        """Retruns a stacked tensor from a dict of tensors.

        Arguments:
            dictionary: Dictionary of tensors with shape (B, N, C1),
                where C1 can vary across the tensors.
            keys: Dictionary keys of the tensors to stack.

        Returns:
            Stacked tensor with shape (B, N, C2), where
                C2 = sum(C1).
        """
        return torch.dstack([dictionary[k] for k in keys])

    def forward(
        self,
        inputs: Dict[str, torch.Tensor],
        targets: List[Dict[str, torch.Tensor]]
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """Returns the loss given a prediction and ground truth.

        Arguments:
            inputs: Dictionary of model predictions with shape (B, N, C).
            targets: Dictionary of ground truth values with shape (B, M, C).

        Returns:
            total_loss: Sum over all loss values with shape (B, ).
            losses: Dictionary of loss values with shape (B, ).
        """
        # Get input data device and dtype
        device = self._get_input_device(inputs)
        dtype = self._get_input_dtype(inputs)

        # Initialize losses
        batch_losses = []

        # Decollate inputs
        inputs: List[Dict[str, torch.Tensor]] = decollate_batch(inputs, detach=False, pad=False)

        # Get loss for each item in the batch
        for input, target in zip(inputs, targets):
            # Append zero and continue if no targets are present
            if not all([t.numel() for t in target.values()]):
                batch_losses.append({
                    name: torch.zeros((), device=device, dtype=dtype, requires_grad=True)
                    for name in self.loss_weights.keys()
                })
                continue

            # Insert dummy batch dimension
            input = {k: v.unsqueeze(0) for k, v in input.items()}
            target = {k: v.unsqueeze(0) for k, v in target.items()}

            if self.anassigner is not None:
                # Get assignment
                i, j = self.anassigner(input, target)

                # Apply loss criterion
                losses = self.criterion(input, target, indices=(i, j))

            else:
                # Get loss values
                losses = {
                    name: func(
                        self._dstack_dict(input, self.loss_inputs[name]),
                        self._dstack_dict(target, [f"gt_{n}" for n in self.loss_inputs[name]])
                    )
                    for name, func in self.losses.items()
                }

            # Weight loss values
            for k, weight in self.loss_weights.items():
                losses[k] *= weight

            # Add losses to the batch
            batch_losses.append(losses)

        # Catch no loss configuration
        if not self.losses:
            return (torch.zeros(1, device=device, dtype=dtype, requires_grad=True),
                    {k: torch.zeros(1, device=device, dtype=dtype) for k in self.losses.keys()})

        # Collate losses (revert decollating)
        batch_losses: Dict[str, torch.Tensor] = default_collate(batch_losses)

        # Reduce batch losses
        if self.reduction != 'none':
            batch_losses = {k: self.reduction_fn(v) for k, v in batch_losses.items()}

        # Get total loss
        total_loss = torch.stack(tuple(batch_losses.values())).sum(dim=-1)

        return total_loss, batch_losses


def _get_loss(name: str) -> nn.modules.loss._Loss:
    """Returns a pytorch or custom loss function given its name.

    Attributes:
        name: Name of the loss function (class).

    Returns:
        Instance of a loss function.
    """
    try:
        return getattr(nn, name)()
    except AttributeError:
        return globals()[name]()
    except Exception as e:
        raise e


class DistillationLoss(nn.modules.loss._Loss):
    def __init__(self,
                 temperature: float = 4.0,
                 distill_class: bool = True,
                 distill_bbox: bool = True,
                 distill_mode: str = 'matched',
                 reduction: str = 'mean',
                 **kwargs):
        """Knowledge Distillation Loss for 3D Object Detection.

        This loss combines soft target distillation for classification logits
        and regression outputs from a teacher model.

        Arguments:
            temperature: Temperature for softening probability distributions.
                Higher temperature produces softer distributions. Default: 4.0
            distill_class: Whether to distill classification logits. Default: True
            distill_bbox: Whether to distill bounding box regression outputs. Default: True
            distill_mode: Distillation mode. Options:
                - 'all': Distill all queries (may include background boxes)
                - 'matched': Only distill queries matched to GT (recommended)
                Default: 'matched'
            reduction: Reduction mode for loss values. One of 'none', 'mean', 'sum'.

        Reference:
            - Hinton et al. "Distilling the Knowledge in a Neural Network" (2015)
            - https://arxiv.org/abs/1503.02531
        """
        super().__init__(**kwargs)

        self.temperature = temperature
        self.distill_class = distill_class
        self.distill_bbox = distill_bbox
        self.distill_mode = distill_mode
        self.reduction = reduction

        # Check input arguments
        if self.reduction not in {'none', 'mean', 'sum'}:
            raise ValueError(
                f"Invalid value for arg 'reduction': '{self.reduction}'"
                f"\n Supported reduction modes: 'none', 'mean', 'sum'"
            )

        if self.distill_mode not in {'all', 'matched'}:
            raise ValueError(
                f"Invalid value for arg 'distill_mode': '{self.distill_mode}'"
                f"\n Supported modes: 'all', 'matched'"
            )

    def distillation_loss_class(self,
                                student_logits: torch.Tensor,
                                teacher_logits: torch.Tensor) -> torch.Tensor:
        """Compute KL divergence loss for classification logits.

        Arguments:
            student_logits: Student model classification logits (B, N, num_classes)
            teacher_logits: Teacher model classification logits (B, N, num_classes)

        Returns:
            loss: KL divergence loss between soft distributions
        """
        # Apply temperature scaling and softmax
        student_soft = F.log_softmax(student_logits / self.temperature, dim=-1)
        teacher_soft = F.softmax(teacher_logits / self.temperature, dim=-1)

        # Compute KL divergence: KL(teacher || student)
        # Scale by T^2 to maintain magnitude as described in the paper
        loss = F.kl_div(student_soft, teacher_soft, reduction='none') * (self.temperature ** 2)

        # Reduce over class dimension
        loss = loss.sum(dim=-1)  # (B, N)

        return loss

    def distillation_loss_bbox(self,
                               student_bbox: torch.Tensor,
                               teacher_bbox: torch.Tensor) -> torch.Tensor:
        """Compute L1 loss for bounding box regression outputs.

        Arguments:
            student_bbox: Student model bbox predictions (B, N, 8)
                Format: (x, y, z, l, w, h, sin_a, cos_a)
            teacher_bbox: Teacher model bbox predictions (B, N, 8)

        Returns:
            loss: L1 loss between student and teacher bbox predictions
        """
        # Compute L1 loss for bbox regression
        loss = F.l1_loss(student_bbox, teacher_bbox, reduction='none')

        # Reduce over bbox dimension
        loss = loss.mean(dim=-1)  # (B, N)

        return loss

    def forward(self,
                student_outputs: Dict[str, torch.Tensor],
                teacher_outputs: Dict[str, torch.Tensor],
                indices: Tuple[torch.Tensor, ...] = None) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """Compute distillation loss between student and teacher outputs.

        Arguments:
            student_outputs: Dictionary containing student model outputs:
                - 'class': Classification logits (B, N_student, num_classes)
                - 'center': Center predictions (B, N_student, 3)
                - 'size': Size predictions (B, N_student, 3)
                - 'angle': Angle predictions (B, N_student, 2) as (sin, cos)
            teacher_outputs: Dictionary containing teacher model outputs with same structure
                - Note: N_teacher can be different from N_student
            indices: Optional tuple from Hungarian matching.
                Required when distill_mode='matched'. Shape: (B, M) where M is number of GT boxes.
                Supported formats:
                - (student_pred_indices, gt_indices)
                - (student_pred_indices, gt_indices, teacher_pred_indices)

        Returns:
            total_loss: Total distillation loss
            losses: Dictionary of individual loss components
        """
        losses = {}

        # Check if indices are required but not provided
        if self.distill_mode == 'matched' and indices is None:
            raise ValueError(
                "distill_mode='matched' requires indices from Hungarian matching, "
                "but indices=None was provided"
            )

        # Get student and teacher shapes
        B, N_student = student_outputs['class'].shape[:2]
        N_teacher = teacher_outputs['class'].shape[1]
        device = student_outputs['class'].device

        # Handle different distillation modes
        if self.distill_mode == 'matched' and indices is not None:
            # Matched mode: Only distill queries matched to GT boxes
            if len(indices) == 3:
                student_pred_indices, gt_indices, teacher_pred_indices = indices
            else:
                student_pred_indices, gt_indices = indices
                teacher_pred_indices = None

            # For each sample in batch, select matched student and teacher queries
            student_selected = []
            teacher_selected = []

            for b in range(B):
                # Get matched indices for this sample
                student_matched_idx = student_pred_indices[b]  # Indices of matched student queries

                if len(student_matched_idx) > 0:
                    if teacher_pred_indices is not None:
                        teacher_matched_idx = teacher_pred_indices[b]
                    else:
                        # Backward compatible fallback: re-use student indices
                        teacher_matched_idx = torch.clamp(student_matched_idx, 0, N_teacher - 1)

                    student_selected.append(student_matched_idx)
                    teacher_selected.append(teacher_matched_idx)

            # Create selection masks
            has_matches = len(student_selected) > 0 and any(len(idx) > 0 for idx in student_selected)

        elif self.distill_mode == 'all':
            # All mode: Distill all queries
            # If N_student != N_teacher, we distill min(N_student, N_teacher) queries
            has_matches = True
            student_selected = None  # Will use slicing instead
            teacher_selected = None
        else:
            has_matches = False

        # Classification distillation
        if has_matches and self.distill_class and 'class' in student_outputs and 'class' in teacher_outputs:
            if self.distill_mode == 'matched':
                # Extract matched queries for each sample
                student_class_list = []
                teacher_class_list = []

                for b in range(B):
                    if b < len(student_selected) and len(student_selected[b]) > 0:
                        student_class_list.append(student_outputs['class'][b, student_selected[b]])
                        teacher_class_list.append(teacher_outputs['class'][b, teacher_selected[b]])

                if student_class_list:
                    student_class = torch.cat(student_class_list, dim=0)  # (total_matched, num_classes)
                    teacher_class = torch.cat(teacher_class_list, dim=0)

                    class_loss = self.distillation_loss_class(student_class, teacher_class)
                    losses['distill_class'] = class_loss
            else:
                # 'all' mode: distill min(N_student, N_teacher) queries
                N_distill = min(N_student, N_teacher)
                student_class = student_outputs['class'][:, :N_distill]
                teacher_class = teacher_outputs['class'][:, :N_distill]

                class_loss = self.distillation_loss_class(student_class, teacher_class)
                losses['distill_class'] = class_loss

        # Bounding box distillation
        if has_matches and self.distill_bbox:
            bbox_keys = ['center', 'size', 'angle']

            # Check if all bbox components are present
            if all(k in student_outputs and k in teacher_outputs for k in bbox_keys):
                if self.distill_mode == 'matched':
                    # Extract matched queries for each sample
                    student_bbox_list = []
                    teacher_bbox_list = []

                    for b in range(B):
                        if b < len(student_selected) and len(student_selected[b]) > 0:
                            student_bbox_b = torch.cat([student_outputs[k][b, student_selected[b]] for k in bbox_keys], dim=-1)
                            teacher_bbox_b = torch.cat([teacher_outputs[k][b, teacher_selected[b]] for k in bbox_keys], dim=-1)
                            student_bbox_list.append(student_bbox_b)
                            teacher_bbox_list.append(teacher_bbox_b)

                    if student_bbox_list:
                        student_bbox = torch.cat(student_bbox_list, dim=0)  # (total_matched, 8)
                        teacher_bbox = torch.cat(teacher_bbox_list, dim=0)

                        bbox_loss = self.distillation_loss_bbox(student_bbox, teacher_bbox)
                        losses['distill_bbox'] = bbox_loss
                else:
                    # 'all' mode: distill min(N_student, N_teacher) queries
                    N_distill = min(N_student, N_teacher)
                    student_bbox = torch.cat([student_outputs[k][:, :N_distill] for k in bbox_keys], dim=-1)
                    teacher_bbox = torch.cat([teacher_outputs[k][:, :N_distill] for k in bbox_keys], dim=-1)

                    bbox_loss = self.distillation_loss_bbox(student_bbox, teacher_bbox)
                    losses['distill_bbox'] = bbox_loss

        # Apply reduction
        if self.reduction == 'mean':
            losses = {k: v.mean() for k, v in losses.items()}
        elif self.reduction == 'sum':
            losses = {k: v.sum() for k, v in losses.items()}

        # Compute total loss
        if losses:
            total_loss = sum(losses.values())
        else:
            # Fallback if no losses computed
            device = list(student_outputs.values())[0].device
            total_loss = torch.tensor(0.0, device=device, requires_grad=True)

        return total_loss, losses


class KDLoss(nn.modules.loss._Loss):
    def __init__(self,
                 student_loss: nn.Module,
                 distillation_loss: DistillationLoss,
                 alpha: float = 0.5,
                 **kwargs):
        """Combined Knowledge Distillation and Student Loss.

        This loss combines the standard detection loss (student loss) with
        the distillation loss from the teacher model.

        Arguments:
            student_loss: The standard detection loss (e.g., Loss module)
            distillation_loss: The distillation loss module (DistillationLoss)
            alpha: Weight for distillation loss.
                total_loss = alpha * distill_loss + (1 - alpha) * student_loss
                Default: 0.5
        """
        super().__init__(**kwargs)

        self.student_loss = student_loss
        self.distillation_loss = distillation_loss
        self.alpha = alpha

        if not (0.0 <= self.alpha <= 1.0):
            raise ValueError(
                f"Invalid value for arg 'alpha': {self.alpha}"
                f"\n Alpha must be in range [0.0, 1.0]"
            )

    def forward(self,
                student_outputs: Dict[str, torch.Tensor],
                teacher_outputs: Dict[str, torch.Tensor],
                targets: List[Dict[str, torch.Tensor]]) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """Compute combined distillation and student loss.

        Arguments:
            student_outputs: Student model outputs dictionary
            teacher_outputs: Teacher model outputs dictionary (detached from computation graph)
            targets: Ground truth labels

        Returns:
            total_loss: Combined total loss
            losses: Dictionary containing all loss components
        """
        # Compute student loss (detection loss with ground truth)
        student_total_loss, student_losses = self.student_loss(student_outputs, targets)

        # Get matching indices if distillation uses 'matched' mode
        indices = None
        if self.distillation_loss.distill_mode == 'matched':
            # Get the anassigner from student_loss (if available)
            if hasattr(self.student_loss, 'anassigner') and self.student_loss.anassigner is not None:
                # Need to process per-sample since targets is a list
                from dprt.utils.data import decollate_batch

                # Decollate inputs
                inputs_list = decollate_batch(student_outputs, detach=False, pad=False)

                # Collect all indices for the batch
                batch_student_pred_indices = []
                batch_gt_indices = []
                batch_teacher_pred_indices = []

                for sample_idx, (input_dict, target) in enumerate(zip(inputs_list, targets)):
                    # Skip if no targets
                    if not all([t.numel() for t in target.values()]):
                        # Empty target - add empty indices
                        empty_idx = torch.tensor([], dtype=torch.long, device=input_dict['class'].device)
                        batch_student_pred_indices.append(empty_idx)
                        batch_gt_indices.append(torch.tensor([], dtype=torch.long, device=input_dict['class'].device))
                        batch_teacher_pred_indices.append(empty_idx)
                        continue

                    # Insert dummy batch dimension for anassigner
                    input_single = {k: v.unsqueeze(0) for k, v in input_dict.items()}
                    target_single = {k: v.unsqueeze(0) for k, v in target.items()}

                    teacher_input_single = {
                        k: teacher_outputs[k][sample_idx].unsqueeze(0) for k in ('class', 'center', 'size', 'angle')
                    }

                    # Get matching indices for student and teacher separately
                    student_pred_idx, student_gt_idx = self.student_loss.anassigner(input_single, target_single)
                    teacher_pred_idx, teacher_gt_idx = self.student_loss.anassigner(teacher_input_single, target_single)

                    student_pred_idx = student_pred_idx[0]
                    student_gt_idx = student_gt_idx[0]
                    teacher_pred_idx = teacher_pred_idx[0]
                    teacher_gt_idx = teacher_gt_idx[0]

                    # Align teacher indices to student GT order to ensure GT-aligned distillation
                    teacher_pred_idx_list = teacher_pred_idx.detach().cpu().tolist()
                    teacher_gt_idx_list = teacher_gt_idx.detach().cpu().tolist()
                    student_gt_idx_list = student_gt_idx.detach().cpu().tolist()

                    teacher_idx_by_gt = {
                        gt: pred for pred, gt in zip(teacher_pred_idx_list, teacher_gt_idx_list)
                    }
                    common_positions = [
                        pos for pos, gt in enumerate(student_gt_idx_list) if gt in teacher_idx_by_gt
                    ]

                    if common_positions:
                        aligned_student_pred_idx = student_pred_idx[common_positions]
                        aligned_gt_idx = student_gt_idx[common_positions]
                        aligned_teacher_pred_idx = torch.tensor(
                            [teacher_idx_by_gt[int(gt.item())] for gt in aligned_gt_idx],
                            dtype=torch.long,
                            device=aligned_student_pred_idx.device,
                        )
                    else:
                        aligned_student_pred_idx = torch.tensor([], dtype=torch.long, device=student_pred_idx.device)
                        aligned_gt_idx = torch.tensor([], dtype=torch.long, device=student_gt_idx.device)
                        aligned_teacher_pred_idx = torch.tensor([], dtype=torch.long, device=teacher_pred_idx.device)

                    batch_student_pred_indices.append(aligned_student_pred_idx)
                    batch_gt_indices.append(aligned_gt_idx)
                    batch_teacher_pred_indices.append(aligned_teacher_pred_idx)

                # Stack into batch format
                indices = (batch_student_pred_indices, batch_gt_indices, batch_teacher_pred_indices)
            else:
                # Fallback: warn and use 'all' mode
                import warnings
                warnings.warn(
                    "distill_mode='matched' but student_loss has no anassigner. "
                    "Falling back to 'all' mode for distillation.",
                    RuntimeWarning
                )

        # Compute distillation loss (soft targets from teacher)
        distill_total_loss, distill_losses = self.distillation_loss(
            student_outputs, teacher_outputs, indices=indices
        )

        # Combine losses with alpha weighting
        total_loss = self.alpha * distill_total_loss + (1.0 - self.alpha) * student_total_loss

        # Combine loss dictionaries
        all_losses = {
            'total_loss': total_loss,
            'student_loss': student_total_loss,
            'distill_loss': distill_total_loss,
            **{f'student_{k}': v for k, v in student_losses.items()},
            **distill_losses
        }

        return total_loss, all_losses


def build_loss(*args, **kwargs):
    return Loss.from_config(*args, **kwargs)

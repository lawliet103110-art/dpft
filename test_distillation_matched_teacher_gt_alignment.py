import torch

from dprt.training.loss import DistillationLoss


def test_matched_mode_uses_teacher_indices_aligned_to_gt():
    loss_fn = DistillationLoss(
        temperature=1.0,
        distill_class=True,
        distill_bbox=False,
        distill_mode='matched',
        reduction='mean',
    )

    student_outputs = {
        'class': torch.tensor([[[2.0, -2.0], [-2.0, 2.0]]], dtype=torch.float32),
        'center': torch.zeros((1, 2, 3), dtype=torch.float32),
        'size': torch.zeros((1, 2, 3), dtype=torch.float32),
        'angle': torch.zeros((1, 2, 2), dtype=torch.float32),
    }
    teacher_outputs = {
        'class': torch.tensor([[[-2.0, 2.0], [2.0, -2.0]]], dtype=torch.float32),
        'center': torch.zeros((1, 2, 3), dtype=torch.float32),
        'size': torch.zeros((1, 2, 3), dtype=torch.float32),
        'angle': torch.zeros((1, 2, 2), dtype=torch.float32),
    }

    # Student->GT: student query 1 matches GT 0
    student_pred_indices = [torch.tensor([1], dtype=torch.long)]
    gt_indices = [torch.tensor([0], dtype=torch.long)]
    # Teacher->GT: teacher query 0 matches GT 0 (different from student query)
    teacher_pred_indices = [torch.tensor([0], dtype=torch.long)]

    total_loss, losses = loss_fn(
        student_outputs,
        teacher_outputs,
        indices=(student_pred_indices, gt_indices, teacher_pred_indices),
    )
    fallback_total_loss, _ = loss_fn(
        student_outputs,
        teacher_outputs,
        indices=(student_pred_indices, gt_indices),
    )

    assert 'distill_class' in losses
    assert fallback_total_loss > 0
    assert torch.isclose(total_loss, torch.tensor(0.0), atol=1e-6)


if __name__ == "__main__":
    test_matched_mode_uses_teacher_indices_aligned_to_gt()
    print("OK")

"""Regression check for teacher Hungarian alignment in matched KD mode.

This is a source-level guard to ensure matched mode keeps supporting
teacher-specific assignment indices aligned by GT order.
"""

from pathlib import Path


def test_matched_mode_supports_teacher_aligned_indices() -> None:
    loss_file = Path(__file__).resolve().parent / "src" / "dprt" / "training" / "loss.py"
    text = loss_file.read_text(encoding="utf-8")

    assert "if len(indices) == 3:" in text
    assert "teacher_pred_indices" in text
    assert "indices = (batch_student_pred_indices, batch_gt_indices, batch_teacher_pred_indices)" in text
    assert "Align teacher indices to student GT order" in text


if __name__ == "__main__":
    test_matched_mode_supports_teacher_aligned_indices()
    print("matched teacher Hungarian alignment source-level test passed.")

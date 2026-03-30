"""Regression check: top_k should be equivalent to all when K covers all queries."""

from pathlib import Path


def test_topk_full_k_uses_all_path() -> None:
    loss_file = Path(__file__).resolve().parent / "src" / "dprt" / "training" / "loss.py"
    text = loss_file.read_text(encoding="utf-8")

    assert "topk_equivalent_all = False" in text
    assert "if effective_k == N_distill:" in text
    assert "topk_equivalent_all = True" in text
    assert "if topk_equivalent_all:" in text
    assert "student_outputs['class'][:, :N_distill]" in text
    assert "teacher_outputs['class'][:, :N_distill]" in text


if __name__ == "__main__":
    test_topk_full_k_uses_all_path()
    print("top_k==all source-level test passed.")


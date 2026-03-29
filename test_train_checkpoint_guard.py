"""Focused regression test for KD teacher-checkpoint guard in train.py.

This repository does not guarantee a full runtime test environment (e.g., torch),
so this test validates the safeguard at source level.
"""

from pathlib import Path


def test_teacher_checkpoint_guard_exists_in_train_main() -> None:
    train_file = Path("/home/runner/work/dpft/dpft/src/dprt/train.py")
    content = train_file.read_text(encoding="utf-8")

    # Guard condition: detect accidental use of teacher checkpoint as resume checkpoint.
    assert "teacher_checkpoint = distill_cfg.get('teacher_checkpoint')" in content
    assert "osp.abspath(checkpoint) == osp.abspath(teacher_checkpoint)" in content
    assert "Treating it as teacher-only checkpoint and starting student training from epoch 0." in content


if __name__ == "__main__":
    test_teacher_checkpoint_guard_exists_in_train_main()
    print("train checkpoint guard source-level test passed.")

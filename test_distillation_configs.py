"""Validate distillation experiment config files.

Focused checks for:
1) Top-k sweep configs (K from small to large)
2) Ablation configs (only cls KD / only bbox KD)
"""

import json
from pathlib import Path


def load_config(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def test_topk_sweep_configs(config_dir: Path) -> None:
    expected = {
        "kradar_distillation_topk_k10.json": 10,
        "kradar_distillation_topk_k25.json": 25,
        "kradar_distillation_topk_k50.json": 50,
        "kradar_distillation_topk_k100.json": 100,
        "kradar_distillation_topk_k200.json": 200,
    }

    for filename, k in expected.items():
        cfg = load_config(config_dir / filename)
        dist = cfg["train"]["distillation"]
        assert dist["distill_mode"] == "top_k", f"{filename}: distill_mode != top_k"
        assert dist["top_k"] == k, f"{filename}: top_k != {k}"


def test_kd_ablation_configs(config_dir: Path) -> None:
    only_cls = load_config(config_dir / "kradar_distillation_only_cls_kd.json")
    dist_cls = only_cls["train"]["distillation"]
    assert dist_cls["distill_class"] is True
    assert dist_cls["distill_bbox"] is False

    only_bbox = load_config(config_dir / "kradar_distillation_only_bbox_kd.json")
    dist_bbox = only_bbox["train"]["distillation"]
    assert dist_bbox["distill_class"] is False
    assert dist_bbox["distill_bbox"] is True


if __name__ == "__main__":
    root = Path(__file__).resolve().parent
    config_dir = root / "config"
    test_topk_sweep_configs(config_dir)
    test_kd_ablation_configs(config_dir)
    print("Distillation experiment configs validated successfully.")

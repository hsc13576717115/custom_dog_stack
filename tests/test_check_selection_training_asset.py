from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
PATH = ROOT / "scripts/check_selection_training_asset.py"
SPEC = importlib.util.spec_from_file_location("check_selection_training_asset", PATH)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def create_selection(tmp_path: Path, asset: str, accepted: bool = True) -> Path:
    selection = tmp_path / "run/evaluation/selection.json"
    selection.parent.mkdir(parents=True)
    selection.write_text(json.dumps({"accepted": accepted}), encoding="utf-8")
    params = tmp_path / "run/params/env.yaml"
    params.parent.mkdir(parents=True)
    params.write_text(
        "scene:\n  robot:\n    spawn:\n      asset_path: " + asset + "\n",
        encoding="utf-8",
    )
    return selection


def test_expected_training_asset_is_accepted(tmp_path: Path) -> None:
    selection = create_selection(tmp_path, "/assets/custom_dog_selective_collision.urdf")
    result = MODULE.verify(selection, "custom_dog_selective_collision.urdf")
    assert result["accepted"] is True


def test_stale_or_unaccepted_selection_is_rejected(tmp_path: Path) -> None:
    stale = create_selection(tmp_path, "/assets/custom_dog.urdf")
    with pytest.raises(ValueError, match="expected"):
        MODULE.verify(stale, "custom_dog_selective_collision.urdf")
    stale.write_text(json.dumps({"accepted": False}), encoding="utf-8")
    with pytest.raises(ValueError, match="not accepted"):
        MODULE.verify(stale, "custom_dog_selective_collision.urdf")


def test_every_post_collision_gate_calls_asset_checker() -> None:
    scripts = (
        "train_next_closed_loop_stage_gated.sh",
        "train_gait_robust_stage_gated.sh",
        "train_dynamics_teacher_stage_gated.sh",
        "train_next_self_righting_stage_gated.sh",
        "train_next_terrain_stage_gated.sh",
        "train_history213_distillation_gated.sh",
    )
    for name in scripts:
        source = (ROOT / "scripts" / name).read_text(encoding="utf-8")
        assert "check_selection_training_asset.py" in source
        assert "custom_dog_selective_collision.urdf" in source

from __future__ import annotations

import importlib.util
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PATH = ROOT / "scripts/audit_gated_pipeline.py"
SPEC = importlib.util.spec_from_file_location("audit_gated_pipeline", PATH)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def write_json(path: Path, value: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")


def test_accepted_selection_requires_both_acceptance_and_prefix(tmp_path: Path) -> None:
    log_root = tmp_path / "logs"
    rejected = log_root / "run1/evaluation/selection.json"
    write_json(rejected, {"accepted": True, "selected": "source"})
    result = MODULE.accepted_selection(
        log_root, "*/evaluation/selection.json", prefix="B_"
    )
    assert result["status"] == "failed"

    accepted = log_root / "run2/evaluation/selection.json"
    write_json(
        accepted,
        {"accepted": True, "selected": "B_500", "selected_candidate": "/candidate"},
    )
    result = MODULE.accepted_selection(
        log_root, "*/evaluation/selection.json", prefix="B_"
    )
    assert result["status"] == "complete"
    assert result["selected"] == "B_500"


def test_post_collision_selection_requires_cleaned_asset_snapshot(tmp_path: Path) -> None:
    log_root = tmp_path / "logs"
    selection = log_root / "run/evaluation/selection.json"
    write_json(
        selection,
        {"accepted": True, "selected": "B_500", "selected_candidate": "/candidate"},
    )
    env = selection.parent.parent / "params/env.yaml"
    env.parent.mkdir(parents=True, exist_ok=True)
    env.write_text(
        "scene:\n  robot:\n    spawn:\n      asset_path: /assets/custom_dog.urdf\n",
        encoding="utf-8",
    )
    result = MODULE.accepted_selection(
        log_root,
        "*/evaluation/selection.json",
        prefix="B_",
        required_asset="custom_dog_selective_collision.urdf",
    )
    assert result["status"] == "stale_asset"
    assert result["accepted"] is False

    env.write_text(
        "scene:\n  robot:\n    spawn:\n"
        "      asset_path: /assets/custom_dog_selective_collision.urdf\n",
        encoding="utf-8",
    )
    result = MODULE.accepted_selection(
        log_root,
        "*/evaluation/selection.json",
        prefix="B_",
        required_asset="custom_dog_selective_collision.urdf",
    )
    assert result["status"] == "complete"


def test_second_seed_is_training_until_final_checkpoint_and_gate(tmp_path: Path) -> None:
    log_root = tmp_path / "logs"
    run = log_root / "date_closed_loop_robust_foundation_seed73"
    run.mkdir(parents=True)
    (run / "model_300.pt").touch()
    result = MODULE.second_seed(log_root)
    assert result["status"] == "training"
    assert result["latest_checkpoint"] == 300

    (run / "model_999.pt").touch()
    result = MODULE.second_seed(log_root)
    assert result["status"] == "evaluating"
    write_json(
        run / "evaluation/routed_robust_foundation_selection.json",
        {"accepted": True, "selected": "RFR_700", "selected_candidate": "/routed"},
    )
    assert MODULE.second_seed(log_root)["status"] == "complete"


def test_requirements_do_not_treat_manual_ready_as_automatic_completion(tmp_path: Path) -> None:
    source = PATH.read_text(encoding="utf-8")
    assert "required = list(stages)" in source
    assert 'stages[name]["status"] == "complete"' in source
    assert "pending_revalidation" in source
    assert MODULE.keyboard_review(tmp_path)["status"] == "ready_for_user"
    write_json(
        tmp_path / "reports/keyboard_sim2sim_review.json",
        {
            "accepted": True,
            "reviewed_candidates": ["model_975", "model_999"],
            "notes": "Keyboard review completed.",
        },
    )
    assert MODULE.keyboard_review(tmp_path)["status"] == "complete"


def test_keyboard_review_candidates_follow_durable_ranking(tmp_path: Path) -> None:
    write_json(
        tmp_path / "reports/stage_a_six_checkpoint_ranking.json",
        {
            "top_candidates": [
                {"label": "model_999"},
                {"label": "model_975"},
                {"label": "model_875"},
                {"label": "model_900"},
            ]
        },
    )
    result = MODULE.keyboard_review(tmp_path)
    assert result["status"] == "ready_for_user"
    assert result["candidates"] == ["model_875", "model_975", "model_999"]

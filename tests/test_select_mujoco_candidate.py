from __future__ import annotations

import importlib.util
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PATH = ROOT / "scripts" / "select_mujoco_candidate.py"
SPEC = importlib.util.spec_from_file_location("select_mujoco_candidate", PATH)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def row(candidate: str, *, vx_error: float, gates: dict[str, bool]) -> dict[str, object]:
    return {
        "candidate": candidate,
        "command_index": 0,
        "command_vx": 0.5,
        "command_vy": 0.0,
        "command_wz": 0.0,
        "error_vx": vx_error,
        "error_vy": 0.01,
        "error_wz": 0.01,
        "max_tilt_deg": 2.0,
        "hip_outward_max_deg": 10.0,
        "absolute_gates": gates,
        "relative_gates": {"action_delta": True},
    }


def test_selector_prefers_all_gate_pass_over_lower_tracking_error() -> None:
    evaluation = {
        "stage": "A",
        "rows": [
            row("unsafe_low_error", vx_error=0.01, gates={"not_fallen": False}),
            row("accepted", vx_error=0.08, gates={"not_fallen": True}),
        ],
    }
    result = MODULE.select(evaluation, top=2)
    assert result["accepted"]
    assert result["selected"] == "accepted"


def test_selector_reports_best_rejected_candidate_without_claiming_acceptance() -> None:
    evaluation = {
        "stage": "A",
        "rows": [
            row("two_failures", vx_error=0.01, gates={"height": False, "tilt": False}),
            row("one_nonsafety_failure", vx_error=0.03, gates={"wz": False}),
        ],
    }
    result = MODULE.select(evaluation, top=2)
    assert not result["accepted"]
    assert result["selected"] == "one_nonsafety_failure"


def test_selector_can_keep_baseline_for_comparison_but_only_rank_stage_candidates() -> None:
    evaluation = {
        "stage": "A",
        "rows": [
            row("source", vx_error=0.01, gates={"not_fallen": True}),
            row("SC_100", vx_error=0.04, gates={"not_fallen": True}),
            row("SC_200", vx_error=0.03, gates={"not_fallen": True}),
        ],
    }
    result = MODULE.select(evaluation, top=3, label_prefix="SC_")
    assert result["accepted"]
    assert result["selected"] == "SC_200"
    assert all(item["label"].startswith("SC_") for item in result["top_candidates"])


def test_selector_rejects_prefix_without_candidates() -> None:
    evaluation = {
        "stage": "A",
        "rows": [row("source", vx_error=0.01, gates={"not_fallen": True})],
    }
    try:
        MODULE.select(evaluation, top=3, label_prefix="SC_")
    except ValueError as error:
        assert "SC_" in str(error)
    else:
        raise AssertionError("missing stage candidates must not silently select the baseline")

from __future__ import annotations

import importlib.util
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PATH = ROOT / "scripts/select_self_righting_candidate.py"
SPEC = importlib.util.spec_from_file_location("select_self_righting_candidate", PATH)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def result(candidate: str, times: list[float | None]) -> dict[str, object]:
    return {
        "candidate": candidate,
        "passes_all": all(time is not None for time in times),
        "states_passed": sum(time is not None for time in times),
        "total_states": len(times),
        "results": [{"success_time_s": time} for time in times],
    }


def test_all_pose_success_precedes_faster_partial_candidate() -> None:
    selected = MODULE.select(
        [
            ("partial", result("partial", [0.5, 0.5, 0.5, None])),
            ("complete", result("complete", [2.0, 2.5, 2.2, 2.1])),
        ]
    )
    assert selected["accepted"]
    assert selected["selected"] == "complete"


def test_complete_candidates_rank_by_worst_recovery_time() -> None:
    selected = MODULE.select(
        [
            ("slow_tail", result("slow_tail", [1.0, 1.0, 1.0, 3.0])),
            ("balanced", result("balanced", [1.8, 1.8, 1.8, 1.8])),
        ]
    )
    assert selected["selected"] == "balanced"

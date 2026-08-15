from __future__ import annotations

import importlib.util
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PATH = ROOT / "scripts/select_terrain_candidate.py"
SPEC = importlib.util.spec_from_file_location("select_terrain_candidate", PATH)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def flat_row(label: str, passed: bool = True) -> dict[str, object]:
    return {
        "candidate": label,
        "candidate_path": f"/{label}",
        "command_index": 0,
        "command_vx": 0.5,
        "command_vy": 0.0,
        "command_wz": 0.0,
        "error_vx": 0.03,
        "error_vy": 0.01,
        "error_wz": 0.01,
        "max_tilt_deg": 3.0,
        "hip_outward_max_deg": 10.0,
        "absolute_gates": {"not_fallen": passed},
        "relative_gates": {},
    }


def terrain(label: str, passed: bool, success: float) -> tuple[str, dict[str, object]]:
    return (
        label,
        {
            "stage": "T0",
            "passes_all": passed,
            "commands_passed": 1 if passed else 0,
            "total_commands": 1,
            "rows": [{"success_rate": success}],
            "terrain_families_passed": 1 if passed else 0,
            "total_terrain_families": 1,
            "family_commands_passed": 1 if passed else 0,
            "total_family_commands": 1,
            "terrain_families": [{"success_rate": success}],
            "family_commands": [{"success_rate": success}],
        },
    )


def test_both_flat_and_terrain_must_pass() -> None:
    result = MODULE.select(
        {"rows": [flat_row("T0_100"), flat_row("T0_200")]},
        [terrain("T0_100", False, 0.90), terrain("T0_200", True, 0.98)],
    )
    assert result["accepted"]
    assert result["selected"] == "T0_200"


def test_label_mismatch_is_rejected() -> None:
    try:
        MODULE.select({"rows": [flat_row("T0_100")]}, [terrain("T0_200", True, 1.0)])
    except ValueError as error:
        assert "labels differ" in str(error)
    else:
        raise AssertionError("mismatched evidence must not be silently combined")


def test_flat_source_baseline_can_be_present_without_terrain_result() -> None:
    result = MODULE.select(
        {"rows": [flat_row("source"), flat_row("T0_100")]},
        [terrain("T0_100", True, 0.97)],
    )
    assert result["accepted"]
    assert result["selected"] == "T0_100"


def test_failed_family_command_cell_blocks_acceptance() -> None:
    label, evidence = terrain("T0_100", True, 0.99)
    evidence["passes_all"] = False
    evidence["family_commands_passed"] = 5
    evidence["total_family_commands"] = 6
    result = MODULE.select({"rows": [flat_row(label)]}, [(label, evidence)])
    assert not result["accepted"]
    assert result["ranked"][0]["terrain_family_command_failures"] == 1

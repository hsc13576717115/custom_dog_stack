from __future__ import annotations

import importlib.util
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT_DIR = ROOT / "scripts"
MODULE_PATH = SCRIPT_DIR / "select_stand_bias_candidate.py"
sys.path.insert(0, str(SCRIPT_DIR))
SPEC = importlib.util.spec_from_file_location("select_stand_bias_candidate", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def _row(label: str, accepted: bool, hip: float) -> dict[str, object]:
    return {
        "candidate": label,
        "candidate_path": f"/tmp/{label}",
        "command_index": 0,
        "command_vx": 0.0,
        "command_vy": 0.0,
        "command_wz": 0.0,
        "error_vx": 0.0,
        "error_vy": 0.0,
        "error_wz": 0.0,
        "max_tilt_deg": 2.0,
        "hip_outward_max_deg": hip,
        "absolute_gates": {"standing": accepted},
        "relative_gates": {},
    }


def test_selector_uses_smallest_bias_that_passes() -> None:
    result = MODULE.select(
        {
            "stage": "A",
            "rows": [
                _row("STAND_BIAS_20", False, 13.0),
                _row("STAND_BIAS_40", True, 11.5),
                _row("STAND_BIAS_60", True, 10.0),
            ],
        }
    )
    assert result["accepted"] is True
    assert result["selected"] == "STAND_BIAS_40"
    assert result["selected_candidate"] == "/tmp/STAND_BIAS_40"

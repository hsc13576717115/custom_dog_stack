from __future__ import annotations

import importlib.util
import sys
from pathlib import Path


MODULE_PATH = Path(__file__).resolve().parents[1] / "scripts/evaluate_mujoco_grid.py"
SPEC = importlib.util.spec_from_file_location("custom_dog_evaluate_mujoco_grid", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def test_parse_args_accepts_repeated_custom_commands(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setattr(
        sys,
        "argv",
        [
            str(MODULE_PATH),
            "--candidate",
            f"baseline={tmp_path}",
            "--baseline-label",
            "baseline",
            "--command",
            "1.0",
            "0.0",
            "0.2",
            "--command",
            "1.5",
            "0.0",
            "-0.3",
            "--output-csv",
            str(tmp_path / "metrics.csv"),
            "--output-json",
            str(tmp_path / "metrics.json"),
        ],
    )

    args = MODULE.parse_args()
    assert args.command == [[1.0, 0.0, 0.2], [1.5, 0.0, -0.3]]


def test_high_speed_vx_gate_uses_relative_tolerance() -> None:
    row = {
        "command_vx": 1.5,
        "error_vx": 0.20,
        "error_vy": 0.05,
        "error_wz": 0.08,
        "min_height_m": 0.25,
        "max_tilt_deg": 8.0,
    }
    assert all(MODULE.absolute_gates(row).values())

    row["error_vx"] = 0.23
    assert not MODULE.absolute_gates(row)["vx"]

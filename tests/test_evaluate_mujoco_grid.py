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


def test_pure_axis_gates_include_decoupling_and_standing_metrics() -> None:
    row = {
        "command_vx": 0.0,
        "command_vy": 0.0,
        "command_wz": 0.2,
        "error_vx": 0.01,
        "error_vy": 0.01,
        "error_wz": 0.02,
        "measured_vx": 0.02,
        "measured_vy": 0.03,
        "measured_wz": 0.20,
        "min_height_m": 0.30,
        "mean_height_m": 0.31,
        "max_tilt_deg": 6.0,
        "world_displacement_m": 0.4,
        "yaw_integral_bias_rad": 0.3,
        "duration_s": 10.0,
        "warmup_s": 2.0,
        "contact_transitions_per_leg": [4, 5, 4, 5],
    }
    gates = MODULE.absolute_gates(row)
    assert gates["pure_yaw_xy_drift"]
    assert gates["pure_yaw_integral_bias_rate"]
    assert gates["gait_transitions"]

    row["world_displacement_m"] = 0.41
    assert not MODULE.absolute_gates(row)["pure_yaw_xy_drift"]

    row["world_displacement_m"] = 0.4
    row["yaw_integral_bias_rad"] = -0.41
    assert not MODULE.absolute_gates(row)["pure_yaw_integral_bias_rate"]


def test_zero_command_gate_checks_standing_height_and_tilt() -> None:
    row = {
        "command_vx": 0.0,
        "command_vy": 0.0,
        "command_wz": 0.0,
        "error_vx": 0.0,
        "error_vy": 0.0,
        "error_wz": 0.0,
        "measured_vx": 0.0,
        "measured_vy": 0.0,
        "measured_wz": 0.0,
        "min_height_m": 0.30,
        "mean_height_m": 0.33,
        "max_tilt_deg": 3.0,
        "hip_outward_max_deg": 12.0,
        "world_displacement_m": 0.0,
    }
    gates = MODULE.absolute_gates(row)
    assert gates["standing_height"] and gates["standing_tilt"]

    row["mean_height_m"] = 0.336
    assert not MODULE.absolute_gates(row)["standing_height"]

    row["mean_height_m"] = 0.309
    assert not MODULE.absolute_gates(row)["standing_height"]

    row["mean_height_m"] = 0.33
    row["max_tilt_deg"] = 3.01
    assert not MODULE.absolute_gates(row)["standing_tilt"]

    row["max_tilt_deg"] = 3.0
    row["hip_outward_max_deg"] = 12.01
    assert not MODULE.absolute_gates(row)["hip_outward"]


def test_low_speed_motion_requires_each_leg_to_step() -> None:
    row = {
        "command_vx": 0.05,
        "command_vy": 0.0,
        "command_wz": 0.0,
        "error_vx": 0.01,
        "error_vy": 0.01,
        "error_wz": 0.01,
        "measured_vx": 0.04,
        "measured_vy": 0.0,
        "measured_wz": 0.0,
        "min_height_m": 0.30,
        "mean_height_m": 0.32,
        "max_tilt_deg": 2.0,
        "hip_outward_max_deg": 18.0,
        "world_displacement_m": 0.3,
        "duration_s": 10.0,
        "warmup_s": 2.0,
        "contact_transitions_per_leg": [2, 3, 2, 3],
    }
    assert MODULE.absolute_gates(row)["gait_transitions"]

    row["contact_transitions_per_leg"] = [2, 3, 1, 3]
    assert not MODULE.absolute_gates(row)["gait_transitions"]


def test_lateral_and_yaw_commands_allow_more_hip_abduction() -> None:
    row = {
        "command_vx": 0.0,
        "command_vy": 0.0,
        "command_wz": 0.1,
        "error_vx": 0.01,
        "error_vy": 0.01,
        "error_wz": 0.01,
        "measured_vx": 0.0,
        "measured_vy": 0.0,
        "measured_wz": 0.1,
        "min_height_m": 0.30,
        "mean_height_m": 0.32,
        "max_tilt_deg": 2.0,
        "hip_outward_max_deg": 25.0,
        "world_displacement_m": 0.1,
        "yaw_integral_bias_rad": 0.1,
        "duration_s": 10.0,
        "warmup_s": 2.0,
        "contact_transitions_per_leg": [2, 2, 2, 2],
    }
    assert MODULE.absolute_gates(row)["hip_outward"]

    row["hip_outward_max_deg"] = 25.01
    assert not MODULE.absolute_gates(row)["hip_outward"]


def test_csv_fieldnames_include_command_specific_gate_columns() -> None:
    rows = [
        {"candidate": "standing", "absolute_gates/standing_height": True},
        {"candidate": "forward", "absolute_gates/pure_vx_decoupled": True},
        {"candidate": "yaw", "absolute_gates/pure_yaw_xy_drift": False},
    ]

    assert MODULE.csv_fieldnames(rows) == [
        "candidate",
        "absolute_gates/standing_height",
        "absolute_gates/pure_vx_decoupled",
        "absolute_gates/pure_yaw_xy_drift",
    ]


def test_stage_grids_are_cumulative_and_reach_requested_envelopes() -> None:
    assert set(MODULE.STAGE_GRIDS["A"]).issubset(MODULE.STAGE_GRIDS["B"])
    assert set(MODULE.STAGE_GRIDS["A"]).issubset(MODULE.STAGE_GRIDS["C"])
    assert set(MODULE.STAGE_GRIDS["A"]).issubset(MODULE.STAGE_GRIDS["D"])
    assert (0.8, 0.0, 0.0) in MODULE.STAGE_GRIDS["B"]
    assert (1.5, 0.0, 0.0) in MODULE.STAGE_GRIDS["C"]
    assert (3.0, 0.0, 0.0) in MODULE.STAGE_GRIDS["D"]
    assert (0.0, 0.6, 0.0) in MODULE.STAGE_GRIDS["D"]
    assert (0.0, 0.0, 2.0) in MODULE.STAGE_GRIDS["D"]


def test_speed_adaptive_height_gate_tracks_033_to_028_band() -> None:
    row = {
        "command_vx": 3.0,
        "command_vy": 0.0,
        "command_wz": 0.0,
        "error_vx": 0.0,
        "error_vy": 0.0,
        "error_wz": 0.0,
        "measured_vx": 3.0,
        "measured_vy": 0.0,
        "measured_wz": 0.0,
        "min_height_m": 0.25,
        "mean_height_m": 0.28,
        "max_tilt_deg": 5.0,
        "world_displacement_m": 20.0,
        "duration_s": 10.0,
        "warmup_s": 2.0,
        "contact_transitions_per_leg": [10, 10, 10, 10],
    }
    assert MODULE.absolute_gates(row)["body_height_target"]
    row["mean_height_m"] = 0.32
    assert not MODULE.absolute_gates(row)["body_height_target"]


def test_self_collision_gate_allows_at_most_one_percent_contact_steps() -> None:
    row = {
        "command_vx": 0.0,
        "command_vy": 0.0,
        "command_wz": 0.1,
        "error_vx": 0.0,
        "error_vy": 0.0,
        "error_wz": 0.0,
        "measured_vx": 0.0,
        "measured_vy": 0.0,
        "measured_wz": 0.1,
        "min_height_m": 0.30,
        "mean_height_m": 0.33,
        "max_tilt_deg": 2.0,
        "world_displacement_m": 0.0,
        "duration_s": 10.0,
        "warmup_s": 2.0,
        "self_collision_contact_steps": 8,
        "self_collision_sample_steps": 800,
    }
    assert MODULE.absolute_gates(row)["self_collision"]
    row["self_collision_contact_steps"] = 9
    assert not MODULE.absolute_gates(row)["self_collision"]


def test_illegal_ground_contact_gate_allows_at_most_one_percent_steps() -> None:
    row = {
        "command_vx": 0.0,
        "command_vy": 0.0,
        "command_wz": 0.0,
        "error_vx": 0.0,
        "error_vy": 0.0,
        "error_wz": 0.0,
        "measured_vx": 0.0,
        "measured_vy": 0.0,
        "measured_wz": 0.0,
        "min_height_m": 0.31,
        "mean_height_m": 0.32,
        "max_tilt_deg": 2.0,
        "world_displacement_m": 0.0,
        "illegal_ground_contact_steps": 8,
        "illegal_ground_sample_steps": 800,
    }
    assert MODULE.absolute_gates(row)["illegal_ground_contact"]
    row["illegal_ground_contact_steps"] = 9
    assert not MODULE.absolute_gates(row)["illegal_ground_contact"]

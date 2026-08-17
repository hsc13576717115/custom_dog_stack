#!/usr/bin/env python3
"""Evaluate exported policies on a reproducible fixed-command MuJoCo grid."""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import subprocess
import sys
from pathlib import Path

import numpy as np


NUMBER = r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?"
TRACKING_RE = re.compile(
    rf"tracking: command=\[([^]]+)\], policy_command=\[([^]]+)\], "
    rf"measured=\[([^]]+)\], abs_error=\[([^]]+)\]"
)
STABILITY_RE = re.compile(
    rf"stability: min_height=({NUMBER}) m, mean_height=({NUMBER}) m, max_tilt=({NUMBER}) deg"
)
POSTURE_RE = re.compile(
    rf"posture: mean_hip_outward=\[([^]]+)\] deg, max_hip_outward=({NUMBER}) deg"
)
SMOOTHNESS_RE = re.compile(
    rf"smoothness: mean_action_delta2=({NUMBER}), mean_action_second_delta2=({NUMBER})"
)
SLIP_RE = re.compile(rf"foot_slip: mean_contact_speed=({NUMBER}) m/s")
IMPACT_RE = re.compile(
    rf"foot_impact: mean_velocity=({NUMBER}) m/s, max_velocity=({NUMBER}) m/s"
)
DISPLACEMENT_RE = re.compile(rf"world displacement=\[({NUMBER}), ({NUMBER})\] m")
YAW_INTEGRAL_RE = re.compile(
    rf"yaw_integral: measured=({NUMBER}) rad, requested=({NUMBER}) rad, bias=({NUMBER}) rad"
)
CONTACT_RE = re.compile(r"contacts: duty=\[([^]]+)\], transitions=\[([^]]+)\]")
SELF_COLLISION_RE = re.compile(
    rf"self_collision: contact_steps=(\d+)/(\d+), mean_pairs=({NUMBER}), max_pairs=(\d+)"
)
ILLEGAL_GROUND_CONTACT_RE = re.compile(
    rf"illegal_ground_contact: contact_steps=(\d+)/(\d+), "
    rf"mean_pairs=({NUMBER}), max_pairs=(\d+)"
)

DEFAULT_GRID = (
    (0.0, 0.0, 0.0),
    (0.05, 0.0, 0.0),
    (-0.05, 0.0, 0.0),
    (0.15, 0.0, 0.0),
    (-0.15, 0.0, 0.0),
    (0.45, 0.0, 0.0),
    (-0.45, 0.0, 0.0),
    (0.0, 0.10, 0.0),
    (0.0, -0.10, 0.0),
    (0.0, 0.0, 0.10),
    (0.0, 0.0, -0.10),
    (0.0, 0.0, 0.25),
    (0.0, 0.0, -0.25),
    (0.30, 0.05, 0.15),
    (-0.30, -0.05, -0.15),
)


def _stage_grid(
    *boundary_commands: tuple[float, float, float],
) -> tuple[tuple[float, float, float], ...]:
    return tuple(dict.fromkeys((*DEFAULT_GRID, *boundary_commands)))


STAGE_GRIDS = {
    "A": DEFAULT_GRID,
    "B": _stage_grid(
        (0.80, 0.0, 0.0), (-0.80, 0.0, 0.0),
        (0.0, 0.20, 0.0), (0.0, -0.20, 0.0),
        (0.0, 0.0, 0.50), (0.0, 0.0, -0.50),
        (0.60, 0.15, 0.35), (-0.60, -0.15, -0.35),
    ),
    "C": _stage_grid(
        (0.80, 0.0, 0.0), (-0.80, 0.0, 0.0),
        (1.50, 0.0, 0.0), (-1.50, 0.0, 0.0),
        (0.0, 0.40, 0.0), (0.0, -0.40, 0.0),
        (0.0, 0.0, 1.00), (0.0, 0.0, -1.00),
        (1.10, 0.30, 0.70), (-1.10, -0.30, -0.70),
    ),
    "D": _stage_grid(
        (0.80, 0.0, 0.0), (-0.80, 0.0, 0.0),
        (1.50, 0.0, 0.0), (-1.50, 0.0, 0.0),
        (3.00, 0.0, 0.0), (-3.00, 0.0, 0.0),
        (0.0, 0.60, 0.0), (0.0, -0.60, 0.0),
        (0.0, 0.0, 2.00), (0.0, 0.0, -2.00),
        (2.20, 0.45, 1.40), (-2.20, -0.45, -1.40),
    ),
}


def vector(text: str) -> list[float]:
    return [float(value.strip()) for value in text.split(",")]


def parse_metrics(output: str) -> dict[str, object]:
    matches = {
        "tracking": TRACKING_RE.search(output),
        "stability": STABILITY_RE.search(output),
        "posture": POSTURE_RE.search(output),
        "smoothness": SMOOTHNESS_RE.search(output),
        "slip": SLIP_RE.search(output),
        "impact": IMPACT_RE.search(output),
        "displacement": DISPLACEMENT_RE.search(output),
        "yaw_integral": YAW_INTEGRAL_RE.search(output),
        "contacts": CONTACT_RE.search(output),
        "self_collision": SELF_COLLISION_RE.search(output),
        "illegal_ground_contact": ILLEGAL_GROUND_CONTACT_RE.search(output),
    }
    missing = [name for name, match in matches.items() if match is None]
    if missing:
        raise ValueError(f"MuJoCo output is missing metrics: {missing}\n{output}")

    tracking = matches["tracking"]
    stability = matches["stability"]
    posture = matches["posture"]
    smoothness = matches["smoothness"]
    slip = matches["slip"]
    impact = matches["impact"]
    displacement = matches["displacement"]
    yaw_integral = matches["yaw_integral"]
    contacts = matches["contacts"]
    self_collision = matches["self_collision"]
    illegal_ground_contact = matches["illegal_ground_contact"]
    assert tracking and stability and posture and smoothness and slip and impact
    command = vector(tracking.group(1))
    policy_command = vector(tracking.group(2))
    measured = vector(tracking.group(3))
    error = vector(tracking.group(4))
    hip_per_leg = vector(posture.group(1))
    result = {
        "command_vx": command[0],
        "command_vy": command[1],
        "command_wz": command[2],
        "policy_command_vx": policy_command[0],
        "policy_command_vy": policy_command[1],
        "policy_command_wz": policy_command[2],
        "measured_vx": measured[0],
        "measured_vy": measured[1],
        "measured_wz": measured[2],
        "error_vx": error[0],
        "error_vy": error[1],
        "error_wz": error[2],
        "min_height_m": float(stability.group(1)),
        "mean_height_m": float(stability.group(2)),
        "max_tilt_deg": float(stability.group(3)),
        "hip_outward_mean_deg": sum(hip_per_leg) / len(hip_per_leg),
        "hip_outward_mean_per_leg_deg": hip_per_leg,
        "hip_outward_max_deg": float(posture.group(2)),
        "action_delta": float(smoothness.group(1)),
        "action_delta2": float(smoothness.group(2)),
        "foot_slip_m_s": float(slip.group(1)),
        "foot_impact_mean_m_s": float(impact.group(1)),
        "foot_impact_max_m_s": float(impact.group(2)),
    }
    if displacement is not None:
        dx, dy = float(displacement.group(1)), float(displacement.group(2))
        result["world_displacement_x_m"] = dx
        result["world_displacement_y_m"] = dy
        result["world_displacement_m"] = float(np.hypot(dx, dy))
    if yaw_integral is not None:
        result["yaw_integral_measured_rad"] = float(yaw_integral.group(1))
        result["yaw_integral_requested_rad"] = float(yaw_integral.group(2))
        result["yaw_integral_bias_rad"] = float(yaw_integral.group(3))
    if contacts is not None:
        result["contact_duty_per_leg"] = vector(contacts.group(1))
        result["contact_transitions_per_leg"] = [
            int(value.strip()) for value in contacts.group(2).split(",")
        ]
    if self_collision is not None:
        result["self_collision_contact_steps"] = int(self_collision.group(1))
        result["self_collision_sample_steps"] = int(self_collision.group(2))
        result["self_collision_mean_pairs"] = float(self_collision.group(3))
        result["self_collision_max_pairs"] = int(self_collision.group(4))
    if illegal_ground_contact is not None:
        result["illegal_ground_contact_steps"] = int(illegal_ground_contact.group(1))
        result["illegal_ground_sample_steps"] = int(illegal_ground_contact.group(2))
        result["illegal_ground_mean_pairs"] = float(illegal_ground_contact.group(3))
        result["illegal_ground_max_pairs"] = int(illegal_ground_contact.group(4))
    return result


def candidate_spec(value: str) -> tuple[str, Path]:
    if "=" not in value:
        raise argparse.ArgumentTypeError("candidate must be LABEL=PATH")
    label, raw_path = value.split("=", 1)
    if not label:
        raise argparse.ArgumentTypeError("candidate label cannot be empty")
    return label, Path(raw_path).resolve()


def parse_args() -> argparse.Namespace:
    project_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--candidate",
        action="append",
        type=candidate_spec,
        required=True,
        metavar="LABEL=PATH",
        help="Candidate directory containing exported/policy.onnx and params/deploy.yaml",
    )
    parser.add_argument("--baseline-label", required=True)
    parser.add_argument(
        "--absolute-only",
        action="store_true",
        help="Disable baseline-relative style gates for command-envelope expansion",
    )
    parser.add_argument(
        "--stage",
        choices=tuple(STAGE_GRIDS),
        default="A",
        help="Use the cumulative command grid for a closed-loop curriculum stage",
    )
    parser.add_argument("--duration", type=float, default=10.0)
    parser.add_argument("--warmup", type=float, default=2.0)
    parser.add_argument(
        "--command",
        action="append",
        nargs=3,
        type=float,
        metavar=("VX", "VY", "WZ"),
        help="Command to evaluate; repeat for a custom grid (default: first-stage grid)",
    )
    parser.add_argument("--output-csv", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument(
        "--runner",
        type=Path,
        default=project_root / "sim2sim" / "custom_dog" / "run_sim2sim.py",
    )
    parser.add_argument(
        "--mjcf",
        type=Path,
        default=project_root / "sim2sim" / "custom_dog" / "custom_dog.xml",
    )
    return parser.parse_args()


def mujoco_runner_command(runner: Path) -> list[str]:
    """Use the dedicated MuJoCo environment when running a Python runner."""

    if runner.suffix != ".py":
        return [str(runner)]
    configured = os.environ.get("CUSTOM_DOG_MUJOCO_PYTHON")
    candidates = (
        [Path(configured).expanduser()]
        if configured
        else [
            Path.home() / "miniconda3" / "envs" / "custom_dog_mujoco" / "bin" / "python",
            Path.home() / ".conda" / "envs" / "custom_dog_mujoco" / "bin" / "python",
            Path.home() / "anaconda3" / "envs" / "custom_dog_mujoco" / "bin" / "python",
        ]
    )
    python = next((candidate for candidate in candidates if candidate.is_file()), None)
    if python is None:
        raise SystemExit(
            "MuJoCo runner Python was not found; run ./scripts/setup_mujoco.sh "
            "or set CUSTOM_DOG_MUJOCO_PYTHON"
        )
    return [str(python), str(runner)]


def absolute_gates(row: dict[str, object]) -> dict[str, bool]:
    vx_threshold = max(0.10, 0.15 * abs(float(row["command_vx"])))
    gates = {
        "vx": float(row["error_vx"]) <= vx_threshold,
        "vy": float(row["error_vy"]) <= 0.07,
        "wz": float(row["error_wz"]) <= 0.10,
        "height": float(row["min_height_m"]) >= 0.23,
        "tilt": float(row["max_tilt_deg"]) < 10.0,
        "not_fallen": not (
            float(row["min_height_m"]) < 0.18 or float(row["max_tilt_deg"]) >= 45.0
        ),
    }
    if "world_displacement_m" not in row:
        return gates

    vx = abs(float(row["command_vx"]))
    vy = abs(float(row["command_vy"]))
    wz = abs(float(row["command_wz"]))
    measured_vx = abs(float(row["measured_vx"]))
    measured_vy = abs(float(row["measured_vy"]))
    measured_wz = abs(float(row["measured_wz"]))
    pure_vx = vx >= 0.03 and vy < 0.03 and wz < 0.05
    pure_vy = vy >= 0.03 and vx < 0.03 and wz < 0.05
    pure_yaw = wz >= 0.05 and vx < 0.03 and vy < 0.03
    standing = vx < 0.03 and vy < 0.03 and wz < 0.05
    if "mean_height_m" in row:
        equivalent_speed = float(np.sqrt(vx * vx + vy * vy + (0.35 * wz) ** 2))
        crouch_blend = float(np.clip((equivalent_speed - 0.10) / (3.0 - 0.10), 0.0, 1.0))
        target_height = 0.33 + crouch_blend * (0.28 - 0.33)
        gates["body_height_target"] = (
            abs(float(row["mean_height_m"]) - target_height) <= 0.025
        )
    if "self_collision_contact_steps" in row:
        sample_steps = max(int(row["self_collision_sample_steps"]), 1)
        gates["self_collision"] = (
            int(row["self_collision_contact_steps"]) / sample_steps <= 0.01
        )
    if "illegal_ground_contact_steps" in row:
        sample_steps = max(int(row["illegal_ground_sample_steps"]), 1)
        gates["illegal_ground_contact"] = (
            int(row["illegal_ground_contact_steps"]) / sample_steps <= 0.01
        )
    if "hip_outward_max_deg" in row:
        if standing:
            hip_limit_deg = 12.0
        elif vy >= 0.03 or wz >= 0.05:
            hip_limit_deg = 25.0
        else:
            hip_limit_deg = 18.0
        gates["hip_outward"] = float(row["hip_outward_max_deg"]) <= hip_limit_deg
    if pure_vx:
        gates["pure_vx_decoupled"] = measured_vy <= 0.05 and measured_wz <= 0.05
    if pure_vy:
        gates["pure_vy_decoupled"] = measured_vx <= 0.05 and measured_wz <= 0.08
    if pure_yaw:
        duration = max(float(row.get("duration_s", 1.0)) - float(row.get("warmup_s", 0.0)), 1.0e-6)
        gates["pure_yaw_xy_drift"] = float(row["world_displacement_m"]) / duration <= 0.05
        if "yaw_integral_bias_rad" in row:
            gates["pure_yaw_integral_bias_rate"] = (
                abs(float(row["yaw_integral_bias_rad"])) / duration <= 0.05
            )
    if standing:
        gates["standing_height"] = 0.310 <= float(row["mean_height_m"]) <= 0.335
        gates["standing_tilt"] = float(row["max_tilt_deg"]) <= 3.0
    transitions = row.get("contact_transitions_per_leg")
    motion_commanded = vx >= 0.03 or vy >= 0.03 or wz >= 0.05
    if transitions is not None and motion_commanded:
        gates["gait_transitions"] = min(transitions) >= 2
    return gates


def relative_gates(row: dict[str, object], baseline: dict[str, object]) -> dict[str, bool]:
    def within_ratio(key: str, ratio: float = 1.20) -> bool:
        reference = float(baseline[key])
        value = float(row[key])
        return value <= reference * ratio + 1.0e-12

    return {
        "hip_mean": float(row["hip_outward_mean_deg"])
        <= float(baseline["hip_outward_mean_deg"]) + 1.0e-12,
        "hip_max_preferred": float(row["hip_outward_max_deg"]) < 35.0,
        "action_delta": within_ratio("action_delta"),
        "action_delta2": within_ratio("action_delta2"),
        "foot_slip": within_ratio("foot_slip_m_s"),
    }


def flatten_row(row: dict[str, object]) -> dict[str, object]:
    flattened = dict(row)
    per_leg = flattened.pop("hip_outward_mean_per_leg_deg")
    for leg, value in zip(("fr", "fl", "rr", "rl"), per_leg):
        flattened[f"hip_outward_mean_{leg}_deg"] = value
    for group in ("absolute_gates", "relative_gates"):
        values = flattened.pop(group, {})
        for name, value in values.items():
            flattened[f"{group}/{name}"] = value
    for field in ("contact_duty_per_leg", "contact_transitions_per_leg"):
        values = flattened.pop(field, None)
        if values is not None:
            for leg, value in zip(("fr", "fl", "rr", "rl"), values):
                flattened[f"{field}/{leg}"] = value
    return flattened


def csv_fieldnames(rows: list[dict[str, object]]) -> list[str]:
    """Return stable union fieldnames for rows with command-specific gates."""

    return list(dict.fromkeys(field for row in rows for field in row))


def main() -> None:
    args = parse_args()
    labels = [label for label, _ in args.candidate]
    if len(labels) != len(set(labels)):
        raise SystemExit("Candidate labels must be unique")
    if args.baseline_label not in labels:
        raise SystemExit(f"Unknown baseline label: {args.baseline_label}")
    if args.duration <= 0 or not 0 <= args.warmup < args.duration:
        raise SystemExit("Require duration > warmup >= 0")
    if args.command and args.stage != "A":
        raise SystemExit("--command cannot be combined with a non-default --stage")
    grid = (
        tuple(tuple(command) for command in args.command)
        if args.command
        else STAGE_GRIDS[args.stage]
    )

    rows: list[dict[str, object]] = []
    runner_command = mujoco_runner_command(args.runner.resolve())
    for label, candidate in args.candidate:
        policy = candidate / "exported" / "policy.onnx"
        deploy = candidate / "params" / "deploy.yaml"
        for required in (policy, deploy):
            if not required.is_file():
                raise SystemExit(f"Missing candidate artifact: {required}")
        stand_policy = candidate / "exported" / "stand_policy.onnx"
        stand_deploy = candidate / "params" / "stand_deploy.yaml"
        if stand_policy.is_file() != stand_deploy.is_file():
            raise SystemExit(
                f"Candidate must contain both routed stand artifacts or neither: {candidate}"
            )
        routed_args = (
            [
                "--stand-policy",
                str(stand_policy),
                "--stand-deploy-yaml",
                str(stand_deploy),
            ]
            if stand_policy.is_file()
            else []
        )
        for command_index, command in enumerate(grid):
            print(f"[{label}] command {command_index + 1}/{len(grid)}: {command}", flush=True)
            encoder = candidate / "exported" / "encoder.onnx"
            process = subprocess.run(
                runner_command + [
                    "--mjcf",
                    str(args.mjcf.resolve()),
                    "--policy",
                    str(policy),
                    *( ["--encoder", str(encoder)] if encoder.is_file() else [] ),
                    "--deploy-yaml",
                    str(deploy),
                    *routed_args,
                    "--command",
                    *(str(value) for value in command),
                    "--duration",
                    str(args.duration),
                    "--warmup",
                    str(args.warmup),
                ],
                check=True,
                capture_output=True,
                text=True,
            )
            metrics = parse_metrics(process.stdout)
            metrics.update(
                {
                    "candidate": label,
                    "candidate_path": str(candidate),
                    "command_index": command_index,
                    "duration_s": args.duration,
                    "warmup_s": args.warmup,
                }
            )
            metrics["absolute_gates"] = absolute_gates(metrics)
            rows.append(metrics)

    baseline_rows = {
        int(row["command_index"]): row
        for row in rows
        if row["candidate"] == args.baseline_label
    }
    for row in rows:
        baseline = baseline_rows[int(row["command_index"])]
        row["relative_gates"] = {} if args.absolute_only else relative_gates(row, baseline)
        row["passes_absolute"] = all(row["absolute_gates"].values())
        row["passes_relative"] = args.absolute_only or (
            row["candidate"] == args.baseline_label
            or all(row["relative_gates"].values())
        )
        row["passes_all"] = bool(row["passes_absolute"] and row["passes_relative"])

    summaries = {}
    for label in labels:
        selected = [row for row in rows if row["candidate"] == label]
        standing = [
            row
            for row in selected
            if abs(float(row["command_vx"])) < 0.03
            and abs(float(row["command_vy"])) < 0.03
            and abs(float(row["command_wz"])) < 0.05
        ]
        pure_yaw = [
            row
            for row in selected
            if abs(float(row["command_vx"])) < 0.03
            and abs(float(row["command_vy"])) < 0.03
            and abs(float(row["command_wz"])) >= 0.05
        ]
        moving = [
            row
            for row in selected
            if abs(float(row["command_vx"])) >= 0.03
            or abs(float(row["command_vy"])) >= 0.03
            or abs(float(row["command_wz"])) >= 0.05
        ]

        def measured_duration(row: dict[str, object]) -> float:
            return max(
                float(row.get("duration_s", 1.0)) - float(row.get("warmup_s", 0.0)),
                1.0e-6,
            )

        summaries[label] = {
            "commands_passed_absolute": sum(bool(row["passes_absolute"]) for row in selected),
            "commands_passed_all": sum(bool(row["passes_all"]) for row in selected),
            "total_commands": len(selected),
            "passes_grid": all(bool(row["passes_all"]) for row in selected),
            "max_errors": {
                axis: max(float(row[f"error_{axis}"]) for row in selected)
                for axis in ("vx", "vy", "wz")
            },
            "min_height_m": min(float(row["min_height_m"]) for row in selected),
            "max_tilt_deg": max(float(row["max_tilt_deg"]) for row in selected),
            "mean_hip_outward_deg": sum(float(row["hip_outward_mean_deg"]) for row in selected)
            / len(selected),
            "max_hip_outward_deg": max(float(row["hip_outward_max_deg"]) for row in selected),
            "standing_mean_height_m": (
                float(standing[0]["mean_height_m"]) if standing else None
            ),
            "standing_max_tilt_deg": (
                float(standing[0]["max_tilt_deg"]) if standing else None
            ),
            "max_pure_yaw_xy_drift_m_s": (
                max(float(row["world_displacement_m"]) / measured_duration(row) for row in pure_yaw)
                if pure_yaw
                else None
            ),
            "max_pure_yaw_integral_bias_rad": (
                max(abs(float(row["yaw_integral_bias_rad"])) for row in pure_yaw)
                if pure_yaw
                else None
            ),
            "max_pure_yaw_integral_bias_rate_rad_s": (
                max(
                    abs(float(row["yaw_integral_bias_rad"])) / measured_duration(row)
                    for row in pure_yaw
                )
                if pure_yaw
                else None
            ),
            "min_ground_contact_transitions_moving": (
                min(min(row["contact_transitions_per_leg"]) for row in moving)
                if moving
                else None
            ),
        }

    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    flat_rows = [flatten_row(row) for row in rows]
    with args.output_csv.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=csv_fieldnames(flat_rows))
        writer.writeheader()
        writer.writerows(flat_rows)
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    with args.output_json.open("w", encoding="utf-8") as stream:
        json.dump(
            {
                "baseline_label": args.baseline_label,
                "stage": args.stage,
                "absolute_only": args.absolute_only,
                "duration_s": args.duration,
                "warmup_s": args.warmup,
                "grid": grid,
                "summaries": summaries,
                "rows": rows,
            },
            stream,
            indent=2,
        )
        stream.write("\n")
    print(json.dumps(summaries, indent=2))


if __name__ == "__main__":
    main()

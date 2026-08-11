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

DEFAULT_GRID = (
    (0.3, 0.0, 0.0),
    (0.5, 0.0, 0.2),
    (0.5, 0.0, -0.2),
    (0.5, 0.1, 0.0),
    (0.5, -0.1, 0.0),
    (0.5, 0.1, 0.2),
)


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
    assert tracking and stability and posture and smoothness and slip and impact
    command = vector(tracking.group(1))
    policy_command = vector(tracking.group(2))
    measured = vector(tracking.group(3))
    error = vector(tracking.group(4))
    hip_per_leg = vector(posture.group(1))
    return {
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
    default = Path.home() / ".conda" / "envs" / "custom_dog_mujoco" / "bin" / "python"
    python = Path(configured).expanduser() if configured else default
    if not python.is_file():
        raise SystemExit(
            "MuJoCo runner Python was not found at "
            f"{python}; run ./scripts/setup_mujoco.sh or set CUSTOM_DOG_MUJOCO_PYTHON"
        )
    return [str(python), str(runner)]


def absolute_gates(row: dict[str, object]) -> dict[str, bool]:
    vx_threshold = max(0.10, 0.15 * abs(float(row["command_vx"])))
    return {
        "vx": float(row["error_vx"]) <= vx_threshold,
        "vy": float(row["error_vy"]) <= 0.07,
        "wz": float(row["error_wz"]) <= 0.10,
        "height": float(row["min_height_m"]) >= 0.23,
        "tilt": float(row["max_tilt_deg"]) < 10.0,
        "not_fallen": not (
            float(row["min_height_m"]) < 0.18 or float(row["max_tilt_deg"]) >= 45.0
        ),
    }


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
    return flattened


def main() -> None:
    args = parse_args()
    labels = [label for label, _ in args.candidate]
    if len(labels) != len(set(labels)):
        raise SystemExit("Candidate labels must be unique")
    if args.baseline_label not in labels:
        raise SystemExit(f"Unknown baseline label: {args.baseline_label}")
    if args.duration <= 0 or not 0 <= args.warmup < args.duration:
        raise SystemExit("Require duration > warmup >= 0")
    grid = tuple(tuple(command) for command in args.command) if args.command else DEFAULT_GRID

    rows: list[dict[str, object]] = []
    runner_command = mujoco_runner_command(args.runner.resolve())
    for label, candidate in args.candidate:
        policy = candidate / "exported" / "policy.onnx"
        deploy = candidate / "params" / "deploy.yaml"
        for required in (policy, deploy):
            if not required.is_file():
                raise SystemExit(f"Missing candidate artifact: {required}")
        for command_index, command in enumerate(grid):
            print(f"[{label}] command {command_index + 1}/{len(grid)}: {command}", flush=True)
            process = subprocess.run(
                runner_command + [
                    "--mjcf",
                    str(args.mjcf.resolve()),
                    "--policy",
                    str(policy),
                    "--deploy-yaml",
                    str(deploy),
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
        row["relative_gates"] = relative_gates(row, baseline)
        row["passes_absolute"] = all(row["absolute_gates"].values())
        row["passes_relative"] = (
            row["candidate"] == args.baseline_label
            or all(row["relative_gates"].values())
        )
        row["passes_all"] = bool(row["passes_absolute"] and row["passes_relative"])

    summaries = {}
    for label in labels:
        selected = [row for row in rows if row["candidate"] == label]
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
        }

    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    flat_rows = [flatten_row(row) for row in rows]
    with args.output_csv.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(flat_rows[0]))
        writer.writeheader()
        writer.writerows(flat_rows)
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    with args.output_json.open("w", encoding="utf-8") as stream:
        json.dump(
            {
                "baseline_label": args.baseline_label,
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

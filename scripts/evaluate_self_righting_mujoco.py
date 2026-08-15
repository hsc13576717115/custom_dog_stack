#!/usr/bin/env python3
"""Evaluate an exported self-righting policy from deterministic fall poses."""

from __future__ import annotations

import argparse
import json
import re
import subprocess
from pathlib import Path


STAGE_STATES = {
    "R0": ("recovery-belly",),
    "R1": ("recovery-belly", "recovery-back", "recovery-left", "recovery-right"),
    "R2": ("recovery-belly", "recovery-back", "recovery-left", "recovery-right"),
}
R2_PERTURBATIONS = (
    {
        "initial_state": "recovery-belly",
        "orientation_offset_deg": (25.0, -30.0, 20.0),
        "linear_velocity": (0.25, -0.18, 0.10),
        "angular_velocity": (0.65, -0.45, 0.35),
        "joint_noise": 0.25,
        "seed": 17,
    },
    {
        "initial_state": "recovery-back",
        "orientation_offset_deg": (-30.0, 25.0, -20.0),
        "linear_velocity": (-0.22, 0.20, -0.08),
        "angular_velocity": (-0.60, 0.55, -0.40),
        "joint_noise": 0.25,
        "seed": 23,
    },
    {
        "initial_state": "recovery-left",
        "orientation_offset_deg": (20.0, 30.0, 25.0),
        "linear_velocity": (0.18, 0.25, 0.08),
        "angular_velocity": (0.45, 0.70, -0.35),
        "joint_noise": 0.25,
        "seed": 31,
    },
    {
        "initial_state": "recovery-right",
        "orientation_offset_deg": (-20.0, -30.0, -25.0),
        "linear_velocity": (-0.18, -0.25, 0.08),
        "angular_velocity": (-0.45, -0.70, 0.35),
        "joint_noise": 0.25,
        "seed": 47,
    },
)
SUCCESS_PATTERN = re.compile(r"self_righting: success at ([0-9.]+) s")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("candidate", type=Path, help="Directory containing exported/policy.onnx and params/deploy.yaml")
    parser.add_argument("--stage", choices=tuple(STAGE_STATES), required=True)
    parser.add_argument("--duration", type=float, default=7.0)
    parser.add_argument("--runner", type=Path, default=Path("sim2sim/custom_dog/run_sim2sim.py"))
    parser.add_argument("--mjcf", type=Path, default=Path("sim2sim/custom_dog/custom_dog.xml"))
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def evaluate_state(
    python: str,
    runner: Path,
    mjcf: Path,
    policy: Path,
    deploy_yaml: Path,
    initial_state: str,
    duration: float,
    orientation_offset_deg: tuple[float, float, float] = (0.0, 0.0, 0.0),
    linear_velocity: tuple[float, float, float] = (0.0, 0.0, 0.0),
    angular_velocity: tuple[float, float, float] = (0.0, 0.0, 0.0),
    joint_noise: float = 0.0,
    seed: int = 0,
) -> dict[str, object]:
    command = [
        python,
        str(runner),
        "--mjcf",
        str(mjcf),
        "--policy",
        str(policy),
        "--deploy-yaml",
        str(deploy_yaml),
        "--initial-state",
        initial_state,
        "--duration",
        str(duration),
        "--warmup",
        "0.1",
        "--recovery-orientation-offset-deg",
        *(str(value) for value in orientation_offset_deg),
        "--recovery-linear-velocity",
        *(str(value) for value in linear_velocity),
        "--recovery-angular-velocity",
        *(str(value) for value in angular_velocity),
        "--recovery-joint-noise",
        str(joint_noise),
        "--recovery-seed",
        str(seed),
    ]
    result = subprocess.run(command, check=False, capture_output=True, text=True)
    match = SUCCESS_PATTERN.search(result.stdout)
    return {
        "initial_state": initial_state,
        "orientation_offset_deg": list(orientation_offset_deg),
        "linear_velocity": list(linear_velocity),
        "angular_velocity": list(angular_velocity),
        "joint_noise": joint_noise,
        "seed": seed,
        "passed": result.returncode == 0 and match is not None,
        "success_time_s": float(match.group(1)) if match is not None else None,
        "returncode": result.returncode,
        "stdout": result.stdout,
        "stderr": result.stderr,
    }


def main() -> None:
    args = parse_args()
    candidate = args.candidate.resolve()
    policy = candidate / "exported" / "policy.onnx"
    deploy_yaml = candidate / "params" / "deploy.yaml"
    for path in (policy, deploy_yaml, args.runner, args.mjcf):
        if not path.is_file():
            raise FileNotFoundError(path)

    import sys

    cases = [{"initial_state": state} for state in STAGE_STATES[args.stage]]
    if args.stage == "R2":
        cases.extend(R2_PERTURBATIONS)
    rows = [
        evaluate_state(
            sys.executable,
            args.runner.resolve(),
            args.mjcf.resolve(),
            policy,
            deploy_yaml,
            str(case["initial_state"]),
            args.duration,
            orientation_offset_deg=tuple(case.get("orientation_offset_deg", (0.0, 0.0, 0.0))),
            linear_velocity=tuple(case.get("linear_velocity", (0.0, 0.0, 0.0))),
            angular_velocity=tuple(case.get("angular_velocity", (0.0, 0.0, 0.0))),
            joint_noise=float(case.get("joint_noise", 0.0)),
            seed=int(case.get("seed", 0)),
        )
        for case in cases
    ]
    summary = {
        "stage": args.stage,
        "candidate": str(candidate),
        "passes_all": all(bool(row["passed"]) for row in rows),
        "states_passed": sum(bool(row["passed"]) for row in rows),
        "total_states": len(rows),
        "results": rows,
    }
    encoded = json.dumps(summary, indent=2)
    print(encoded)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(encoded + "\n", encoding="utf-8")
    if not summary["passes_all"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()

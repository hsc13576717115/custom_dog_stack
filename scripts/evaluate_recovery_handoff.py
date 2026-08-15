#!/usr/bin/env python3
"""Verify recovery-policy -> locomotion-policy handoff in MuJoCo."""

from __future__ import annotations

import argparse
import json
import re
import subprocess
from pathlib import Path


STATES = ("recovery-belly", "recovery-back", "recovery-left", "recovery-right")
SUCCESS_RE = re.compile(r"self_righting: success at ([0-9.]+) s")
HANDOFF_RE = re.compile(r"recovery_handoff: locomotion command released at ([0-9.]+) s")


def build_command(
    *,
    python: str,
    runner: Path,
    mjcf: Path,
    recovery_candidate: Path,
    locomotion_candidate: Path,
    stand_candidate: Path,
    initial_state: str,
    duration: float,
) -> list[str]:
    def artifact(candidate: Path, relative: str) -> str:
        return str((candidate / relative).resolve())

    return [
        python,
        str(runner.resolve()),
        "--mjcf",
        str(mjcf.resolve()),
        "--policy",
        artifact(locomotion_candidate, "exported/policy.onnx"),
        "--deploy-yaml",
        artifact(locomotion_candidate, "params/deploy.yaml"),
        "--stand-policy",
        artifact(stand_candidate, "exported/policy.onnx"),
        "--stand-deploy-yaml",
        artifact(stand_candidate, "params/deploy.yaml"),
        "--recovery-policy",
        artifact(recovery_candidate, "exported/policy.onnx"),
        "--recovery-deploy-yaml",
        artifact(recovery_candidate, "params/deploy.yaml"),
        "--initial-state",
        initial_state,
        "--command",
        "0.0",
        "0.0",
        "0.0",
        "--duration",
        str(duration),
        "--warmup",
        "0.0",
        "--policy-blend",
        "0.30",
        "--recovery-locomotion-hold",
        "1.0",
    ]


def evaluate_state(command: list[str], cwd: Path, initial_state: str) -> dict[str, object]:
    result = subprocess.run(command, cwd=cwd, check=False, capture_output=True, text=True)
    success = SUCCESS_RE.search(result.stdout)
    handoff = HANDOFF_RE.search(result.stdout)
    passed = result.returncode == 0 and success is not None and handoff is not None
    return {
        "initial_state": initial_state,
        "passed": passed,
        "returncode": result.returncode,
        "recovery_success_time_s": float(success.group(1)) if success else None,
        "locomotion_release_time_s": float(handoff.group(1)) if handoff else None,
        "stdout": result.stdout,
        "stderr": result.stderr,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--recovery-candidate", type=Path, required=True)
    parser.add_argument("--locomotion-candidate", type=Path, required=True)
    parser.add_argument("--stand-candidate", type=Path, required=True)
    parser.add_argument("--mjcf", type=Path, required=True)
    parser.add_argument("--runner", type=Path, default=Path("sim2sim/custom_dog/run_sim2sim.py"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--duration", type=float, default=12.0)
    parser.add_argument("--python", default="python")
    args = parser.parse_args()

    candidates = (
        args.recovery_candidate,
        args.locomotion_candidate,
        args.stand_candidate,
    )
    for candidate in candidates:
        for relative in ("exported/policy.onnx", "params/deploy.yaml"):
            path = candidate / relative
            if not path.is_file():
                raise FileNotFoundError(path)
    if not args.mjcf.is_file() or not args.runner.is_file():
        raise FileNotFoundError("MJCF or sim2sim runner is missing")
    if args.duration <= 2.0:
        raise ValueError("handoff duration must leave time for recovery and locomotion hold")

    rows = []
    for state in STATES:
        command = build_command(
            python=args.python,
            runner=args.runner,
            mjcf=args.mjcf,
            recovery_candidate=args.recovery_candidate,
            locomotion_candidate=args.locomotion_candidate,
            stand_candidate=args.stand_candidate,
            initial_state=state,
            duration=args.duration,
        )
        rows.append(evaluate_state(command, args.runner.resolve().parents[2], state))
    result = {
        "states_passed": sum(bool(row["passed"]) for row in rows),
        "total_states": len(rows),
        "passes_all": all(bool(row["passed"]) for row in rows),
        "recovery_candidate": str(args.recovery_candidate.resolve()),
        "locomotion_candidate": str(args.locomotion_candidate.resolve()),
        "stand_candidate": str(args.stand_candidate.resolve()),
        "rows": rows,
    }
    encoded = json.dumps(result, indent=2) + "\n"
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(encoded, encoding="utf-8")
    print(encoded, end="")
    if not result["passes_all"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()

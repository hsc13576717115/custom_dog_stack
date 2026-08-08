#!/usr/bin/env python3
"""Validate a policy trace against deploy.yaml and the exported ONNX policy."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np
import yaml


def columns(rows: list[dict[str, str]], prefix: str, count: int) -> np.ndarray:
    names = [f"{prefix}_{index}" for index in range(count)]
    missing = [name for name in names if name not in rows[0]]
    if missing:
        raise ValueError(f"Trace is missing columns: {missing}")
    return np.asarray([[float(row[name]) for name in names] for row in rows])


def maximum_error(name: str, actual: np.ndarray, expected: np.ndarray) -> float:
    error = float(np.max(np.abs(actual - expected)))
    print(f"{name}: max_abs_error={error:.6g}")
    return error


def observation_term(values: np.ndarray, cfg: dict[str, object]) -> np.ndarray:
    lower, upper = cfg["clip"]
    scale = np.asarray(cfg["scale"])
    return np.clip(values, lower, upper) * scale


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("trace", type=Path)
    parser.add_argument("candidate", type=Path, help="Directory containing exported/ and params/")
    parser.add_argument("--tolerance", type=float, default=1e-5)
    args = parser.parse_args()

    with args.trace.open(encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream))
    if not rows:
        raise ValueError(f"Trace contains no data rows: {args.trace}")

    deploy_path = args.candidate / "params" / "deploy.yaml"
    policy_path = args.candidate / "exported" / "policy.onnx"
    with deploy_path.open(encoding="utf-8") as stream:
        cfg = yaml.safe_load(stream)

    obs = columns(rows, "obs", 45).astype(np.float32)
    action = columns(rows, "action", 12)
    target = columns(rows, "target_q", 12)
    joint_q = columns(rows, "joint_q", 12)
    joint_dq = columns(rows, "joint_dq", 12)
    ang_vel = columns(rows, "ang_vel", 3)
    gravity = columns(rows, "projected_gravity", 3)

    obs_cfg = cfg["observations"]
    default_q = np.asarray(cfg["default_joint_pos"], dtype=np.float64)
    expected_terms = {
        "base_ang_vel": observation_term(ang_vel, obs_cfg["base_ang_vel"]),
        "projected_gravity": observation_term(gravity, obs_cfg["projected_gravity"]),
        "velocity_commands": obs[:, 6:9],
        "joint_pos_rel": observation_term(
            joint_q - default_q, obs_cfg["joint_pos_rel"]
        ),
        "joint_vel_rel": observation_term(joint_dq, obs_cfg["joint_vel_rel"]),
        "last_action": observation_term(
            np.vstack((np.zeros((1, 12)), action[:-1])),
            obs_cfg["last_action"],
        ),
    }
    term_slices = {
        "base_ang_vel": slice(0, 3),
        "projected_gravity": slice(3, 6),
        "velocity_commands": slice(6, 9),
        "joint_pos_rel": slice(9, 21),
        "joint_vel_rel": slice(21, 33),
        "last_action": slice(33, 45),
    }
    expected_obs = np.concatenate(list(expected_terms.values()), axis=1)

    action_cfg = cfg["actions"]["JointPositionAction"]
    expected_target = (
        np.asarray(action_cfg["offset"])
        + np.asarray(action_cfg["scale"]) * action
    )
    clip = np.asarray(action_cfg["clip"])
    expected_target = np.clip(expected_target, clip[:, 0], clip[:, 1])

    try:
        import onnxruntime as ort
    except ModuleNotFoundError as exc:
        raise SystemExit(
            "onnxruntime is not installed in this Python environment; run this script "
            "through the custom_dog_mujoco Conda environment"
        ) from exc

    session = ort.InferenceSession(str(policy_path.resolve()), providers=["CPUExecutionProvider"])
    input_name = session.get_inputs()[0].name
    output_name = session.get_outputs()[0].name
    expected_action = np.vstack(
        [session.run([output_name], {input_name: row[np.newaxis, :]})[0][0] for row in obs]
    )

    for name, expected in expected_terms.items():
        maximum_error(f"observation/{name}", obs[:, term_slices[name]], expected)

    errors = {
        "observation contract": maximum_error("observation contract", obs, expected_obs),
        "ONNX action": maximum_error("ONNX action", action, expected_action),
        "processed target": maximum_error("processed target", target, expected_target),
    }
    print(
        f"rows={len(rows)}, command_mean={np.mean(obs[:, 6:9], axis=0).tolist()}, "
        f"max_abs_action={float(np.max(np.abs(action))):.6g}"
    )
    failed = [name for name, error in errors.items() if error > args.tolerance]
    if failed:
        raise SystemExit(f"FAIL (tolerance={args.tolerance}): {', '.join(failed)}")
    print(f"PASS: all contracts are within tolerance {args.tolerance}")


if __name__ == "__main__":
    main()

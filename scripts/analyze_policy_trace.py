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
    scale = np.asarray(cfg["scale"])
    clip = cfg.get("clip")
    if clip is None:
        return values * scale
    lower, upper = clip
    return np.clip(values, lower, upper) * scale


def stack_history(values: np.ndarray, history_length: int) -> np.ndarray:
    """Flatten per-term history oldest-to-newest using Isaac Lab startup fill."""

    if values.ndim != 2 or values.shape[0] == 0:
        raise ValueError(f"History source must be a non-empty matrix, got {values.shape}")
    if history_length < 1:
        raise ValueError(f"history_length must be positive, got {history_length}")
    row_indices = np.arange(values.shape[0])[:, np.newaxis]
    delays = np.arange(history_length - 1, -1, -1)[np.newaxis, :]
    indices = np.maximum(row_indices - delays, 0)
    return values[indices].reshape(values.shape[0], history_length * values.shape[1])


def policy_command(raw_command: np.ndarray, cfg: dict[str, object]) -> np.ndarray:
    """Apply the optional deploy-time command calibration to raw requests."""

    result = raw_command.copy()
    calibration_root = cfg.get("command_calibration") or {}
    if not isinstance(calibration_root, dict):
        raise ValueError("command_calibration must be a mapping")
    for index, name in enumerate(("lin_vel_x", "lin_vel_y", "ang_vel_z")):
        calibration = calibration_root.get(name)
        if calibration is None:
            continue
        requested = np.asarray(calibration["requested"], dtype=np.float64)
        policy = np.asarray(calibration["policy"], dtype=np.float64)
        if (
            requested.ndim != 1
            or requested.size < 2
            or policy.shape != requested.shape
            or not np.isfinite(requested).all()
            or not np.isfinite(policy).all()
            or np.any(np.diff(requested) <= 0.0)
            or np.any(np.diff(policy) < 0.0)
        ):
            raise ValueError(f"Invalid command_calibration.{name} configuration")
        zero_indices = np.flatnonzero(np.isclose(requested, 0.0, atol=1.0e-12))
        if zero_indices.size != 1 or not np.isclose(policy[zero_indices[0]], 0.0, atol=1.0e-12):
            raise ValueError(f"command_calibration.{name} must map zero request to zero")
        result[:, index] = np.interp(result[:, index], requested, policy)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("trace", type=Path)
    parser.add_argument("candidate", type=Path, help="Directory containing exported/ and params/")
    parser.add_argument(
        "--expected-command",
        nargs=3,
        type=float,
        metavar=("VX", "VY", "YAW"),
        required=True,
        help="Independent raw velocity command expected in every trace row",
    )
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

    obs_cfg = cfg["observations"]
    observation_dim = sum(
        len(term["scale"]) * int(term.get("history_length", 1))
        for term in obs_cfg.values()
    )
    obs = columns(rows, "obs", observation_dim).astype(np.float32)
    action = columns(rows, "action", 12)
    target = columns(rows, "target_q", 12)
    joint_q = columns(rows, "joint_q", 12)
    joint_dq = columns(rows, "joint_dq", 12)
    ang_vel = columns(rows, "ang_vel", 3)
    gravity = columns(rows, "projected_gravity", 3)
    base_lin_vel = (
        columns(rows, "base_lin_vel", 3)
        if "base_lin_vel_xy" in obs_cfg
        else None
    )

    default_q = np.asarray(cfg["default_joint_pos"], dtype=np.float64)
    command_cfg = cfg["commands"]["base_velocity"]
    external_ranges = command_cfg.get("external_ranges", command_cfg["ranges"])
    raw_command = np.broadcast_to(np.asarray(args.expected_command), (len(rows), 3))
    range_values = np.asarray(
        [
            external_ranges["lin_vel_x"],
            external_ranges["lin_vel_y"],
            external_ranges["ang_vel_z"],
        ],
        dtype=np.float64,
    )
    if np.any(raw_command[0] < range_values[:, 0]) or np.any(raw_command[0] > range_values[:, 1]):
        raise ValueError(f"Expected command is outside external_ranges: {args.expected_command}")
    calibrated_command = policy_command(raw_command, cfg)
    raw_terms = {
        "base_ang_vel": observation_term(ang_vel, obs_cfg["base_ang_vel"]),
        "projected_gravity": observation_term(gravity, obs_cfg["projected_gravity"]),
        "velocity_commands": observation_term(
            calibrated_command, obs_cfg["velocity_commands"]
        ),
        "joint_pos_rel": observation_term(
            joint_q - default_q, obs_cfg["joint_pos_rel"]
        ),
        "joint_vel_rel": observation_term(joint_dq, obs_cfg["joint_vel_rel"]),
        "last_action": observation_term(
            np.vstack((np.zeros((1, 12)), action[:-1])),
            obs_cfg["last_action"],
        ),
    }
    if "gait_phase" in obs_cfg:
        phase_cfg = obs_cfg["gait_phase"]["params"]
        period = float(phase_cfg["period"])
        threshold = float(phase_cfg.get("command_threshold", 0.1))
        phase = np.arange(len(rows), dtype=np.float64) * float(cfg["step_dt"]) / period
        gait_phase = np.column_stack(
            (np.sin(phase * 2.0 * np.pi), np.cos(phase * 2.0 * np.pi))
        )
        gait_phase[np.linalg.norm(calibrated_command, axis=1) <= threshold] = 0.0
        raw_terms["gait_phase"] = observation_term(gait_phase, obs_cfg["gait_phase"])
    if "trot_clock" in obs_cfg:
        clock_cfg = obs_cfg["trot_clock"]["params"]
        threshold = float(clock_cfg.get("command_threshold", 0.1))
        min_frequency = float(clock_cfg.get("min_frequency", 1.4))
        max_frequency = float(clock_cfg.get("max_frequency", 3.2))
        full_speed = float(clock_cfg.get("full_speed", 3.0))
        yaw_speed_scale = float(clock_cfg.get("yaw_speed_scale", 0.35))
        motion_speed = np.linalg.norm(calibrated_command[:, :2], axis=1)
        motion_speed += yaw_speed_scale * np.abs(calibrated_command[:, 2])
        blend = np.clip(motion_speed / full_speed, 0.0, 1.0)
        frequency = min_frequency + blend * (max_frequency - min_frequency)
        phase = np.arange(len(rows), dtype=np.float64) * float(cfg["step_dt"]) * frequency
        foot_phase = (phase[:, np.newaxis] + np.asarray((0.0, 0.5, 0.5, 0.0))) % 1.0
        trot_clock = np.sin(2.0 * np.pi * foot_phase)
        trot_clock[motion_speed <= threshold] = 0.0
        raw_terms["trot_clock"] = observation_term(trot_clock, obs_cfg["trot_clock"])
    if "base_lin_vel_xy" in obs_cfg:
        assert base_lin_vel is not None
        raw_terms["base_lin_vel_xy"] = observation_term(
            base_lin_vel[:, :2], obs_cfg["base_lin_vel_xy"]
        )

    expected_terms = {
        name: stack_history(raw_terms[name], int(term.get("history_length", 1)))
        for name, term in obs_cfg.items()
    }
    term_slices: dict[str, slice] = {}
    offset = 0
    for name, values in expected_terms.items():
        term_slices[name] = slice(offset, offset + values.shape[1])
        offset += values.shape[1]
    expected_obs = np.concatenate(list(expected_terms.values()), axis=1)
    if expected_obs.shape[1] != observation_dim:
        raise ValueError(
            f"Expected observation width {observation_dim}, got {expected_obs.shape[1]}"
        )

    action_cfg = cfg["actions"]["JointPositionAction"]
    expected_target = (
        np.asarray(action_cfg["offset"])
        + np.asarray(action_cfg["scale"]) * action
    )
    clip = np.asarray(action_cfg["clip"])
    expected_target = np.clip(expected_target, clip[:, 0], clip[:, 1])

    bias_cfg = cfg.get("joint_target_bias")
    if bias_cfg is not None:
        bias = np.asarray(bias_cfg["values"], dtype=np.float64)
        vx_min, vx_max = np.asarray(bias_cfg["vx_range"], dtype=np.float64)
        if (
            bias.shape != (12,)
            or not np.isfinite(bias).all()
            or not np.isfinite([vx_min, vx_max]).all()
            or vx_max <= vx_min
        ):
            raise ValueError("Invalid joint_target_bias configuration")
        blend = np.clip((raw_command[:, 0] - vx_min) / (vx_max - vx_min), 0.0, 1.0)
        expected_target += blend[:, np.newaxis] * bias

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
        f"rows={len(rows)}, command_mean="
        f"{np.mean(obs[:, term_slices['velocity_commands']], axis=0).tolist()}, "
        f"max_abs_action={float(np.max(np.abs(action))):.6g}"
    )
    failed = [name for name, error in errors.items() if error > args.tolerance]
    if failed:
        raise SystemExit(f"FAIL (tolerance={args.tolerance}): {', '.join(failed)}")
    print(f"PASS: all contracts are within tolerance {args.tolerance}")


if __name__ == "__main__":
    main()

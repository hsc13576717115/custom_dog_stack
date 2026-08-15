#!/usr/bin/env python3
"""Compare aggregate dynamics from two phase-aligned policy trace CSV files."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np


def load_columns(path: Path, prefix: str) -> np.ndarray:
    with path.open(encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream))
    names = []
    index = 0
    while rows and f"{prefix}_{index}" in rows[0]:
        names.append(f"{prefix}_{index}")
        index += 1
    if not rows or not names:
        raise ValueError(f"{path} has no {prefix}_N columns")
    return np.asarray([[float(row[name]) for name in names] for row in rows], dtype=np.float64)


def load_time(path: Path) -> np.ndarray:
    with path.open(encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream))
    return np.asarray([float(row["time_s"]) for row in rows], dtype=np.float64)


def summarize(path: Path, warmup: float) -> dict[str, object]:
    time = load_time(path)
    selected = time >= warmup
    if not np.any(selected):
        raise ValueError(f"{path} has no samples at or after {warmup} s")
    values = {
        name: load_columns(path, name)[selected]
        for name in ("action", "target_q", "joint_q", "joint_dq", "ang_vel", "base_lin_vel")
    }
    target_error = values["target_q"] - values["joint_q"]
    action_delta = np.diff(values["action"], axis=0)
    return {
        "path": str(path.resolve()),
        "samples": int(np.sum(selected)),
        "mean_body_velocity": values["base_lin_vel"].mean(axis=0).tolist(),
        "mean_body_angular_velocity": values["ang_vel"].mean(axis=0).tolist(),
        "rms_joint_velocity": float(np.sqrt(np.mean(np.square(values["joint_dq"])))),
        "rms_target_tracking_error_rad": float(np.sqrt(np.mean(np.square(target_error)))),
        "max_target_tracking_error_rad": float(np.max(np.abs(target_error))),
        "rms_action": float(np.sqrt(np.mean(np.square(values["action"])))),
        "rms_action_delta": float(np.sqrt(np.mean(np.square(action_delta)))),
        "mean_action_per_joint": values["action"].mean(axis=0).tolist(),
        "std_action_per_joint": values["action"].std(axis=0).tolist(),
        "mean_joint_position": values["joint_q"].mean(axis=0).tolist(),
        "std_joint_position": values["joint_q"].std(axis=0).tolist(),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("left", type=Path)
    parser.add_argument("right", type=Path)
    parser.add_argument("--left-label", default="left")
    parser.add_argument("--right-label", default="right")
    parser.add_argument("--warmup", type=float, default=2.0)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    left = summarize(args.left, args.warmup)
    right = summarize(args.right, args.warmup)
    result = {args.left_label: left, args.right_label: right}
    payload = json.dumps(result, indent=2, sort_keys=True)
    print(payload)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(payload + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()

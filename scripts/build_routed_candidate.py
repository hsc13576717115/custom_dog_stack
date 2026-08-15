#!/usr/bin/env python3
"""Build a validated locomotion-plus-stand routed candidate directory."""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import yaml


CONTRACT_KEYS = (
    "observations",
    "actions",
    "joint_ids_map",
    "default_joint_pos",
    "stiffness",
    "damping",
    "step_dt",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("locomotion", type=Path)
    parser.add_argument("stand", type=Path)
    parser.add_argument("output", type=Path)
    return parser.parse_args()


def artifacts(candidate: Path) -> tuple[Path, Path]:
    candidate = candidate.resolve()
    policy = candidate / "exported" / "policy.onnx"
    deploy = candidate / "params" / "deploy.yaml"
    for path in (policy, deploy):
        if not path.is_file():
            raise SystemExit(f"Missing candidate artifact: {path}")
    return policy, deploy


def load_deploy(path: Path) -> dict[str, object]:
    with path.open(encoding="utf-8") as stream:
        config = yaml.safe_load(stream)
    if not isinstance(config, dict):
        raise SystemExit(f"Deploy contract is not a mapping: {path}")
    return config


def main() -> None:
    args = parse_args()
    locomotion_policy, locomotion_deploy = artifacts(args.locomotion)
    stand_policy, stand_deploy = artifacts(args.stand)
    locomotion_cfg = load_deploy(locomotion_deploy)
    stand_cfg = load_deploy(stand_deploy)
    mismatches = [key for key in CONTRACT_KEYS if locomotion_cfg.get(key) != stand_cfg.get(key)]
    if mismatches:
        raise SystemExit(f"Stand/locomotion deployment contracts differ: {mismatches}")

    output = args.output.resolve()
    exported = output / "exported"
    params = output / "params"
    exported.mkdir(parents=True, exist_ok=True)
    params.mkdir(parents=True, exist_ok=True)
    shutil.copy2(locomotion_policy, exported / "policy.onnx")
    shutil.copy2(locomotion_deploy, params / "deploy.yaml")
    shutil.copy2(stand_policy, exported / "stand_policy.onnx")
    shutil.copy2(stand_deploy, params / "stand_deploy.yaml")
    metadata = {
        "locomotion_candidate": str(args.locomotion.resolve()),
        "stand_candidate": str(args.stand.resolve()),
        "routing": {
            "stand_enter_planar": 0.015,
            "stand_enter_yaw": 0.025,
            "stand_exit_planar": 0.025,
            "stand_exit_yaw": 0.04,
            "blend_seconds": 0.30,
        },
    }
    (output / "routing.json").write_text(
        json.dumps(metadata, indent=2) + "\n", encoding="utf-8"
    )
    print(output)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Prepare an isolated runtime config for unitree_rl_lab's Go2 controller."""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path

import yaml


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--template", type=Path, required=True)
    parser.add_argument("--policy-dir", type=Path, required=True)
    parser.add_argument("--source-binary", type=Path, required=True)
    parser.add_argument("--runtime-dir", type=Path, required=True)
    args = parser.parse_args()

    policy_dir = args.policy_dir.resolve()
    if not (policy_dir / "exported" / "policy.onnx").is_file():
        raise FileNotFoundError(f"Missing policy.onnx under {policy_dir}")
    if not (policy_dir / "params" / "deploy.yaml").is_file():
        raise FileNotFoundError(f"Missing deploy.yaml under {policy_dir}")
    if not args.source_binary.is_file():
        raise FileNotFoundError(f"Missing unitree_rl_lab controller: {args.source_binary}")

    with args.template.open(encoding="utf-8") as stream:
        config = yaml.safe_load(stream)
    config["FSM"]["Velocity"]["policy_dir"] = str(policy_dir)

    config_dir = args.runtime_dir / "config"
    binary_dir = args.runtime_dir / "build"
    config_dir.mkdir(parents=True, exist_ok=True)
    binary_dir.mkdir(parents=True, exist_ok=True)
    with (config_dir / "config.yaml").open("w", encoding="utf-8") as stream:
        yaml.safe_dump(config, stream, sort_keys=False)
    target_binary = binary_dir / "go2_ctrl"
    shutil.copy2(args.source_binary, target_binary)
    target_binary.chmod(0o755)
    print(target_binary)


if __name__ == "__main__":
    main()

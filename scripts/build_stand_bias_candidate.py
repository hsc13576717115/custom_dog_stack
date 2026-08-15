#!/usr/bin/env python3
"""Build a stand candidate with a symmetric policy-order hip target bias."""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import yaml


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--hip-bias", type=float, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not 0.0 <= args.hip_bias <= 0.12:
        raise SystemExit("--hip-bias must be in [0.0, 0.12] rad")
    source = args.source.resolve()
    policy = source / "exported/policy.onnx"
    deploy = source / "params/deploy.yaml"
    for path in (policy, deploy):
        if not path.is_file():
            raise SystemExit(f"Missing stand artifact: {path}")

    output = args.output.resolve()
    (output / "exported").mkdir(parents=True, exist_ok=True)
    (output / "params").mkdir(parents=True, exist_ok=True)
    shutil.copy2(policy, output / "exported/policy.onnx")
    config = yaml.safe_load(deploy.read_text(encoding="utf-8"))
    bias = float(args.hip_bias)
    # Policy hip order maps to FL, FR, RL, RR in the canonical SDK order.
    config["constant_joint_target_bias"] = [
        -bias,
        bias,
        -bias,
        bias,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
    ]
    (output / "params/deploy.yaml").write_text(
        yaml.safe_dump(config, sort_keys=False), encoding="utf-8"
    )
    metadata = {
        "source_candidate": str(source),
        "hip_bias_rad": bias,
        "scope": "stand expert only",
    }
    (output / "stand_bias.json").write_text(
        json.dumps(metadata, indent=2) + "\n", encoding="utf-8"
    )
    print(output)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Verify that a selection was trained with the required robot asset."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import yaml


def training_asset(selection_path: Path) -> str:
    env_path = selection_path.parent.parent / "params/env.yaml"
    try:
        snapshot = yaml.load(env_path.read_text(encoding="utf-8"), Loader=yaml.BaseLoader)
        value = snapshot["scene"]["robot"]["spawn"]["asset_path"]
    except (FileNotFoundError, OSError, TypeError, KeyError, yaml.YAMLError) as error:
        raise ValueError(f"cannot read training asset from {env_path}: {error}") from error
    if not isinstance(value, str) or not value:
        raise ValueError(f"training asset path is missing from {env_path}")
    return value


def verify(selection_path: Path, expected_basename: str) -> dict[str, object]:
    selection = json.loads(selection_path.read_text(encoding="utf-8"))
    if selection.get("accepted") is not True:
        raise ValueError(f"selection is not accepted: {selection_path}")
    asset_path = training_asset(selection_path)
    actual_basename = Path(asset_path).name
    if actual_basename != expected_basename:
        raise ValueError(
            f"selection used {actual_basename!r}, expected {expected_basename!r}: "
            f"{selection_path}"
        )
    return {
        "selection": str(selection_path.resolve()),
        "training_asset_path": asset_path,
        "expected_asset_basename": expected_basename,
        "accepted": True,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("selection", type=Path)
    parser.add_argument("--expected", required=True, help="Required asset filename")
    args = parser.parse_args()
    print(json.dumps(verify(args.selection.resolve(), args.expected), indent=2))


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Fail unless a selective-collision runtime report satisfies its contract."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def validate_report(
    report: dict[str, object], expected_asset: Path | None = None
) -> list[str]:
    failures: list[str] = []
    if report.get("filtered_pair_count") != report.get("expected_filtered_pair_count"):
        failures.append("filtered-pair count does not match the contract")
    if report.get("nominal_pose_held") is not True:
        failures.append("nominal pose was not held during geometry validation")
    if report.get("nominal_nonfoot_contact_steps") != 0:
        failures.append("nominal stance has persistent non-foot contact")
    forced_steps = report.get("forced_cross_leg_contact_steps")
    if not isinstance(forced_steps, int) or forced_steps <= 0:
        failures.append("forced cross-leg pose did not produce contact")
    if expected_asset is not None:
        expected_asset = expected_asset.resolve()
        expected_hash = hashlib.sha256(expected_asset.read_bytes()).hexdigest()
        report_path = report.get("asset_path")
        if not isinstance(report_path, str) or Path(report_path).resolve() != expected_asset:
            failures.append("report asset path does not match the current selective URDF")
        if report.get("asset_sha256") != expected_hash:
            failures.append("report asset SHA-256 does not match the current selective URDF")
    return failures


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("report", type=Path)
    parser.add_argument("--asset", type=Path, help="Require this exact current asset")
    args = parser.parse_args()

    try:
        report = json.loads(args.report.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise SystemExit(f"Invalid selective-collision report {args.report}: {exc}") from exc
    if not isinstance(report, dict):
        raise SystemExit(f"Selective-collision report must be a JSON object: {args.report}")

    failures = validate_report(report, args.asset)
    if failures:
        details = "; ".join(failures)
        raise SystemExit(f"Selective-collision runtime gate failed: {details}")
    print(f"Selective-collision runtime gate passed: {args.report}")


if __name__ == "__main__":
    main()

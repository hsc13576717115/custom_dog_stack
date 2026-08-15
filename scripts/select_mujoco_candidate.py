#!/usr/bin/env python3
"""Rank MuJoCo grid candidates by explicit gates, never by training reward."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


SAFETY_GATES = {"height", "tilt", "not_fallen"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("evaluation", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--top", type=int, default=3)
    parser.add_argument(
        "--label-prefix",
        help="Only rank candidates whose evaluation label starts with this prefix",
    )
    return parser.parse_args()


def normalized_tracking_error(row: dict[str, object]) -> float:
    vx_limit = max(0.10, 0.15 * abs(float(row["command_vx"])))
    return max(
        float(row["error_vx"]) / vx_limit,
        float(row["error_vy"]) / 0.07,
        float(row["error_wz"]) / 0.10,
    )


def rank_candidate(label: str, rows: list[dict[str, object]]) -> dict[str, object]:
    absolute_failures = []
    relative_failures = []
    safety_failures = 0
    for row in rows:
        failed_absolute = [
            name for name, passed in row.get("absolute_gates", {}).items() if not passed
        ]
        failed_relative = [
            name for name, passed in row.get("relative_gates", {}).items() if not passed
        ]
        safety_failures += sum(name in SAFETY_GATES for name in failed_absolute)
        if failed_absolute:
            absolute_failures.append(
                {"command_index": row["command_index"], "gates": failed_absolute}
            )
        if failed_relative:
            relative_failures.append(
                {"command_index": row["command_index"], "gates": failed_relative}
            )
    max_tracking_ratio = max(normalized_tracking_error(row) for row in rows)
    score = (
        safety_failures,
        len(absolute_failures) + len(relative_failures),
        len(absolute_failures),
        max_tracking_ratio,
        sum(float(row["max_tilt_deg"]) for row in rows) / len(rows),
        sum(float(row["hip_outward_max_deg"]) for row in rows) / len(rows),
        label,
    )
    return {
        "label": label,
        "candidate": rows[0].get("candidate_path"),
        "accepted": not absolute_failures and not relative_failures,
        "score": list(score[:-1]),
        "safety_failures": safety_failures,
        "absolute_command_failures": absolute_failures,
        "relative_command_failures": relative_failures,
        "max_normalized_tracking_error": max_tracking_ratio,
        "sort_key": score,
    }


def select(
    evaluation: dict[str, object],
    top: int,
    label_prefix: str | None = None,
) -> dict[str, object]:
    if top <= 0:
        raise ValueError("top must be positive")
    grouped: dict[str, list[dict[str, object]]] = {}
    for row in evaluation["rows"]:
        label = str(row["candidate"])
        if label_prefix is None or label.startswith(label_prefix):
            grouped.setdefault(label, []).append(row)
    if not grouped:
        suffix = f" matching prefix {label_prefix!r}" if label_prefix else ""
        raise ValueError(f"evaluation contains no candidates{suffix}")
    ranked = sorted(
        (rank_candidate(label, rows) for label, rows in grouped.items()),
        key=lambda item: item["sort_key"],
    )
    for item in ranked:
        item.pop("sort_key")
    selected = ranked[0]
    return {
        "evaluation": evaluation.get("stage", "A"),
        "accepted": bool(selected["accepted"]),
        "selected": selected["label"],
        "selected_candidate": selected["candidate"],
        "top_candidates": ranked[:top],
        "selection_rule": (
            (f"label prefix {label_prefix!r}; " if label_prefix else "")
            + "safety failures, failed command/gate groups, absolute command failures, "
            "worst normalized tracking error, mean tilt, mean maximum hip outward"
        ),
    }


def main() -> None:
    args = parse_args()
    evaluation = json.loads(args.evaluation.read_text(encoding="utf-8"))
    result = select(evaluation, args.top, args.label_prefix)
    encoded = json.dumps(result, indent=2)
    print(encoded)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(encoded + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()

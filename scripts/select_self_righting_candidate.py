#!/usr/bin/env python3
"""Select a self-righting checkpoint by canonical-pose success and recovery time."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def result_spec(value: str) -> tuple[str, Path]:
    if "=" not in value:
        raise argparse.ArgumentTypeError("result must be LABEL=PATH")
    label, path = value.split("=", 1)
    return label, Path(path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--result", action="append", type=result_spec, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def select(results: list[tuple[str, dict[str, object]]]) -> dict[str, object]:
    ranked = []
    for label, result in results:
        success_times = [
            float(row["success_time_s"])
            for row in result["results"]
            if row["success_time_s"] is not None
        ]
        total = int(result["total_states"])
        passed = int(result["states_passed"])
        passes_all = bool(result["passes_all"])
        max_time = max(success_times) if len(success_times) == total else float("inf")
        mean_time = sum(success_times) / len(success_times) if success_times else float("inf")
        ranked.append(
            {
                "label": label,
                "candidate": result["candidate"],
                "passes_all": passes_all,
                "states_passed": passed,
                "total_states": total,
                "max_success_time_s": max_time if passes_all else None,
                "mean_success_time_s": mean_time if passes_all else None,
                "sort_key": (not passes_all, -passed, max_time, mean_time, label),
            }
        )
    ranked.sort(key=lambda item: item["sort_key"])
    for item in ranked:
        item.pop("sort_key")
    return {
        "accepted": bool(ranked[0]["passes_all"]),
        "selected": ranked[0]["label"],
        "selected_candidate": ranked[0]["candidate"],
        "ranked": ranked,
        "selection_rule": "all canonical poses, then worst recovery time, then mean recovery time",
    }


def main() -> None:
    args = parse_args()
    results = [
        (label, json.loads(path.read_text(encoding="utf-8")))
        for label, path in args.result
    ]
    selected = select(results)
    encoded = json.dumps(selected, indent=2, allow_nan=False)
    print(encoded)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(encoded + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()

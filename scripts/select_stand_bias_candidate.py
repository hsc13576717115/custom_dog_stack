#!/usr/bin/env python3
"""Select the smallest stand hip bias that passes every absolute gate."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

from select_mujoco_candidate import rank_candidate


BIAS_LABEL = re.compile(r"^STAND_BIAS_(\d+)$")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("evaluation", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def select(evaluation: dict[str, object]) -> dict[str, object]:
    grouped: dict[str, list[dict[str, object]]] = {}
    for row in evaluation["rows"]:
        grouped.setdefault(str(row["candidate"]), []).append(row)
    ranked = [rank_candidate(label, rows) for label, rows in grouped.items()]
    candidates = []
    for item in ranked:
        match = BIAS_LABEL.fullmatch(str(item["label"]))
        if match and item["accepted"]:
            candidates.append((int(match.group(1)), item))
    if candidates:
        selected = min(candidates, key=lambda pair: (pair[0], pair[1]["sort_key"]))[1]
    else:
        selected = min(ranked, key=lambda item: item["sort_key"])
    for item in ranked:
        item.pop("sort_key")
    ranked.sort(
        key=lambda item: (
            not bool(item["accepted"]),
            int(BIAS_LABEL.fullmatch(str(item["label"])).group(1))
            if BIAS_LABEL.fullmatch(str(item["label"]))
            else 10**9,
            item["score"],
        )
    )
    return {
        "evaluation": evaluation.get("stage", "A"),
        "accepted": bool(selected["accepted"]),
        "selected": selected["label"],
        "selected_candidate": selected["candidate"],
        "top_candidates": ranked[:3],
        "selection_rule": "smallest symmetric hip bias passing every absolute gate",
    }


def main() -> None:
    args = parse_args()
    evaluation = json.loads(args.evaluation.read_text(encoding="utf-8"))
    result = select(evaluation)
    encoded = json.dumps(result, indent=2)
    print(encoded)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(encoded + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()

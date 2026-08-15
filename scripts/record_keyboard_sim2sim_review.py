#!/usr/bin/env python3
"""Record the required human keyboard review for ranked Stage-A candidates."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path


def ranked_labels(ranking: dict[str, object], limit: int = 3) -> list[str]:
    candidates = ranking.get("top_candidates")
    if not isinstance(candidates, list):
        raise ValueError("ranking does not contain top_candidates")
    labels = [
        str(item["label"])
        for item in candidates[:limit]
        if isinstance(item, dict) and isinstance(item.get("label"), str)
    ]
    if len(labels) < 2:
        raise ValueError("ranking must contain at least two candidate labels")
    return labels


def build_record(
    ranking: dict[str, object], reviewed: list[str], accepted: str, notes: str
) -> dict[str, object]:
    ranked = ranked_labels(ranking)
    reviewed_unique = list(dict.fromkeys(reviewed))
    if len(set(reviewed_unique) & set(ranked)) < 2:
        raise ValueError(f"review at least two ranked candidates: {ranked}")
    if accepted not in reviewed_unique:
        raise ValueError("accepted candidate must be included in reviewed candidates")
    if accepted not in ranked:
        raise ValueError(f"accepted candidate must be one of the ranked candidates: {ranked}")
    if not notes.strip():
        raise ValueError("review notes must not be empty")
    return {
        "accepted": True,
        "accepted_candidate": accepted,
        "reviewed_candidates": reviewed_unique,
        "ranked_candidates": ranked,
        "notes": notes.strip(),
        "recorded_at": datetime.now(timezone.utc).isoformat(),
    }


def main() -> None:
    project_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reviewed", nargs="+", required=True)
    parser.add_argument("--accepted", required=True)
    parser.add_argument("--notes", required=True)
    parser.add_argument(
        "--ranking",
        type=Path,
        default=project_root / "reports/stage_a_six_checkpoint_ranking.json",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=project_root / "reports/keyboard_sim2sim_review.json",
    )
    args = parser.parse_args()
    ranking = json.loads(args.ranking.read_text(encoding="utf-8"))
    record = build_record(ranking, args.reviewed, args.accepted, args.notes)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(record, indent=2))


if __name__ == "__main__":
    main()

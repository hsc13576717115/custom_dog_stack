#!/usr/bin/env python3
"""Select a terrain checkpoint only when terrain and flat-regression gates pass."""

from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path


def result_spec(value: str) -> tuple[str, Path]:
    if "=" not in value:
        raise argparse.ArgumentTypeError("terrain result must be LABEL=PATH")
    label, path = value.split("=", 1)
    return label, Path(path)


def load_flat_ranker():
    path = Path(__file__).with_name("select_mujoco_candidate.py")
    spec = importlib.util.spec_from_file_location("select_mujoco_candidate", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def select(
    flat_evaluation: dict[str, object],
    terrain_results: list[tuple[str, dict[str, object]]],
) -> dict[str, object]:
    ranker = load_flat_ranker()
    grouped: dict[str, list[dict[str, object]]] = {}
    for row in flat_evaluation["rows"]:
        grouped.setdefault(str(row["candidate"]), []).append(row)
    terrain_by_label = dict(terrain_results)
    if not set(terrain_by_label).issubset(grouped):
        raise ValueError(
            "flat and terrain labels differ: "
            f"flat={sorted(grouped)}, terrain={sorted(terrain_by_label)}"
        )
    grouped = {label: grouped[label] for label in terrain_by_label}

    ranked = []
    for label, rows in grouped.items():
        flat = ranker.rank_candidate(label, rows)
        flat.pop("sort_key")
        terrain = terrain_by_label[label]
        terrain_failures = int(terrain["total_commands"]) - int(
            terrain["commands_passed"]
        )
        family_failures = int(terrain.get("total_terrain_families", 0)) - int(
            terrain.get("terrain_families_passed", 0)
        )
        family_command_failures = int(terrain.get("total_family_commands", 0)) - int(
            terrain.get("family_commands_passed", 0)
        )
        success_rows = list(terrain["rows"])
        success_rows.extend(terrain.get("terrain_families", []))
        success_rows.extend(terrain.get("family_commands", []))
        min_success_rate = min(float(row["success_rate"]) for row in success_rows)
        accepted = bool(flat["accepted"]) and bool(terrain["passes_all"])
        score = (
            not accepted,
            len(flat["absolute_command_failures"]),
            family_command_failures,
            family_failures,
            terrain_failures,
            -min_success_rate,
            float(flat["max_normalized_tracking_error"]),
            label,
        )
        ranked.append(
            {
                "label": label,
                "candidate": flat["candidate"],
                "accepted": accepted,
                "flat_accepted": bool(flat["accepted"]),
                "terrain_accepted": bool(terrain["passes_all"]),
                "flat_absolute_command_failures": flat["absolute_command_failures"],
                "terrain_command_failures": terrain_failures,
                "terrain_family_failures": family_failures,
                "terrain_family_command_failures": family_command_failures,
                "terrain_min_success_rate": min_success_rate,
                "max_normalized_flat_tracking_error": flat[
                    "max_normalized_tracking_error"
                ],
                "sort_key": score,
            }
        )
    ranked.sort(key=lambda item: item["sort_key"])
    for item in ranked:
        item.pop("sort_key")
    winner = ranked[0]
    return {
        "stage": terrain_results[0][1]["stage"],
        "accepted": bool(winner["accepted"]),
        "selected": winner["label"],
        "selected_candidate": winner["candidate"],
        "ranked": ranked,
        "selection_rule": (
            "pass complete Stage-D flat MuJoCo regression and every grouped Isaac terrain "
            "family-command cell; then flat, family-command, family and command failures, "
            "worst terrain success rate, "
            "and flat tracking error"
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--flat-grid", type=Path, required=True)
    parser.add_argument("--terrain-result", action="append", type=result_spec, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    flat = json.loads(args.flat_grid.read_text(encoding="utf-8"))
    terrain = [
        (label, json.loads(path.read_text(encoding="utf-8")))
        for label, path in args.terrain_result
    ]
    result = select(flat, terrain)
    encoded = json.dumps(result, indent=2)
    print(encoded)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(encoded + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()

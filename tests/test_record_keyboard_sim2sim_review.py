from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
PATH = ROOT / "scripts/record_keyboard_sim2sim_review.py"
SPEC = importlib.util.spec_from_file_location("record_keyboard_sim2sim_review", PATH)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


RANKING = {
    "top_candidates": [
        {"label": "model_999"},
        {"label": "model_975"},
        {"label": "model_875"},
    ]
}


def test_record_requires_two_ranked_reviews_and_notes() -> None:
    record = MODULE.build_record(
        RANKING,
        ["model_999", "model_975"],
        "model_999",
        "Better low-speed stepping and less body tilt.",
    )
    assert record["accepted_candidate"] == "model_999"
    assert record["accepted"] is True
    with pytest.raises(ValueError, match="at least two"):
        MODULE.build_record(RANKING, ["model_999"], "model_999", "notes")
    with pytest.raises(ValueError, match="notes"):
        MODULE.build_record(
            RANKING, ["model_999", "model_975"], "model_999", "  "
        )


def test_accepted_candidate_must_be_ranked_and_reviewed() -> None:
    with pytest.raises(ValueError, match="included"):
        MODULE.build_record(
            RANKING, ["model_999", "model_975"], "model_875", "notes"
        )
    with pytest.raises(ValueError, match="ranked"):
        MODULE.build_record(
            RANKING, ["model_999", "model_975", "other"], "other", "notes"
        )

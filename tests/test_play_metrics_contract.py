from __future__ import annotations

import ast
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PLAY = ROOT / "rl/scripts/play.py"


def test_fixed_command_metrics_are_machine_readable_and_track_resets() -> None:
    source = PLAY.read_text(encoding="utf-8")
    ast.parse(source)
    assert '"--metrics_json"' in source
    assert "done_count +=" in source
    assert "environments_done |= done_mask" in source
    assert '"success_rate"' in source
    assert '"height_p05_m"' in source
    assert '"tilt_p95_deg"' in source
    assert '"contact_transitions_min_per_leg"' in source


def test_metrics_output_requires_a_fixed_command() -> None:
    source = PLAY.read_text(encoding="utf-8")
    assert 'parser.error("--metrics_json requires --fixed_command")' in source

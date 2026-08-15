from __future__ import annotations

from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_training_starts_from_selected_rf700() -> None:
    script = (ROOT / "scripts/train_robust_stand_fix.sh").read_text()
    assert "model_700.pt" in script
    assert "CUSTOM_DOG_MAX_ITERATIONS:-150" in script
    assert "CUSTOM_DOG_LOAD_OPTIMIZER:-0" in script
    assert "CUSTOM_DOG_RESET_POLICY_STD:-0.10" in script


def test_evaluation_uses_absolute_stage_a_gate() -> None:
    script = (ROOT / "scripts/evaluate_robust_stand_fix.sh").read_text()
    assert "--absolute-only" in script
    assert "--baseline-label RF_700" in script
    assert "robust_stand_fix_selection.json" in script
    assert "RSF_${iteration}" in script


def test_latest_wrapper_uses_only_formal_run_name() -> None:
    script = (ROOT / "scripts/evaluate_latest_robust_stand_fix.sh").read_text()
    assert "*_robust_stand_fix_from_rf700" in script
    assert "robust_stand_fix_config_smoke" not in script
    assert "CUSTOM_DOG_RUN_DIR" in script

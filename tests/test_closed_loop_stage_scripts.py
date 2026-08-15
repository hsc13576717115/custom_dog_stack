from __future__ import annotations

import os
from pathlib import Path
import shutil
import subprocess


ROOT = Path(__file__).resolve().parents[1]


def test_expansion_evaluator_uses_cumulative_absolute_grid_and_selective_model() -> None:
    source = (ROOT / "scripts/evaluate_closed_loop_stage.sh").read_text(encoding="utf-8")
    assert "--absolute-only" in source
    assert '--stage "${stage}"' in source
    assert "generate_selective_mujoco.py" in source
    assert "build_routed_candidate.py" in source
    assert "CUSTOM_DOG_STAND_CANDIDATE" in source
    assert "select_mujoco_candidate.py" in source


def test_stage_gate_requires_previous_explicit_selection() -> None:
    source = (ROOT / "scripts/train_next_closed_loop_stage_gated.sh").read_text(
        encoding="utf-8"
    )
    assert ".accepted == true and .selected_candidate != null" in source
    assert 'startsWith("SC_")' not in source
    assert 'startswith("SC_")' in source
    assert "CUSTOM_DOG_STAND_CANDIDATE" in source
    assert "train_closed_loop_stage.sh" in source


def test_collision_gate_requires_two_routed_seeds_and_composes_stand() -> None:
    gate = (ROOT / "scripts/train_selective_collision_stage_a_gated.sh").read_text(
        encoding="utf-8"
    )
    evaluator = (ROOT / "scripts/evaluate_selective_collision_stage_a.sh").read_text(
        encoding="utf-8"
    )
    assert "routed_stage_a_selection.json" in gate
    assert "routed_robust_foundation_selection.json" in gate
    assert 'startswith("RFR_")' in gate
    assert "CUSTOM_DOG_STAND_CANDIDATE" in gate
    assert "build_routed_candidate.py" in evaluator
    assert "validate_selective_self_collision.py" in gate
    assert "--label-prefix SC_" in evaluator


def test_gated_pipeline_interleaves_locomotion_and_recovery_on_one_gpu() -> None:
    source = (ROOT / "scripts/queue_gated_training_pipeline.sh").read_text(
        encoding="utf-8"
    )
    expected = (
        "stage_b",
        "recovery_r0",
        "stage_c",
        "recovery_r1",
        "stage_d",
        "recovery_r2",
    )
    positions = [source.index(f'queue_after "${{{name}}}"') for name in expected]
    assert positions == sorted(positions)
    assert "train_self_righting_r0_gated.sh" in source
    assert "train_next_self_righting_stage_gated.sh\" R2" in source


def test_second_seed_requires_accepted_routed_seed42_and_reuses_stand_expert() -> None:
    source = (ROOT / "scripts/train_robust_foundation_seed73_gated.sh").read_text(
        encoding="utf-8"
    )
    assert "routed_stage_a_selection.json" in source
    assert 'startswith("ROUTED_")' in source
    assert "CUSTOM_DOG_STAND_CANDIDATE" in source
    assert "evaluate_routed_robust_foundation.sh" in source


def test_routed_second_seed_evaluator_only_ranks_new_seed_checkpoints() -> None:
    source = (ROOT / "scripts/evaluate_routed_robust_foundation.sh").read_text(
        encoding="utf-8"
    )
    assert "build_routed_candidate.py" in source
    assert 'label="RFR_${iteration}"' in source
    assert "--absolute-only" in source
    assert "routed_robust_foundation_selection.json" in source


def test_checkpoint_export_retries_and_rejects_stale_onnx() -> None:
    source = (ROOT / "scripts/export_checkpoint_candidates.sh").read_text(
        encoding="utf-8"
    )
    assert "CUSTOM_DOG_EXPORT_ATTEMPTS" in source
    assert "CUSTOM_DOG_EXPORT_RETRY_DELAY" in source
    assert "export_signature_before" in source
    assert "export_signature_after" in source
    assert "Failed to produce a fresh ONNX export" in source


def _fake_export_project(tmp_path: Path, play_export_body: str) -> tuple[Path, Path]:
    project = tmp_path / "project"
    scripts = project / "scripts"
    run_dir = project / "run"
    scripts.mkdir(parents=True)
    (run_dir / "params").mkdir(parents=True)
    shutil.copy2(ROOT / "scripts/export_checkpoint_candidates.sh", scripts)
    exporter = scripts / "export_checkpoint_candidates.sh"
    exporter.chmod(0o755)
    play_export = scripts / "play_export.sh"
    play_export.write_text(play_export_body, encoding="utf-8")
    play_export.chmod(0o755)
    (run_dir / "model_10.pt").write_bytes(b"checkpoint")
    (run_dir / "params/deploy.yaml").write_text("observations: 51\n", encoding="utf-8")
    return exporter, run_dir


def test_checkpoint_export_recovers_from_one_127_failure(tmp_path: Path) -> None:
    exporter, run_dir = _fake_export_project(
        tmp_path,
        """#!/usr/bin/env bash
set -eu
checkpoint="$1"
run_dir="$(dirname "${checkpoint}")"
counter="${run_dir}/attempts"
attempt=0
[[ ! -f "${counter}" ]] || attempt="$(cat "${counter}")"
attempt=$((attempt + 1))
printf '%s' "${attempt}" > "${counter}"
if [[ ${attempt} -eq 1 ]]; then
    exit 127
fi
mkdir -p "${run_dir}/exported"
printf 'fresh-onnx' > "${run_dir}/exported/policy.onnx"
""",
    )
    env = os.environ.copy()
    env.update({"CUSTOM_DOG_EXPORT_RETRY_DELAY": "0"})
    result = subprocess.run(
        [str(exporter), str(run_dir), "FakeTask-v0", "model_10.pt"],
        check=False,
        capture_output=True,
        text=True,
        env=env,
    )
    assert result.returncode == 0, result.stderr
    assert (run_dir / "attempts").read_text(encoding="utf-8") == "2"
    assert (
        run_dir / "evaluation/candidates/model_10/exported/policy.onnx"
    ).read_bytes() == b"fresh-onnx"


def test_checkpoint_export_rejects_unchanged_stale_policy(tmp_path: Path) -> None:
    exporter, run_dir = _fake_export_project(
        tmp_path,
        """#!/usr/bin/env bash
exit 0
""",
    )
    (run_dir / "exported").mkdir()
    (run_dir / "exported/policy.onnx").write_bytes(b"stale-onnx")
    env = os.environ.copy()
    env.update(
        {
            "CUSTOM_DOG_EXPORT_ATTEMPTS": "1",
            "CUSTOM_DOG_EXPORT_RETRY_DELAY": "0",
        }
    )
    result = subprocess.run(
        [str(exporter), str(run_dir), "FakeTask-v0", "model_10.pt"],
        check=False,
        capture_output=True,
        text=True,
        env=env,
    )
    assert result.returncode == 1
    assert "Failed to produce a fresh ONNX export" in result.stderr

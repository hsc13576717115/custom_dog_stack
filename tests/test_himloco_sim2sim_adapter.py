from __future__ import annotations

import ast
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
RUNNER = ROOT / "sim2sim/custom_dog/run_sim2sim.py"
TELEOP = ROOT / "scripts/teleop_mujoco_policy.sh"
HIMLOCO_TELEOP = ROOT / "scripts/teleop_himloco_checkpoint.sh"


def _load_history_update():
    tree = ast.parse(RUNNER.read_text(encoding="utf-8"))
    node = next(
        item
        for item in tree.body
        if isinstance(item, ast.FunctionDef) and item.name == "update_current_first_history"
    )
    namespace = {"np": np}
    exec(compile(ast.Module(body=[node], type_ignores=[]), str(RUNNER), "exec"), namespace)
    return namespace["update_current_first_history"]


def test_himloco_history_is_current_first_and_zero_initialized() -> None:
    update = _load_history_update()
    history = np.zeros(6, dtype=np.float32)

    history = update(history, np.array([1.0, 2.0], dtype=np.float32))
    np.testing.assert_array_equal(history, [1.0, 2.0, 0.0, 0.0, 0.0, 0.0])

    history = update(history, np.array([3.0, 4.0], dtype=np.float32))
    np.testing.assert_array_equal(history, [3.0, 4.0, 1.0, 2.0, 0.0, 0.0])


def test_runner_declares_himloco_dual_network_contract() -> None:
    source = RUNNER.read_text(encoding="utf-8")
    assert '"velocity_commands",\n    "base_ang_vel"' in source
    assert 'self.cfg.get("use_encoder", False)' in source
    assert "self.single_observation_dim * self.history_frames" in source
    assert "encoder_outputs[0].shape[-1] != 19" in source
    assert "--encoder" in source


def test_teleop_auto_detects_himloco_encoder() -> None:
    source = TELEOP.read_text(encoding="utf-8")
    assert '${candidate}/exported/encoder.onnx' in source
    assert '--encoder "${candidate}/exported/encoder.onnx"' in source


def test_checkpoint_teleop_exports_then_launches_mujoco() -> None:
    source = HIMLOCO_TELEOP.read_text(encoding="utf-8")
    assert "--export-only" in source
    assert "--checkpoint" in source
    assert 'teleop_mujoco_policy.sh" "${run_dir}"' in source

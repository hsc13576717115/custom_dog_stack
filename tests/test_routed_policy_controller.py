from __future__ import annotations

import ast
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
RUNNER = ROOT / "sim2sim/custom_dog/run_sim2sim.py"
EVALUATOR = ROOT / "scripts/evaluate_mujoco_grid.py"
TELEOP = ROOT / "scripts/teleop_mujoco_policy.sh"


def _load_router():
    tree = ast.parse(RUNNER.read_text())
    node = next(
        item
        for item in tree.body
        if isinstance(item, ast.ClassDef) and item.name == "RoutedPolicyController"
    )
    module = ast.Module(
        body=[
            ast.ImportFrom(
                module="__future__",
                names=[ast.alias(name="annotations")],
                level=0,
            ),
            node,
        ],
        type_ignores=[],
    )
    namespace = {
        "np": np,
        "PolicyController": object,
        "quintic_smoothstep": lambda value: value * value * value * (10 - 15 * value + 6 * value * value),
    }
    exec(compile(ast.fix_missing_locations(module), str(RUNNER), "exec"), namespace)
    return namespace["RoutedPolicyController"]


class _FakeController:
    def __init__(self, action: float):
        self.command = np.zeros(3)
        self.requested_action = np.full(12, action)
        self.reset_seed = None
        self.applied = None

    def set_command(self, command):
        self.command = np.asarray(command, dtype=np.float64).copy()

    def reset_history(self, previous_action=None):
        self.reset_seed = np.asarray(previous_action).copy()

    def infer_action(self, _model, _data):
        return np.zeros(1), self.requested_action.copy()

    def apply_action(self, _model, _data, _observation, action):
        self.applied = np.asarray(action).copy()


def _router():
    router_type = _load_router()
    router = router_type.__new__(router_type)
    router.locomotion = _FakeController(0.8)
    router.stand = _FakeController(0.2)
    router.stand_enter_planar = 0.015
    router.stand_enter_yaw = 0.025
    router.stand_exit_planar = 0.025
    router.stand_exit_yaw = 0.04
    router.blend_steps = 3
    router.mode = router.LOCOMOTION
    router.current_action = np.zeros(12)
    router._blend_start_action = np.zeros(12)
    router._blend_index = router.blend_steps
    return router


def test_router_uses_hysteresis_around_zero_command() -> None:
    router = _router()
    router.set_command(np.zeros(3))
    assert router.mode == router.STAND

    router.set_command(np.array([0.02, 0.0, 0.0]))
    assert router.mode == router.STAND

    router.set_command(np.array([0.03, 0.0, 0.0]))
    assert router.mode == router.LOCOMOTION

    router.set_command(np.zeros(3))
    assert router.mode == router.STAND
    router.set_command(np.array([0.0, 0.0, 0.05]))
    assert router.mode == router.LOCOMOTION


def test_router_blends_the_first_action_after_switch() -> None:
    router = _router()
    router.set_command(np.zeros(3))
    router.update(None, None)

    assert np.all(router.current_action > 0.0)
    assert np.all(router.current_action < router.stand.requested_action)
    np.testing.assert_allclose(router.stand.applied, router.current_action)


def test_grid_evaluator_auto_detects_routed_artifacts() -> None:
    source = EVALUATOR.read_text()
    assert 'candidate / "exported" / "stand_policy.onnx"' in source
    assert 'candidate / "params" / "stand_deploy.yaml"' in source
    assert '"--stand-policy"' in source
    assert '"--stand-deploy-yaml"' in source


def test_teleop_auto_detects_routed_artifacts() -> None:
    source = TELEOP.read_text()
    assert '${candidate}/exported/stand_policy.onnx' in source
    assert '${candidate}/params/stand_deploy.yaml' in source
    assert '--stand-policy "${candidate}/exported/stand_policy.onnx"' in source
    assert '--stand-deploy-yaml "${candidate}/params/stand_deploy.yaml"' in source


def test_policy_controller_supports_a_constant_target_bias() -> None:
    source = RUNNER.read_text(encoding="utf-8")
    assert 'self.cfg.get("constant_joint_target_bias", [0.0] * 12)' in source
    assert "self.constant_target_bias_policy + blend * self.target_bias_policy" in source

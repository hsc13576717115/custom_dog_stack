from __future__ import annotations

import ast
from pathlib import Path
from threading import Lock
from types import SimpleNamespace

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RUNNER = PROJECT_ROOT / "sim2sim/custom_dog/run_sim2sim.py"


def _load_controls():
    tree = ast.parse(RUNNER.read_text(encoding="utf-8"))
    class_node = next(
        node
        for node in tree.body
        if isinstance(node, ast.ClassDef) and node.name == "InteractiveControls"
    )
    module = ast.Module(
        body=[
            ast.ImportFrom(
                module="__future__",
                names=[ast.alias(name="annotations")],
                level=0,
            ),
            class_node,
        ],
        type_ignores=[],
    )
    keys = {
        name: index
        for index, name in enumerate(("P", "R", "V", "W", "S", "A", "D", "Q", "E", "X"))
    }
    glfw = SimpleNamespace(**{f"KEY_{name}": value for name, value in keys.items()})
    namespace = {
        "Lock": Lock,
        "PolicyController": object,
        "mujoco": SimpleNamespace(glfw=SimpleNamespace(glfw=glfw)),
        "np": np,
    }
    exec(compile(ast.fix_missing_locations(module), str(RUNNER), "exec"), namespace)
    return namespace["InteractiveControls"], glfw


def _controller():
    return SimpleNamespace(
        command=np.zeros(3, dtype=np.float64),
        command_ranges=np.asarray([[-0.45, 0.45], [-0.1, 0.1], [-0.25, 0.25]]),
    )


def test_velocity_key_resumes_directly_from_policy_hold() -> None:
    controls_type, glfw = _load_controls()
    controls = controls_type(_controller(), initial_mode=controls_type.POLICY_HOLD)

    controls.key_callback(glfw.KEY_W)

    mode, command, _revision = controls.snapshot()
    assert mode == controls_type.VELOCITY
    np.testing.assert_allclose(command, [0.1, 0.0, 0.0])


def test_stop_key_enters_smooth_fix_stand_and_zeros_command() -> None:
    controls_type, glfw = _load_controls()
    controller = _controller()
    controller.command[:] = [0.3, 0.1, -0.2]
    controls = controls_type(controller, initial_mode=controls_type.VELOCITY)

    controls.key_callback(glfw.KEY_X)

    mode, command, _revision = controls.snapshot()
    assert mode == controls_type.FIX_STAND
    np.testing.assert_allclose(command, np.zeros(3))
    assert controls.complete_fix_stand() is not None
    assert controls.snapshot()[0] == controls_type.POLICY_HOLD

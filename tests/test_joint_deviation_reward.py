from __future__ import annotations

import ast
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest


PROJECT_ROOT = Path(__file__).resolve().parents[1]
REWARDS = PROJECT_ROOT / "rl/src/custom_dog_rl/tasks/locomotion/mdp/rewards.py"


class _SceneEntityCfg:
    def __init__(self, name: str, joint_ids: tuple[int, ...] = (0, 1)) -> None:
        self.name = name
        self.joint_ids = joint_ids


class _NumpyTorch:
    class linalg:
        @staticmethod
        def vector_norm(value: np.ndarray, dim: int) -> np.ndarray:
            return np.linalg.norm(value, axis=dim)

    abs = staticmethod(np.abs)
    full_like = staticmethod(np.full_like)
    logical_and = staticmethod(np.logical_and)
    ones_like = staticmethod(np.ones_like)
    square = staticmethod(np.square)
    sum = staticmethod(lambda value, dim: np.sum(value, axis=dim))
    where = staticmethod(np.where)


def _load_joint_deviation_l2():
    tree = ast.parse(REWARDS.read_text(encoding="utf-8"))
    function = next(
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == "joint_deviation_l2"
    )
    module = ast.Module(
        body=[
            ast.ImportFrom(
                module="__future__",
                names=[ast.alias(name="annotations")],
                level=0,
            ),
            function,
        ],
        type_ignores=[],
    )
    namespace = {"SceneEntityCfg": _SceneEntityCfg, "torch": _NumpyTorch}
    exec(compile(ast.fix_missing_locations(module), str(REWARDS), "exec"), namespace)
    return namespace["joint_deviation_l2"]


class _CommandManager:
    def __init__(self, command: np.ndarray) -> None:
        self.command = command
        self.requested_name: str | None = None

    def get_command(self, name: str) -> np.ndarray:
        self.requested_name = name
        return self.command


def _make_env(command: np.ndarray, body_velocity: np.ndarray | None = None):
    batch_size = command.shape[0]
    if body_velocity is None:
        body_velocity = np.zeros((batch_size, 3), dtype=np.float64)
    data = SimpleNamespace(
        joint_pos=np.ones((batch_size, 2), dtype=np.float64),
        default_joint_pos=np.zeros((batch_size, 2), dtype=np.float64),
        root_lin_vel_b=body_velocity,
    )
    return SimpleNamespace(
        scene={"robot": SimpleNamespace(data=data)},
        command_manager=_CommandManager(command),
    )


def test_pure_yaw_command_does_not_receive_standing_scale():
    reward = _load_joint_deviation_l2()
    command = np.asarray(
        [
            [0.0, 0.0, 0.0],
            [0.0, 0.0, 0.2],
            [0.1, 0.0, 0.0],
            [0.0, 0.0, 0.0],
        ]
    )
    body_velocity = np.zeros((4, 3), dtype=np.float64)
    body_velocity[3, 0] = 0.3
    env = _make_env(command, body_velocity)

    actual = reward(
        env,
        stand_still_scale=4.0,
        velocity_threshold=0.2,
        command_name="test_velocity",
    )

    np.testing.assert_allclose(actual, [8.0, 2.0, 2.0, 2.0])
    assert env.command_manager.requested_name == "test_velocity"


def test_command_thresholds_control_standing_classification():
    reward = _load_joint_deviation_l2()
    env = _make_env(
        np.asarray(
            [
                [0.02, 0.0, 0.04],
                [0.031, 0.0, 0.04],
                [0.02, 0.0, 0.051],
            ]
        )
    )

    actual = reward(
        env,
        stand_still_scale=3.0,
        planar_command_threshold=0.03,
        yaw_command_threshold=0.05,
    )

    np.testing.assert_allclose(actual, [6.0, 2.0, 2.0])


@pytest.mark.parametrize(
    ("parameter", "value"),
    [("planar_command_threshold", -0.01), ("yaw_command_threshold", -0.01)],
)
def test_command_thresholds_must_be_non_negative(parameter: str, value: float):
    reward = _load_joint_deviation_l2()
    env = _make_env(np.zeros((1, 3), dtype=np.float64))

    with pytest.raises(ValueError, match="command thresholds must be non-negative"):
        reward(env, **{parameter: value})

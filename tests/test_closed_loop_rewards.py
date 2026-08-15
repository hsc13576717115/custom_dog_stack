from __future__ import annotations

import ast
import math
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
    mean = staticmethod(lambda value, dim: np.mean(value, axis=dim))
    sign = staticmethod(np.sign)
    square = staticmethod(np.square)
    sum = staticmethod(lambda value, dim: np.sum(value, axis=dim))

    @staticmethod
    def clamp(value: np.ndarray, min: float | None = None, max: float | None = None):
        lower = -np.inf if min is None else min
        upper = np.inf if max is None else max
        return np.clip(value, lower, upper)


def _load_reward(name: str):
    tree = ast.parse(REWARDS.read_text(encoding="utf-8"))
    function = next(
        node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == name
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
    namespace = {"SceneEntityCfg": _SceneEntityCfg, "torch": _NumpyTorch, "math": math}
    exec(compile(ast.fix_missing_locations(module), str(REWARDS), "exec"), namespace)
    return namespace[name]


def _make_env(command: np.ndarray, yaw_rate: np.ndarray, height: np.ndarray):
    batch_size = command.shape[0]
    data = SimpleNamespace(
        root_ang_vel_b=np.column_stack((np.zeros(batch_size), np.zeros(batch_size), yaw_rate)),
        root_lin_vel_b=np.zeros((batch_size, 3)),
        root_pos_w=np.column_stack((np.zeros(batch_size), np.zeros(batch_size), height)),
        joint_pos=np.zeros((batch_size, 2)),
        default_joint_pos=np.zeros((batch_size, 2)),
        projected_gravity_b=np.column_stack(
            (np.zeros(batch_size), np.zeros(batch_size), -np.ones(batch_size))
        ),
    )
    manager = SimpleNamespace(get_command=lambda _name: command)
    return SimpleNamespace(scene={"robot": SimpleNamespace(data=data)}, command_manager=manager)


def test_yaw_overspeed_penalizes_only_pure_signed_overshoot():
    reward = _load_reward("yaw_overspeed_relative_l2")
    command = np.asarray(
        [
            [0.0, 0.0, 0.25],
            [0.0, 0.0, -0.25],
            [0.0, 0.0, 0.25],
            [0.1, 0.0, 0.25],
            [0.0, 0.0, 0.0],
        ]
    )
    env = _make_env(command, np.asarray([0.30, -0.35, 0.20, 0.40, 0.10]), np.full(5, 0.32))

    actual = reward(env)

    np.testing.assert_allclose(actual, [0.04, 0.16, 0.0, 0.0, 0.0])


def test_standing_height_band_is_zero_inside_and_while_moving():
    reward = _load_reward("standing_base_height_band_l2")
    command = np.asarray(
        [
            [0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0],
            [0.1, 0.0, 0.0],
        ]
    )
    env = _make_env(command, np.zeros(4), np.asarray([0.30, 0.32, 0.35, 0.28]))

    actual = reward(env)

    np.testing.assert_allclose(actual, [0.25, 0.0, 1.0, 0.0])


@pytest.mark.parametrize(
    "kwargs",
    [
        {"lower_height": 0.33, "upper_height": 0.31},
        {"error_std": 0.0},
        {"max_error": 0.0},
    ],
)
def test_standing_height_rejects_invalid_parameters(kwargs: dict[str, float]):
    reward = _load_reward("standing_base_height_band_l2")
    env = _make_env(np.zeros((1, 3)), np.zeros(1), np.full(1, 0.32))

    with pytest.raises(ValueError):
        reward(env, **kwargs)


def test_standing_joint_pose_is_normalized_and_zero_while_moving():
    reward = _load_reward("standing_joint_deviation_normalized_l2")
    command = np.asarray([[0.0, 0.0, 0.0], [0.1, 0.0, 0.0]])
    env = _make_env(command, np.zeros(2), np.full(2, 0.32))
    env.scene["robot"].data.joint_pos[:] = np.asarray([[0.1, -0.1], [0.1, -0.1]])

    actual = reward(env, asset_cfg=_SceneEntityCfg("robot"), position_std=0.1)

    np.testing.assert_allclose(actual, [1.0, 0.0])


def test_standing_orientation_uses_three_degree_normalization():
    reward = _load_reward("standing_orientation_normalized_l2")
    command = np.asarray([[0.0, 0.0, 0.0], [0.1, 0.0, 0.0]])
    env = _make_env(command, np.zeros(2), np.full(2, 0.32))
    tilt = math.radians(3.0)
    env.scene["robot"].data.projected_gravity_b[:] = np.asarray(
        [[math.sin(tilt), 0.0, -math.cos(tilt)], [math.sin(tilt), 0.0, -math.cos(tilt)]]
    )

    actual = reward(env, tilt_std_deg=3.0)

    np.testing.assert_allclose(actual, [1.0, 0.0])

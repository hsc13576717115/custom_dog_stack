from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

REPO_ROOT = Path(__file__).resolve().parents[1]
RL_SOURCE = REPO_ROOT / "rl" / "src"
sys.path.insert(0, str(RL_SOURCE))

_MODULE_PATH = RL_SOURCE / "custom_dog_rl" / "agents" / "routed_teacher.py"
_SPEC = importlib.util.spec_from_file_location("custom_dog_routed_teacher_test", _MODULE_PATH)
assert _SPEC is not None and _SPEC.loader is not None
_MODULE = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_MODULE)
CommandRoutedTeacher = _MODULE.CommandRoutedTeacher
PrivilegedOmniRoutedTeacher = _MODULE.PrivilegedOmniRoutedTeacher


class ConstantActor(torch.nn.Module):
    def __init__(self, value: float):
        super().__init__()
        self.register_buffer("value", torch.tensor(value))

    def forward(self, observation: torch.Tensor) -> torch.Tensor:
        return self.value.expand(observation.shape[0], 2)


def test_command_routed_teacher_selects_and_blends_experts():
    teacher = CommandRoutedTeacher(
        ConstantActor(1.0),
        ConstantActor(-1.0),
        command_index=1,
        reverse_threshold=-0.05,
        blend_width=0.10,
    )
    observation = torch.tensor(
        [
            [0.0, 0.10],
            [0.0, -0.10],
            [0.0, -0.05],
        ]
    )

    action = teacher(observation)

    assert torch.allclose(action[0], torch.tensor([1.0, 1.0]))
    assert torch.allclose(action[1], torch.tensor([-1.0, -1.0]))
    assert torch.allclose(action[2], torch.tensor([0.0, 0.0]))


def test_command_routed_teacher_rejects_invalid_observation_shape():
    teacher = CommandRoutedTeacher(ConstantActor(1.0), ConstantActor(-1.0), command_index=2)
    with pytest.raises(ValueError, match="command index 2"):
        teacher(torch.zeros(3, 2))


class InputAwareActor(torch.nn.Module):
    def __init__(self, expected_dim: int, value: float):
        super().__init__()
        self.expected_dim = expected_dim
        self.value = value

    def forward(self, observation: torch.Tensor) -> torch.Tensor:
        assert observation.shape[1] == self.expected_dim
        return torch.full((observation.shape[0], 12), self.value, dtype=observation.dtype)


def test_privileged_omni_router_preserves_forward_and_routes_missing_axes():
    teacher = PrivilegedOmniRoutedTeacher(
        InputAwareActor(45, 1.0),
        InputAwareActor(47, -1.0),
        blend_width=0.0,
    )
    observations = torch.zeros(7, 47)
    observations[1, 6] = 0.3
    observations[2, 6] = 0.3
    observations[2, 8] = 0.2
    observations[3, 6] = -0.2
    observations[4, 7] = 0.1
    observations[5, 8] = -0.2
    observations[6, 6] = 0.2
    observations[6, 7] = -0.1

    actions = teacher(observations)

    assert torch.all(actions[:3] == 1.0)  # stop, forward and vx+wz
    assert torch.all(actions[3:] == -1.0)  # reverse, lateral, pure yaw and vx+vy


def test_privileged_omni_router_rejects_short_observations():
    teacher = PrivilegedOmniRoutedTeacher(
        InputAwareActor(45, 1.0),
        InputAwareActor(47, -1.0),
    )
    with pytest.raises(ValueError, match="at least 45"):
        teacher(torch.zeros(2, 44))

"""Command-routed frozen teachers for behavior distillation."""

from __future__ import annotations

import copy
from pathlib import Path

import torch
from torch import nn


class CommandRoutedTeacher(nn.Module):
    """Select a forward or reverse actor from the commanded x velocity.

    Both actors consume the same deployable policy observation.  A narrow
    linear blend around the route threshold avoids a discontinuous action
    target when a command crosses zero.
    """

    def __init__(
        self,
        forward_actor: nn.Module,
        reverse_actor: nn.Module,
        command_index: int = 6,
        reverse_threshold: float = -0.05,
        blend_width: float = 0.05,
    ):
        super().__init__()
        if command_index < 0:
            raise ValueError("command_index must be non-negative")
        if blend_width < 0.0:
            raise ValueError("blend_width must be non-negative")
        self.forward_actor = forward_actor
        self.reverse_actor = reverse_actor
        self.command_index = command_index
        self.reverse_threshold = reverse_threshold
        self.blend_width = blend_width
        self.requires_grad_(False)
        self.eval()

    def forward(self, observation: torch.Tensor) -> torch.Tensor:
        if observation.ndim != 2 or observation.shape[1] <= self.command_index:
            raise ValueError(
                f"Expected batched observations containing command index {self.command_index}, "
                f"got shape {tuple(observation.shape)}"
            )
        forward_action = self.forward_actor(observation)
        reverse_action = self.reverse_actor(observation)
        command_vx = observation[:, self.command_index]
        if self.blend_width == 0.0:
            reverse_weight = (command_vx < self.reverse_threshold).to(observation.dtype)
        else:
            upper = self.reverse_threshold + 0.5 * self.blend_width
            reverse_weight = torch.clamp((upper - command_vx) / self.blend_width, 0.0, 1.0)
        return torch.lerp(forward_action, reverse_action, reverse_weight.unsqueeze(-1))


class PrivilegedOmniRoutedTeacher(nn.Module):
    """Route deployable forward commands and privileged omni commands.

    The privileged observation appends true planar velocity to the unchanged
    45-D deployable contract. Positive vx and vx+wz commands retain the
    accepted forward actor. Reverse, lateral and pure-yaw commands use the
    velocity-aware omni actor during distillation only.
    """

    def __init__(
        self,
        forward_actor: nn.Module,
        omni_actor: nn.Module,
        *,
        forward_observation_dim: int = 45,
        command_indices: tuple[int, int, int] = (6, 7, 8),
        reverse_threshold: float = 0.05,
        lateral_threshold: float = 0.025,
        yaw_threshold: float = 0.025,
        pure_yaw_forward_threshold: float = 0.05,
        blend_width: float = 0.025,
    ):
        super().__init__()
        if forward_observation_dim <= 0:
            raise ValueError("forward_observation_dim must be positive")
        if len(command_indices) != 3 or min(command_indices) < 0:
            raise ValueError("command_indices must contain three non-negative indices")
        if min(reverse_threshold, lateral_threshold, yaw_threshold) < 0.0:
            raise ValueError("routing thresholds must be non-negative")
        if pure_yaw_forward_threshold < 0.0 or blend_width < 0.0:
            raise ValueError("routing thresholds and blend_width must be non-negative")
        self.forward_actor = forward_actor
        self.omni_actor = omni_actor
        self.forward_observation_dim = forward_observation_dim
        self.command_indices = command_indices
        self.reverse_threshold = reverse_threshold
        self.lateral_threshold = lateral_threshold
        self.yaw_threshold = yaw_threshold
        self.pure_yaw_forward_threshold = pure_yaw_forward_threshold
        self.blend_width = blend_width
        self.requires_grad_(False)
        self.eval()

    def _above(self, value: torch.Tensor, threshold: float) -> torch.Tensor:
        if self.blend_width == 0.0:
            return (value > threshold).to(value.dtype)
        return torch.clamp((value - threshold) / self.blend_width, 0.0, 1.0)

    def _below(self, value: torch.Tensor, threshold: float) -> torch.Tensor:
        if self.blend_width == 0.0:
            return (value < threshold).to(value.dtype)
        return torch.clamp((threshold - value) / self.blend_width, 0.0, 1.0)

    def forward(self, observation: torch.Tensor) -> torch.Tensor:
        required_dim = max(self.forward_observation_dim, max(self.command_indices) + 1)
        if observation.ndim != 2 or observation.shape[1] < required_dim:
            raise ValueError(
                f"Expected batched privileged observations with at least {required_dim} values, "
                f"got shape {tuple(observation.shape)}"
            )
        forward_action = self.forward_actor(observation[:, : self.forward_observation_dim])
        omni_action = self.omni_actor(observation)
        vx = observation[:, self.command_indices[0]]
        vy = observation[:, self.command_indices[1]]
        wz = observation[:, self.command_indices[2]]

        reverse_weight = self._above(-vx, self.reverse_threshold)
        lateral_weight = self._above(torch.abs(vy), self.lateral_threshold)
        yaw_activity = self._above(torch.abs(wz), self.yaw_threshold)
        near_zero_forward = self._below(torch.abs(vx), self.pure_yaw_forward_threshold)
        pure_yaw_weight = yaw_activity * near_zero_forward
        omni_weight = torch.maximum(
            reverse_weight,
            torch.maximum(lateral_weight, pure_yaw_weight),
        )
        return torch.lerp(forward_action, omni_action, omni_weight.unsqueeze(-1))


def _actor_state_dict(checkpoint_path: str | Path) -> dict[str, torch.Tensor]:
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    state_dict = checkpoint.get("model_state_dict")
    if not isinstance(state_dict, dict):
        raise ValueError(f"Checkpoint does not contain model_state_dict: {checkpoint_path}")
    actor_state = {
        key.removeprefix("actor."): value
        for key, value in state_dict.items()
        if key.startswith("actor.")
    }
    if not actor_state:
        raise ValueError(f"Checkpoint does not contain actor parameters: {checkpoint_path}")
    return actor_state


def configure_routed_teacher(
    policy,
    reverse_checkpoint: str | Path,
    *,
    command_index: int = 6,
    reverse_threshold: float = -0.05,
    blend_width: float = 0.05,
) -> None:
    """Replace a loaded distillation teacher with a frozen two-actor router."""

    if not getattr(policy, "loaded_teacher", False):
        raise ValueError("The forward teacher must be loaded before routing is configured")
    if isinstance(policy.teacher, CommandRoutedTeacher):
        raise ValueError("The distillation teacher is already command-routed")

    forward_actor = policy.teacher
    reverse_actor = copy.deepcopy(forward_actor)
    reverse_actor.load_state_dict(_actor_state_dict(reverse_checkpoint), strict=True)
    policy.teacher = CommandRoutedTeacher(
        forward_actor,
        reverse_actor,
        command_index=command_index,
        reverse_threshold=reverse_threshold,
        blend_width=blend_width,
    ).to(next(policy.student.parameters()).device)
    policy.teacher_obs_normalizer.eval()


def configure_privileged_omni_teacher(
    policy,
    forward_checkpoint: str | Path,
    *,
    forward_observation_dim: int = 45,
    command_indices: tuple[int, int, int] = (6, 7, 8),
    reverse_threshold: float = 0.05,
    lateral_threshold: float = 0.025,
    yaw_threshold: float = 0.025,
    pure_yaw_forward_threshold: float = 0.05,
    blend_width: float = 0.025,
) -> None:
    """Route a loaded privileged omni teacher with a frozen 45-D forward actor."""

    if not getattr(policy, "loaded_teacher", False):
        raise ValueError("The privileged omni teacher must be loaded before routing is configured")
    if isinstance(policy.teacher, (CommandRoutedTeacher, PrivilegedOmniRoutedTeacher)):
        raise ValueError("The distillation teacher is already command-routed")

    forward_actor = copy.deepcopy(policy.student)
    forward_actor.load_state_dict(_actor_state_dict(forward_checkpoint), strict=True)
    policy.teacher = PrivilegedOmniRoutedTeacher(
        forward_actor,
        policy.teacher,
        forward_observation_dim=forward_observation_dim,
        command_indices=command_indices,
        reverse_threshold=reverse_threshold,
        lateral_threshold=lateral_threshold,
        yaw_threshold=yaw_threshold,
        pure_yaw_forward_threshold=pure_yaw_forward_threshold,
        blend_width=blend_width,
    ).to(next(policy.student.parameters()).device)
    policy.teacher_obs_normalizer.eval()

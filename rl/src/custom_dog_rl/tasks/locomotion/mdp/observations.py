"""Observation terms specific to custom-dog locomotion tasks."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from isaaclab.managers import SceneEntityCfg

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def base_lin_vel_xy(
    env: "ManagerBasedRLEnv",
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Return root x/y velocity expressed in the robot base frame."""

    return env.scene[asset_cfg.name].data.root_lin_vel_b[:, :2]


def command_gait_phase(
    env: "ManagerBasedRLEnv",
    period: float,
    command_name: str = "base_velocity",
    command_threshold: float = 0.1,
) -> torch.Tensor:
    """Return a two-channel gait clock and suppress it for stand commands."""

    if period <= 0.0:
        raise ValueError("period must be positive")
    if command_threshold < 0.0:
        raise ValueError("command_threshold must be non-negative")

    global_phase = (env.episode_length_buf * env.step_dt) % period / period
    phase = torch.stack(
        (
            torch.sin(global_phase * 2.0 * torch.pi),
            torch.cos(global_phase * 2.0 * torch.pi),
        ),
        dim=-1,
    )
    command = env.command_manager.get_command(command_name)
    moving = torch.linalg.vector_norm(command, dim=1) > command_threshold
    return phase * moving.unsqueeze(1)

"""Reward terms specific to the custom-dog velocity tasks."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from isaaclab.managers import SceneEntityCfg

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def track_ang_vel_z_l2(
    env: "ManagerBasedRLEnv",
    command_name: str,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Return squared yaw-rate tracking error in the robot body frame.

    The existing exponential term gives the main yaw-tracking signal.  This
    auxiliary error term keeps a small, non-saturating gradient when a fast
    forward gait develops a persistent yaw bias.  Since it compares against
    the commanded yaw rate, turning commands are not penalized as such.
    """

    asset = env.scene[asset_cfg.name]
    command = env.command_manager.get_command(command_name)[:, 2]
    error = command - asset.data.root_ang_vel_b[:, 2]
    return torch.square(error)


def track_lin_vel_xy_low_speed_relative_l1(
    env: "ManagerBasedRLEnv",
    command_name: str,
    command_min: float,
    command_max: float,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Penalize low-speed tracking error relative to the requested speed.

    A fixed absolute error can make a stationary policy look acceptable at a
    small command.  Normalizing by the command magnitude supplies a useful
    gradient in that band while excluding exact standing commands.
    """

    asset = env.scene[asset_cfg.name]
    command = env.command_manager.get_command(command_name)
    command_speed = torch.linalg.vector_norm(command[:, :2], dim=1)
    active = (command_speed >= command_min) & (command_speed <= command_max)
    error = torch.linalg.vector_norm(
        command[:, :2] - asset.data.root_lin_vel_b[:, :2],
        dim=1,
    )
    denominator = torch.clamp(command_speed, min=command_min)
    return (error / denominator) * active

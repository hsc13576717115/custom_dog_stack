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


def command_trot_phase(
    env: "ManagerBasedRLEnv",
    command_name: str = "base_velocity",
    command_threshold: float = 0.1,
    min_frequency: float = 1.4,
    max_frequency: float = 3.2,
    full_speed: float = 3.0,
    yaw_speed_scale: float = 0.35,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return a speed-adaptive global trot phase and moving mask."""

    if command_threshold < 0.0:
        raise ValueError("command_threshold must be non-negative")
    if min_frequency <= 0.0 or max_frequency < min_frequency:
        raise ValueError("invalid trot frequency range")
    if full_speed <= 0.0 or yaw_speed_scale < 0.0:
        raise ValueError("full_speed must be positive and yaw_speed_scale non-negative")

    command = env.command_manager.get_command(command_name)
    motion_speed = torch.linalg.vector_norm(command[:, :2], dim=1)
    motion_speed += yaw_speed_scale * torch.abs(command[:, 2])
    frequency_blend = torch.clamp(motion_speed / full_speed, min=0.0, max=1.0)
    frequency = min_frequency + frequency_blend * (max_frequency - min_frequency)
    phase = torch.remainder(env.episode_length_buf * env.step_dt * frequency, 1.0)
    return phase, motion_speed > command_threshold


def command_trot_clock(
    env: "ManagerBasedRLEnv",
    command_name: str = "base_velocity",
    command_threshold: float = 0.1,
    min_frequency: float = 1.4,
    max_frequency: float = 3.2,
    full_speed: float = 3.0,
    yaw_speed_scale: float = 0.35,
) -> torch.Tensor:
    """Return four foot clocks in explicit FR, FL, RR, RL order.

    FR+RL and FL+RR form the two diagonal trot pairs.  A positive clock is
    the desired stance half-cycle; clocks are zero for standing commands.
    """

    phase, moving = command_trot_phase(
        env,
        command_name=command_name,
        command_threshold=command_threshold,
        min_frequency=min_frequency,
        max_frequency=max_frequency,
        full_speed=full_speed,
        yaw_speed_scale=yaw_speed_scale,
    )
    offsets = phase.new_tensor((0.0, 0.5, 0.5, 0.0))
    foot_phase = torch.remainder(phase.unsqueeze(1) + offsets.unsqueeze(0), 1.0)
    clocks = torch.sin(2.0 * torch.pi * foot_phase)
    return clocks * moving.unsqueeze(1)

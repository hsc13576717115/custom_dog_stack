"""Termination terms for dedicated self-righting policies."""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

import torch

from isaaclab.assets import Articulation
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import ContactSensor

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def recovery_success_state(
    env: ManagerBasedRLEnv,
    minimum_height: float,
    maximum_tilt_deg: float,
    maximum_angular_velocity: float,
    minimum_contact_feet: int,
    contact_force_threshold: float,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    sensor_cfg: SceneEntityCfg = SceneEntityCfg("contact_forces", body_names=".*_foot"),
) -> torch.Tensor:
    """Return the deployable hand-off predicate before its dwell requirement."""

    if minimum_height <= 0.0:
        raise ValueError("minimum_height must be positive")
    if not 0.0 < maximum_tilt_deg < 90.0:
        raise ValueError("maximum_tilt_deg must be in (0, 90)")
    if maximum_angular_velocity <= 0.0 or contact_force_threshold <= 0.0:
        raise ValueError("velocity and contact thresholds must be positive")
    if not 1 <= minimum_contact_feet <= 4:
        raise ValueError("minimum_contact_feet must be in [1, 4]")

    asset: Articulation = env.scene[asset_cfg.name]
    sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    foot_forces = torch.linalg.vector_norm(
        sensor.data.net_forces_w[:, sensor_cfg.body_ids],
        dim=-1,
    )
    contact_count = torch.sum(foot_forces >= contact_force_threshold, dim=1)
    upright_cosine = -asset.data.projected_gravity_b[:, 2]
    angular_speed = torch.linalg.vector_norm(asset.data.root_ang_vel_b, dim=1)
    return (
        (asset.data.root_pos_w[:, 2] >= minimum_height)
        & (upright_cosine >= math.cos(math.radians(maximum_tilt_deg)))
        & (angular_speed <= maximum_angular_velocity)
        & (contact_count >= minimum_contact_feet)
    )


def recovery_success_dwell(
    env: ManagerBasedRLEnv,
    dwell_time_s: float,
    **success_params,
) -> torch.Tensor:
    """Terminate successfully only after the hand-off state remains stable."""

    if dwell_time_s <= 0.0:
        raise ValueError("dwell_time_s must be positive")
    required_steps = max(1, math.ceil(dwell_time_s / env.step_dt))
    stable = recovery_success_state(env, **success_params)
    counter = getattr(env, "_custom_dog_recovery_success_dwell", None)
    if counter is None:
        counter = torch.zeros(env.num_envs, dtype=torch.long, device=env.device)
        setattr(env, "_custom_dog_recovery_success_dwell", counter)
    counter[:] = torch.where(stable, counter + 1, torch.zeros_like(counter))
    return counter >= required_steps

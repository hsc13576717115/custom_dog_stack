"""Reset events for recovery-capable custom-dog locomotion tasks."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from isaaclab.assets import Articulation
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils import math as math_utils

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv


HIP_JOINT_NAMES = ("FR_hip_joint", "FL_hip_joint", "RR_hip_joint", "RL_hip_joint")
THIGH_JOINT_NAMES = ("FR_thigh_joint", "FL_thigh_joint", "RR_thigh_joint", "RL_thigh_joint")
CALF_JOINT_NAMES = ("FR_calf_joint", "FL_calf_joint", "RR_calf_joint", "RL_calf_joint")


def _joint_ids(asset: Articulation, names: tuple[str, ...]) -> torch.Tensor:
    cache_name = "_custom_dog_recovery_joint_ids"
    cached = getattr(asset, cache_name, None)
    if cached is None:
        resolved = []
        for name in HIP_JOINT_NAMES + THIGH_JOINT_NAMES + CALF_JOINT_NAMES:
            joint_ids, _ = asset.find_joints(name)
            if len(joint_ids) != 1:
                raise ValueError(f"Expected exactly one joint named {name}, got {joint_ids}")
            resolved.append(joint_ids[0])
        cached = torch.tensor(resolved, device=asset.device, dtype=torch.long)
        setattr(asset, cache_name, cached)

    index_by_name = {
        name: index
        for index, name in enumerate(HIP_JOINT_NAMES + THIGH_JOINT_NAMES + CALF_JOINT_NAMES)
    }
    return cached[[index_by_name[name] for name in names]]


def reset_recovery_or_standing(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor,
    prone_probability: float,
    prone_root_height: float,
    prone_thigh_position: float,
    prone_calf_position: float,
    prone_joint_noise: float,
    standing_joint_noise: float,
    root_xy_range: tuple[float, float],
    yaw_range: tuple[float, float],
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
):
    """Reset a mix of normal standing and measured folded-prone states.

    The prone state keeps the trunk level, with hips at zero and the supplied
    thigh/calf angles.  Its low root height deliberately permits base-ground
    contact so the policy learns to recover instead of terminating at reset.
    """

    if not 0.0 <= prone_probability <= 1.0:
        raise ValueError("prone_probability must be in [0, 1]")
    if prone_joint_noise < 0.0 or standing_joint_noise < 0.0:
        raise ValueError("joint noise values must be non-negative")

    asset: Articulation = env.scene[asset_cfg.name]
    device = asset.device
    count = len(env_ids)
    if count == 0:
        return

    default_root_state = asset.data.default_root_state[env_ids].clone()
    root_pose = default_root_state[:, :7].clone()
    root_pose[:, :3] += env.scene.env_origins[env_ids]
    root_pose[:, :2] += torch.empty((count, 2), device=device).uniform_(*root_xy_range)
    yaw = torch.empty(count, device=device).uniform_(*yaw_range)
    yaw_quat = math_utils.quat_from_euler_xyz(torch.zeros_like(yaw), torch.zeros_like(yaw), yaw)
    root_pose[:, 3:7] = math_utils.quat_mul(root_pose[:, 3:7], yaw_quat)
    root_velocity = torch.zeros((count, 6), device=device)

    joint_position = asset.data.default_joint_pos[env_ids].clone()
    if standing_joint_noise > 0.0:
        joint_position += torch.empty_like(joint_position).uniform_(
            -standing_joint_noise, standing_joint_noise
        )
    joint_velocity = torch.zeros_like(joint_position)

    prone_rows = torch.nonzero(
        torch.rand(count, device=device) < prone_probability,
        as_tuple=False,
    ).squeeze(-1)
    prone_reset_mask = getattr(env, "_custom_dog_prone_reset_mask", None)
    if prone_reset_mask is None:
        prone_reset_mask = torch.zeros(env.num_envs, dtype=torch.bool, device=device)
        setattr(env, "_custom_dog_prone_reset_mask", prone_reset_mask)
    prone_reset_mask[env_ids] = False
    prone_reset_mask[env_ids[prone_rows]] = True
    if len(prone_rows) > 0:
        root_pose[prone_rows, 2] = env.scene.env_origins[env_ids[prone_rows], 2] + prone_root_height
        hip_ids = _joint_ids(asset, HIP_JOINT_NAMES)
        thigh_ids = _joint_ids(asset, THIGH_JOINT_NAMES)
        calf_ids = _joint_ids(asset, CALF_JOINT_NAMES)
        joint_position[prone_rows[:, None], hip_ids] = 0.0
        joint_position[prone_rows[:, None], thigh_ids] = prone_thigh_position
        joint_position[prone_rows[:, None], calf_ids] = prone_calf_position
        if prone_joint_noise > 0.0:
            prone_joint_ids = torch.cat((hip_ids, thigh_ids, calf_ids))
            joint_position[prone_rows[:, None], prone_joint_ids] += torch.empty(
                (len(prone_rows), len(prone_joint_ids)), device=device
            ).uniform_(-prone_joint_noise, prone_joint_noise)

    asset.write_root_pose_to_sim(root_pose, env_ids=env_ids)
    asset.write_root_velocity_to_sim(root_velocity, env_ids=env_ids)
    asset.write_joint_state_to_sim(joint_position, joint_velocity, env_ids=env_ids)

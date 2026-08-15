"""Reset events for recovery-capable custom-dog locomotion tasks."""

from __future__ import annotations

from typing import TYPE_CHECKING

import math

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


def reset_self_righting_states(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor,
    orientation_probabilities: tuple[float, float, float, float],
    arbitrary_orientation_probability: float,
    root_height_range: tuple[float, float],
    hip_position_range: tuple[float, float],
    thigh_position_range: tuple[float, float],
    calf_position_range: tuple[float, float],
    max_linear_velocity: float,
    max_angular_velocity: float,
    root_xy_range: tuple[float, float] = (-0.25, 0.25),
    yaw_range: tuple[float, float] = (-math.pi, math.pi),
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
):
    """Reset dedicated recovery episodes across belly, back, sides, and SO(3)."""

    if len(orientation_probabilities) != 4:
        raise ValueError("orientation_probabilities must contain belly/back/left/right")
    if any(probability < 0.0 for probability in orientation_probabilities):
        raise ValueError("orientation probabilities must be non-negative")
    if abs(sum(orientation_probabilities) - 1.0) > 1.0e-6:
        raise ValueError("orientation probabilities must sum to one")
    if not 0.0 <= arbitrary_orientation_probability <= 1.0:
        raise ValueError("arbitrary_orientation_probability must be in [0, 1]")
    for name, value_range in (
        ("root_height_range", root_height_range),
        ("hip_position_range", hip_position_range),
        ("thigh_position_range", thigh_position_range),
        ("calf_position_range", calf_position_range),
        ("root_xy_range", root_xy_range),
        ("yaw_range", yaw_range),
    ):
        if value_range[0] > value_range[1]:
            raise ValueError(f"{name} lower bound must not exceed upper bound")
    if max_linear_velocity < 0.0 or max_angular_velocity < 0.0:
        raise ValueError("initial velocity limits must be non-negative")

    asset: Articulation = env.scene[asset_cfg.name]
    device = asset.device
    count = len(env_ids)
    if count == 0:
        return

    probabilities = torch.tensor(
        orientation_probabilities,
        dtype=torch.float,
        device=device,
    )
    mode = torch.multinomial(probabilities, count, replacement=True)
    roll_values = torch.tensor(
        (0.0, math.pi, 0.5 * math.pi, -0.5 * math.pi),
        dtype=torch.float,
        device=device,
    )
    roll = roll_values[mode]
    pitch = torch.zeros(count, dtype=torch.float, device=device)
    yaw = torch.empty(count, device=device).uniform_(*yaw_range)
    orientation = math_utils.quat_from_euler_xyz(roll, pitch, yaw)
    arbitrary = torch.rand(count, device=device) < arbitrary_orientation_probability
    if torch.any(arbitrary):
        orientation[arbitrary] = math_utils.random_orientation(int(arbitrary.sum().item()), device)
        mode[arbitrary] = 4

    default_root_state = asset.data.default_root_state[env_ids].clone()
    root_pose = default_root_state[:, :7].clone()
    root_pose[:, :3] += env.scene.env_origins[env_ids]
    root_pose[:, :2] += torch.empty((count, 2), device=device).uniform_(*root_xy_range)
    root_pose[:, 2] = env.scene.env_origins[env_ids, 2] + torch.empty(
        count, device=device
    ).uniform_(*root_height_range)
    root_pose[:, 3:7] = orientation
    root_velocity = torch.empty((count, 6), device=device)
    root_velocity[:, :3].uniform_(-max_linear_velocity, max_linear_velocity)
    root_velocity[:, 3:].uniform_(-max_angular_velocity, max_angular_velocity)

    joint_position = asset.data.default_joint_pos[env_ids].clone()
    hip_ids = _joint_ids(asset, HIP_JOINT_NAMES)
    thigh_ids = _joint_ids(asset, THIGH_JOINT_NAMES)
    calf_ids = _joint_ids(asset, CALF_JOINT_NAMES)
    for joint_ids, position_range in (
        (hip_ids, hip_position_range),
        (thigh_ids, thigh_position_range),
        (calf_ids, calf_position_range),
    ):
        joint_position[:, joint_ids] = torch.empty(
            (count, len(joint_ids)), device=device
        ).uniform_(*position_range)
    joint_velocity = torch.empty_like(joint_position).uniform_(
        -max_angular_velocity,
        max_angular_velocity,
    )

    recovery_mode = getattr(env, "_custom_dog_recovery_mode", None)
    if recovery_mode is None:
        recovery_mode = torch.full((env.num_envs,), -1, dtype=torch.long, device=device)
        setattr(env, "_custom_dog_recovery_mode", recovery_mode)
    recovery_mode[env_ids] = mode
    recovery_dwell = getattr(env, "_custom_dog_recovery_success_dwell", None)
    if recovery_dwell is not None:
        recovery_dwell[env_ids] = 0

    asset.write_root_pose_to_sim(root_pose, env_ids=env_ids)
    asset.write_root_velocity_to_sim(root_velocity, env_ids=env_ids)
    asset.write_joint_state_to_sim(joint_position, joint_velocity, env_ids=env_ids)


def record_privileged_dynamics_context(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor | None,
    nominal_base_com: tuple[float, float, float],
    mass_ratio_scale: float = 0.10,
    com_scales: tuple[float, float, float] = (0.015, 0.015, 0.010),
    stiffness_ratio_scale: float = 0.15,
    damping_ratio_scale: float = 0.35,
    joint_friction_scale: float = 0.03,
    material_centers: tuple[float, float, float] = (0.875, 0.825, 0.04),
    material_scales: tuple[float, float, float] = (0.325, 0.325, 0.04),
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
):
    """Record normalized dynamics values after startup randomization.

    PhysX exposes mass, center-of-mass, and material tensors through CPU
    views.  Reading them once here avoids a CPU/GPU synchronization in every
    policy step while still giving a privileged teacher the exact randomized
    environment parameters.
    """

    positive_scales = (
        mass_ratio_scale,
        *com_scales,
        stiffness_ratio_scale,
        damping_ratio_scale,
        joint_friction_scale,
        *material_scales,
    )
    if any(scale <= 0.0 for scale in positive_scales):
        raise ValueError("all dynamics-context scales must be positive")

    asset: Articulation = env.scene[asset_cfg.name]
    device = asset.device
    if env_ids is None:
        selected = torch.arange(env.num_envs, device=device, dtype=torch.long)
        selected_cpu = torch.arange(env.num_envs, device="cpu", dtype=torch.long)
    else:
        selected = env_ids.to(device=device, dtype=torch.long)
        selected_cpu = env_ids.to(device="cpu", dtype=torch.long)

    context = getattr(env, "_custom_dog_startup_dynamics_context", None)
    if context is None:
        context = torch.zeros((env.num_envs, 10), device=device, dtype=torch.float)
        setattr(env, "_custom_dog_startup_dynamics_context", context)

    masses = asset.root_physx_view.get_masses()[selected_cpu].to(device=device)
    # PhysX mass buffers remain on CPU, unlike joint/actuator tensors.
    default_masses = asset.data.default_mass[selected_cpu].to(device=device)
    mass_ratio = masses.sum(dim=1) / torch.clamp(default_masses.sum(dim=1), min=1.0e-6)

    base_ids, _ = asset.find_bodies("base")
    if len(base_ids) != 1:
        raise ValueError(f"Expected exactly one base body, got {base_ids}")
    coms = asset.root_physx_view.get_coms()[selected_cpu, base_ids[0], :3].to(device=device)
    nominal_com = coms.new_tensor(nominal_base_com)
    com_scale = coms.new_tensor(com_scales)

    stiffness_ratios = []
    damping_ratios = []
    for actuator in asset.actuators.values():
        joint_ids = actuator.joint_indices
        default_stiffness = asset.data.default_joint_stiffness[selected, joint_ids]
        default_damping = asset.data.default_joint_damping[selected, joint_ids]
        stiffness_ratios.append(
            actuator.stiffness[selected]
            / torch.clamp(default_stiffness, min=1.0e-6)
        )
        damping_ratios.append(
            actuator.damping[selected]
            / torch.clamp(default_damping, min=1.0e-6)
        )
    mean_stiffness_ratio = torch.cat(stiffness_ratios, dim=1).mean(dim=1)
    mean_damping_ratio = torch.cat(damping_ratios, dim=1).mean(dim=1)
    mean_joint_friction = asset.data.joint_friction_coeff[selected].mean(dim=1)

    materials = asset.root_physx_view.get_material_properties()[selected_cpu]
    mean_material = materials.mean(dim=1).to(device=device)
    material_center = mean_material.new_tensor(material_centers)
    material_scale = mean_material.new_tensor(material_scales)

    context[selected] = torch.cat(
        (
            ((mass_ratio - 1.0) / mass_ratio_scale).unsqueeze(1),
            (coms - nominal_com) / com_scale,
            ((mean_stiffness_ratio - 1.0) / stiffness_ratio_scale).unsqueeze(1),
            ((mean_damping_ratio - 1.0) / damping_ratio_scale).unsqueeze(1),
            (mean_joint_friction / joint_friction_scale).unsqueeze(1),
            (mean_material - material_center) / material_scale,
        ),
        dim=1,
    )

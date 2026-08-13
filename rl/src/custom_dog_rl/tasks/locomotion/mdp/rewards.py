"""Reward terms specific to the custom-dog velocity tasks."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from isaaclab.managers import ManagerTermBase, SceneEntityCfg
from isaaclab.sensors import ContactSensor
from isaaclab.utils.math import quat_apply_inverse

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


HIP_JOINT_NAMES = ("FR_hip_joint", "FL_hip_joint", "RR_hip_joint", "RL_hip_joint")


def _desired_trot_stance(
    env: "ManagerBasedRLEnv",
    command_name: str,
    command_threshold: float,
    duty_factor: float,
    min_frequency: float,
    max_frequency: float,
    full_speed: float,
    yaw_speed_scale: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Build the FR/FL/RR/RL diagonal trot schedule shared by gait rewards."""

    if not 0.0 < duty_factor < 1.0:
        raise ValueError("duty_factor must be in (0, 1)")
    from .observations import command_trot_phase

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
    return foot_phase < duty_factor, moving


def trot_contact_schedule(
    env: "ManagerBasedRLEnv",
    sensor_cfg: SceneEntityCfg,
    command_name: str = "base_velocity",
    command_threshold: float = 0.1,
    duty_factor: float = 0.52,
    min_frequency: float = 1.4,
    max_frequency: float = 3.2,
    full_speed: float = 3.0,
    yaw_speed_scale: float = 0.35,
) -> torch.Tensor:
    """Reward matching the desired diagonal trot contact state."""

    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    desired_stance, moving = _desired_trot_stance(
        env,
        command_name,
        command_threshold,
        duty_factor,
        min_frequency,
        max_frequency,
        full_speed,
        yaw_speed_scale,
    )
    is_contact = contact_sensor.data.current_contact_time[:, sensor_cfg.body_ids] > 0.0
    return torch.mean((desired_stance == is_contact).float(), dim=1) * moving


def trot_stance_swing_tracking(
    env: "ManagerBasedRLEnv",
    sensor_cfg: SceneEntityCfg,
    asset_cfg: SceneEntityCfg,
    command_name: str = "base_velocity",
    command_threshold: float = 0.1,
    duty_factor: float = 0.52,
    min_frequency: float = 1.4,
    max_frequency: float = 3.2,
    full_speed: float = 3.0,
    yaw_speed_scale: float = 0.35,
    stance_velocity_std: float = 0.35,
    swing_force_std: float = 25.0,
) -> torch.Tensor:
    """Keep scheduled stance feet still and scheduled swing feet unloaded."""

    if stance_velocity_std <= 0.0 or swing_force_std <= 0.0:
        raise ValueError("trot tracking standard deviations must be positive")
    asset = env.scene[asset_cfg.name]
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    desired_stance, moving = _desired_trot_stance(
        env,
        command_name,
        command_threshold,
        duty_factor,
        min_frequency,
        max_frequency,
        full_speed,
        yaw_speed_scale,
    )
    foot_speed = torch.linalg.vector_norm(
        asset.data.body_lin_vel_w[:, asset_cfg.body_ids, :2], dim=2
    )
    vertical_force = torch.abs(contact_sensor.data.net_forces_w[:, sensor_cfg.body_ids, 2])
    stance_score = torch.exp(-torch.square(foot_speed / stance_velocity_std))
    swing_score = torch.exp(-torch.square(vertical_force / swing_force_std))
    score = torch.where(desired_stance, stance_score, swing_score)
    return torch.mean(score, dim=1) * moving


def joint_deviation_l2(
    env: "ManagerBasedRLEnv",
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    stand_still_scale: float = 1.0,
    velocity_threshold: float = 0.3,
) -> torch.Tensor:
    """Penalize whole-body squared deviation, stronger during zero-command stand."""

    if stand_still_scale < 1.0:
        raise ValueError("stand_still_scale must be at least 1")
    if velocity_threshold < 0.0:
        raise ValueError("velocity_threshold must be non-negative")
    asset = env.scene[asset_cfg.name]
    error = asset.data.joint_pos[:, asset_cfg.joint_ids] - asset.data.default_joint_pos[:, asset_cfg.joint_ids]
    deviation = torch.sum(torch.square(error), dim=1)
    command = env.command_manager.get_command("base_velocity")
    command_speed = torch.linalg.vector_norm(command[:, :2], dim=1)
    body_speed = torch.linalg.vector_norm(asset.data.root_lin_vel_b[:, :2], dim=1)
    standing = torch.logical_and(command_speed <= 1e-6, body_speed <= velocity_threshold)
    scale = torch.where(standing, torch.full_like(deviation, stand_still_scale), torch.ones_like(deviation))
    return scale * deviation


def _hip_joint_ids(env: "ManagerBasedRLEnv", asset_cfg: SceneEntityCfg) -> torch.Tensor:
    cache_name = "_custom_dog_hip_joint_ids"
    cached = getattr(env, cache_name, None)
    if cached is None:
        asset = env.scene[asset_cfg.name]
        indices = []
        for name in HIP_JOINT_NAMES:
            joint_ids, _ = asset.find_joints(name)
            if len(joint_ids) != 1:
                raise ValueError(f"Expected exactly one joint named {name}, got {joint_ids}")
            indices.append(joint_ids[0])
        cached = torch.tensor(indices, dtype=torch.long, device=env.device)
        setattr(env, cache_name, cached)
    return cached


def hip_nominal_l2(
    env: "ManagerBasedRLEnv",
    target_positions: tuple[float, float, float, float],
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Penalize deviation from a compact, symmetric hip stance."""

    if len(target_positions) != len(HIP_JOINT_NAMES):
        raise ValueError("target_positions must contain FR, FL, RR, RL hip targets")
    asset = env.scene[asset_cfg.name]
    target = torch.tensor(target_positions, dtype=asset.data.joint_pos.dtype, device=env.device)
    error = asset.data.joint_pos[:, _hip_joint_ids(env, asset_cfg)] - target
    return torch.mean(torch.square(error), dim=1)


def hip_outward_excess_l2(
    env: "ManagerBasedRLEnv",
    outward_limit: float,
    lateral_limit_gain: float = 0.0,
    command_name: str = "base_velocity",
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Penalize excessive hip splay while allowing commanded side stepping."""

    if outward_limit <= 0.0:
        raise ValueError("outward_limit must be positive")
    if lateral_limit_gain < 0.0:
        raise ValueError("lateral_limit_gain must be non-negative")
    asset = env.scene[asset_cfg.name]
    outward_sign = torch.tensor((-1.0, 1.0, -1.0, 1.0), device=env.device)
    outward_angle = asset.data.joint_pos[:, _hip_joint_ids(env, asset_cfg)] * outward_sign
    lateral_command = torch.abs(env.command_manager.get_command(command_name)[:, 1])
    adaptive_limit = outward_limit + lateral_limit_gain * lateral_command
    excess = torch.clamp(outward_angle - adaptive_limit.unsqueeze(1), min=0.0)
    return torch.mean(torch.square(excess), dim=1)


def hip_outward_speed_style_l2(
    env: "ManagerBasedRLEnv",
    standing_limit: float = 0.28,
    walking_limit: float = 0.34,
    high_speed_limit: float = 0.42,
    walking_speed: float = 0.35,
    high_speed: float = 1.5,
    lateral_limit_gain: float = 0.10,
    yaw_limit_gain: float = 0.04,
    command_name: str = "base_velocity",
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Softly discourage excessive hip splay while leaving speed-dependent motion free.

    The allowed outward angle grows from standing to walking and then to the
    high-speed regime.  Lateral and yaw commands receive additional allowance,
    so this term is a style regularizer rather than a fixed hip target.
    """

    if not (0.0 < walking_speed < high_speed):
        raise ValueError("walking_speed must be positive and less than high_speed")
    if not (0.0 < standing_limit <= walking_limit <= high_speed_limit):
        raise ValueError("hip limits must be positive and non-decreasing")
    asset = env.scene[asset_cfg.name]
    command = env.command_manager.get_command(command_name)
    speed = torch.linalg.vector_norm(command[:, :2], dim=1) + 0.25 * torch.abs(command[:, 2])
    walk_blend = torch.clamp(speed / walking_speed, min=0.0, max=1.0)
    fast_blend = torch.clamp((speed - walking_speed) / (high_speed - walking_speed), min=0.0, max=1.0)
    limit = standing_limit + walk_blend * (walking_limit - standing_limit)
    limit = limit + fast_blend * (high_speed_limit - walking_limit)
    limit = limit + lateral_limit_gain * torch.abs(command[:, 1]) + yaw_limit_gain * torch.abs(command[:, 2])

    outward_sign = torch.tensor((-1.0, 1.0, -1.0, 1.0), device=env.device)
    outward = asset.data.joint_pos[:, _hip_joint_ids(env, asset_cfg)] * outward_sign
    excess = torch.clamp(outward - limit.unsqueeze(1), min=0.0)
    return torch.mean(torch.square(excess), dim=1)


def foot_clearance_speed_style(
    env: "ManagerBasedRLEnv",
    sensor_cfg: SceneEntityCfg,
    asset_cfg: SceneEntityCfg,
    target_height: float = 0.065,
    std: float = 0.045,
    command_name: str = "base_velocity",
    command_threshold: float = 0.15,
    yaw_speed_scale: float = 0.0,
) -> torch.Tensor:
    """Reward moving feet that pass through a modest clearance band.

    Contact gates the term to the swing phase.  The height is measured from
    the flat-ground world frame and is deliberately a soft band, not a body
    height target or a hard foot trajectory.
    """

    if target_height <= 0.0 or std <= 0.0 or yaw_speed_scale < 0.0:
        raise ValueError("target_height and std must be positive")
    asset = env.scene[asset_cfg.name]
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    in_contact = contact_sensor.data.current_contact_time[:, sensor_cfg.body_ids] > 0.0
    command = env.command_manager.get_command(command_name)
    motion = torch.linalg.vector_norm(command[:, :2], dim=1) + yaw_speed_scale * torch.abs(command[:, 2])
    moving = motion > command_threshold
    swing = (~in_contact) & moving.unsqueeze(1)
    foot_z = asset.data.body_pos_w[:, asset_cfg.body_ids, 2]
    foot_speed = torch.linalg.vector_norm(asset.data.body_lin_vel_w[:, asset_cfg.body_ids, :2], dim=2)
    height_error = torch.square(foot_z - target_height)
    motion_gate = torch.tanh(4.0 * foot_speed)
    per_foot = torch.exp(-height_error / (std * std)) * motion_gate * swing
    return torch.mean(per_foot, dim=1)


def foot_soft_landing_l2(
    env: "ManagerBasedRLEnv",
    sensor_cfg: SceneEntityCfg,
    asset_cfg: SceneEntityCfg,
    command_name: str = "base_velocity",
    vertical_speed_std: float = 0.8,
    command_threshold: float = 0.15,
    yaw_speed_scale: float = 0.0,
) -> torch.Tensor:
    """Penalize high downward foot velocity on the first contact frame."""

    if vertical_speed_std <= 0.0 or yaw_speed_scale < 0.0:
        raise ValueError("vertical_speed_std must be positive")
    asset = env.scene[asset_cfg.name]
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    first_contact = contact_sensor.compute_first_contact(env.step_dt)[:, sensor_cfg.body_ids]
    downward_speed = torch.clamp(-asset.data.body_lin_vel_w[:, asset_cfg.body_ids, 2], min=0.0)
    command = env.command_manager.get_command(command_name)
    motion = torch.linalg.vector_norm(command[:, :2], dim=1) + yaw_speed_scale * torch.abs(command[:, 2])
    moving = (motion > command_threshold).unsqueeze(1)
    return torch.mean(torch.square(downward_speed / vertical_speed_std) * first_contact * moving, dim=1)


class FootSoftLandingPreviousVelocity(ManagerTermBase):
    """Penalize touchdown using the foot velocity immediately before contact.

    The contact-frame velocity is already constrained by the ground solver and
    therefore underestimates impact severity.  Keeping one control step of
    history gives the reward a useful pre-impact signal without adding policy
    observations or a gait clock.
    """

    def __init__(self, cfg, env):
        super().__init__(cfg, env)
        self._previous_vertical_velocity: torch.Tensor | None = None
        self._initialized = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)

    def reset(self, env_ids=None) -> None:
        if env_ids is None:
            env_ids = slice(None)
        if self._previous_vertical_velocity is not None:
            self._previous_vertical_velocity[env_ids] = 0.0
        self._initialized[env_ids] = False

    def __call__(
        self,
        env: "ManagerBasedRLEnv",
        sensor_cfg: SceneEntityCfg,
        asset_cfg: SceneEntityCfg,
        command_name: str = "base_velocity",
        vertical_speed_std: float = 0.8,
        command_threshold: float = 0.15,
        yaw_speed_scale: float = 0.0,
    ) -> torch.Tensor:
        if vertical_speed_std <= 0.0 or command_threshold <= 0.0 or yaw_speed_scale < 0.0:
            raise ValueError("invalid soft-landing speed or command thresholds")
        asset = env.scene[asset_cfg.name]
        contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
        current_vertical_velocity = asset.data.body_lin_vel_w[:, asset_cfg.body_ids, 2]
        if (
            self._previous_vertical_velocity is None
            or self._previous_vertical_velocity.shape != current_vertical_velocity.shape
        ):
            self._previous_vertical_velocity = current_vertical_velocity.clone()

        first_contact = contact_sensor.compute_first_contact(env.step_dt)[:, sensor_cfg.body_ids]
        downward_speed = torch.clamp(-self._previous_vertical_velocity, min=0.0)
        command = env.command_manager.get_command(command_name)
        motion = torch.linalg.vector_norm(command[:, :2], dim=1)
        motion += yaw_speed_scale * torch.abs(command[:, 2])
        moving = (motion > command_threshold).unsqueeze(1)
        value = torch.mean(
            torch.square(downward_speed / vertical_speed_std) * first_contact * moving,
            dim=1,
        )
        value = torch.where(self._initialized, value, torch.zeros_like(value))
        self._previous_vertical_velocity.copy_(current_vertical_velocity)
        self._initialized[:] = True
        return value


def feet_air_time_command_aware(
    env: "ManagerBasedRLEnv",
    sensor_cfg: SceneEntityCfg,
    threshold: float = 0.16,
    command_name: str = "base_velocity",
    command_threshold: float = 0.10,
    yaw_speed_scale: float = 0.35,
) -> torch.Tensor:
    """Reward a useful swing duration for planar and pure-yaw commands.

    Isaac Lab's stock ``feet_air_time`` term gates motion on ``command[:, :2]``
    only.  That makes a pure yaw request look stationary, so an all-feet-down
    policy receives no incentive to lift a foot.  This term keeps the same
    phase-free first-contact signal while treating yaw as a controllable
    motion component.  It changes rewards only; the policy contract stays 45
    observations and 12 actions.
    """

    if threshold < 0.0 or command_threshold <= 0.0 or yaw_speed_scale < 0.0:
        raise ValueError("invalid air-time command thresholds or yaw scale")
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    first_contact = contact_sensor.compute_first_contact(env.step_dt)[:, sensor_cfg.body_ids]
    last_air_time = contact_sensor.data.last_air_time[:, sensor_cfg.body_ids]
    command = env.command_manager.get_command(command_name)
    motion = torch.linalg.vector_norm(command[:, :2], dim=1) + yaw_speed_scale * torch.abs(command[:, 2])
    moving = motion > command_threshold
    per_foot = (last_air_time - threshold) * first_contact
    return torch.mean(per_foot * moving.unsqueeze(1), dim=1)


def pure_axis_swing_count(
    env: "ManagerBasedRLEnv",
    sensor_cfg: SceneEntityCfg,
    command_name: str = "base_velocity",
    forward_deadband: float = 0.05,
    lateral_minimum: float = 0.08,
    yaw_minimum: float = 0.12,
    target_airborne: float = 1.5,
    airborne_std: float = 0.75,
) -> torch.Tensor:
    """Reward one or two swing feet only during pure lateral/yaw commands.

    A velocity-only objective has a stable local optimum where all four feet
    remain planted for small pure ``vy`` or ``wz`` commands.  This term makes
    that posture score zero and rewards a physically plausible one-to-two-foot
    swing state.  It does not prescribe a clock or a joint pose, and is not
    active for forward or ``vx+wz`` commands.
    """

    if forward_deadband < 0.0 or lateral_minimum <= 0.0 or yaw_minimum <= 0.0:
        raise ValueError("invalid pure-axis command thresholds")
    if not (0.0 < target_airborne < 4.0 and airborne_std > 0.0):
        raise ValueError("invalid swing-count target")
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    in_contact = contact_sensor.data.current_contact_time[:, sensor_cfg.body_ids] > 0.0
    airborne_count = torch.sum((~in_contact).to(torch.float), dim=1)
    command = env.command_manager.get_command(command_name)
    pure_axis = (torch.abs(command[:, 0]) <= forward_deadband) & (
        (torch.abs(command[:, 1]) >= lateral_minimum)
        | (torch.abs(command[:, 2]) >= yaw_minimum)
    )
    score = torch.exp(-torch.square((airborne_count - target_airborne) / airborne_std))
    planted_score = torch.exp(
        torch.as_tensor(
            -((target_airborne / airborne_std) ** 2),
            dtype=score.dtype,
            device=score.device,
        )
    )
    normalized = torch.clamp((score - planted_score) / (1.0 - planted_score), min=0.0)
    return normalized * pure_axis


def pure_axis_swing_direction(
    env: "ManagerBasedRLEnv",
    sensor_cfg: SceneEntityCfg,
    asset_cfg: SceneEntityCfg,
    command_name: str = "base_velocity",
    forward_deadband: float = 0.05,
    lateral_minimum: float = 0.08,
    yaw_minimum: float = 0.12,
    target_speed: float = 0.25,
) -> torch.Tensor:
    """Reward lateral or tangential travel of airborne feet for pure axes.

    Lateral velocity tracking alone only rewards the body after an appropriate
    foot has already been lifted and placed.  This geometric intermediate
    reward supplies a direction for that swing: body-y for ``vy`` and tangent
    to the body centre for ``wz``.  The reward is phase-free, applies only to
    airborne feet, and leaves forward plus ``vx+wz`` locomotion untouched.
    """

    if forward_deadband < 0.0 or lateral_minimum <= 0.0 or yaw_minimum <= 0.0:
        raise ValueError("invalid pure-axis command thresholds")
    if target_speed <= 0.0:
        raise ValueError("target_speed must be positive")
    asset = env.scene[asset_cfg.name]
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    swing = contact_sensor.data.current_contact_time[:, sensor_cfg.body_ids] <= 0.0
    command = env.command_manager.get_command(command_name)
    lateral_active = torch.abs(command[:, 1]) >= lateral_minimum
    yaw_active = torch.abs(command[:, 2]) >= yaw_minimum
    pure_axis = (torch.abs(command[:, 0]) <= forward_deadband) & (lateral_active | yaw_active)

    num_feet = len(asset_cfg.body_ids)
    root_quat = asset.data.root_quat_w.unsqueeze(1).expand(-1, num_feet, -1).reshape(-1, 4)
    foot_pos_w = asset.data.body_pos_w[:, asset_cfg.body_ids] - asset.data.root_pos_w.unsqueeze(1)
    foot_vel_w = asset.data.body_lin_vel_w[:, asset_cfg.body_ids] - asset.data.root_lin_vel_w.unsqueeze(1)
    foot_pos_b = quat_apply_inverse(root_quat, foot_pos_w.reshape(-1, 3)).reshape(-1, num_feet, 3)
    foot_vel_b = quat_apply_inverse(root_quat, foot_vel_w.reshape(-1, 3)).reshape(-1, num_feet, 3)

    direction = torch.zeros_like(foot_vel_b[:, :, :2])
    direction[:, :, 1] += torch.sign(command[:, 1]).unsqueeze(1) * lateral_active.unsqueeze(1)
    tangent = torch.stack((-foot_pos_b[:, :, 1], foot_pos_b[:, :, 0]), dim=2)
    tangent = tangent / torch.clamp(torch.linalg.vector_norm(tangent, dim=2, keepdim=True), min=1.0e-4)
    direction += tangent * torch.sign(command[:, 2]).view(-1, 1, 1) * yaw_active.view(-1, 1, 1)
    direction = direction / torch.clamp(torch.linalg.vector_norm(direction, dim=2, keepdim=True), min=1.0)
    progress = torch.sum(foot_vel_b[:, :, :2] * direction, dim=2)
    normalized_progress = torch.clamp(progress / target_speed, min=0.0, max=1.0)
    return torch.mean(normalized_progress * swing * pure_axis.unsqueeze(1), dim=1)


class ActionSmoothness2(ManagerTermBase):
    """Penalize the discrete second derivative of the policy action.

    ``ActionManager`` exposes only the current and previous action.  Keeping a
    private previous-previous buffer makes this term independent of observation
    history, so the deployed policy remains exactly 45-dimensional.
    """

    def __init__(self, cfg, env):
        super().__init__(cfg, env)
        action = env.action_manager.action
        self._previous_previous_action = torch.zeros_like(action)
        self._initialized = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)

    def reset(self, env_ids=None) -> None:
        if env_ids is None:
            env_ids = slice(None)
        self._previous_previous_action[env_ids] = 0.0
        self._initialized[env_ids] = False

    def __call__(self, env) -> torch.Tensor:
        current = env.action_manager.action
        previous = env.action_manager.prev_action
        second_difference = current - 2.0 * previous + self._previous_previous_action
        value = torch.mean(torch.square(second_difference), dim=1)
        value = torch.where(self._initialized, value, torch.zeros_like(value))
        self._previous_previous_action[:] = previous
        self._initialized[:] = True
        return value


def foot_impact_velocity_l2(
    env: "ManagerBasedRLEnv",
    sensor_cfg: SceneEntityCfg,
    asset_cfg: SceneEntityCfg,
    command_name: str = "base_velocity",
    impact_speed_std: float = 0.8,
    command_threshold: float = 0.15,
    yaw_speed_scale: float = 0.25,
) -> torch.Tensor:
    """Penalize high total foot speed on first contact, softly scaled."""

    if impact_speed_std <= 0.0 or command_threshold <= 0.0 or yaw_speed_scale < 0.0:
        raise ValueError("invalid impact speed or command thresholds")
    asset = env.scene[asset_cfg.name]
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    first_contact = contact_sensor.compute_first_contact(env.step_dt)[:, sensor_cfg.body_ids]
    foot_speed = torch.linalg.vector_norm(asset.data.body_lin_vel_w[:, asset_cfg.body_ids, :], dim=2)
    command = env.command_manager.get_command(command_name)
    moving = (
        torch.linalg.vector_norm(command[:, :2], dim=1) + yaw_speed_scale * torch.abs(command[:, 2])
    ) > command_threshold
    return torch.mean(torch.square(foot_speed / impact_speed_std) * first_contact * moving.unsqueeze(1), dim=1)


def stance_foot_placement_l2(
    env: "ManagerBasedRLEnv",
    sensor_cfg: SceneEntityCfg,
    asset_cfg: SceneEntityCfg,
    nominal_positions: tuple[float, ...],
    stance_time: float = 0.24,
    position_std: float = 0.07,
    command_name: str = "base_velocity",
    command_threshold: float = 0.12,
    max_error: float = 0.18,
) -> torch.Tensor:
    """Keep stance feet near a geometry-based Raibert landing point.

    The target is expressed in the body frame using the actual CAD home-foot
    positions.  Commanded body velocity and yaw contribute a small predicted
    stance displacement, preserving natural hip motion during side steps and
    turns while discouraging wide outward splay.
    """

    if len(nominal_positions) != 12:
        raise ValueError("nominal_positions must contain four xyz triplets")
    if stance_time <= 0.0 or position_std <= 0.0 or max_error <= 0.0:
        raise ValueError("stance_time, position_std and max_error must be positive")
    asset = env.scene[asset_cfg.name]
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    contact = contact_sensor.data.current_contact_time[:, sensor_cfg.body_ids] > 0.0
    root_pos = asset.data.root_pos_w.unsqueeze(1)
    foot_pos_w = asset.data.body_pos_w[:, asset_cfg.body_ids] - root_pos
    num_feet = foot_pos_w.shape[1]
    root_quat = asset.data.root_quat_w.unsqueeze(1).expand(-1, num_feet, -1).reshape(-1, 4)
    foot_pos_b = quat_apply_inverse(root_quat, foot_pos_w.reshape(-1, 3)).reshape(-1, num_feet, 3)
    nominal = torch.tensor(nominal_positions, dtype=foot_pos_b.dtype, device=env.device).view(1, 4, 3)
    command = env.command_manager.get_command(command_name)
    target = nominal.expand(foot_pos_b.shape[0], -1, -1).clone()
    half_stance = 0.5 * stance_time
    target[:, :, 0] += half_stance * command[:, 0:1]
    target[:, :, 1] += half_stance * command[:, 1:2]
    # Planar omega x r = (-wz * y, wz * x).
    target[:, :, 0] -= half_stance * command[:, 2:3] * nominal[:, :, 1]
    target[:, :, 1] += half_stance * command[:, 2:3] * nominal[:, :, 0]
    error = foot_pos_b[:, :, :2] - target[:, :, :2]
    error_norm = torch.linalg.vector_norm(error, dim=2)
    error_norm = torch.clamp(error_norm, max=max_error)
    moving = (torch.linalg.vector_norm(command[:, :2], dim=1) + 0.25 * torch.abs(command[:, 2])) > command_threshold
    gated = torch.square(error_norm / position_std) * contact * moving.unsqueeze(1)
    return torch.mean(gated, dim=1)


def stance_foot_lateral_placement_l2(
    env: "ManagerBasedRLEnv",
    sensor_cfg: SceneEntityCfg,
    asset_cfg: SceneEntityCfg,
    nominal_positions: tuple[float, ...],
    stance_time: float = 0.24,
    position_std: float = 0.06,
    command_name: str = "base_velocity",
    command_threshold: float = 0.12,
    max_error: float = 0.12,
) -> torch.Tensor:
    """Keep stance-foot width compact without prescribing a hip angle.

    Only the body-frame lateral coordinate is shaped.  Forward step length is
    therefore left to the policy, while lateral and yaw commands shift the
    target so that genuine side stepping and steering remain available.
    """

    if len(nominal_positions) != 12:
        raise ValueError("nominal_positions must contain four xyz triplets")
    if stance_time <= 0.0 or position_std <= 0.0 or max_error <= 0.0:
        raise ValueError("stance_time, position_std and max_error must be positive")
    asset = env.scene[asset_cfg.name]
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    contact = contact_sensor.data.current_contact_time[:, sensor_cfg.body_ids] > 0.0
    root_pos = asset.data.root_pos_w.unsqueeze(1)
    foot_pos_w = asset.data.body_pos_w[:, asset_cfg.body_ids] - root_pos
    num_feet = foot_pos_w.shape[1]
    root_quat = asset.data.root_quat_w.unsqueeze(1).expand(-1, num_feet, -1).reshape(-1, 4)
    foot_pos_b = quat_apply_inverse(root_quat, foot_pos_w.reshape(-1, 3)).reshape(-1, num_feet, 3)
    nominal = torch.tensor(nominal_positions, dtype=foot_pos_b.dtype, device=env.device).view(1, 4, 3)
    command = env.command_manager.get_command(command_name)
    target_y = nominal[:, :, 1].expand(foot_pos_b.shape[0], -1).clone()
    target_y += 0.5 * stance_time * command[:, 1:2]
    target_y += 0.5 * stance_time * command[:, 2:3] * nominal[:, :, 0]
    error = torch.clamp(torch.abs(foot_pos_b[:, :, 1] - target_y), max=max_error)
    moving = (
        torch.linalg.vector_norm(command[:, :2], dim=1) + 0.25 * torch.abs(command[:, 2])
    ) > command_threshold
    gated = torch.square(error / position_std) * contact * moving.unsqueeze(1)
    return torch.mean(gated, dim=1)


def speed_adaptive_feet_gait(
    env: "ManagerBasedRLEnv",
    period: float,
    sensor_cfg: SceneEntityCfg,
    threshold: float = 0.55,
    walk_speed: float = 0.75,
    trot_speed: float = 1.25,
    command_name: str = "base_velocity",
) -> torch.Tensor:
    """Use a walk contact pattern at low speed and diagonal trot at high speed."""

    if not (0.0 < walk_speed < trot_speed and period > 0.0):
        raise ValueError("invalid gait speed or period")
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    is_contact = contact_sensor.data.current_contact_time[:, sensor_cfg.body_ids] > 0.0
    command = env.command_manager.get_command(command_name)
    speed = torch.linalg.vector_norm(command[:, :2], dim=1)
    trot_blend = torch.clamp((speed - walk_speed) / (trot_speed - walk_speed), min=0.0, max=1.0)
    # Body order is FR, FL, RR, RL. Walk is sequential; trot is diagonal.
    walk_offset = (0.0, 0.5, 0.75, 0.25)
    trot_offset = (0.0, 0.5, 0.5, 0.0)
    phase = ((env.episode_length_buf * env.step_dt) % period / period).unsqueeze(1)
    walk_reward = torch.zeros(env.num_envs, dtype=torch.float, device=env.device)
    trot_reward = torch.zeros(env.num_envs, dtype=torch.float, device=env.device)
    for i, (walk_phase_offset, trot_phase_offset) in enumerate(zip(walk_offset, trot_offset)):
        walk_stance = ((phase[:, 0] + walk_phase_offset) % 1.0) < threshold
        trot_stance = ((phase[:, 0] + trot_phase_offset) % 1.0) < threshold
        walk_reward += (walk_stance == is_contact[:, i]).float()
        trot_reward += (trot_stance == is_contact[:, i]).float()
    reward = (1.0 - trot_blend) * walk_reward + trot_blend * trot_reward
    reward *= speed > 0.1
    return reward


def pure_axis_feet_gait(
    env: "ManagerBasedRLEnv",
    period: float,
    sensor_cfg: SceneEntityCfg,
    threshold: float = 0.55,
    forward_deadband: float = 0.05,
    lateral_minimum: float = 0.10,
    yaw_minimum: float = 0.15,
    command_name: str = "base_velocity",
) -> torch.Tensor:
    """Reward a diagonal alternating contact pattern only for pure side/yaw commands.

    The forward expert already owns the high-speed ``vx`` and ``vx+wz`` gait.
    Applying a fixed clock there would overwrite its useful cadence.  Pure
    lateral and pure yaw requests otherwise admit a stationary all-feet-down
    solution, so they alone receive a clocked trot contact target.
    """

    if period <= 0.0:
        raise ValueError("period must be positive")
    if not 0.0 < threshold < 1.0:
        raise ValueError("threshold must be in (0, 1)")
    if forward_deadband < 0.0 or lateral_minimum <= 0.0 or yaw_minimum <= 0.0:
        raise ValueError("invalid pure-axis command thresholds")

    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    is_contact = contact_sensor.data.current_contact_time[:, sensor_cfg.body_ids] > 0.0
    command = env.command_manager.get_command(command_name)
    active = (torch.abs(command[:, 0]) <= forward_deadband) & (
        (torch.abs(command[:, 1]) >= lateral_minimum)
        | (torch.abs(command[:, 2]) >= yaw_minimum)
    )

    global_phase = ((env.episode_length_buf * env.step_dt) % period / period).unsqueeze(1)
    # Explicit FR, FL, RR, RL ordering: FR+RL and FL+RR alternate.
    offsets = (0.0, 0.5, 0.5, 0.0)
    reward = torch.zeros(env.num_envs, dtype=torch.float, device=env.device)
    for index, offset in enumerate(offsets):
        stance = ((global_phase[:, 0] + offset) % 1.0) < threshold
        reward += (stance == is_contact[:, index]).float()
    return reward * active


def speed_adaptive_base_height_l2(
    env: "ManagerBasedRLEnv",
    standing_height: float,
    crouched_height: float,
    crouch_start_speed: float,
    crouch_full_speed: float,
    yaw_speed_scale: float = 0.0,
    command_name: str = "base_velocity",
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Track a lower body-height target only as commanded planar speed increases."""

    if crouched_height > standing_height:
        raise ValueError("crouched_height must not exceed standing_height")
    if crouch_full_speed <= crouch_start_speed:
        raise ValueError("crouch_full_speed must exceed crouch_start_speed")
    if yaw_speed_scale < 0.0:
        raise ValueError("yaw_speed_scale must be non-negative")
    asset = env.scene[asset_cfg.name]
    command = env.command_manager.get_command(command_name)
    planar_speed_sq = torch.sum(torch.square(command[:, :2]), dim=1)
    speed = torch.sqrt(planar_speed_sq + torch.square(yaw_speed_scale * command[:, 2]))
    blend = torch.clamp(
        (speed - crouch_start_speed) / (crouch_full_speed - crouch_start_speed),
        min=0.0,
        max=1.0,
    )
    target_height = standing_height + blend * (crouched_height - standing_height)
    return torch.square(asset.data.root_pos_w[:, 2] - target_height)


def recovery_upright_height(
    env: "ManagerBasedRLEnv",
    prone_height: float,
    standing_height: float,
    prone_only: bool = True,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Reward lifting a level trunk from the measured prone height to standing."""

    if standing_height <= prone_height:
        raise ValueError("standing_height must exceed prone_height")
    asset = env.scene[asset_cfg.name]
    normalized_height = torch.clamp(
        (asset.data.root_pos_w[:, 2] - prone_height) / (standing_height - prone_height),
        min=0.0,
        max=1.0,
    )
    upright = torch.clamp(-asset.data.projected_gravity_b[:, 2], min=0.0, max=1.0)
    reward = normalized_height * upright
    if prone_only:
        prone_mask = getattr(env, "_custom_dog_prone_reset_mask", None)
        if prone_mask is None:
            raise RuntimeError("Recovery reset mask was not initialized by reset_recovery_or_standing")
        reward *= prone_mask
    return reward


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


def track_lin_vel_xy_l2(
    env: "ManagerBasedRLEnv",
    command_name: str,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Return squared planar velocity tracking error without exponential saturation."""

    asset = env.scene[asset_cfg.name]
    command = env.command_manager.get_command(command_name)[:, :2]
    error = command - asset.data.root_lin_vel_b[:, :2]
    return torch.sum(torch.square(error), dim=1)


def track_lin_vel_xy_relative_l1(
    env: "ManagerBasedRLEnv",
    command_name: str,
    command_min: float = 0.05,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Track non-zero planar commands relative to their requested magnitude."""

    if command_min <= 0.0:
        raise ValueError("command_min must be positive")
    asset = env.scene[asset_cfg.name]
    command = env.command_manager.get_command(command_name)[:, :2]
    command_norm = torch.linalg.vector_norm(command, dim=1)
    error_norm = torch.linalg.vector_norm(command - asset.data.root_lin_vel_b[:, :2], dim=1)
    active = command_norm >= command_min
    return torch.where(active, error_norm / torch.clamp(command_norm, min=command_min), torch.zeros_like(error_norm))


def track_ang_vel_z_relative_l1(
    env: "ManagerBasedRLEnv",
    command_name: str,
    command_min: float = 0.05,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Track non-zero yaw commands relative to their requested magnitude."""

    if command_min <= 0.0:
        raise ValueError("command_min must be positive")
    asset = env.scene[asset_cfg.name]
    command = env.command_manager.get_command(command_name)[:, 2]
    error = torch.abs(command - asset.data.root_ang_vel_b[:, 2])
    magnitude = torch.abs(command)
    active = magnitude >= command_min
    return torch.where(active, error / torch.clamp(magnitude, min=command_min), torch.zeros_like(error))


def track_velocity_components_relative_l1(
    env: "ManagerBasedRLEnv",
    command_name: str,
    command_min: tuple[float, float, float] = (0.1, 0.1, 0.15),
    axis_weights: tuple[float, float, float] = (1.0, 1.0, 1.0),
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Average normalized tracking error over each commanded velocity axis.

    A vector-norm reward can trade lateral or yaw accuracy for forward speed
    when all axes are commanded together.  Normalizing active vx, vy and wz
    components separately keeps each joystick axis visible to PPO without
    imposing any robot-specific pose or gait target.
    """

    if len(command_min) != 3 or any(value <= 0.0 for value in command_min):
        raise ValueError("command_min must contain three positive values")
    if len(axis_weights) != 3 or any(value <= 0.0 for value in axis_weights):
        raise ValueError("axis_weights must contain three positive values")
    asset = env.scene[asset_cfg.name]
    command = env.command_manager.get_command(command_name)
    measured = torch.stack(
        (
            asset.data.root_lin_vel_b[:, 0],
            asset.data.root_lin_vel_b[:, 1],
            asset.data.root_ang_vel_b[:, 2],
        ),
        dim=1,
    )
    minimum = torch.tensor(command_min, dtype=command.dtype, device=env.device)
    weights = torch.tensor(axis_weights, dtype=command.dtype, device=env.device)
    active = torch.abs(command) >= minimum
    relative_error = torch.abs(command - measured) / torch.maximum(torch.abs(command), minimum)
    active_weights = active * weights
    active_weight_sum = torch.clamp(torch.sum(active_weights, dim=1), min=1.0)
    return torch.sum(relative_error * active_weights, dim=1) / active_weight_sum


def inactive_velocity_axes_l2(
    env: "ManagerBasedRLEnv",
    command_name: str,
    command_min: tuple[float, float, float] = (0.10, 0.025, 0.05),
    axis_weights: tuple[float, float, float] = (1.0, 1.0, 1.0),
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Penalize motion on axes that the operator did not command.

    Component-relative tracking deliberately ignores zero-command axes. That
    can let a fast ``vx+wz`` gait trade yaw tracking for lateral drift. This
    complementary term acts only below each axis threshold, so commanded
    side-stepping and turning remain unconstrained.
    """

    if len(command_min) != 3 or any(value <= 0.0 for value in command_min):
        raise ValueError("command_min must contain three positive values")
    if len(axis_weights) != 3 or any(value <= 0.0 for value in axis_weights):
        raise ValueError("axis_weights must contain three positive values")
    asset = env.scene[asset_cfg.name]
    command = env.command_manager.get_command(command_name)
    measured = torch.stack(
        (
            asset.data.root_lin_vel_b[:, 0],
            asset.data.root_lin_vel_b[:, 1],
            asset.data.root_ang_vel_b[:, 2],
        ),
        dim=1,
    )
    minimum = torch.tensor(command_min, dtype=command.dtype, device=env.device)
    weights = torch.tensor(axis_weights, dtype=command.dtype, device=env.device)
    inactive = torch.abs(command) < minimum
    return torch.sum(torch.square(measured) * inactive * weights, dim=1)


def high_speed_turn_lateral_drift_l2(
    env: "ManagerBasedRLEnv",
    command_name: str,
    min_forward_speed: float = 0.75,
    min_yaw_rate: float = 0.10,
    lateral_command_deadband: float = 0.025,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Penalize body-y drift only for high-speed vx+wz commands."""

    if min_forward_speed <= 0.0 or min_yaw_rate <= 0.0 or lateral_command_deadband <= 0.0:
        raise ValueError("high-speed turn thresholds must be positive")
    command = env.command_manager.get_command(command_name)
    active = (
        (command[:, 0] >= min_forward_speed)
        & (torch.abs(command[:, 1]) < lateral_command_deadband)
        & (torch.abs(command[:, 2]) >= min_yaw_rate)
    )
    lateral_velocity = env.scene[asset_cfg.name].data.root_lin_vel_b[:, 1]
    return torch.square(lateral_velocity) * active


def track_velocity_components_progress(
    env: "ManagerBasedRLEnv",
    command_name: str,
    command_min: tuple[float, float, float] = (0.1, 0.1, 0.15),
    axis_weights: tuple[float, float, float] = (1.0, 1.0, 1.0),
    max_progress: tuple[float, float, float] = (1.0, 1.0, 1.2),
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Reward signed progress independently along commanded vx, vy and wz."""

    if len(command_min) != 3 or any(value <= 0.0 for value in command_min):
        raise ValueError("command_min must contain three positive values")
    if len(axis_weights) != 3 or any(value <= 0.0 for value in axis_weights):
        raise ValueError("axis_weights must contain three positive values")
    if len(max_progress) != 3 or any(value <= 0.0 for value in max_progress):
        raise ValueError("max_progress must contain three positive values")
    asset = env.scene[asset_cfg.name]
    command = env.command_manager.get_command(command_name)
    measured = torch.stack(
        (
            asset.data.root_lin_vel_b[:, 0],
            asset.data.root_lin_vel_b[:, 1],
            asset.data.root_ang_vel_b[:, 2],
        ),
        dim=1,
    )
    minimum = torch.tensor(command_min, dtype=command.dtype, device=env.device)
    weights = torch.tensor(axis_weights, dtype=command.dtype, device=env.device)
    limits = torch.tensor(max_progress, dtype=command.dtype, device=env.device)
    active = torch.abs(command) >= minimum
    signed_progress = torch.sign(command) * measured
    signed_progress = torch.clamp(signed_progress, min=-limits, max=limits)
    normalized = signed_progress / torch.maximum(torch.abs(command), minimum)
    active_weights = active * weights
    weight_sum = torch.clamp(torch.sum(active_weights, dim=1), min=1.0)
    return torch.sum(normalized * active_weights, dim=1) / weight_sum


def planar_command_progress(
    env: "ManagerBasedRLEnv",
    command_name: str,
    command_min: float = 0.10,
    max_progress: float = 1.0,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Reward signed planar speed along the requested command direction."""

    if command_min <= 0.0 or max_progress <= 0.0:
        raise ValueError("command_min and max_progress must be positive")
    asset = env.scene[asset_cfg.name]
    command = env.command_manager.get_command(command_name)[:, :2]
    command_norm = torch.linalg.vector_norm(command, dim=1)
    direction = command / torch.clamp(command_norm, min=command_min).unsqueeze(1)
    progress = torch.sum(asset.data.root_lin_vel_b[:, :2] * direction, dim=1)
    progress = torch.clamp(progress, min=-max_progress, max=max_progress)
    return torch.where(command_norm >= command_min, progress, torch.zeros_like(progress))


def yaw_command_progress(
    env: "ManagerBasedRLEnv",
    command_name: str,
    command_min: float = 0.15,
    max_progress: float = 1.2,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Reward signed yaw rate in the direction requested by the command."""

    if command_min <= 0.0 or max_progress <= 0.0:
        raise ValueError("command_min and max_progress must be positive")
    asset = env.scene[asset_cfg.name]
    command = env.command_manager.get_command(command_name)[:, 2]
    progress = torch.sign(command) * asset.data.root_ang_vel_b[:, 2]
    progress = torch.clamp(progress, min=-max_progress, max=max_progress)
    return torch.where(torch.abs(command) >= command_min, progress, torch.zeros_like(progress))


def lateral_command_progress(
    env: "ManagerBasedRLEnv",
    command_name: str,
    command_min: float = 0.20,
    max_progress: float = 0.8,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Reward signed body-y speed for commands dominated by lateral motion."""

    if command_min <= 0.0 or max_progress <= 0.0:
        raise ValueError("command_min and max_progress must be positive")
    asset = env.scene[asset_cfg.name]
    command = env.command_manager.get_command(command_name)
    lateral = command[:, 1]
    progress = torch.sign(lateral) * asset.data.root_lin_vel_b[:, 1]
    progress = torch.clamp(progress, min=-max_progress, max=max_progress)
    active = (torch.abs(lateral) >= command_min) & (torch.abs(lateral) >= torch.abs(command[:, 0]))
    return torch.where(active, progress, torch.zeros_like(progress))


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

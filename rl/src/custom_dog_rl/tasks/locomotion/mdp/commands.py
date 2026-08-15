"""Command generators specific to the custom-dog velocity tasks."""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING

import torch

from isaaclab.envs.mdp import UniformVelocityCommand
from isaaclab.utils import configclass
from unitree_rl_lab.tasks.locomotion.mdp.commands.velocity_command import (
    UniformLevelVelocityCommandCfg,
)

from .usage_window import UsageWindowStatistics

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv


class MixedForwardVelocityCommand(UniformVelocityCommand):
    """Sample a low-speed/full-range mixture while retaining zero-command environments."""

    cfg: "MixedForwardVelocityCommandCfg"

    def __init__(self, cfg: "MixedForwardVelocityCommandCfg", env: "ManagerBasedEnv"):
        low, high = cfg.low_speed_range
        if not 0.0 <= cfg.rel_low_speed_envs <= 1.0:
            raise ValueError("rel_low_speed_envs must be in [0, 1].")
        if low > high:
            raise ValueError("low_speed_range lower bound must not exceed its upper bound.")
        if low < cfg.ranges.lin_vel_x[0] or high > cfg.ranges.lin_vel_x[1]:
            raise ValueError("low_speed_range must be contained in ranges.lin_vel_x.")
        super().__init__(cfg, env)

    def _resample_command(self, env_ids: Sequence[int]):
        super()._resample_command(env_ids)
        if len(env_ids) == 0 or self.cfg.rel_low_speed_envs == 0.0:
            return

        env_ids_tensor = torch.as_tensor(env_ids, dtype=torch.long, device=self.device)
        use_low_speed = torch.rand(len(env_ids_tensor), device=self.device) < self.cfg.rel_low_speed_envs
        # Standing is an independent, exact-zero command mode.  The mixture
        # probability therefore describes only the remaining moving samples.
        use_low_speed &= ~self.is_standing_env[env_ids_tensor]
        low_speed_env_ids = env_ids_tensor[use_low_speed]
        if len(low_speed_env_ids) == 0:
            return

        low, high = self.cfg.low_speed_range
        sampled_speed = torch.empty(len(low_speed_env_ids), device=self.device).uniform_(low, high)
        self.vel_command_b[low_speed_env_ids, 0] = sampled_speed


@configclass
class MixedForwardVelocityCommandCfg(UniformLevelVelocityCommandCfg):
    """Configuration for a forward-speed mixture command."""

    class_type: type = MixedForwardVelocityCommand

    rel_low_speed_envs: float = 0.5
    """Fraction of non-standing samples drawn from :attr:`low_speed_range`."""

    low_speed_range: tuple[float, float] = (0.1, 0.5)
    """Forward-speed interval oversampled by the mixture (m/s)."""


class RecoveryVelocityCommand(UniformVelocityCommand):
    """Hold zero command briefly after a prone reset, then release the sampled command."""

    cfg: "RecoveryVelocityCommandCfg"

    def __init__(self, cfg: "RecoveryVelocityCommandCfg", env: "ManagerBasedEnv"):
        if cfg.recovery_duration_s < 0.0:
            raise ValueError("recovery_duration_s must be non-negative.")
        if len(cfg.omni_mode_probabilities) != 4:
            raise ValueError("omni_mode_probabilities must contain forward/lateral/yaw/combined")
        if any(probability < 0.0 for probability in cfg.omni_mode_probabilities):
            raise ValueError("omni_mode_probabilities must be non-negative")
        if abs(sum(cfg.omni_mode_probabilities) - 1.0) > 1.0e-6:
            raise ValueError("omni_mode_probabilities must sum to one")
        if not 0.0 <= cfg.lateral_min_fraction <= 1.0:
            raise ValueError("lateral_min_fraction must be in [0, 1]")
        if not 0.0 <= cfg.yaw_min_fraction <= 1.0:
            raise ValueError("yaw_min_fraction must be in [0, 1]")
        super().__init__(cfg, env)
        self._sampled_vel_command_b = torch.zeros_like(self.vel_command_b)

    def _resample_command(self, env_ids: Sequence[int]):
        super()._resample_command(env_ids)
        if self.cfg.omni_mixture:
            self._resample_omni_mixture(env_ids)
        self._sampled_vel_command_b[env_ids] = self.vel_command_b[env_ids]

    def _resample_omni_mixture(self, env_ids: Sequence[int]) -> None:
        """Oversample pure and combined planar/yaw behaviors for early adaptation."""

        env_ids_tensor = torch.as_tensor(env_ids, dtype=torch.long, device=self.device)
        active = ~self.is_standing_env[env_ids_tensor]
        active_ids = env_ids_tensor[active]
        if len(active_ids) == 0:
            return
        probabilities = torch.tensor(
            self.cfg.omni_mode_probabilities,
            dtype=self.vel_command_b.dtype,
            device=self.device,
        )
        mode = torch.multinomial(probabilities, len(active_ids), replacement=True)
        self.vel_command_b[active_ids] = 0.0

        x_low, x_high = self.cfg.ranges.lin_vel_x
        y_low, y_high = self.cfg.ranges.lin_vel_y
        yaw_low, yaw_high = self.cfg.ranges.ang_vel_z

        def sample_magnitude(lower: float, upper: float) -> torch.Tensor:
            """Sample safely when an axis is disabled or has a fixed magnitude."""

            upper = max(0.0, upper)
            if upper == 0.0:
                return torch.zeros(len(active_ids), device=self.device)
            lower = min(max(0.0, lower), upper)
            if lower == upper:
                return torch.full((len(active_ids),), upper, device=self.device)
            return torch.empty(len(active_ids), device=self.device).uniform_(lower, upper)

        x_mag = sample_magnitude(max(0.1, x_low), max(0.0, x_high))
        y_upper = max(abs(y_low), abs(y_high))
        yaw_upper = max(abs(yaw_low), abs(yaw_high))
        y_mag = sample_magnitude(
            max(0.05, self.cfg.lateral_min_fraction * y_upper),
            y_upper,
        )
        yaw_mag = sample_magnitude(
            max(0.1, self.cfg.yaw_min_fraction * yaw_upper),
            yaw_upper,
        )
        y_sign = torch.where(torch.rand(len(active_ids), device=self.device) < 0.5, -1.0, 1.0)
        yaw_sign = torch.where(torch.rand(len(active_ids), device=self.device) < 0.5, -1.0, 1.0)

        forward = mode == 0
        lateral = mode == 1
        yaw = mode == 2
        combined = mode == 3
        self.vel_command_b[active_ids[forward], 0] = x_mag[forward]
        self.vel_command_b[active_ids[lateral], 1] = y_mag[lateral] * y_sign[lateral]
        self.vel_command_b[active_ids[yaw], 2] = yaw_mag[yaw] * yaw_sign[yaw]
        self.vel_command_b[active_ids[combined], 0] = x_mag[combined]
        self.vel_command_b[active_ids[combined], 1] = y_mag[combined] * y_sign[combined]
        self.vel_command_b[active_ids[combined], 2] = yaw_mag[combined] * yaw_sign[combined]

    def reset(self, env_ids: Sequence[int] | None = None) -> dict[str, float]:
        extras = super().reset(env_ids)
        if env_ids is None:
            env_ids = slice(None)
        prone_mask = getattr(self._env, "_custom_dog_prone_reset_mask", None)
        if prone_mask is not None:
            selected_mask = prone_mask if isinstance(env_ids, slice) else prone_mask[env_ids]
            selected_commands = self.vel_command_b[env_ids]
            selected_commands[selected_mask] = 0.0
            self.vel_command_b[env_ids] = selected_commands
        return extras

    def _update_command(self):
        # The parent mutates vel_command_b for standing/heading modes, so first
        # restore the last sampled target before applying those modes again.
        self.vel_command_b.copy_(self._sampled_vel_command_b)
        super()._update_command()

        prone_mask = getattr(self._env, "_custom_dog_prone_reset_mask", None)
        if prone_mask is None or self.cfg.recovery_duration_s == 0.0:
            return
        elapsed_s = self._env.episode_length_buf * self._env.step_dt
        recovery_active = prone_mask & (elapsed_s < self.cfg.recovery_duration_s)
        self.vel_command_b[recovery_active] = 0.0


@configclass
class RecoveryVelocityCommandCfg(UniformLevelVelocityCommandCfg):
    """Configuration for a velocity command with a prone-recovery hold window."""

    class_type: type = RecoveryVelocityCommand

    recovery_duration_s: float = 1.0
    """Seconds of zero command after a prone reset before releasing the sampled command."""

    omni_mixture: bool = False
    """Use balanced pure/combined command modes for early omni-directional adaptation."""

    omni_mode_probabilities: tuple[float, float, float, float] = (0.25, 0.25, 0.25, 0.25)
    """Probabilities for forward, pure lateral, pure yaw and combined samples."""

    lateral_min_fraction: float = 0.5
    """Minimum pure/combined lateral magnitude as a fraction of its limit."""

    yaw_min_fraction: float = 0.5
    """Minimum pure/combined yaw magnitude as a fraction of its limit."""


class SpeedBandOmniVelocityCommand(RecoveryVelocityCommand):
    """Oversample the low-speed walk band while retaining full-speed omni commands."""

    cfg: "SpeedBandOmniVelocityCommandCfg"

    def __init__(self, cfg: "SpeedBandOmniVelocityCommandCfg", env: "ManagerBasedEnv"):
        low, high = cfg.low_speed_range
        if not 0.0 <= cfg.rel_low_speed_forward <= 1.0:
            raise ValueError("rel_low_speed_forward must be in [0, 1]")
        if not (0.0 <= low <= high <= cfg.ranges.lin_vel_x[1]):
            raise ValueError("low_speed_range must be inside the forward command range")
        super().__init__(cfg, env)

    def _resample_command(self, env_ids: Sequence[int]):
        super()._resample_command(env_ids)
        if len(env_ids) == 0 or self.cfg.rel_low_speed_forward == 0.0:
            return
        env_ids_tensor = torch.as_tensor(env_ids, dtype=torch.long, device=self.device)
        sampled = self.vel_command_b[env_ids_tensor]
        has_forward = sampled[:, 0] > 1.0e-6
        use_low_speed = torch.rand(len(env_ids_tensor), device=self.device) < self.cfg.rel_low_speed_forward
        use_low_speed &= has_forward & ~self.is_standing_env[env_ids_tensor]
        low_ids = env_ids_tensor[use_low_speed]
        if len(low_ids) > 0:
            low, high = self.cfg.low_speed_range
            self.vel_command_b[low_ids, 0] = torch.empty(len(low_ids), device=self.device).uniform_(low, high)
        # RecoveryVelocityCommand restores this buffer on every control step.
        self._sampled_vel_command_b[env_ids_tensor] = self.vel_command_b[env_ids_tensor]


@configclass
class SpeedBandOmniVelocityCommandCfg(RecoveryVelocityCommandCfg):
    """Configuration for low-speed/full-speed omni command sampling."""

    class_type: type = SpeedBandOmniVelocityCommand

    rel_low_speed_forward: float = 0.55
    """Fraction of forward-bearing samples drawn from the walk-speed band."""

    low_speed_range: tuple[float, float] = (0.10, 0.60)
    """Oversampled forward-speed interval in m/s."""

class MovingSteeringVelocityCommand(SpeedBandOmniVelocityCommand):
    """Disentangle lateral steering and yaw while an existing gait is active."""

    cfg: "MovingSteeringVelocityCommandCfg"

    def __init__(self, cfg: "MovingSteeringVelocityCommandCfg", env: "ManagerBasedEnv"):
        if len(cfg.steering_mode_probabilities) != 4:
            raise ValueError(
                "steering_mode_probabilities must contain forward/forward+lateral/forward+yaw/combined"
            )
        if any(probability < 0.0 for probability in cfg.steering_mode_probabilities):
            raise ValueError("steering_mode_probabilities must be non-negative")
        if abs(sum(cfg.steering_mode_probabilities) - 1.0) > 1.0e-6:
            raise ValueError("steering_mode_probabilities must sum to one")
        if not 0.0 <= cfg.rel_high_speed_forward <= 1.0:
            raise ValueError("rel_high_speed_forward must be in [0, 1]")
        if cfg.rel_low_speed_forward + cfg.rel_high_speed_forward > 1.0:
            raise ValueError("low/high speed mixture probabilities must sum to at most one")
        high_low, high_high = cfg.high_speed_range
        if cfg.rel_high_speed_forward > 0.0 and not (
            0.0 <= high_low <= high_high <= cfg.ranges.lin_vel_x[1]
        ):
            raise ValueError("high_speed_range must be inside the forward command range")
        if any(mode not in (0, 1, 2, 3) for mode in cfg.high_speed_modes):
            raise ValueError("high_speed_modes must contain only steering mode indices 0..3")
        super().__init__(cfg, env)

    def _resample_command(self, env_ids: Sequence[int]):
        # Bypass SpeedBandOmniVelocityCommand's pure-axis mixture. The base
        # class still owns standing/recovery state and sampled-command storage.
        RecoveryVelocityCommand._resample_command(self, env_ids)
        if len(env_ids) == 0:
            return

        env_ids_tensor = torch.as_tensor(env_ids, dtype=torch.long, device=self.device)
        active_ids = env_ids_tensor[~self.is_standing_env[env_ids_tensor]]
        if len(active_ids) == 0:
            return

        probabilities = torch.tensor(
            self.cfg.steering_mode_probabilities,
            dtype=self.vel_command_b.dtype,
            device=self.device,
        )
        mode = torch.multinomial(probabilities, len(active_ids), replacement=True)
        x_low, x_high = self.cfg.ranges.lin_vel_x
        y_low, y_high = self.cfg.ranges.lin_vel_y
        yaw_low, yaw_high = self.cfg.ranges.ang_vel_z

        x = torch.empty(len(active_ids), device=self.device).uniform_(x_low, x_high)
        draw = torch.rand(len(active_ids), device=self.device)
        high_eligible = torch.zeros(len(active_ids), dtype=torch.bool, device=self.device)
        for high_mode in self.cfg.high_speed_modes:
            high_eligible |= mode == high_mode
        use_high = (draw < self.cfg.rel_high_speed_forward) & high_eligible
        use_low = (~high_eligible) | ((draw >= self.cfg.rel_high_speed_forward) & (
            draw < self.cfg.rel_high_speed_forward + self.cfg.rel_low_speed_forward
        ))
        if torch.any(use_high):
            high, high_limit = self.cfg.high_speed_range
            x[use_high] = torch.empty(
                int(torch.sum(use_high).item()), device=self.device
            ).uniform_(high, high_limit)
        if torch.any(use_low):
            low, high = self.cfg.low_speed_range
            x[use_low] = torch.empty(int(torch.sum(use_low).item()), device=self.device).uniform_(low, high)

        def signed_magnitude(upper: float, minimum_fraction: float) -> torch.Tensor:
            upper = abs(upper)
            lower = minimum_fraction * upper
            magnitude = torch.empty(len(active_ids), device=self.device).uniform_(lower, upper)
            sign = torch.where(torch.rand(len(active_ids), device=self.device) < 0.5, -1.0, 1.0)
            return magnitude * sign

        y = signed_magnitude(max(abs(y_low), abs(y_high)), self.cfg.lateral_min_fraction)
        yaw = signed_magnitude(max(abs(yaw_low), abs(yaw_high)), self.cfg.yaw_min_fraction)
        self.vel_command_b[active_ids] = 0.0
        self.vel_command_b[active_ids, 0] = x
        self.vel_command_b[active_ids[mode == 1], 1] = y[mode == 1]
        self.vel_command_b[active_ids[mode == 2], 2] = yaw[mode == 2]
        self.vel_command_b[active_ids[mode == 3], 1] = y[mode == 3]
        self.vel_command_b[active_ids[mode == 3], 2] = yaw[mode == 3]
        self._sampled_vel_command_b[env_ids_tensor] = self.vel_command_b[env_ids_tensor]


@configclass
class MovingSteeringVelocityCommandCfg(SpeedBandOmniVelocityCommandCfg):
    """Configuration for motion-conditioned, axis-separated steering commands."""

    class_type: type = MovingSteeringVelocityCommand

    steering_mode_probabilities: tuple[float, float, float, float] = (0.35, 0.30, 0.25, 0.10)
    """Probabilities for forward, forward+lateral, forward+yaw and combined commands."""

    rel_high_speed_forward: float = 0.0
    """Fraction of forward-bearing samples drawn from the high-speed band."""

    high_speed_range: tuple[float, float] = (0.90, 1.50)
    """High-speed forward interval used by the second-stage steering task."""

    high_speed_modes: tuple[int, ...] = (0, 1, 2, 3)
    """Steering modes eligible for high-speed sampling."""


class UsageWeightedSteeringVelocityCommand(MovingSteeringVelocityCommand):
    """Usage-weighted steering with complete-window tracking and diagnostics."""

    cfg: "UsageWeightedSteeringVelocityCommandCfg"

    def __init__(self, cfg: "UsageWeightedSteeringVelocityCommandCfg", env: "ManagerBasedEnv"):
        super().__init__(cfg, env)
        self.window_statistics = UsageWindowStatistics(
            num_envs=self.num_envs,
            device=self.device,
            speed_bin_edges=cfg.speed_bin_edges,
            success_thresholds=cfg.axis_success_thresholds,
        )

        hip_ids = []
        for name in cfg.hip_joint_names:
            ids, _ = self.robot.find_joints(name)
            if len(ids) != 1:
                raise ValueError(f"Expected exactly one hip joint named {name}, got {ids}")
            hip_ids.append(ids[0])
        self._hip_joint_ids = torch.tensor(hip_ids, dtype=torch.long, device=self.device)
        self._hip_outward_sign = torch.tensor((-1.0, 1.0, -1.0, 1.0), device=self.device)

        foot_ids, _ = self.robot.find_bodies(list(cfg.foot_body_names), preserve_order=True)
        if len(foot_ids) != len(cfg.foot_body_names):
            raise ValueError(f"Expected feet {cfg.foot_body_names}, got body ids {foot_ids}")
        self._foot_body_ids = torch.tensor(foot_ids, dtype=torch.long, device=self.device)
        self._contact_sensor = env.scene.sensors[cfg.contact_sensor_name]
        contact_ids, _ = self._contact_sensor.find_bodies(list(cfg.foot_body_names), preserve_order=True)
        if len(contact_ids) != len(cfg.foot_body_names):
            raise ValueError(f"Contact sensor does not expose all feet: {cfg.foot_body_names}")
        self._contact_body_ids = contact_ids

        self._episode_axis_error_sum = torch.zeros(self.num_envs, 3, device=self.device)
        self._episode_axis_steps = torch.zeros(self.num_envs, 3, device=self.device)
        self._episode_mode_samples = torch.zeros(self.num_envs, 4, device=self.device)
        self._episode_direction_samples = torch.zeros(self.num_envs, 3, device=self.device)
        self._episode_direction_vx_error_sum = torch.zeros(self.num_envs, 2, device=self.device)
        self._episode_direction_vx_steps = torch.zeros(self.num_envs, 2, device=self.device)
        self._episode_inactive_abs_sum = torch.zeros(self.num_envs, 2, device=self.device)
        self._episode_inactive_steps = torch.zeros(self.num_envs, 2, device=self.device)
        self._episode_min_height = torch.full((self.num_envs,), torch.inf, device=self.device)
        self._episode_max_tilt = torch.zeros(self.num_envs, device=self.device)
        self._episode_hip_sum = torch.zeros(self.num_envs, device=self.device)
        self._episode_hip_count = torch.zeros(self.num_envs, device=self.device)
        self._episode_hip_max = torch.full((self.num_envs,), -torch.inf, device=self.device)
        self._episode_slip_sum = torch.zeros(self.num_envs, device=self.device)
        self._episode_slip_count = torch.zeros(self.num_envs, device=self.device)
        self._episode_impact_sum = torch.zeros(self.num_envs, device=self.device)
        self._episode_impact_count = torch.zeros(self.num_envs, device=self.device)
        self._episode_action_delta_sum = torch.zeros(self.num_envs, device=self.device)
        self._episode_action_delta2_sum = torch.zeros(self.num_envs, device=self.device)
        self._episode_action_steps = torch.zeros(self.num_envs, device=self.device)
        self._previous_action_delta: torch.Tensor | None = None
        self._action_initialized = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)

    def _as_env_ids(self, env_ids: Sequence[int] | slice | None) -> torch.Tensor:
        if env_ids is None or isinstance(env_ids, slice):
            return torch.arange(self.num_envs, device=self.device)
        return torch.as_tensor(env_ids, dtype=torch.long, device=self.device)

    def _resample(self, env_ids: Sequence[int]):
        ids = self._as_env_ids(env_ids)
        if len(ids) == 0:
            return

        complete = (self.time_left[ids] <= 0.0) & (self.window_statistics.steps[ids] > 0)
        self.window_statistics.complete(ids[complete])
        self.window_statistics.discard(ids[~complete])
        super()._resample(ids)

        sampled = self._sampled_vel_command_b[ids].clone()
        sampled[self.is_standing_env[ids]] = 0.0
        has_lateral = torch.abs(sampled[:, 1]) > 1.0e-6
        has_yaw = torch.abs(sampled[:, 2]) > 1.0e-6
        modes = torch.zeros(len(ids), dtype=torch.long, device=self.device)
        modes[has_lateral & ~has_yaw] = 1
        modes[~has_lateral & has_yaw] = 2
        modes[has_lateral & has_yaw] = 3
        modes[self.is_standing_env[ids]] = -1
        self.window_statistics.begin(ids, sampled, modes)
        for mode in range(4):
            self._episode_mode_samples[ids, mode] += modes == mode
        direction = torch.ones(len(ids), dtype=torch.long, device=self.device)
        direction[sampled[:, 0] < -1.0e-6] = 0
        direction[self.is_standing_env[ids]] = 2
        for index in range(3):
            self._episode_direction_samples[ids, index] += direction == index

    def _update_metrics(self):
        measured = torch.stack(
            (
                self.robot.data.root_lin_vel_b[:, 0],
                self.robot.data.root_lin_vel_b[:, 1],
                self.robot.data.root_ang_vel_b[:, 2],
            ),
            dim=1,
        )
        executed = self._env.episode_length_buf > 0
        self.window_statistics.update(measured, env_mask=executed)

        active_axis = (torch.abs(self.vel_command_b) > 1.0e-6) & executed.unsqueeze(1)
        error = torch.abs(measured - self.vel_command_b)
        self._episode_axis_error_sum += error * active_axis
        self._episode_axis_steps += active_axis
        for index, direction_mask in enumerate(
            (self.vel_command_b[:, 0] < -1.0e-6, self.vel_command_b[:, 0] > 1.0e-6)
        ):
            selected = direction_mask & executed
            self._episode_direction_vx_error_sum[selected, index] += error[selected, 0]
            self._episode_direction_vx_steps[selected, index] += 1
        inactive = torch.stack(
            (torch.abs(self.vel_command_b[:, 1]) < 0.025, torch.abs(self.vel_command_b[:, 2]) < 0.05),
            dim=1,
        ) & executed.unsqueeze(1)
        inactive_error = torch.abs(measured[:, 1:3])
        self._episode_inactive_abs_sum += inactive_error * inactive
        self._episode_inactive_steps += inactive

        height = self.robot.data.root_pos_w[:, 2]
        self._episode_min_height[executed] = torch.minimum(
            self._episode_min_height[executed], height[executed]
        )
        gravity = self.robot.data.projected_gravity_b
        gravity_norm = torch.linalg.vector_norm(gravity, dim=1).clamp_min(1.0e-6)
        tilt = torch.acos(torch.clamp(-gravity[:, 2] / gravity_norm, min=-1.0, max=1.0))
        self._episode_max_tilt[executed] = torch.maximum(self._episode_max_tilt[executed], tilt[executed])

        hip_outward = self.robot.data.joint_pos[:, self._hip_joint_ids] * self._hip_outward_sign
        hip_mean = torch.mean(hip_outward, dim=1)
        hip_max = torch.max(hip_outward, dim=1).values
        self._episode_hip_sum[executed] += hip_mean[executed]
        self._episode_hip_count[executed] += 1
        self._episode_hip_max[executed] = torch.maximum(self._episode_hip_max[executed], hip_max[executed])

        in_contact = self._contact_sensor.data.current_contact_time[:, self._contact_body_ids] > 0.0
        foot_velocity = self.robot.data.body_lin_vel_w[:, self._foot_body_ids]
        horizontal_speed = torch.linalg.vector_norm(foot_velocity[:, :, :2], dim=2)
        contact_count = torch.sum(in_contact, dim=1)
        has_contact = executed & (contact_count > 0)
        slip = torch.sum(horizontal_speed * in_contact, dim=1) / contact_count.clamp_min(1)
        self._episode_slip_sum[has_contact] += slip[has_contact]
        self._episode_slip_count[has_contact] += 1

        first_contact = self._contact_sensor.compute_first_contact(self._env.step_dt)[:, self._contact_body_ids]
        impact_count = torch.sum(first_contact, dim=1)
        has_impact = executed & (impact_count > 0)
        impact = torch.sum(torch.linalg.vector_norm(foot_velocity, dim=2) * first_contact, dim=1)
        impact /= impact_count.clamp_min(1)
        self._episode_impact_sum[has_impact] += impact[has_impact]
        self._episode_impact_count[has_impact] += 1

        if hasattr(self._env, "action_manager"):
            current = self._env.action_manager.action
            previous = self._env.action_manager.prev_action
            delta = current - previous
            if self._previous_action_delta is None:
                self._previous_action_delta = torch.zeros_like(delta)
            initialized = executed & self._action_initialized
            self._episode_action_delta_sum[initialized] += torch.mean(torch.abs(delta[initialized]), dim=1)
            delta2 = delta - self._previous_action_delta
            self._episode_action_delta2_sum[initialized] += torch.mean(torch.abs(delta2[initialized]), dim=1)
            self._episode_action_steps[initialized] += 1
            self._previous_action_delta[:] = delta
            self._action_initialized[executed] = True

    @staticmethod
    def _ratio(numerator: torch.Tensor, denominator: torch.Tensor, ids: torch.Tensor) -> float:
        count = torch.sum(denominator[ids])
        if count <= 0:
            return 0.0
        return float((torch.sum(numerator[ids]) / count).item())

    @staticmethod
    def _finite_extreme(values: torch.Tensor, ids: torch.Tensor, maximum: bool) -> float:
        selected = values[ids]
        selected = selected[torch.isfinite(selected)]
        if len(selected) == 0:
            return 0.0
        result = torch.max(selected) if maximum else torch.min(selected)
        return float(result.item())

    def reset(self, env_ids: Sequence[int] | None = None) -> dict[str, float]:
        ids = self._as_env_ids(env_ids)
        xy_error_sum = torch.sum(self._episode_axis_error_sum[:, :2], dim=1)
        xy_steps = torch.sum(self._episode_axis_steps[:, :2], dim=1)
        extras = {
            "error_vel_xy": self._ratio(xy_error_sum, xy_steps, ids),
            "error_vel_yaw": self._ratio(
                self._episode_axis_error_sum[:, 2], self._episode_axis_steps[:, 2], ids
            ),
            "tracking_error/vx": self._ratio(
                self._episode_axis_error_sum[:, 0], self._episode_axis_steps[:, 0], ids
            ),
            "tracking_error/vy": self._ratio(
                self._episode_axis_error_sum[:, 1], self._episode_axis_steps[:, 1], ids
            ),
            "tracking_error/wz": self._ratio(
                self._episode_axis_error_sum[:, 2], self._episode_axis_steps[:, 2], ids
            ),
            "body/min_height": self._finite_extreme(self._episode_min_height, ids, maximum=False),
            "body/max_tilt": self._finite_extreme(self._episode_max_tilt, ids, maximum=True),
            "style/hip_outward_mean": self._ratio(
                self._episode_hip_sum, self._episode_hip_count, ids
            ),
            "style/hip_outward_max": self._finite_extreme(self._episode_hip_max, ids, maximum=True),
            "feet/slip": self._ratio(self._episode_slip_sum, self._episode_slip_count, ids),
            "feet/impact_velocity": self._ratio(
                self._episode_impact_sum, self._episode_impact_count, ids
            ),
            "action/delta": self._ratio(
                self._episode_action_delta_sum, self._episode_action_steps, ids
            ),
            "action/delta2": self._ratio(
                self._episode_action_delta2_sum, self._episode_action_steps, ids
            ),
        }
        for mode, name in enumerate(("vx", "vx_vy", "vx_wz", "combined")):
            extras[f"command_samples/{name}"] = float(torch.sum(self._episode_mode_samples[ids, mode]).item())
        for direction, name in enumerate(("backward", "forward", "standing")):
            extras[f"command_samples/{name}"] = float(
                torch.sum(self._episode_direction_samples[ids, direction]).item()
            )
        for direction, name in enumerate(("backward", "forward")):
            extras[f"tracking_error/vx_{name}"] = self._ratio(
                self._episode_direction_vx_error_sum[:, direction],
                self._episode_direction_vx_steps[:, direction],
                ids,
            )
        extras["drift/inactive_vy_abs"] = self._ratio(
            self._episode_inactive_abs_sum[:, 0], self._episode_inactive_steps[:, 0], ids
        )
        extras["drift/inactive_wz_abs"] = self._ratio(
            self._episode_inactive_abs_sum[:, 1], self._episode_inactive_steps[:, 1], ids
        )

        self.command_counter[ids] = 0
        self.window_statistics.discard(ids)
        self._episode_axis_error_sum[ids] = 0.0
        self._episode_axis_steps[ids] = 0.0
        self._episode_mode_samples[ids] = 0.0
        self._episode_direction_samples[ids] = 0.0
        self._episode_direction_vx_error_sum[ids] = 0.0
        self._episode_direction_vx_steps[ids] = 0.0
        self._episode_inactive_abs_sum[ids] = 0.0
        self._episode_inactive_steps[ids] = 0.0
        self._episode_min_height[ids] = torch.inf
        self._episode_max_tilt[ids] = 0.0
        self._episode_hip_sum[ids] = 0.0
        self._episode_hip_count[ids] = 0.0
        self._episode_hip_max[ids] = -torch.inf
        self._episode_slip_sum[ids] = 0.0
        self._episode_slip_count[ids] = 0.0
        self._episode_impact_sum[ids] = 0.0
        self._episode_impact_count[ids] = 0.0
        self._episode_action_delta_sum[ids] = 0.0
        self._episode_action_delta2_sum[ids] = 0.0
        self._episode_action_steps[ids] = 0.0
        self._action_initialized[ids] = False
        if self._previous_action_delta is not None:
            self._previous_action_delta[ids] = 0.0
        self._resample(ids)
        return extras


@configclass
class UsageWeightedSteeringVelocityCommandCfg(MovingSteeringVelocityCommandCfg):
    """Configuration for the first-stage real-usage command distribution."""

    class_type: type = UsageWeightedSteeringVelocityCommand

    speed_bin_edges: tuple[float, ...] = (0.35, 0.55)
    axis_success_thresholds: tuple[float, float, float] = (0.10, 0.07, 0.10)
    hip_joint_names: tuple[str, str, str, str] = (
        "FR_hip_joint",
        "FL_hip_joint",
        "RR_hip_joint",
        "RL_hip_joint",
    )
    foot_body_names: tuple[str, str, str, str] = (
        "FR_foot",
        "FL_foot",
        "RR_foot",
        "RL_foot",
    )
    contact_sensor_name: str = "contact_forces"


class BidirectionalUsageWeightedSteeringVelocityCommand(UsageWeightedSteeringVelocityCommand):
    """Add a bounded backward band while retaining usage-window diagnostics."""

    cfg: "BidirectionalUsageWeightedSteeringVelocityCommandCfg"

    def __init__(
        self,
        cfg: "BidirectionalUsageWeightedSteeringVelocityCommandCfg",
        env: "ManagerBasedEnv",
    ):
        if not 0.0 <= cfg.rel_backward_envs <= 1.0:
            raise ValueError("rel_backward_envs must be in [0, 1]")
        low, high = cfg.backward_speed_range
        if not 0.0 < low <= high:
            raise ValueError("backward_speed_range must contain positive increasing magnitudes")
        super().__init__(cfg, env)

    def _resample_command(self, env_ids: Sequence[int]):
        super()._resample_command(env_ids)
        if len(env_ids) == 0 or self.cfg.rel_backward_envs == 0.0:
            return

        ids = torch.as_tensor(env_ids, dtype=torch.long, device=self.device)
        moving = ~self.is_standing_env[ids]
        use_backward = torch.rand(len(ids), device=self.device) < self.cfg.rel_backward_envs
        backward_ids = ids[moving & use_backward]
        if len(backward_ids) == 0:
            return

        low, high = self.cfg.backward_speed_range
        magnitude = torch.empty(len(backward_ids), device=self.device).uniform_(low, high)
        self.vel_command_b[backward_ids, 0] = -magnitude
        self._sampled_vel_command_b[ids] = self.vel_command_b[ids]


@configclass
class BidirectionalUsageWeightedSteeringVelocityCommandCfg(UsageWeightedSteeringVelocityCommandCfg):
    """Usage-weighted commands with an explicit low-speed reverse fraction."""

    class_type: type = BidirectionalUsageWeightedSteeringVelocityCommand

    rel_backward_envs: float = 0.40
    backward_speed_range: tuple[float, float] = (0.15, 0.35)


class BidirectionalMovingSteeringVelocityCommand(MovingSteeringVelocityCommand):
    """Explicitly oversample a bounded backward band during gait discovery."""

    cfg: "BidirectionalMovingSteeringVelocityCommandCfg"

    def __init__(self, cfg: "BidirectionalMovingSteeringVelocityCommandCfg", env: "ManagerBasedEnv"):
        if not 0.0 <= cfg.rel_backward_envs <= 1.0:
            raise ValueError("rel_backward_envs must be in [0, 1]")
        low, high = cfg.backward_speed_range
        if not 0.0 < low <= high:
            raise ValueError("backward_speed_range must contain positive increasing magnitudes")
        super().__init__(cfg, env)

    def _resample_command(self, env_ids: Sequence[int]):
        super()._resample_command(env_ids)
        if len(env_ids) == 0 or self.cfg.rel_backward_envs == 0.0:
            return

        env_ids_tensor = torch.as_tensor(env_ids, dtype=torch.long, device=self.device)
        active = ~self.is_standing_env[env_ids_tensor]
        use_backward = torch.rand(len(env_ids_tensor), device=self.device) < self.cfg.rel_backward_envs
        backward_ids = env_ids_tensor[active & use_backward]
        if len(backward_ids) == 0:
            return
        low, high = self.cfg.backward_speed_range
        magnitude = torch.empty(len(backward_ids), device=self.device).uniform_(low, high)
        self.vel_command_b[backward_ids, 0] = -magnitude
        self._sampled_vel_command_b[env_ids_tensor] = self.vel_command_b[env_ids_tensor]


@configclass
class BidirectionalMovingSteeringVelocityCommandCfg(MovingSteeringVelocityCommandCfg):
    """Configuration for an explicit forward/backward steering mixture."""

    class_type: type = BidirectionalMovingSteeringVelocityCommand

    rel_backward_envs: float = 0.35
    """Fraction of moving environments assigned a backward command."""

    backward_speed_range: tuple[float, float] = (0.25, 0.80)
    """Backward speed magnitudes used during the first curriculum stage."""


class StratifiedOmniVelocityCommand(RecoveryVelocityCommand):
    """Sample signed three-axis commands from identifiable motion buckets.

    The sampler deliberately exposes pure forward, pure lateral, pure yaw and
    combined commands.  Unlike the earlier forward-only experiments, the x
    bucket is signed whenever the configured range is signed.  This keeps the
    policy contract at three command values while making every joystick axis
    visible during the first training stage.
    """

    cfg: "StratifiedOmniVelocityCommandCfg"

    def __init__(self, cfg: "StratifiedOmniVelocityCommandCfg", env: "ManagerBasedEnv"):
        if len(cfg.bucket_probabilities) != 4:
            raise ValueError("bucket_probabilities must contain forward/lateral/yaw/combined")
        if any(probability < 0.0 for probability in cfg.bucket_probabilities):
            raise ValueError("bucket_probabilities must be non-negative")
        if abs(sum(cfg.bucket_probabilities) - 1.0) > 1.0e-6:
            raise ValueError("bucket_probabilities must sum to one")
        if cfg.minimum_command_magnitude < 0.0:
            raise ValueError("minimum_command_magnitude must be non-negative")
        if (cfg.flat_terrain_type_count is None) != (cfg.rough_terrain_ranges is None):
            raise ValueError(
                "flat_terrain_type_count and rough_terrain_ranges must be configured together"
            )
        if cfg.flat_terrain_type_count is not None and cfg.flat_terrain_type_count <= 0:
            raise ValueError("flat_terrain_type_count must be positive")
        if not 0.0 <= cfg.rel_low_speed_x <= 1.0:
            raise ValueError("rel_low_speed_x must be in [0, 1]")
        if cfg.rel_low_speed_x > 0.0:
            low, high = cfg.low_speed_x_range
            if not 0.0 < low <= high:
                raise ValueError("low_speed_x_range must contain positive increasing magnitudes")
            maximum = max(abs(cfg.ranges.lin_vel_x[0]), abs(cfg.ranges.lin_vel_x[1]))
            if high > maximum:
                raise ValueError("low_speed_x_range must fit inside ranges.lin_vel_x")
        if not 0.0 <= cfg.rel_low_speed_yaw <= 1.0:
            raise ValueError("rel_low_speed_yaw must be in [0, 1]")
        if cfg.rel_low_speed_yaw > 0.0:
            low, high = cfg.low_speed_yaw_range
            if not 0.0 < low <= high:
                raise ValueError("low_speed_yaw_range must contain positive increasing magnitudes")
            maximum = max(abs(cfg.ranges.ang_vel_z[0]), abs(cfg.ranges.ang_vel_z[1]))
            if high > maximum:
                raise ValueError("low_speed_yaw_range must fit inside ranges.ang_vel_z")
        if not 0.0 <= cfg.rel_high_speed_yaw <= 1.0:
            raise ValueError("rel_high_speed_yaw must be in [0, 1]")
        if cfg.rel_low_speed_yaw + cfg.rel_high_speed_yaw > 1.0:
            raise ValueError("low/high yaw mixture probabilities must sum to at most one")
        if cfg.rel_high_speed_yaw > 0.0:
            low, high = cfg.high_speed_yaw_range
            if not 0.0 < low <= high:
                raise ValueError("high_speed_yaw_range must contain positive increasing magnitudes")
            maximum = max(abs(cfg.ranges.ang_vel_z[0]), abs(cfg.ranges.ang_vel_z[1]))
            if high > maximum:
                raise ValueError("high_speed_yaw_range must fit inside ranges.ang_vel_z")
        if not 0.0 <= cfg.rel_low_speed_y <= 1.0:
            raise ValueError("rel_low_speed_y must be in [0, 1]")
        if cfg.rel_low_speed_y > 0.0:
            low, high = cfg.low_speed_y_range
            if not 0.0 < low <= high:
                raise ValueError("low_speed_y_range must contain positive increasing magnitudes")
            maximum = max(abs(cfg.ranges.lin_vel_y[0]), abs(cfg.ranges.lin_vel_y[1]))
            if high > maximum:
                raise ValueError("low_speed_y_range must fit inside ranges.lin_vel_y")
        super().__init__(cfg, env)
        for axis in ("vx", "vy", "wz"):
            for direction in ("negative", "positive"):
                self.metrics[f"signed_error_{axis}_{direction}"] = torch.zeros(
                    self.num_envs, device=self.device
                )
                self.metrics[f"signed_fraction_{axis}_{direction}"] = torch.zeros(
                    self.num_envs, device=self.device
                )

    @staticmethod
    def _sample_axis(
        lower: float,
        upper: float,
        count: int,
        device: str,
        minimum_magnitude: float,
        negative_probability: float | None = None,
    ) -> torch.Tensor:
        """Draw values across a possibly signed interval without a zero dead-zone."""

        if count == 0:
            return torch.empty(0, device=device)
        if lower > upper:
            raise ValueError("command range lower bound must not exceed upper bound")
        negative_span = max(0.0, -lower)
        positive_span = max(0.0, upper)
        total_span = negative_span + positive_span
        if total_span <= 1.0e-8:
            return torch.zeros(count, device=device)

        if negative_probability is not None:
            if not 0.0 <= negative_probability <= 1.0:
                raise ValueError("negative_probability must be in [0, 1]")
            sign_probability = 1.0 - negative_probability
        else:
            sign_probability = positive_span / total_span
        signs = torch.where(
            torch.rand(count, device=device) < sign_probability,
            torch.ones(count, device=device),
            -torch.ones(count, device=device),
        )
        max_magnitude = torch.where(signs > 0.0, upper, -lower)
        min_magnitude = torch.minimum(
            max_magnitude,
            torch.full_like(max_magnitude, minimum_magnitude),
        )
        magnitude = min_magnitude + torch.rand(count, device=device) * (max_magnitude - min_magnitude)
        return signs * magnitude

    def _update_metrics(self):
        """Expose signed command errors so opposite directions cannot cancel in logs."""

        super()._update_metrics()
        measured = torch.stack(
            (
                self.robot.data.root_lin_vel_b[:, 0],
                self.robot.data.root_lin_vel_b[:, 1],
                self.robot.data.root_ang_vel_b[:, 2],
            ),
            dim=1,
        )
        error = torch.abs(measured - self.vel_command_b)
        max_command_steps = self.cfg.resampling_time_range[1] / self._env.step_dt
        for axis_index, axis in enumerate(("vx", "vy", "wz")):
            for direction, active in (
                ("negative", self.vel_command_b[:, axis_index] < -1.0e-6),
                ("positive", self.vel_command_b[:, axis_index] > 1.0e-6),
            ):
                self.metrics[f"signed_error_{axis}_{direction}"] += (
                    error[:, axis_index] * active / max_command_steps
                )
                self.metrics[f"signed_fraction_{axis}_{direction}"] += (
                    active / max_command_steps
                )

    def _resample_command(self, env_ids: Sequence[int]):
        RecoveryVelocityCommand._resample_command(self, env_ids)
        if len(env_ids) == 0:
            return
        env_ids_tensor = torch.as_tensor(env_ids, dtype=torch.long, device=self.device)
        active_ids = env_ids_tensor[~self.is_standing_env[env_ids_tensor]]
        if len(active_ids) == 0:
            self._sampled_vel_command_b[env_ids_tensor] = self.vel_command_b[env_ids_tensor]
            return

        probabilities = torch.tensor(
            self.cfg.bucket_probabilities,
            dtype=self.vel_command_b.dtype,
            device=self.device,
        )
        bucket = torch.multinomial(probabilities, len(active_ids), replacement=True)
        self.vel_command_b[active_ids] = 0.0
        x_low, x_high = self.cfg.ranges.lin_vel_x
        y_low, y_high = self.cfg.ranges.lin_vel_y
        yaw_low, yaw_high = self.cfg.ranges.ang_vel_z
        x = self._sample_axis(
            x_low,
            x_high,
            len(active_ids),
            self.device,
            self.cfg.minimum_command_magnitude,
            self.cfg.negative_x_probability,
        )
        y = self._sample_axis(y_low, y_high, len(active_ids), self.device, self.cfg.minimum_command_magnitude)
        yaw = self._sample_axis(yaw_low, yaw_high, len(active_ids), self.device, self.cfg.minimum_command_magnitude)

        forward = bucket == 0
        lateral = bucket == 1
        yaw_only = bucket == 2
        combined = bucket == 3
        x_bearing = forward | combined
        use_low_speed_x = x_bearing & (
            torch.rand(len(active_ids), device=self.device) < self.cfg.rel_low_speed_x
        )
        if torch.any(use_low_speed_x):
            low, high = self.cfg.low_speed_x_range
            magnitude = torch.empty(
                int(torch.sum(use_low_speed_x).item()), device=self.device
            ).uniform_(low, high)
            sign = torch.where(x[use_low_speed_x] < 0.0, -1.0, 1.0)
            x[use_low_speed_x] = sign * magnitude
        yaw_bearing = yaw_only | combined
        yaw_mixture_draw = torch.rand(len(active_ids), device=self.device)
        use_low_speed_yaw = yaw_bearing & (yaw_mixture_draw < self.cfg.rel_low_speed_yaw)
        use_high_speed_yaw = yaw_bearing & (
            yaw_mixture_draw >= self.cfg.rel_low_speed_yaw
        ) & (
            yaw_mixture_draw
            < self.cfg.rel_low_speed_yaw + self.cfg.rel_high_speed_yaw
        )
        if torch.any(use_low_speed_yaw):
            low, high = self.cfg.low_speed_yaw_range
            magnitude = torch.empty(
                int(torch.sum(use_low_speed_yaw).item()), device=self.device
            ).uniform_(low, high)
            sign = torch.where(yaw[use_low_speed_yaw] < 0.0, -1.0, 1.0)
            yaw[use_low_speed_yaw] = sign * magnitude
        if torch.any(use_high_speed_yaw):
            low, high = self.cfg.high_speed_yaw_range
            magnitude = torch.empty(
                int(torch.sum(use_high_speed_yaw).item()), device=self.device
            ).uniform_(low, high)
            sign = torch.where(yaw[use_high_speed_yaw] < 0.0, -1.0, 1.0)
            yaw[use_high_speed_yaw] = sign * magnitude
        y_bearing = lateral | combined
        use_low_speed_y = y_bearing & (
            torch.rand(len(active_ids), device=self.device) < self.cfg.rel_low_speed_y
        )
        if torch.any(use_low_speed_y):
            low, high = self.cfg.low_speed_y_range
            magnitude = torch.empty(
                int(torch.sum(use_low_speed_y).item()), device=self.device
            ).uniform_(low, high)
            sign = torch.where(y[use_low_speed_y] < 0.0, -1.0, 1.0)
            y[use_low_speed_y] = sign * magnitude
        self.vel_command_b[active_ids[forward], 0] = x[forward]
        self.vel_command_b[active_ids[lateral], 1] = y[lateral]
        self.vel_command_b[active_ids[yaw_only], 2] = yaw[yaw_only]
        self.vel_command_b[active_ids[combined], 0] = x[combined]
        if self.cfg.combined_include_lateral:
            self.vel_command_b[active_ids[combined], 1] = y[combined]
        self.vel_command_b[active_ids[combined], 2] = yaw[combined]
        self._sampled_vel_command_b[env_ids_tensor] = self.vel_command_b[env_ids_tensor]
        self._apply_terrain_command_ranges(env_ids_tensor)

    def _apply_terrain_command_ranges(self, env_ids: torch.Tensor) -> None:
        """Keep the full command envelope on flat columns and clamp rough terrain."""

        if self.cfg.flat_terrain_type_count is None:
            return
        terrain = self._env.scene.terrain
        if not hasattr(terrain, "terrain_types"):
            raise RuntimeError("terrain-aware commands require generated terrain types")
        rough = terrain.terrain_types[env_ids] >= self.cfg.flat_terrain_type_count
        rough_ids = env_ids[rough]
        if len(rough_ids) == 0:
            return
        ranges = self.cfg.rough_terrain_ranges
        assert ranges is not None
        for axis, limits in enumerate(
            (ranges.lin_vel_x, ranges.lin_vel_y, ranges.ang_vel_z)
        ):
            self.vel_command_b[rough_ids, axis].clamp_(float(limits[0]), float(limits[1]))
        self._sampled_vel_command_b[rough_ids] = self.vel_command_b[rough_ids]


@configclass
class StratifiedOmniVelocityCommandCfg(RecoveryVelocityCommandCfg):
    """Configuration for the signed, bucketed Omni-45 command curriculum."""

    class_type: type = StratifiedOmniVelocityCommand

    bucket_probabilities: tuple[float, float, float, float] = (0.35, 0.25, 0.20, 0.20)
    """Probabilities for forward, lateral, yaw and combined command buckets."""

    minimum_command_magnitude: float = 0.08
    """Avoid near-zero moving commands that are indistinguishable from standing."""

    rel_low_speed_x: float = 0.0
    """Fraction of x-bearing samples redrawn from :attr:`low_speed_x_range`."""

    low_speed_x_range: tuple[float, float] = (0.15, 0.40)
    """Signed x-command magnitudes oversampled for low-speed gait refinement."""

    rel_low_speed_yaw: float = 0.0
    """Fraction of yaw-bearing samples redrawn from :attr:`low_speed_yaw_range`."""

    low_speed_yaw_range: tuple[float, float] = (0.08, 0.40)
    """Signed yaw-command magnitudes oversampled for low-rate turning refinement."""

    rel_high_speed_yaw: float = 0.0
    """Fraction of yaw-bearing samples redrawn from :attr:`high_speed_yaw_range`."""

    high_speed_yaw_range: tuple[float, float] = (0.8, 1.0)
    """Signed yaw magnitudes oversampled near the current curriculum boundary."""

    rel_low_speed_y: float = 0.0
    """Fraction of lateral-bearing samples redrawn from :attr:`low_speed_y_range`."""

    low_speed_y_range: tuple[float, float] = (0.05, 0.20)
    """Signed lateral-command magnitudes oversampled for low-speed stepping."""

    negative_x_probability: float | None = None
    """Optional probability of a negative x command in the x-bearing bucket.

    ``None`` preserves range-proportional sampling.  A configured value is
    useful for a staged teacher experiment where reverse examples must be
    visible without changing lateral or yaw bucket semantics.
    """

    combined_include_lateral: bool = True
    """Whether the combined bucket samples all axes instead of the common vx+wz case."""

    flat_terrain_type_count: int | None = None
    """Number of leading generated-terrain columns that retain the full ranges."""

    rough_terrain_ranges: RecoveryVelocityCommandCfg.Ranges | None = None
    """Per-axis ranges applied to all remaining non-flat terrain columns."""

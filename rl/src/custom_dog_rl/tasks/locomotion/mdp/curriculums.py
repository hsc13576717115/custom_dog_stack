"""Command curricula used by the custom-dog velocity tasks."""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def forward_vel_cmd_levels(
    env: ManagerBasedRLEnv,
    env_ids: Sequence[int],
    reward_term_name: str = "track_lin_vel_xy",
    increment: float = 0.25,
    success_threshold: float = 0.75,
) -> torch.Tensor:
    """Expand only the forward-speed upper bound after tracking succeeds."""
    command_term = env.command_manager.get_term("base_velocity")
    ranges = command_term.cfg.ranges
    limit_ranges = command_term.cfg.limit_ranges
    reward_term = env.reward_manager.get_term_cfg(reward_term_name)
    mean_reward = (
        torch.mean(env.reward_manager._episode_sums[reward_term_name][env_ids])
        / env.max_episode_length_s
    )

    if env.common_step_counter % env.max_episode_length == 0:
        if mean_reward > reward_term.weight * success_threshold:
            lower = float(limit_ranges.lin_vel_x[0])
            upper = min(float(ranges.lin_vel_x[1]) + increment, float(limit_ranges.lin_vel_x[1]))
            ranges.lin_vel_x = [lower, upper]

    return torch.tensor(float(ranges.lin_vel_x[1]), device=env.device)


def stratified_omni_cmd_levels(
    env: ManagerBasedRLEnv,
    env_ids: Sequence[int],
    lin_reward_term: str = "track_lin_vel_xy",
    yaw_reward_term: str = "track_ang_vel_z",
    increments: tuple[float, float, float] = (0.25, 0.05, 0.10),
    lin_success_threshold: float = 0.70,
    yaw_success_threshold: float = 0.70,
    min_lin_command: float = 0.08,
    min_yaw_command: float = 0.08,
) -> dict[str, float]:
    """Expand signed vx/vy/wz buckets after both planar and yaw tracking succeed.

    The command generator starts with a learnable envelope and expands it
    symmetrically toward ``limit_ranges``.  Reward-manager episode sums are
    already weighted and time-integrated, so the normalization below recovers
    the mean unweighted tracking reward before comparing it with the threshold.
    """

    if len(increments) != 3 or any(value <= 0.0 for value in increments):
        raise ValueError("increments must contain three positive values")
    if not (0.0 < lin_success_threshold <= 1.0 and 0.0 < yaw_success_threshold <= 1.0):
        raise ValueError("success thresholds must be in (0, 1]")
    if min_lin_command <= 0.0 or min_yaw_command <= 0.0:
        raise ValueError("minimum active command thresholds must be positive")

    command_term = env.command_manager.get_term("base_velocity")
    ranges = command_term.cfg.ranges
    limits = command_term.cfg.limit_ranges
    lin_cfg = env.reward_manager.get_term_cfg(lin_reward_term)
    yaw_cfg = env.reward_manager.get_term_cfg(yaw_reward_term)
    lin_denominator = max(abs(float(lin_cfg.weight)) * env.max_episode_length_s, 1.0e-6)
    yaw_denominator = max(abs(float(yaw_cfg.weight)) * env.max_episode_length_s, 1.0e-6)

    state_name = "_custom_dog_omni_curriculum_state"
    state = getattr(env, state_name, None)
    if state is None:
        state = {
            "last_evaluation_step": int(env.common_step_counter),
            "lin_sum": torch.zeros((), device=env.device),
            "yaw_sum": torch.zeros((), device=env.device),
            "lin_count": torch.zeros((), device=env.device),
            "yaw_count": torch.zeros((), device=env.device),
            "last_lin_score": 0.0,
            "last_yaw_score": 0.0,
        }
        setattr(env, state_name, state)

    # CurriculumManager runs before RewardManager and CommandManager reset, so
    # these tensors still describe the completed episode and its old command.
    if env.common_step_counter > 0 and len(env_ids) > 0:
        completed_commands = command_term.command[env_ids]
        lin_active = torch.linalg.vector_norm(completed_commands[:, :2], dim=1) >= min_lin_command
        yaw_active = torch.abs(completed_commands[:, 2]) >= min_yaw_command
        lin_scores = env.reward_manager._episode_sums[lin_reward_term][env_ids] / lin_denominator
        yaw_scores = env.reward_manager._episode_sums[yaw_reward_term][env_ids] / yaw_denominator
        state["lin_sum"] += torch.sum(lin_scores[lin_active])
        state["yaw_sum"] += torch.sum(yaw_scores[yaw_active])
        state["lin_count"] += torch.sum(lin_active)
        state["yaw_count"] += torch.sum(yaw_active)

    evaluation_due = (
        int(env.common_step_counter) - state["last_evaluation_step"] >= env.max_episode_length
    )
    if evaluation_due:
        lin_count = float(state["lin_count"].item())
        yaw_count = float(state["yaw_count"].item())
        lin_score = float((state["lin_sum"] / max(lin_count, 1.0)).item())
        yaw_score = float((state["yaw_sum"] / max(yaw_count, 1.0)).item())
        state["last_lin_score"] = lin_score
        state["last_yaw_score"] = yaw_score
        state["last_evaluation_step"] = int(env.common_step_counter)
        state["lin_sum"].zero_()
        state["yaw_sum"].zero_()
        state["lin_count"].zero_()
        state["yaw_count"].zero_()

        if lin_count > 0.0 and yaw_count > 0.0 and lin_score >= lin_success_threshold and yaw_score >= yaw_success_threshold:
            ranges.lin_vel_x = (
                max(float(limits.lin_vel_x[0]), float(ranges.lin_vel_x[0]) - increments[0]),
                min(float(limits.lin_vel_x[1]), float(ranges.lin_vel_x[1]) + increments[0]),
            )
            ranges.lin_vel_y = (
                max(float(limits.lin_vel_y[0]), float(ranges.lin_vel_y[0]) - increments[1]),
                min(float(limits.lin_vel_y[1]), float(ranges.lin_vel_y[1]) + increments[1]),
            )
            ranges.ang_vel_z = (
                max(float(limits.ang_vel_z[0]), float(ranges.ang_vel_z[0]) - increments[2]),
                min(float(limits.ang_vel_z[1]), float(ranges.ang_vel_z[1]) + increments[2]),
            )

    return {
        "vx_max": float(ranges.lin_vel_x[1]),
        "vy_max": float(ranges.lin_vel_y[1]),
        "wz_max": float(ranges.ang_vel_z[1]),
        "lin_score": float(state["last_lin_score"]),
        "yaw_score": float(state["last_yaw_score"]),
    }


def usage_command_window_levels(
    env: ManagerBasedRLEnv,
    env_ids: Sequence[int],
    command_name: str = "base_velocity",
    min_windows: int = 50,
    success_rate_threshold: float = 0.70,
    required_consecutive_windows: int = 3,
    increments: tuple[float, float, float] = (0.15, 0.025, 0.075),
) -> dict[str, float]:
    """Expand each command axis from complete, correctly attributed windows."""
    if min_windows <= 0 or required_consecutive_windows <= 0:
        raise ValueError("window requirements must be positive")
    if not 0.0 < success_rate_threshold <= 1.0:
        raise ValueError("success_rate_threshold must be in (0, 1]")
    if len(increments) != 3 or any(value <= 0.0 for value in increments):
        raise ValueError("increments must contain three positive values")

    command_term = env.command_manager.get_term(command_name)
    statistics = getattr(command_term, "window_statistics", None)
    if statistics is None:
        raise TypeError(f"Command term {command_name} does not expose complete-window statistics")

    state_name = "_custom_dog_usage_curriculum_state"
    state = getattr(env, state_name, None)
    if state is None:
        state = {
            "streaks": [0, 0, 0],
            "mae": [0.0, 0.0, 0.0],
            "success_rate": [0.0, 0.0, 0.0],
            "counts": [0.0, 0.0, 0.0],
            "populated_buckets": [0.0, 0.0, 0.0],
            "min_bucket_count": [0.0, 0.0, 0.0],
            "mode_counts": [0.0, 0.0, 0.0, 0.0],
        }
        setattr(env, state_name, state)

    snapshot = statistics.snapshot()
    counts = snapshot["counts"]
    ranges = command_term.cfg.ranges
    limits = command_term.cfg.limit_ranges

    state["mode_counts"] = [float(torch.sum(counts[mode, :, 0]).item()) for mode in range(4)]
    for axis in range(3):
        summary = statistics.axis_summary(axis, min_windows)
        count = float(summary["count"])
        state["counts"][axis] = count
        state["populated_buckets"][axis] = float(summary["populated_buckets"])
        state["min_bucket_count"][axis] = float(summary["min_bucket_count"])
        if not summary["ready"]:
            continue

        mae = float(summary["error_sum"]) / count
        success_rate = float(summary["successes"]) / count
        state["mae"][axis] = mae
        state["success_rate"][axis] = success_rate
        if mae <= float(statistics.success_thresholds[axis].item()) and success_rate >= success_rate_threshold:
            state["streaks"][axis] += 1
        else:
            state["streaks"][axis] = 0
        statistics.clear_axis(axis)

        if state["streaks"][axis] < required_consecutive_windows:
            continue
        state["streaks"][axis] = 0
        if axis == 0:
            ranges.lin_vel_x = (
                float(limits.lin_vel_x[0]),
                min(float(ranges.lin_vel_x[1]) + increments[0], float(limits.lin_vel_x[1])),
            )
        elif axis == 1:
            current = max(abs(float(ranges.lin_vel_y[0])), abs(float(ranges.lin_vel_y[1])))
            maximum = max(abs(float(limits.lin_vel_y[0])), abs(float(limits.lin_vel_y[1])))
            expanded = min(current + increments[1], maximum)
            ranges.lin_vel_y = (-expanded, expanded)
        else:
            current = max(abs(float(ranges.ang_vel_z[0])), abs(float(ranges.ang_vel_z[1])))
            maximum = max(abs(float(limits.ang_vel_z[0])), abs(float(limits.ang_vel_z[1])))
            expanded = min(current + increments[2], maximum)
            ranges.ang_vel_z = (-expanded, expanded)

    return {
        "range/vx_max": float(ranges.lin_vel_x[1]),
        "range/vy_max": float(ranges.lin_vel_y[1]),
        "range/wz_max": float(ranges.ang_vel_z[1]),
        "tracking_error/vx": state["mae"][0],
        "tracking_error/vy": state["mae"][1],
        "tracking_error/wz": state["mae"][2],
        "success_rate/vx": state["success_rate"][0],
        "success_rate/vy": state["success_rate"][1],
        "success_rate/wz": state["success_rate"][2],
        "pass_streak/vx": float(state["streaks"][0]),
        "pass_streak/vy": float(state["streaks"][1]),
        "pass_streak/wz": float(state["streaks"][2]),
        "populated_buckets/vx": state["populated_buckets"][0],
        "populated_buckets/vy": state["populated_buckets"][1],
        "populated_buckets/wz": state["populated_buckets"][2],
        "min_bucket_count/vx": state["min_bucket_count"][0],
        "min_bucket_count/vy": state["min_bucket_count"][1],
        "min_bucket_count/wz": state["min_bucket_count"][2],
        "command_samples/vx": state["mode_counts"][0],
        "command_samples/vx_vy": state["mode_counts"][1],
        "command_samples/vx_wz": state["mode_counts"][2],
        "command_samples/combined": state["mode_counts"][3],
    }

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

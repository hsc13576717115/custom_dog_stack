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

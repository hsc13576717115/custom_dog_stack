"""Per-command-window statistics for usage-weighted velocity training."""

from __future__ import annotations

from collections.abc import Sequence

import torch


class UsageWindowStatistics:
    """Accumulate tracking errors without crossing command resample boundaries."""

    def __init__(
        self,
        num_envs: int,
        device: str,
        speed_bin_edges: Sequence[float],
        success_thresholds: Sequence[float],
        num_modes: int = 4,
    ) -> None:
        if num_envs <= 0 or num_modes <= 0:
            raise ValueError("num_envs and num_modes must be positive")
        if len(success_thresholds) != 3 or any(value <= 0.0 for value in success_thresholds):
            raise ValueError("success_thresholds must contain three positive values")
        if any(left >= right for left, right in zip(speed_bin_edges, speed_bin_edges[1:])):
            raise ValueError("speed_bin_edges must be strictly increasing")

        self.num_envs = num_envs
        self.num_modes = num_modes
        self.num_bins = len(speed_bin_edges) + 1
        self.device = device
        self.speed_bin_edges = torch.tensor(speed_bin_edges, dtype=torch.float, device=device)
        self.success_thresholds = torch.tensor(success_thresholds, dtype=torch.float, device=device)

        self.commands = torch.zeros(num_envs, 3, device=device)
        self.modes = torch.full((num_envs,), -1, dtype=torch.long, device=device)
        self.error_sums = torch.zeros(num_envs, 3, device=device)
        self.steps = torch.zeros(num_envs, dtype=torch.long, device=device)

        shape = (num_modes, self.num_bins, 3)
        self.completed_error_sums = torch.zeros(shape, device=device)
        self.completed_counts = torch.zeros(shape, device=device)
        self.completed_successes = torch.zeros(shape, device=device)

    def begin(self, env_ids: torch.Tensor, commands: torch.Tensor, modes: torch.Tensor) -> None:
        """Start a new window after the previous one was completed or discarded."""
        if commands.shape != (len(env_ids), 3) or modes.shape != (len(env_ids),):
            raise ValueError("commands or modes have an invalid shape")
        self.discard(env_ids)
        self.commands[env_ids] = commands
        self.modes[env_ids] = modes

    def update(self, measured_velocity: torch.Tensor, env_mask: torch.Tensor | None = None) -> None:
        """Add one control step to every active, non-standing command window."""
        if measured_velocity.shape != self.commands.shape:
            raise ValueError("measured_velocity must match the command tensor shape")
        active = self.modes >= 0
        if env_mask is not None:
            if env_mask.shape != active.shape:
                raise ValueError("env_mask must have one entry per environment")
            active &= env_mask
        self.error_sums[active] += torch.abs(measured_velocity[active] - self.commands[active])
        self.steps[active] += 1

    def complete(self, env_ids: torch.Tensor) -> None:
        """Commit complete windows into mode, speed-bin and axis statistics."""
        if len(env_ids) == 0:
            return
        valid = (self.modes[env_ids] >= 0) & (self.steps[env_ids] > 0)
        valid_ids = env_ids[valid]
        if len(valid_ids) == 0:
            self.discard(env_ids)
            return

        modes = self.modes[valid_ids]
        commands = self.commands[valid_ids]
        mean_error = self.error_sums[valid_ids] / self.steps[valid_ids].unsqueeze(1)
        speed_bins = torch.bucketize(torch.abs(commands[:, 0]), self.speed_bin_edges)
        axis_active = torch.abs(commands) > 1.0e-6

        for mode in range(self.num_modes):
            for speed_bin in range(self.num_bins):
                group = (modes == mode) & (speed_bins == speed_bin)
                if not torch.any(group):
                    continue
                for axis in range(3):
                    selected = group & axis_active[:, axis]
                    if not torch.any(selected):
                        continue
                    errors = mean_error[selected, axis]
                    self.completed_error_sums[mode, speed_bin, axis] += torch.sum(errors)
                    self.completed_counts[mode, speed_bin, axis] += errors.numel()
                    self.completed_successes[mode, speed_bin, axis] += torch.sum(
                        errors <= self.success_thresholds[axis]
                    )

        self.discard(env_ids)

    def discard(self, env_ids: torch.Tensor) -> None:
        """Clear active windows without adding incomplete data to curriculum statistics."""
        self.commands[env_ids] = 0.0
        self.modes[env_ids] = -1
        self.error_sums[env_ids] = 0.0
        self.steps[env_ids] = 0

    def snapshot(self) -> dict[str, torch.Tensor]:
        """Return immutable copies of the accumulated complete-window statistics."""
        return {
            "error_sums": self.completed_error_sums.clone(),
            "counts": self.completed_counts.clone(),
            "successes": self.completed_successes.clone(),
        }

    def axis_summary(self, axis: int, min_windows_per_bucket: int) -> dict[str, float | bool]:
        """Summarize an axis only after every populated mode/speed bucket is ready."""
        if axis not in (0, 1, 2):
            raise ValueError("axis must be 0, 1 or 2")
        if min_windows_per_bucket <= 0:
            raise ValueError("min_windows_per_bucket must be positive")

        counts = self.completed_counts[:, :, axis]
        populated = counts > 0
        populated_count = int(torch.count_nonzero(populated).item())
        if populated_count == 0:
            return {
                "ready": False,
                "count": 0.0,
                "error_sum": 0.0,
                "successes": 0.0,
                "populated_buckets": 0.0,
                "min_bucket_count": 0.0,
            }

        populated_counts = counts[populated]
        return {
            "ready": bool(torch.all(populated_counts >= min_windows_per_bucket).item()),
            "count": float(torch.sum(populated_counts).item()),
            "error_sum": float(torch.sum(self.completed_error_sums[:, :, axis]).item()),
            "successes": float(torch.sum(self.completed_successes[:, :, axis]).item()),
            "populated_buckets": float(populated_count),
            "min_bucket_count": float(torch.min(populated_counts).item()),
        }

    def clear_axis(self, axis: int) -> None:
        """Clear one evaluated axis while retaining samples for the other axes."""
        if axis not in (0, 1, 2):
            raise ValueError("axis must be 0, 1 or 2")
        self.completed_error_sums[:, :, axis] = 0.0
        self.completed_counts[:, :, axis] = 0.0
        self.completed_successes[:, :, axis] = 0.0

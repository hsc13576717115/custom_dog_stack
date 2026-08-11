from __future__ import annotations

import importlib.util
from pathlib import Path

import torch


MODULE_PATH = (
    Path(__file__).resolve().parents[1]
    / "rl/src/custom_dog_rl/tasks/locomotion/mdp/usage_window.py"
)
SPEC = importlib.util.spec_from_file_location("custom_dog_usage_window", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)
UsageWindowStatistics = MODULE.UsageWindowStatistics


def test_completed_windows_are_attributed_before_resampling() -> None:
    statistics = UsageWindowStatistics(
        num_envs=2,
        device="cpu",
        speed_bin_edges=(0.35, 0.55),
        success_thresholds=(0.10, 0.07, 0.10),
    )

    env = torch.tensor([0])
    statistics.begin(env, torch.tensor([[0.30, 0.00, 0.20]]), torch.tensor([2]))
    statistics.update(torch.tensor([[0.20, 0.00, 0.10], [0.00, 0.00, 0.00]]))
    statistics.update(torch.tensor([[0.20, 0.00, 0.10], [0.00, 0.00, 0.00]]))
    statistics.complete(env)

    statistics.begin(env, torch.tensor([[0.50, 0.10, 0.00]]), torch.tensor([1]))
    statistics.update(torch.tensor([[0.50, 0.00, 0.00], [0.00, 0.00, 0.00]]))
    statistics.complete(env)

    snapshot = statistics.snapshot()
    counts = snapshot["counts"]
    errors = snapshot["error_sums"]
    assert counts[2, 0, 0] == 1
    assert counts[2, 0, 2] == 1
    assert torch.isclose(errors[2, 0, 0], torch.tensor(0.10))
    assert torch.isclose(errors[2, 0, 2], torch.tensor(0.10))
    assert counts[1, 1, 0] == 1
    assert counts[1, 1, 1] == 1
    assert torch.isclose(errors[1, 1, 0], torch.tensor(0.00))
    assert torch.isclose(errors[1, 1, 1], torch.tensor(0.10))
    assert torch.sum(counts) == 4


def test_incomplete_or_reset_windows_are_not_committed() -> None:
    statistics = UsageWindowStatistics(
        num_envs=1,
        device="cpu",
        speed_bin_edges=(0.35,),
        success_thresholds=(0.10, 0.07, 0.10),
    )
    env = torch.tensor([0])
    statistics.begin(env, torch.tensor([[0.30, 0.00, 0.20]]), torch.tensor([2]))
    statistics.update(torch.tensor([[0.00, 0.00, 0.00]]))
    statistics.discard(env)

    snapshot = statistics.snapshot()
    assert torch.sum(snapshot["counts"]) == 0
    assert torch.sum(snapshot["error_sums"]) == 0


def test_axis_evaluation_can_clear_without_erasing_other_axes() -> None:
    statistics = UsageWindowStatistics(
        num_envs=1,
        device="cpu",
        speed_bin_edges=(),
        success_thresholds=(0.10, 0.07, 0.10),
    )
    env = torch.tensor([0])
    statistics.begin(env, torch.tensor([[0.40, 0.00, 0.20]]), torch.tensor([2]))
    statistics.update(torch.tensor([[0.35, 0.00, 0.15]]))
    statistics.complete(env)
    statistics.clear_axis(0)

    snapshot = statistics.snapshot()
    assert torch.sum(snapshot["counts"][:, :, 0]) == 0
    assert torch.sum(snapshot["counts"][:, :, 2]) == 1


def test_axis_summary_requires_every_populated_bucket_to_reach_minimum() -> None:
    statistics = UsageWindowStatistics(
        num_envs=1,
        device="cpu",
        speed_bin_edges=(0.35,),
        success_thresholds=(0.10, 0.07, 0.10),
    )
    statistics.completed_counts[0, 0, 0] = 50
    statistics.completed_counts[2, 1, 0] = 49
    statistics.completed_error_sums[0, 0, 0] = 2.5
    statistics.completed_error_sums[2, 1, 0] = 2.45
    statistics.completed_successes[0, 0, 0] = 45
    statistics.completed_successes[2, 1, 0] = 44

    summary = statistics.axis_summary(axis=0, min_windows_per_bucket=50)
    assert not summary["ready"]
    assert summary["count"] == 99
    assert summary["populated_buckets"] == 2
    assert summary["min_bucket_count"] == 49

    statistics.completed_counts[2, 1, 0] = 50
    summary = statistics.axis_summary(axis=0, min_windows_per_bucket=50)
    assert summary["ready"]
    assert summary["count"] == 100
    assert abs(summary["error_sum"] - 4.95) < 1.0e-5
    assert summary["successes"] == 89

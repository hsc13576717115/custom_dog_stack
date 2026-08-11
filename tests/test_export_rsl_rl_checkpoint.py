from pathlib import Path
import sys

import pytest
import torch


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from export_rsl_rl_checkpoint import build_actor  # noqa: E402


def network_state(prefix: str, input_dim: int) -> dict[str, torch.Tensor]:
    return {
        f"{prefix}.0.weight": torch.randn(4, input_dim),
        f"{prefix}.0.bias": torch.randn(4),
        f"{prefix}.2.weight": torch.randn(2, 4),
        f"{prefix}.2.bias": torch.randn(2),
    }


@pytest.mark.parametrize(("prefix", "input_dim"), (("actor", 45), ("student", 213)))
def test_build_actor_supports_ppo_and_distillation(prefix: str, input_dim: int) -> None:
    actor = build_actor(network_state(prefix, input_dim), "elu")

    assert actor(torch.zeros(3, input_dim)).shape == (3, 2)


def test_build_actor_rejects_unknown_checkpoint() -> None:
    with pytest.raises(ValueError, match="neither actor nor student"):
        build_actor({}, "elu")

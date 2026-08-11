import importlib.util
from pathlib import Path

import pytest
import torch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "expand_actor_observation",
    PROJECT_ROOT / "scripts" / "expand_rsl_rl_actor_observation.py",
)
assert SPEC is not None and SPEC.loader is not None
EXPANDER = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(EXPANDER)


def test_arbitrary_mapping_preserves_linear_output():
    weight = torch.arange(12, dtype=torch.float32).reshape(3, 4)
    indices = [7, 2, 9, 4]
    expanded, actual_indices = EXPANDER.expand_actor_weight(weight, 10, indices)
    source = torch.tensor([1.0, -2.0, 3.0, -4.0])
    target = torch.randn(10)
    target[indices] = source

    torch.testing.assert_close(expanded @ target, weight @ source)
    assert actual_indices == indices
    untouched = [index for index in range(10) if index not in indices]
    assert torch.count_nonzero(expanded[:, untouched]) == 0


@pytest.mark.parametrize(
    "indices, message",
    [([0, 1], "Expected 3"), ([0, 0, 1], "unique"), ([0, 1, 4], "inside"), ([-1, 1, 2], "inside")],
)
def test_invalid_mapping_is_rejected(indices, message):
    with pytest.raises(ValueError, match=message):
        EXPANDER.expand_actor_weight(torch.ones((2, 3)), 4, indices)

import importlib.util
from pathlib import Path

import numpy as np
import pytest


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "analyze_policy_trace", PROJECT_ROOT / "scripts" / "analyze_policy_trace.py"
)
assert SPEC is not None and SPEC.loader is not None
ANALYZER = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(ANALYZER)


def test_stack_history_uses_startup_fill_and_oldest_to_newest_order():
    values = np.asarray([[1.0, 10.0], [2.0, 20.0], [3.0, 30.0]])

    actual = ANALYZER.stack_history(values, 3)

    np.testing.assert_array_equal(
        actual,
        np.asarray(
            [
                [1.0, 10.0, 1.0, 10.0, 1.0, 10.0],
                [1.0, 10.0, 1.0, 10.0, 2.0, 20.0],
                [1.0, 10.0, 2.0, 20.0, 3.0, 30.0],
            ]
        ),
    )


def test_stack_history_preserves_single_frame_contract():
    values = np.arange(12, dtype=np.float64).reshape(3, 4)
    np.testing.assert_array_equal(ANALYZER.stack_history(values, 1), values)


@pytest.mark.parametrize("history_length", [0, -1])
def test_stack_history_rejects_non_positive_length(history_length):
    with pytest.raises(ValueError, match="history_length must be positive"):
        ANALYZER.stack_history(np.ones((2, 3)), history_length)

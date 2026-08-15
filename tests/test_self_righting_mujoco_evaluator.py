from __future__ import annotations

import importlib.util
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = ROOT / "scripts" / "evaluate_self_righting_mujoco.py"
SPEC = importlib.util.spec_from_file_location("evaluate_self_righting_mujoco", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def test_stage_pose_contract_expands_after_r0() -> None:
    assert MODULE.STAGE_STATES["R0"] == ("recovery-belly",)
    assert set(MODULE.STAGE_STATES["R1"]) == {
        "recovery-belly",
        "recovery-back",
        "recovery-left",
        "recovery-right",
    }
    assert MODULE.STAGE_STATES["R2"] == MODULE.STAGE_STATES["R1"]
    assert len(MODULE.R2_PERTURBATIONS) == 4
    assert {case["initial_state"] for case in MODULE.R2_PERTURBATIONS} == set(
        MODULE.STAGE_STATES["R2"]
    )
    assert all(case["joint_noise"] > 0.0 for case in MODULE.R2_PERTURBATIONS)


def test_success_parser_requires_stable_gate_message() -> None:
    success = MODULE.SUCCESS_PATTERN.search("self_righting: success at 2.375 s")
    assert success is not None
    assert float(success.group(1)) == 2.375
    assert MODULE.SUCCESS_PATTERN.search("recovery: first upright state at 1.000 s") is None

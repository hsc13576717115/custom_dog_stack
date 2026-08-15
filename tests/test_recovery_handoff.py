from __future__ import annotations

import importlib.util
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PATH = ROOT / "scripts/evaluate_recovery_handoff.py"
SPEC = importlib.util.spec_from_file_location("evaluate_recovery_handoff", PATH)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def test_handoff_command_contains_both_policy_contracts(tmp_path: Path) -> None:
    recovery = tmp_path / "recovery"
    locomotion = tmp_path / "locomotion"
    stand = tmp_path / "stand"
    command = MODULE.build_command(
        python="python",
        runner=ROOT / "sim2sim/custom_dog/run_sim2sim.py",
        mjcf=ROOT / "sim2sim/custom_dog/custom_dog.xml",
        recovery_candidate=recovery,
        locomotion_candidate=locomotion,
        stand_candidate=stand,
        initial_state="recovery-back",
        duration=12.0,
    )
    assert "--recovery-policy" in command
    assert "--policy" in command
    assert "--stand-policy" in command
    assert command[command.index("--initial-state") + 1] == "recovery-back"
    assert command[command.index("--recovery-locomotion-hold") + 1] == "1.0"


def test_handoff_requires_both_success_markers() -> None:
    result = MODULE.evaluate_state(
        [sys.executable, "-c", "print('self_righting: success at 2.0 s')"],
        ROOT,
        "recovery-belly",
    )
    assert not result["passed"]

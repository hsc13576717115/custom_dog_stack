from __future__ import annotations

import ast
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
ENV = ROOT / "rl/src/custom_dog_rl/tasks/locomotion/velocity_env_cfg.py"
PPO = ROOT / "rl/src/custom_dog_rl/agents/ppo_cfg.py"
REGISTRY = ROOT / "rl/src/custom_dog_rl/tasks/locomotion/__init__.py"


def _class_source(path: Path, name: str) -> str:
    module = ast.parse(path.read_text())
    for node in module.body:
        if isinstance(node, ast.ClassDef) and node.name == name:
            return ast.unparse(node)
    raise AssertionError(f"Missing class {name}")


def test_stand_expert_keeps_51d_contract_and_exact_zero_commands() -> None:
    source = _class_source(ENV, "RobotStandExpertEnvCfg")
    assert "RobotOmniTrotClosedLoopRobustFoundationEnvCfg" in source
    assert "command.rel_standing_envs = 1.0" in source
    assert "self.rewards.trot_contact_schedule = None" in source
    assert "self.rewards.feet_air_time_command_aware = None" in source
    assert "standing_joint_deviation_normalized_l2" in source
    assert "standing_orientation_normalized_l2" in source
    assert "recovery_stable_support" in source


def test_stand_expert_is_registered_with_from_scratch_runner() -> None:
    registry = REGISTRY.read_text()
    runner = _class_source(PPO, "CustomDogStandExpertPPORunnerCfg")
    assert 'id="CustomDog-Stand-ClosedLoop-v1"' in registry
    assert "RobotStandExpertEnvCfg" in registry
    assert "CustomDogStandExpertPPORunnerCfg" in registry
    assert "learning_rate = 0.0005" in runner
    assert "init_noise_std = 0.35" in runner


def test_stand_scripts_train_from_scratch_and_use_zero_command_gate() -> None:
    train = (ROOT / "scripts/train_stand_expert.sh").read_text()
    evaluate = (ROOT / "scripts/evaluate_stand_expert.sh").read_text()
    latest = (ROOT / "scripts/evaluate_latest_stand_expert.sh").read_text()
    assert "--resume" not in train
    assert "CUSTOM_DOG_MAX_ITERATIONS:-300" in train
    assert "--command 0 0 0" in evaluate
    assert "--absolute-only" in evaluate
    assert "stand_selection.json" in evaluate
    assert "*_stand_expert_seed42" in latest

from __future__ import annotations

import ast
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
ENV = ROOT / "rl/src/custom_dog_rl/tasks/locomotion/velocity_env_cfg.py"
REWARDS = ROOT / "rl/src/custom_dog_rl/tasks/locomotion/mdp/rewards.py"
REGISTRY = ROOT / "rl/src/custom_dog_rl/tasks/locomotion/__init__.py"


def _source(path: Path, name: str, kind: type[ast.AST]) -> str:
    module = ast.parse(path.read_text(encoding="utf-8"))
    node = next(node for node in module.body if isinstance(node, kind) and node.name == name)
    return ast.unparse(node)


def test_gait_stage_keeps_51d_contract_and_adapts_frequency_clearance_and_width() -> None:
    source = _source(ENV, "RobotOmniTrotClosedLoopGaitRobustEnvCfg", ast.ClassDef)
    assert "RobotOmniTrotClosedLoopStageDEnvCfg" in source
    assert "target_height_high" in source
    assert "lateral_gain" in source
    assert "yaw_gain" in source
    assert "min_frequency" in source and "max_frequency" in source
    assert "observations.policy" in source
    assert "ObsTerm" not in source


def test_clearance_reward_interpolates_target_from_velocity_command() -> None:
    source = _source(REWARDS, "foot_clearance_speed_style", ast.FunctionDef)
    assert "motion / full_speed" in source
    assert "target_height_high - target_height" in source
    assert "desired_height.unsqueeze(1)" in source


def test_gait_task_and_strict_stage_d_evaluator_are_wired() -> None:
    registry = REGISTRY.read_text(encoding="utf-8")
    evaluator = (ROOT / "scripts/evaluate_gait_robust_stage.sh").read_text(encoding="utf-8")
    gate = (ROOT / "scripts/train_gait_robust_stage_gated.sh").read_text(encoding="utf-8")
    assert "CustomDog-Velocity-OmniTrot-ClosedLoopGaitRobust-v1" in registry
    assert "--stage D" in evaluator
    assert 'CUSTOM_DOG_EVAL_LABEL_PREFIX:-GR_' in evaluator
    assert '--label-prefix "${label_prefix}"' in evaluator
    assert 'startswith("D_")' in gate

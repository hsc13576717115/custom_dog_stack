from __future__ import annotations

import ast
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
ENV_CONFIG = ROOT / "rl/src/custom_dog_rl/tasks/locomotion/velocity_env_cfg.py"
PPO_CONFIG = ROOT / "rl/src/custom_dog_rl/agents/ppo_cfg.py"
REGISTRY = ROOT / "rl/src/custom_dog_rl/tasks/locomotion/__init__.py"


def _class(path: Path, name: str) -> ast.ClassDef:
    module = ast.parse(path.read_text(encoding="utf-8"))
    return next(node for node in module.body if isinstance(node, ast.ClassDef) and node.name == name)


def test_stage_b_inherits_robust_selective_collision_contract() -> None:
    node = _class(ENV_CONFIG, "RobotOmniTrotClosedLoopStageBEnvCfg")
    assert ast.unparse(node.bases[0]) == "RobotOmniTrotClosedLoopSelectiveCollisionEnvCfg"


def test_stage_ranges_expand_in_declared_order() -> None:
    source = ENV_CONFIG.read_text(encoding="utf-8")
    expected = {
        "RobotOmniTrotClosedLoopStageBEnvCfg": ("-0.8, 0.8", "-0.2, 0.2", "-0.5, 0.5"),
        "RobotOmniTrotClosedLoopStageCEnvCfg": ("-1.5, 1.5", "-0.4, 0.4", "-1.0, 1.0"),
        "RobotOmniTrotClosedLoopStageDEnvCfg": ("-3.0, 3.0", "-0.6, 0.6", "-2.0, 2.0"),
    }
    for class_name, ranges in expected.items():
        node = _class(ENV_CONFIG, class_name)
        class_source = ast.unparse(node)
        assert all(value in class_source for value in ranges), (class_name, source)


def test_expansion_runner_is_not_foundation_learning_rate() -> None:
    source = ast.unparse(_class(PPO_CONFIG, "CustomDogOmniTrotClosedLoopExpansionPPORunnerCfg"))
    assert "self.algorithm.learning_rate = 0.0001" in source
    assert "self.policy.init_noise_std = 0.35" in source
    registry = REGISTRY.read_text(encoding="utf-8")
    module = ast.parse(registry)
    task_runners = {}
    for node in module.body:
        if not isinstance(node, ast.Expr) or not isinstance(node.value, ast.Call):
            continue
        call_source = ast.unparse(node.value)
        for stage in ("B", "C", "D"):
            task_id = f"CustomDog-Velocity-OmniTrot-ClosedLoopStage{stage}-v1"
            if task_id in call_source:
                task_runners[stage] = call_source
    assert set(task_runners) == {"B", "C", "D"}
    assert all(
        "CustomDogOmniTrotClosedLoopExpansionPPORunnerCfg" in task_runners[stage]
        for stage in task_runners
    )
    robust_registration = registry.split(
        'id="CustomDog-Velocity-OmniTrot-ClosedLoopRobustFoundation-v1"', 1
    )[1].split("gym.register(", 1)[0]
    assert "CustomDogOmniTrotClosedLoopFoundationPPORunnerCfg" in robust_registration

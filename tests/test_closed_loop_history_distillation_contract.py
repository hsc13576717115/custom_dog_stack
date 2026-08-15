from __future__ import annotations

import ast
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
ENV_CONFIG = ROOT / "rl/src/custom_dog_rl/tasks/locomotion/velocity_env_cfg.py"
PPO_CONFIG = ROOT / "rl/src/custom_dog_rl/agents/ppo_cfg.py"
REGISTRY = ROOT / "rl/src/custom_dog_rl/tasks/locomotion/__init__.py"


def _class_source(path: Path, name: str) -> str:
    module = ast.parse(path.read_text(encoding="utf-8"))
    node = next(node for node in module.body if isinstance(node, ast.ClassDef) and node.name == name)
    return ast.unparse(node)


def test_teacher_contract_is_51d_trot_then_velocity() -> None:
    source = _class_source(ENV_CONFIG, "PrivilegedClosedLoopTeacherObsCfg")
    assert source.index("trot_clock") < source.index("base_lin_vel_xy")
    assert "command_threshold': 0.03" in source
    assert "yaw_command_threshold': 0.05" in source


def test_student_contract_is_213d_and_removes_privileged_terms() -> None:
    source = _class_source(ENV_CONFIG, "RobotClosedLoopHistory213DistillationEnvCfg")
    assert "RobotOmniTrotTerrainT1EnvCfg" in source
    assert "self.observations.policy.trot_clock = None" in source
    assert "self.observations.policy.base_lin_vel_xy = None" in source
    assert "self.observations.policy.dynamics_context = None" in source
    assert "self.observations.teacher.dynamics_context" in source
    assert "term.history_length = 5" in source
    assert "self.observations.policy.velocity_commands.history_length = 1" in source


def test_new_distillation_task_does_not_repurpose_legacy_47d_task() -> None:
    runner = _class_source(PPO_CONFIG, "CustomDogClosedLoopHistory213DistillationRunnerCfg")
    registry = REGISTRY.read_text(encoding="utf-8")
    assert "CustomDogHistory213DistillationRunnerCfg" in runner
    assert "CustomDog-Velocity-ClosedLoop-History213-Distill-v1" in registry
    assert "CustomDog-Velocity-History213-Distill-v1" in registry

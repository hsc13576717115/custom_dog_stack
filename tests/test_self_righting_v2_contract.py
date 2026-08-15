from __future__ import annotations

import ast
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
ENV_CONFIG = ROOT / "rl/src/custom_dog_rl/tasks/locomotion/velocity_env_cfg.py"
EVENTS = ROOT / "rl/src/custom_dog_rl/tasks/locomotion/mdp/events.py"
TERMINATIONS = ROOT / "rl/src/custom_dog_rl/tasks/locomotion/mdp/terminations.py"
REWARDS = ROOT / "rl/src/custom_dog_rl/tasks/locomotion/mdp/rewards.py"
PPO = ROOT / "rl/src/custom_dog_rl/agents/ppo_cfg.py"
TASK_REGISTRY = ROOT / "rl/src/custom_dog_rl/tasks/locomotion/__init__.py"
EVALUATE_RUN = ROOT / "scripts/evaluate_self_righting_run.sh"
TRAIN_R0_GATED = ROOT / "scripts/train_self_righting_r0_gated.sh"


def _class_source(path: Path, name: str) -> str:
    module = ast.parse(path.read_text(encoding="utf-8"))
    node = next(node for node in module.body if isinstance(node, ast.ClassDef) and node.name == name)
    return ast.unparse(node)


def _function_source(path: Path, name: str) -> str:
    module = ast.parse(path.read_text(encoding="utf-8"))
    node = next(
        node for node in module.body if isinstance(node, ast.FunctionDef) and node.name == name
    )
    return ast.unparse(node)


def test_reset_distribution_covers_categorical_and_arbitrary_falls() -> None:
    source = _function_source(EVENTS, "reset_self_righting_states")
    assert "orientation_probabilities" in source
    assert "math_utils.random_orientation" in source
    assert "hip_position_range" in source
    assert "max_linear_velocity" in source
    assert "max_angular_velocity" in source
    assert "_custom_dog_recovery_success_dwell" in source


def test_success_requires_upright_height_low_rate_and_foot_support() -> None:
    state = _function_source(TERMINATIONS, "recovery_success_state")
    dwell = _function_source(TERMINATIONS, "recovery_success_dwell")
    assert "minimum_height" in state
    assert "maximum_tilt_deg" in state
    assert "maximum_angular_velocity" in state
    assert "contact_count >= minimum_contact_feet" in state
    assert "math.ceil(dwell_time_s / env.step_dt)" in dwell
    assert "torch.where(stable, counter + 1" in dwell


def test_recovery_rewards_have_supine_gradient_and_stable_support() -> None:
    orientation = _function_source(REWARDS, "recovery_orientation_progress")
    support = _function_source(REWARDS, "recovery_stable_support")
    assert "0.5 * (1.0 - asset.data.projected_gravity_b[:, 2])" in orientation
    assert "support" in support
    assert "angular_stability" in support


def test_r0_is_zero_command_selective_collision_with_strict_success() -> None:
    source = _class_source(ENV_CONFIG, "RobotSelfRightingR0EnvCfg")
    assert "CUSTOM_DOG_SELECTIVE_SELF_COLLISION_CFG.spawn.copy()" in source
    assert "orientation_probabilities': (1.0, 0.0, 0.0, 0.0)" in source
    assert "ranges.lin_vel_x = (0.0, 0.0)" in source
    assert "minimum_contact_feet': 4" in source
    assert "dwell_time_s': 0.4" in source
    assert "self.terminations.base_contact = None" in source
    assert "self.terminations.bad_orientation = None" in source


def test_r1_and_r2_expand_only_the_reset_distribution() -> None:
    r1 = _class_source(ENV_CONFIG, "RobotSelfRightingR1EnvCfg")
    r2 = _class_source(ENV_CONFIG, "RobotSelfRightingR2EnvCfg")
    assert "(0.25, 0.25, 0.25, 0.25)" in r1
    assert "params['arbitrary_orientation_probability'] = 0.35" in r2
    assert "params['max_linear_velocity'] = 0.5" in r2
    assert "params['max_angular_velocity'] = 1.5" in r2


def test_self_righting_runner_and_tasks_are_independent() -> None:
    runner = _class_source(PPO, "CustomDogSelfRightingPPORunnerCfg")
    registry = TASK_REGISTRY.read_text(encoding="utf-8")
    assert "self.algorithm.learning_rate = 0.0005" in runner
    assert "self.policy.init_noise_std = 0.8" in runner
    assert 'id=f"CustomDog-SelfRighting-{stage}-v2"' in registry
    assert "CustomDogSelfRightingPPORunnerCfg" in registry


def test_recovery_gate_evaluates_every_stage_with_selective_collision_mjcf() -> None:
    evaluator = EVALUATE_RUN.read_text(encoding="utf-8")
    r0_gate = TRAIN_R0_GATED.read_text(encoding="utf-8")
    assert "generate_selective_mujoco.py" in evaluator
    assert "validate_mujoco_self_collision.py" in evaluator
    assert '--mjcf "${selective_mjcf}"' in evaluator
    assert 'evaluate_self_righting_run.sh" R0' in r0_gate

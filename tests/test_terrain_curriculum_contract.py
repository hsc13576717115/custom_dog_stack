from __future__ import annotations

import ast
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
ENV_CONFIG = ROOT / "rl/src/custom_dog_rl/tasks/locomotion/velocity_env_cfg.py"
REGISTRY = ROOT / "rl/src/custom_dog_rl/tasks/locomotion/__init__.py"


def _assignment_source(name: str) -> str:
    module = ast.parse(ENV_CONFIG.read_text(encoding="utf-8"))
    node = next(
        node
        for node in module.body
        if isinstance(node, ast.Assign)
        and any(isinstance(target, ast.Name) and target.id == name for target in node.targets)
    )
    return ast.unparse(node)


def _class_source(name: str) -> str:
    module = ast.parse(ENV_CONFIG.read_text(encoding="utf-8"))
    node = next(node for node in module.body if isinstance(node, ast.ClassDef) and node.name == name)
    return ast.unparse(node)


def test_t0_retains_forty_percent_flat_and_only_mild_terrain() -> None:
    source = _assignment_source("CUSTOM_DOG_TERRAIN_T0_CFG")
    assert "curriculum=True" in source
    assert "proportion=0.4" in source
    assert "proportion=0.3" in source
    assert "noise_range=(0.005, 0.025)" in source
    assert "slope_range=(0.0, 0.12)" in source
    assert "Stairs" not in source


def test_t1_retains_flat_regression_and_adds_only_low_steps() -> None:
    source = _assignment_source("CUSTOM_DOG_TERRAIN_T1_CFG")
    assert "curriculum=True" in source
    assert "proportion=0.4" in source
    assert "step_height_range=(0.02, 0.08)" in source
    assert "holes=False" in source


def test_height_scan_is_privileged_and_tasks_are_registered() -> None:
    t0 = _class_source("RobotOmniTrotTerrainT0EnvCfg")
    assert "RobotOmniTrotDynamicsTeacherEnvCfg" in t0
    assert "command.ranges = command.limit_ranges" in t0
    assert "command.flat_terrain_type_count = 8" in t0
    assert "self.curriculum.terrain_levels = CurrTerm(func=mdp.terrain_levels_vel)" in t0
    assert "lin_vel_x=(-0.8, 0.8)" in t0
    assert "self.rewards.foot_clearance_style = None" in t0
    assert "self.observations.critic.height_scan" in t0
    assert "self.observations.policy.height_scan" not in t0
    registry = REGISTRY.read_text(encoding="utf-8")
    assert 'id=f"CustomDog-Velocity-OmniTrot-Terrain{stage}-v1"' in registry


def test_t1_expands_terrain_training_to_stage_c_but_preserves_stage_d_limits() -> None:
    t1 = _class_source("RobotOmniTrotTerrainT1EnvCfg")
    assert "RobotOmniTrotTerrainT0EnvCfg" in t1
    assert "command.flat_terrain_type_count = 8" in t1
    assert "lin_vel_x=(-1.5, 1.5)" in t1
    assert "lin_vel_y=(-0.4, 0.4)" in t1
    assert "ang_vel_z=(-1.0, 1.0)" in t1


def test_command_sampler_keeps_full_flat_range_and_clamps_only_rough_columns() -> None:
    commands = ROOT / "rl/src/custom_dog_rl/tasks/locomotion/mdp/commands.py"
    module = ast.parse(commands.read_text(encoding="utf-8"))
    node = next(
        node
        for node in module.body
        if isinstance(node, ast.ClassDef) and node.name == "StratifiedOmniVelocityCommand"
    )
    source = ast.unparse(node)
    assert "terrain.terrain_types[env_ids] >= self.cfg.flat_terrain_type_count" in source
    assert "self.vel_command_b[rough_ids, axis].clamp_" in source
    assert "self._sampled_vel_command_b[rough_ids]" in source


def test_terrain_evidence_combines_grouped_isaac_and_full_flat_stage_d_grid() -> None:
    evaluator = (ROOT / "scripts/evaluate_terrain_stage.sh").read_text(encoding="utf-8")
    gate = (ROOT / "scripts/train_next_terrain_stage_gated.sh").read_text(
        encoding="utf-8"
    )
    isaac = (ROOT / "rl/scripts/evaluate_terrain_checkpoint.py").read_text(
        encoding="utf-8"
    )
    assert "evaluate_terrain_checkpoint.py" in evaluator
    assert "--stage D" in evaluator
    assert "select_terrain_candidate.py" in evaluator
    assert 'startswith($prefix)' in gate
    assert '"success_rate": float(row["success_rate"]) >= 0.95' in isaac
    assert '"height_p05"' in isaac
    assert '"tilt_p95"' in isaac
    assert '"gait_transitions"' in isaac
    assert 'entry_point_key="env_cfg_entry_point"' in isaac
    assert '"terrain_families"' in isaac
    assert '"family_commands"' in isaac
    assert "family_mask & (group_ids == command_index)" in isaac
    assert "family_env_ids" in isaac

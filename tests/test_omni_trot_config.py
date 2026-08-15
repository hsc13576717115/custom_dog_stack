from __future__ import annotations

import ast
from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[1]
ENV_CONFIG = ROOT / "rl/src/custom_dog_rl/tasks/locomotion/velocity_env_cfg.py"
TASK_REGISTRY = ROOT / "rl/src/custom_dog_rl/tasks/locomotion/__init__.py"
OBSERVATIONS = ROOT / "rl/src/custom_dog_rl/tasks/locomotion/mdp/observations.py"
REWARDS = ROOT / "rl/src/custom_dog_rl/tasks/locomotion/mdp/rewards.py"
PPO_CONFIG = ROOT / "rl/src/custom_dog_rl/agents/ppo_cfg.py"


def class_node(source: str, name: str) -> ast.ClassDef:
    module = ast.parse(source)
    for node in module.body:
        if isinstance(node, ast.ClassDef) and node.name == name:
            return node
    raise AssertionError(f"Missing class {name}")


class OmniTrotConfigTest(unittest.TestCase):
    def test_target_ranges_clock_and_rewards(self) -> None:
        source = ENV_CONFIG.read_text(encoding="utf-8")
        training = ast.unparse(class_node(source, "RobotOmniTrotEnvCfg"))
        self.assertIn("lin_vel_x=(-3.0, 3.0)", training)
        self.assertIn("lin_vel_y=(-0.6, 0.6)", training)
        self.assertIn("ang_vel_z=(-2.0, 2.0)", training)
        self.assertIn("command_trot_clock", training)
        self.assertIn("trot_contact_schedule", training)
        self.assertIn("trot_stance_swing_tracking", training)
        self.assertIn("speed_adaptive_base_height_l2", training)

    def test_explicit_diagonal_order(self) -> None:
        observation_source = OBSERVATIONS.read_text(encoding="utf-8")
        reward_source = REWARDS.read_text(encoding="utf-8")
        self.assertIn("(0.0, 0.5, 0.5, 0.0)", observation_source)
        self.assertIn("(0.0, 0.5, 0.5, 0.0)", reward_source)

    def test_task_is_registered(self) -> None:
        registry = TASK_REGISTRY.read_text(encoding="utf-8")
        self.assertIn('id="CustomDog-Velocity-OmniTrot-v1"', registry)
        self.assertIn("RobotOmniTrotEnvCfg", registry)
        self.assertIn("CustomDogOmniTrotPPORunnerCfg", registry)

    def test_posture_refinement_contract(self) -> None:
        source = ENV_CONFIG.read_text(encoding="utf-8")
        training = ast.unparse(class_node(source, "RobotOmniTrotPostureEnvCfg"))
        self.assertIn("standing_height': 0.33", training)
        self.assertIn("crouched_height': 0.28", training)
        self.assertIn("hip_outward_speed_style_l2", training)
        self.assertIn("inactive_velocity_axes_l2", training)
        self.assertIn("feet_air_time_command_aware", training)
        self.assertIn("pure_axis_swing_count", training)
        self.assertIn("gait_threshold = 0.025", training)
        self.assertIn("command.rel_low_speed_yaw = 0.6", training)
        self.assertIn("command.low_speed_yaw_range = (0.08, 0.4)", training)

        registry = TASK_REGISTRY.read_text(encoding="utf-8")
        self.assertIn('id="CustomDog-Velocity-OmniTrot-Posture-v2"', registry)
        self.assertIn("CustomDogOmniTrotPosturePPORunnerCfg", registry)

    def test_low_speed_refinement_contract(self) -> None:
        source = ENV_CONFIG.read_text(encoding="utf-8")
        training = ast.unparse(class_node(source, "RobotOmniTrotRefineEnvCfg"))
        self.assertIn("command.minimum_command_magnitude = 0.03", training)
        self.assertIn("command.low_speed_x_range = (0.03, 0.2)", training)
        self.assertIn("command.low_speed_y_range = (0.03, 0.2)", training)
        self.assertIn("command.low_speed_yaw_range = (0.05, 0.3)", training)
        self.assertIn("yaw_deadband = 0.05", training)
        self.assertIn("hip_outward_band_l2", training)
        self.assertIn("lateral_allowance': 0.18", training)
        self.assertIn("yaw_allowance': 0.05", training)
        self.assertIn("paired_lateral_separation_l2", training)
        self.assertIn("motion_swing_count", training)
        self.assertIn("self.rewards.energy.weight = -8e-05", training)
        self.assertIn("self.rewards.flat_orientation_l2.weight = -8.0", training)
        self.assertIn("self.rewards.track_lin_vel_xy_l2", training)
        self.assertIn("params['command_min'] = (0.03, 0.03, 0.05)", training)
        self.assertIn("params['crouched_height'] = 0.29", training)

        registry = TASK_REGISTRY.read_text(encoding="utf-8")
        self.assertIn('id="CustomDog-Velocity-OmniTrot-Refine-v3"', registry)
        self.assertIn("CustomDogOmniTrotRefinePPORunnerCfg", registry)

    def test_closed_loop_foundation_contract(self) -> None:
        source = ENV_CONFIG.read_text(encoding="utf-8")
        training = ast.unparse(class_node(source, "RobotOmniTrotClosedLoopFoundationEnvCfg"))
        self.assertIn("base_lin_vel_xy", training)
        self.assertIn("lin_vel_x=(-0.45, 0.45)", training)
        self.assertIn("lin_vel_y=(-0.1, 0.1)", training)
        self.assertIn("ang_vel_z=(-0.25, 0.25)", training)
        self.assertIn("pure_axis_velocity_decoupling_l2", training)
        self.assertIn("standing_foot_placement_l2", training)
        self.assertIn("standing_height': 0.33", training)
        self.assertIn("crouched_height': 0.28", training)
        self.assertIn("command.resampling_time_range = (6.0, 8.0)", training)

        registry = TASK_REGISTRY.read_text(encoding="utf-8")
        self.assertIn('id="CustomDog-Velocity-OmniTrot-ClosedLoopFoundation-v1"', registry)
        self.assertIn('id="CustomDog-Velocity-OmniTrot-ClosedLoopStageB-v1"', registry)
        self.assertIn('id="CustomDog-Velocity-OmniTrot-ClosedLoopStageC-v1"', registry)
        self.assertIn('id="CustomDog-Velocity-OmniTrot-ClosedLoopStageD-v1"', registry)
        self.assertIn("CustomDogOmniTrotClosedLoopFoundationPPORunnerCfg", registry)

    def test_closed_loop_polish_a1_contract(self) -> None:
        source = ENV_CONFIG.read_text(encoding="utf-8")
        training = ast.unparse(class_node(source, "RobotOmniTrotClosedLoopPolishA1EnvCfg"))
        self.assertIn("command.resampling_time_range = (8.0, 12.0)", training)
        self.assertIn("command.bucket_probabilities = (0.22, 0.13, 0.45, 0.2)", training)
        self.assertIn("command.rel_low_speed_yaw = 0.8", training)
        self.assertIn("command.low_speed_yaw_range = (0.05, 0.18)", training)
        self.assertIn("self.rewards.track_ang_vel_z_l2.weight = -3.0", training)
        self.assertIn("self.rewards.speed_adaptive_base_height.weight = -90.0", training)
        self.assertIn("self.rewards.joint_pos.params['stand_still_scale'] = 1.5", training)

        registry = TASK_REGISTRY.read_text(encoding="utf-8")
        self.assertIn('id="CustomDog-Velocity-OmniTrot-ClosedLoopPolishA1-v1"', registry)
        self.assertIn("CustomDogOmniTrotClosedLoopPolishPPORunnerCfg", registry)

    def test_closed_loop_polish_a2_contract(self) -> None:
        source = ENV_CONFIG.read_text(encoding="utf-8")
        training = ast.unparse(class_node(source, "RobotOmniTrotClosedLoopPolishA2EnvCfg"))
        self.assertIn("command.bucket_probabilities = (0.18, 0.1, 0.52, 0.2)", training)
        self.assertIn("command.rel_low_speed_yaw = 0.5", training)
        self.assertIn("command.rel_high_speed_yaw = 0.4", training)
        self.assertIn("command.high_speed_yaw_range = (0.2, 0.25)", training)
        self.assertIn("yaw_overspeed_relative_l2", training)
        self.assertIn("standing_base_height_band_l2", training)
        self.assertIn("self.rewards.standing_foot_placement.weight = -0.5", training)

        registry = TASK_REGISTRY.read_text(encoding="utf-8")
        self.assertIn('id="CustomDog-Velocity-OmniTrot-ClosedLoopPolishA2-v1"', registry)
        self.assertIn("RobotOmniTrotClosedLoopPolishA2EnvCfg", registry)

    def test_closed_loop_stand_fix_contract(self) -> None:
        source = ENV_CONFIG.read_text(encoding="utf-8")
        training = ast.unparse(class_node(source, "RobotOmniTrotClosedLoopStandFixEnvCfg"))
        self.assertIn("self.commands.base_velocity.rel_standing_envs = 0.35", training)
        self.assertIn("self.rewards.yaw_overspeed_relative = None", training)
        self.assertIn("standing_joint_deviation_normalized_l2", training)
        self.assertIn("standing_orientation_normalized_l2", training)

        registry = TASK_REGISTRY.read_text(encoding="utf-8")
        self.assertIn('id="CustomDog-Velocity-OmniTrot-ClosedLoopStandFix-v1"', registry)
        self.assertIn("RobotOmniTrotClosedLoopStandFixEnvCfg", registry)

    def test_robust_stand_fix_is_zero_command_only(self) -> None:
        source = ENV_CONFIG.read_text()
        training = ast.unparse(class_node(source, "RobotOmniTrotRobustStandFixEnvCfg"))
        registry = TASK_REGISTRY.read_text()
        runner = PPO_CONFIG.read_text()

        self.assertIn("RobotOmniTrotClosedLoopRobustFoundationEnvCfg", training)
        self.assertIn("command.rel_standing_envs = 0.45", training)
        self.assertIn("standing_joint_deviation_normalized_l2", training)
        self.assertIn("standing_orientation_normalized_l2", training)
        self.assertIn("'yaw_command_threshold': 0.05", training)
        self.assertIn('id="CustomDog-Velocity-OmniTrot-RobustStandFix-v1"', registry)
        self.assertIn("CustomDogOmniTrotRobustStandFixPPORunnerCfg", runner)

    def test_closed_loop_cross_physics_contract(self) -> None:
        source = ENV_CONFIG.read_text(encoding="utf-8")
        training = ast.unparse(
            class_node(source, "RobotOmniTrotClosedLoopCrossPhysicsEnvCfg")
        )
        self.assertIn("self.rewards.yaw_overspeed_relative = None", training)
        self.assertIn("randomize_rigid_body_com", training)
        self.assertIn("randomize_actuator_gains", training)
        self.assertIn("randomize_joint_parameters", training)
        self.assertIn("actuator.max_delay = 2", training)
        self.assertNotIn("lin_vel_x=", training)

        registry = TASK_REGISTRY.read_text(encoding="utf-8")
        self.assertIn('id="CustomDog-Velocity-OmniTrot-ClosedLoopCrossPhysics-v1"', registry)
        self.assertIn("CustomDogOmniTrotClosedLoopCrossPhysicsPPORunnerCfg", registry)

    def test_closed_loop_robust_foundation_contract(self) -> None:
        source = ENV_CONFIG.read_text(encoding="utf-8")
        training = ast.unparse(
            class_node(source, "RobotOmniTrotClosedLoopRobustFoundationEnvCfg")
        )
        self.assertIn("scale = (0.2, 0.2, 1.0)", training)
        self.assertIn("Unoise(n_min=-0.1, n_max=0.1)", training)
        self.assertIn("command.rel_standing_envs = 0.25", training)
        self.assertIn("yaw_overspeed_relative_l2", training)
        self.assertNotIn("lin_vel_x=", training)

        registry = TASK_REGISTRY.read_text(encoding="utf-8")
        self.assertIn(
            'id="CustomDog-Velocity-OmniTrot-ClosedLoopRobustFoundation-v1"',
            registry,
        )
        self.assertIn("CustomDogOmniTrotClosedLoopFoundationPPORunnerCfg", registry)

    def test_closed_loop_selective_collision_contract(self) -> None:
        source = ENV_CONFIG.read_text(encoding="utf-8")
        training = ast.unparse(
            class_node(source, "RobotOmniTrotClosedLoopSelectiveCollisionEnvCfg")
        )
        self.assertIn("CUSTOM_DOG_SELECTIVE_SELF_COLLISION_CFG.spawn.copy()", training)

        registry = TASK_REGISTRY.read_text(encoding="utf-8")
        self.assertIn(
            'id="CustomDog-Velocity-OmniTrot-ClosedLoopSelectiveCollision-v1"',
            registry,
        )


if __name__ == "__main__":
    unittest.main()

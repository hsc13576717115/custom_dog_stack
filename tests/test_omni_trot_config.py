from __future__ import annotations

import ast
from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[1]
ENV_CONFIG = ROOT / "rl/src/custom_dog_rl/tasks/locomotion/velocity_env_cfg.py"
TASK_REGISTRY = ROOT / "rl/src/custom_dog_rl/tasks/locomotion/__init__.py"
OBSERVATIONS = ROOT / "rl/src/custom_dog_rl/tasks/locomotion/mdp/observations.py"
REWARDS = ROOT / "rl/src/custom_dog_rl/tasks/locomotion/mdp/rewards.py"


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


if __name__ == "__main__":
    unittest.main()

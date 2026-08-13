from __future__ import annotations

import ast
from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[1]
ENV_CONFIG = ROOT / "rl/src/custom_dog_rl/tasks/locomotion/velocity_env_cfg.py"
TASK_REGISTRY = ROOT / "rl/src/custom_dog_rl/tasks/locomotion/__init__.py"


def class_node(source: str, name: str) -> ast.ClassDef:
    module = ast.parse(source)
    for node in module.body:
        if isinstance(node, ast.ClassDef) and node.name == name:
            return node
    raise AssertionError(f"Missing class {name}")


class HighSpeedOmniConfigTest(unittest.TestCase):
    def test_training_and_play_ranges(self) -> None:
        source = ENV_CONFIG.read_text(encoding="utf-8")
        training = ast.unparse(class_node(source, "RobotOmni45HighSpeedEnvCfg"))
        play = ast.unparse(class_node(source, "RobotOmni45HighSpeedPlayEnvCfg"))

        self.assertIn("lin_vel_x=(-1.0, 1.0)", training)
        self.assertIn("lin_vel_y=(-0.4, 0.4)", training)
        self.assertIn("ang_vel_z=(-1.0, 1.0)", training)
        self.assertIn("lin_vel_x=(-3.0, 3.0)", training)
        self.assertIn("lin_vel_y=(-0.6, 0.6)", training)
        self.assertIn("ang_vel_z=(-2.0, 2.0)", training)
        self.assertIn("ranges = self.commands.base_velocity.limit_ranges", play)

    def test_task_is_registered(self) -> None:
        registry = TASK_REGISTRY.read_text(encoding="utf-8")
        self.assertIn('id="CustomDog-Velocity-Omni45-HighSpeed-v1"', registry)
        self.assertIn("RobotOmni45HighSpeedEnvCfg", registry)
        self.assertIn("CustomDogOmni45HighSpeedPPORunnerCfg", registry)


if __name__ == "__main__":
    unittest.main()

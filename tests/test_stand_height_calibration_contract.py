from __future__ import annotations

import ast
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
ENV_CFG = ROOT / "rl/src/custom_dog_rl/tasks/locomotion/velocity_env_cfg.py"
PPO_CFG = ROOT / "rl/src/custom_dog_rl/agents/ppo_cfg.py"
REGISTRY = ROOT / "rl/src/custom_dog_rl/tasks/locomotion/__init__.py"


def _class_names(path: Path) -> set[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    return {node.name for node in tree.body if isinstance(node, ast.ClassDef)}


def test_height_calibration_has_separate_env_runner_and_registration() -> None:
    assert "RobotStandHeightCalibratedEnvCfg" in _class_names(ENV_CFG)
    assert "RobotStandHeightCalibratedPlayEnvCfg" in _class_names(ENV_CFG)
    assert "CustomDogStandHeightCalibratedPPORunnerCfg" in _class_names(PPO_CFG)
    registry = REGISTRY.read_text(encoding="utf-8")
    assert 'id="CustomDog-Stand-HeightCalibrated-v2"' in registry
    assert "RobotStandHeightHipCalibratedEnvCfg" in _class_names(ENV_CFG)
    assert 'id="CustomDog-Stand-HeightHipCalibrated-v3"' in registry


def test_height_calibration_targets_transfer_offset_without_relaxing_gate() -> None:
    source = ENV_CFG.read_text(encoding="utf-8")
    class_source = source[
        source.index("class RobotStandHeightCalibratedEnvCfg") :
        source.index("class RobotStandHeightCalibratedPlayEnvCfg")
    ]
    assert '["standing_height"] = 0.35' in class_source
    assert '"lower_height": 0.335' in class_source
    assert '"upper_height": 0.355' in class_source


def test_calibrated_train_and_evaluation_are_checkpoint_gated() -> None:
    train = (ROOT / "scripts/train_stand_height_calibrated.sh").read_text(encoding="utf-8")
    evaluate = (ROOT / "scripts/evaluate_stand_height_calibrated.sh").read_text(
        encoding="utf-8"
    )
    assert "model_200.pt" in train
    assert "CUSTOM_DOG_LOAD_OPTIMIZER" in train
    assert "--resume" in train
    assert "--duration 15" in evaluate
    assert "select_mujoco_candidate.py" in evaluate


def test_v3_continues_the_best_height_checkpoint_and_restores_hip_limit() -> None:
    source = ENV_CFG.read_text(encoding="utf-8")
    class_source = source[
        source.index("class RobotStandHeightHipCalibratedEnvCfg") :
        source.index("class RobotStandHeightHipCalibratedPlayEnvCfg")
    ]
    assert '["standing_height"] = 0.365' in class_source
    assert 'self.rewards.hip_outward_band.weight = -20.0' in class_source
    assert 'self.rewards.standing_hip_pose.weight = -3.0' in class_source
    train = (ROOT / "scripts/train_stand_height_hip_calibrated.sh").read_text(
        encoding="utf-8"
    )
    assert "model_319.pt" in train

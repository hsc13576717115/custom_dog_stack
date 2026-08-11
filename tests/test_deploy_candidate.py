import hashlib
import math
import unittest
from pathlib import Path

import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CANDIDATE_DIR = PROJECT_ROOT / "deploy" / "candidates" / "model_4500_yaw_straight"
POLICY_PATH = CANDIDATE_DIR / "exported" / "policy.onnx"
DEPLOY_PATH = CANDIDATE_DIR / "params" / "deploy.yaml"
METADATA_PATH = CANDIDATE_DIR / "metadata.yaml"
OMNI_CANDIDATE_DIR = (
    PROJECT_ROOT / "deploy" / "candidates" / "model_700_compact_omni_balanced"
)
CALIBRATED_OMNI_CANDIDATE_DIR = (
    PROJECT_ROOT / "deploy" / "candidates" / "model_800_omni_stability_calibrated"
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


class DeployCandidateTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        with DEPLOY_PATH.open(encoding="utf-8") as stream:
            cls.deploy = yaml.safe_load(stream)
        with METADATA_PATH.open(encoding="utf-8") as stream:
            cls.metadata = yaml.safe_load(stream)

    def test_artifacts_match_recorded_hashes(self):
        self.assertTrue(POLICY_PATH.is_file())
        self.assertEqual(sha256(POLICY_PATH), self.metadata["onnx_sha256"])
        self.assertEqual(sha256(DEPLOY_PATH), self.metadata["deploy_yaml_sha256"])

    def test_policy_dimensions_and_joint_mapping(self):
        self.assertEqual(self.deploy["policy_contract_version"], "1.2")
        self.assertEqual(self.metadata["policy_contract_version"], "1.2")
        self.assertEqual(sorted(self.deploy["joint_ids_map"]), list(range(12)))
        observation_dim = sum(
            len(term["scale"]) for term in self.deploy["observations"].values()
        )
        action = self.deploy["actions"]["JointPositionAction"]
        self.assertEqual(observation_dim, 45)
        self.assertEqual(len(action["scale"]), 12)
        self.assertEqual(len(action["offset"]), 12)

    def test_target_bias_contract(self):
        bias = self.deploy["joint_target_bias"]
        values = bias["values"]
        vx_min, vx_max = bias["vx_range"]
        self.assertEqual(len(values), 12)
        self.assertTrue(all(math.isfinite(value) for value in values))
        self.assertLess(vx_min, vx_max)
        self.assertEqual(
            values,
            [0.0, 0.0, 0.0, 0.0, 0.02, -0.02, 0.02, -0.02, 0.0, 0.0, 0.0, 0.0],
        )
        self.assertTrue(self.metadata["sim2sim_target_bias_calibrated"])
        self.assertFalse(self.metadata["validated_for_hardware"])

    def test_velocity_command_calibration_contract(self):
        calibration = self.deploy["command_calibration"]["lin_vel_x"]
        requested = calibration["requested"]
        policy = calibration["policy"]
        self.assertEqual(len(requested), len(policy))
        self.assertGreaterEqual(len(requested), 2)
        self.assertEqual(requested[0], 0.0)
        self.assertEqual(requested[-1], 3.0)
        self.assertTrue(all(left < right for left, right in zip(requested, requested[1:])))
        self.assertTrue(all(left <= right for left, right in zip(policy, policy[1:])))
        self.assertTrue(self.metadata["sim2sim_velocity_command_calibrated"])


class OmniDeployCandidateTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.policy_path = OMNI_CANDIDATE_DIR / "exported" / "policy.onnx"
        cls.deploy_path = OMNI_CANDIDATE_DIR / "params" / "deploy.yaml"
        cls.metadata_path = OMNI_CANDIDATE_DIR / "metadata.yaml"
        with cls.deploy_path.open(encoding="utf-8") as stream:
            cls.deploy = yaml.safe_load(stream)
        with cls.metadata_path.open(encoding="utf-8") as stream:
            cls.metadata = yaml.safe_load(stream)

    def test_artifacts_match_recorded_hashes(self):
        self.assertTrue(self.policy_path.is_file())
        self.assertEqual(sha256(self.policy_path), self.metadata["onnx_sha256"])
        self.assertEqual(sha256(self.deploy_path), self.metadata["deploy_yaml_sha256"])

    def test_dimensions_gains_and_safe_command_envelope(self):
        observation_dim = sum(
            len(term["scale"]) for term in self.deploy["observations"].values()
        )
        self.assertEqual(observation_dim, 47)
        self.assertEqual(len(self.deploy["actions"]["JointPositionAction"]["scale"]), 12)
        self.assertEqual(self.deploy["stiffness"], [25.0] * 12)
        self.assertEqual(self.deploy["damping"], [0.5] * 12)
        self.assertEqual(
            self.deploy["commands"]["base_velocity"]["ranges"],
            {
                "lin_vel_x": [0.0, 0.5],
                "lin_vel_y": [-0.25, 0.25],
                "ang_vel_z": [-0.5, 0.5],
                "heading": None,
            },
        )

    def test_default_position_maps_to_compact_sdk_home(self):
        joint_map = self.deploy["joint_ids_map"]
        self.assertEqual(sorted(joint_map), list(range(12)))
        sdk_home = [0.0] * 12
        for policy_index, sdk_index in enumerate(joint_map):
            sdk_home[sdk_index] = self.deploy["default_joint_pos"][policy_index]
        self.assertEqual(
            sdk_home,
            [-0.05, 0.8, -1.5, 0.05, 0.8, -1.5] * 2,
        )
        self.assertTrue(self.metadata["measured_prone_recovery_validated"])
        self.assertFalse(self.metadata["validated_for_hardware"])


class CalibratedOmniDeployCandidateTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.policy_path = CALIBRATED_OMNI_CANDIDATE_DIR / "exported" / "policy.onnx"
        cls.deploy_path = CALIBRATED_OMNI_CANDIDATE_DIR / "params" / "deploy.yaml"
        cls.metadata_path = CALIBRATED_OMNI_CANDIDATE_DIR / "metadata.yaml"
        with cls.deploy_path.open(encoding="utf-8") as stream:
            cls.deploy = yaml.safe_load(stream)
        with cls.metadata_path.open(encoding="utf-8") as stream:
            cls.metadata = yaml.safe_load(stream)

    def test_artifacts_match_recorded_hashes(self):
        self.assertEqual(sha256(self.policy_path), self.metadata["onnx_sha256"])
        self.assertEqual(sha256(self.deploy_path), self.metadata["deploy_yaml_sha256"])

    def test_policy_contract_and_sim2sim_status(self):
        observation_dim = sum(
            len(term["scale"]) for term in self.deploy["observations"].values()
        )
        self.assertEqual(self.deploy["policy_contract_version"], "2.1")
        self.assertEqual(self.metadata["policy_contract_version"], "2.1")
        self.assertEqual(observation_dim, 47)
        self.assertEqual(len(self.deploy["actions"]["JointPositionAction"]["scale"]), 12)
        self.assertTrue(self.metadata["sim2sim_validated"])
        self.assertFalse(self.metadata["validated_for_hardware"])

    def test_external_and_policy_ranges(self):
        command = self.deploy["commands"]["base_velocity"]
        self.assertEqual(
            command["external_ranges"],
            {
                "lin_vel_x": [0.0, 0.6],
                "lin_vel_y": [-0.17, 0.17],
                "ang_vel_z": [-0.6, 0.6],
            },
        )
        self.assertEqual(
            command["policy_ranges"],
            {
                "lin_vel_x": [0.0, 0.62],
                "lin_vel_y": [-0.42, 0.42],
                "ang_vel_z": [-0.6, 0.6],
            },
        )

    def test_three_axis_calibration_is_monotonic_and_bounded(self):
        command = self.deploy["commands"]["base_velocity"]
        axis_keys = ("lin_vel_x", "lin_vel_y", "ang_vel_z")
        for axis in axis_keys:
            calibration = self.deploy["command_calibration"][axis]
            requested = calibration["requested"]
            policy = calibration["policy"]
            self.assertEqual(len(requested), len(policy))
            self.assertGreaterEqual(len(requested), 2)
            self.assertTrue(
                all(left < right for left, right in zip(requested, requested[1:]))
            )
            self.assertTrue(all(left <= right for left, right in zip(policy, policy[1:])))
            self.assertLessEqual(requested[0], command["external_ranges"][axis][0])
            self.assertGreaterEqual(requested[-1], command["external_ranges"][axis][1])
            self.assertGreaterEqual(min(policy), command["policy_ranges"][axis][0])
            self.assertLessEqual(max(policy), command["policy_ranges"][axis][1])


if __name__ == "__main__":
    unittest.main()

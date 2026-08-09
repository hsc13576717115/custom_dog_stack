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


if __name__ == "__main__":
    unittest.main()

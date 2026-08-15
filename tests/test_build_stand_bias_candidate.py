from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[1]
BUILDER = ROOT / "scripts/build_stand_bias_candidate.py"


def _source(root: Path) -> Path:
    (root / "exported").mkdir(parents=True)
    (root / "params").mkdir()
    (root / "exported/policy.onnx").write_bytes(b"stand")
    (root / "params/deploy.yaml").write_text(
        yaml.safe_dump({"observations": {}, "actions": {}}), encoding="utf-8"
    )
    return root


def test_builder_adds_symmetric_policy_order_hip_bias(tmp_path: Path) -> None:
    source = _source(tmp_path / "source")
    output = tmp_path / "biased"
    subprocess.run(
        [
            sys.executable,
            str(BUILDER),
            str(source),
            str(output),
            "--hip-bias",
            "0.06",
        ],
        check=True,
    )
    config = yaml.safe_load((output / "params/deploy.yaml").read_text())
    assert config["constant_joint_target_bias"] == [
        -0.06,
        0.06,
        -0.06,
        0.06,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
    ]


def test_builder_rejects_excessive_bias(tmp_path: Path) -> None:
    source = _source(tmp_path / "source")
    completed = subprocess.run(
        [
            sys.executable,
            str(BUILDER),
            str(source),
            str(tmp_path / "biased"),
            "--hip-bias",
            "0.2",
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    assert completed.returncode != 0
    assert "must be in" in completed.stderr

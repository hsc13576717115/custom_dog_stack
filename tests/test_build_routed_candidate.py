from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[1]
BUILDER = ROOT / "scripts/build_routed_candidate.py"


def _candidate(root: Path, marker: bytes, *, step_dt: float = 0.02) -> Path:
    (root / "exported").mkdir(parents=True)
    (root / "params").mkdir()
    (root / "exported/policy.onnx").write_bytes(marker)
    config = {
        "observations": ["base_ang_vel", "commands"],
        "actions": ["joint_pos"],
        "joint_ids_map": list(range(12)),
        "default_joint_pos": [0.0] * 12,
        "stiffness": [20.0] * 12,
        "damping": [0.5] * 12,
        "step_dt": step_dt,
    }
    (root / "params/deploy.yaml").write_text(
        yaml.safe_dump(config), encoding="utf-8"
    )
    return root


def test_builder_copies_both_policies_and_records_sources(tmp_path: Path) -> None:
    locomotion = _candidate(tmp_path / "locomotion", b"motion")
    stand = _candidate(tmp_path / "stand", b"stand")
    output = tmp_path / "routed"

    subprocess.run(
        [sys.executable, str(BUILDER), str(locomotion), str(stand), str(output)],
        check=True,
        capture_output=True,
        text=True,
    )

    assert (output / "exported/policy.onnx").read_bytes() == b"motion"
    assert (output / "exported/stand_policy.onnx").read_bytes() == b"stand"
    assert (output / "params/deploy.yaml").is_file()
    assert (output / "params/stand_deploy.yaml").is_file()
    metadata = json.loads((output / "routing.json").read_text())
    assert metadata["locomotion_candidate"] == str(locomotion.resolve())
    assert metadata["stand_candidate"] == str(stand.resolve())
    assert metadata["routing"]["stand_exit_planar"] > metadata["routing"]["stand_enter_planar"]


def test_builder_rejects_contract_mismatch(tmp_path: Path) -> None:
    locomotion = _candidate(tmp_path / "locomotion", b"motion")
    stand = _candidate(tmp_path / "stand", b"stand", step_dt=0.03)

    completed = subprocess.run(
        [sys.executable, str(BUILDER), str(locomotion), str(stand), str(tmp_path / "routed")],
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode != 0
    assert "deployment contracts differ" in completed.stderr

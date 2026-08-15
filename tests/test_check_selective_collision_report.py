import importlib.util
import hashlib
from pathlib import Path


MODULE_PATH = Path(__file__).resolve().parents[1] / "scripts" / "check_selective_collision_report.py"
SPEC = importlib.util.spec_from_file_location("check_selective_collision_report", MODULE_PATH)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


def test_accepts_complete_collision_contract():
    assert MODULE.validate_report(
        {
            "filtered_pair_count": 40,
            "expected_filtered_pair_count": 40,
            "nominal_pose_held": True,
            "nominal_nonfoot_contact_steps": 0,
            "forced_cross_leg_contact_steps": 12,
        }
    ) == []


def test_rejects_nominal_contact_and_missing_forced_contact():
    failures = MODULE.validate_report(
        {
            "filtered_pair_count": 39,
            "expected_filtered_pair_count": 40,
            "nominal_pose_held": False,
            "nominal_nonfoot_contact_steps": 200,
            "forced_cross_leg_contact_steps": 0,
        }
    )
    assert failures == [
        "filtered-pair count does not match the contract",
        "nominal pose was not held during geometry validation",
        "nominal stance has persistent non-foot contact",
        "forced cross-leg pose did not produce contact",
    ]


def test_report_asset_fingerprint_must_match(tmp_path: Path):
    asset = tmp_path / "custom_dog_selective_collision.urdf"
    asset.write_text("<robot/>", encoding="utf-8")
    report = {
        "filtered_pair_count": 40,
        "expected_filtered_pair_count": 40,
        "nominal_pose_held": True,
        "nominal_nonfoot_contact_steps": 0,
        "forced_cross_leg_contact_steps": 12,
        "asset_path": str(asset.resolve()),
        "asset_sha256": hashlib.sha256(asset.read_bytes()).hexdigest(),
    }
    assert MODULE.validate_report(report, asset) == []
    asset.write_text("<robot name='changed'/>", encoding="utf-8")
    failures = MODULE.validate_report(report, asset)
    assert "SHA-256" in "; ".join(failures)

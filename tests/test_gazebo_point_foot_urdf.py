from __future__ import annotations

import subprocess
import sys
import xml.etree.ElementTree as ET
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
URDF_DIR = ROOT / "ros2/src/custom_dog_description/urdf"
LEGS = ("FR", "FL", "RR", "RL")


def test_generator_only_removes_thigh_and_calf_collisions(tmp_path: Path) -> None:
    source = URDF_DIR / "custom_dog.urdf"
    output = tmp_path / "point_foot.urdf"
    subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts/generate_gazebo_point_foot_urdf.py"),
            "--input",
            str(source),
            "--output",
            str(output),
        ],
        check=True,
    )

    canonical = ET.parse(source).getroot()
    point_foot = ET.parse(output).getroot()
    assert [link.attrib["name"] for link in canonical.findall("link")] == [
        link.attrib["name"] for link in point_foot.findall("link")
    ]
    assert [joint.attrib["name"] for joint in canonical.findall("joint")] == [
        joint.attrib["name"] for joint in point_foot.findall("joint")
    ]

    for leg in LEGS:
        for suffix in ("thigh", "calf"):
            assert canonical.find(
                f"./link[@name='{leg}_{suffix}']/collision"
            ) is not None
            assert point_foot.find(
                f"./link[@name='{leg}_{suffix}']/collision"
            ) is None
        for suffix in ("hip", "foot"):
            assert point_foot.find(
                f"./link[@name='{leg}_{suffix}']/collision"
            ) is not None

    assert point_foot.find("./link[@name='base']/collision") is not None
    checked_in = URDF_DIR / "custom_dog_gazebo_point_foot.urdf"
    assert output.read_text(encoding="utf-8") == checked_in.read_text(encoding="utf-8")

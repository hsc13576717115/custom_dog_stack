from __future__ import annotations

import subprocess
import sys
import xml.etree.ElementTree as ET
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_generator_replaces_only_thigh_and_calf_collision_proxies(tmp_path: Path) -> None:
    source = ROOT / "ros2/src/custom_dog_description/urdf/custom_dog.urdf"
    output = tmp_path / "selective.urdf"
    subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts/generate_selective_collision_urdf.py"),
            "--input",
            str(source),
            "--output",
            str(output),
        ],
        check=True,
    )
    root = ET.parse(output).getroot()
    for leg in ("FR", "FL", "RR", "RL"):
        thigh_collision = root.find(
            f"./link[@name='{leg}_thigh']/collision/geometry/cylinder"
        )
        thigh_origin = root.find(f"./link[@name='{leg}_thigh']/collision/origin")
        collision = root.find(
            f"./link[@name='{leg}_calf']/collision/geometry/cylinder"
        )
        calf_origin = root.find(f"./link[@name='{leg}_calf']/collision/origin")
        visual = root.find(f"./link[@name='{leg}_calf']/visual/geometry/mesh")
        foot = root.find(f"./link[@name='{leg}_foot']/collision/geometry/sphere")
        assert thigh_collision is not None and thigh_origin is not None
        assert thigh_collision.attrib == {"radius": "0.035", "length": "0.17"}
        assert thigh_origin.attrib["xyz"] == "0 0 -0.09"
        assert collision is not None and calf_origin is not None
        assert collision.attrib == {"radius": "0.022", "length": "0.15"}
        assert calf_origin.attrib["xyz"] == "0 0 -0.09"
        assert visual is not None and "scale" not in visual.attrib
        assert foot is not None and foot.attrib == {"radius": "0.026"}

    source_root = ET.parse(source).getroot()
    source_calf = source_root.find(
        "./link[@name='RL_calf']/collision/geometry/cylinder"
    )
    assert source_calf is not None
    assert source_calf.attrib == {"radius": "0.018", "length": "0.185"}

    checked_in = ROOT / (
        "ros2/src/custom_dog_description/urdf/custom_dog_selective_collision.urdf"
    )
    assert output.read_text(encoding="utf-8") == checked_in.read_text(encoding="utf-8")

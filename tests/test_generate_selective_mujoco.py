from __future__ import annotations

import subprocess
import sys
import xml.etree.ElementTree as ET
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_generator_enables_cross_leg_masks_and_preserves_source(tmp_path: Path) -> None:
    source = ROOT / "sim2sim/custom_dog/custom_dog.xml"
    output = tmp_path / "selective.xml"
    subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts/generate_selective_mujoco.py"),
            "--input",
            str(source),
            "--output",
            str(output),
        ],
        check=True,
    )
    root = ET.parse(output).getroot()
    fr = root.find(".//body[@name='FR_calf']/geom[@class='collision']")
    fl = root.find(".//body[@name='FL_calf']/geom[@class='collision']")
    assert fr is not None and fl is not None
    assert int(fr.attrib["conaffinity"]) & int(fl.attrib["contype"])
    for leg in ("FR", "FL", "RR", "RL"):
        thigh = root.find(
            f".//body[@name='{leg}_thigh']/geom[@class='collision']"
        )
        calf = root.find(f".//body[@name='{leg}_calf']/geom[@class='collision']")
        assert thigh is not None and thigh.attrib["type"] == "capsule"
        assert thigh.attrib["size"] == "0.035"
        assert thigh.attrib["fromto"] == "0 0 -0.005 0 0 -0.175"
        assert "mesh" not in thigh.attrib
        assert calf is not None and calf.attrib["type"] == "capsule"
        assert calf.attrib["size"] == "0.022"
        assert calf.attrib["fromto"] == "0 0 -0.015 0 0 -0.165"
        assert "mesh" not in calf.attrib
    assert "contype" not in ET.parse(source).getroot().find(
        ".//body[@name='FR_calf']/geom[@class='collision']"
    ).attrib

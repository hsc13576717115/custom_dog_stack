from __future__ import annotations

import importlib.util
import xml.etree.ElementTree as ET
from pathlib import Path

import pytest


MODULE_PATH = Path(__file__).resolve().parents[1] / "sim2sim/custom_dog/postprocess_mjcf.py"
SPEC = importlib.util.spec_from_file_location("custom_dog_postprocess_mjcf", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def _root(legs: tuple[str, ...] = ("FR", "FL", "RR", "RL")) -> ET.Element:
    root = ET.fromstring(
        """
        <mujoco>
          <worldbody>
            <body name="base"><geom class="collision" name="base_collision" /></body>
          </worldbody>
        </mujoco>
        """
    )
    world = root.find("worldbody")
    assert world is not None
    for leg in legs:
        body = ET.SubElement(world, "body", {"name": f"{leg}_calf"})
        ET.SubElement(body, "geom", {"class": "visual", "name": f"{leg}_visual"})
        ET.SubElement(body, "geom", {"class": "collision", "name": f"{leg}_collision"})
    return root


def test_cross_leg_masks_filter_same_leg_and_keep_ground() -> None:
    root = _root()
    MODULE.configure_selective_self_collisions(root)

    leg_bits = MODULE.LEG_COLLISION_BITS
    all_leg_bits = sum(leg_bits.values())
    for leg, own_bit in leg_bits.items():
        geom = root.find(f".//geom[@name='{leg}_collision']")
        assert geom is not None
        assert int(geom.attrib["contype"]) == own_bit
        affinity = int(geom.attrib["conaffinity"])
        assert affinity & MODULE.GROUND_COLLISION_BIT
        assert not affinity & own_bit
        assert affinity & (all_leg_bits & ~own_bit) == all_leg_bits & ~own_bit

    base = root.find(".//geom[@name='base_collision']")
    assert base is not None and "contype" not in base.attrib


def test_all_four_legs_are_required() -> None:
    with pytest.raises(ValueError, match="RL"):
        MODULE.configure_selective_self_collisions(_root(("FR", "FL", "RR")))


def test_simulation_contract_sets_hinge_joint_friction() -> None:
    root = ET.fromstring(
        """
        <mujoco>
          <default><default class="collision"><geom /></default></default>
          <worldbody>
            <body name="base">
              <joint name="hip" type="hinge" />
              <joint name="floating" type="free" />
            </body>
          </worldbody>
        </mujoco>
        """
    )
    MODULE.configure_simulation(root)

    hinge = root.find(".//joint[@name='hip']")
    free = root.find(".//joint[@name='floating']")
    assert hinge is not None and float(hinge.attrib["frictionloss"]) == MODULE.JOINT_FRICTION_LOSS_NM
    assert free is not None and "frictionloss" not in free.attrib

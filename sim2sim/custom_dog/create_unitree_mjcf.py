#!/usr/bin/env python3
"""Create the Unitree SDK2 bridge variant from the canonical MJCF."""

from __future__ import annotations

import argparse
import xml.etree.ElementTree as ET
from pathlib import Path


SDK_JOINT_ORDER = [
    "FR_hip_joint",
    "FR_thigh_joint",
    "FR_calf_joint",
    "FL_hip_joint",
    "FL_thigh_joint",
    "FL_calf_joint",
    "RR_hip_joint",
    "RR_thigh_joint",
    "RR_calf_joint",
    "RL_hip_joint",
    "RL_thigh_joint",
    "RL_calf_joint",
]


def motor_name(joint_name: str) -> str:
    return joint_name.removesuffix("_joint")


def replace_actuators(root: ET.Element) -> None:
    actuator = root.find("actuator")
    if actuator is None:
        raise ValueError("MJCF is missing the actuator section")
    actuator.clear()

    for joint_name in SDK_JOINT_ORDER:
        torque = 46.8 if "calf" in joint_name else 23.4
        limit = f"{-torque} {torque}"
        ET.SubElement(
            actuator,
            "motor",
            {
                "name": motor_name(joint_name),
                "joint": joint_name,
                "gear": "1",
                "ctrllimited": "true",
                "ctrlrange": limit,
                "forcelimited": "true",
                "forcerange": limit,
            },
        )


def replace_sensors(root: ET.Element) -> None:
    sensor = root.find("sensor")
    if sensor is None:
        sensor = ET.SubElement(root, "sensor")
    sensor.clear()

    # UnitreeSDK2Bridge reads the first 3 * nu values by raw address. Keep
    # position, velocity and actuator force in this exact order.
    for joint_name in SDK_JOINT_ORDER:
        ET.SubElement(
            sensor,
            "jointpos",
            {"name": f"{motor_name(joint_name)}_pos", "joint": joint_name},
        )
    for joint_name in SDK_JOINT_ORDER:
        ET.SubElement(
            sensor,
            "jointvel",
            {"name": f"{motor_name(joint_name)}_vel", "joint": joint_name},
        )
    for joint_name in SDK_JOINT_ORDER:
        ET.SubElement(
            sensor,
            "jointactuatorfrc",
            {"name": f"{motor_name(joint_name)}_torque", "joint": joint_name},
        )

    ET.SubElement(
        sensor,
        "framequat",
        {"name": "imu_quat", "objtype": "site", "objname": "base_imu_site"},
    )
    ET.SubElement(sensor, "gyro", {"name": "imu_gyro", "site": "base_imu_site"})
    ET.SubElement(sensor, "accelerometer", {"name": "imu_acc", "site": "base_imu_site"})
    ET.SubElement(
        sensor,
        "framepos",
        {"name": "frame_pos", "objtype": "site", "objname": "base_imu_site"},
    )
    ET.SubElement(
        sensor,
        "framelinvel",
        {"name": "frame_vel", "objtype": "site", "objname": "base_imu_site"},
    )


def configure_keyframe(root: ET.Element) -> None:
    home = root.find(".//key[@name='home']")
    if home is None:
        raise ValueError("MJCF is missing the home keyframe")
    home.attrib["ctrl"] = " ".join("0" for _ in SDK_JOINT_ORDER)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    tree = ET.parse(args.input)
    root = tree.getroot()
    root.attrib["model"] = "custom_dog_unitree"
    replace_actuators(root)
    replace_sensors(root)
    configure_keyframe(root)
    ET.indent(tree, space="  ")
    tree.write(args.output, encoding="utf-8", xml_declaration=True)


if __name__ == "__main__":
    main()

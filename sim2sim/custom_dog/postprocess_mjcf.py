#!/usr/bin/env python3
"""Apply the custom-dog simulation contract to converted MJCF."""

from __future__ import annotations

import argparse
import xml.etree.ElementTree as ET
from pathlib import Path


HOME_POSITION = {
    "FR_hip_joint": -0.1,
    "FR_thigh_joint": 0.8,
    "FR_calf_joint": -1.5,
    "FL_hip_joint": 0.1,
    "FL_thigh_joint": 0.8,
    "FL_calf_joint": -1.5,
    "RR_hip_joint": -0.1,
    "RR_thigh_joint": 0.8,
    "RR_calf_joint": -1.5,
    "RL_hip_joint": 0.1,
    "RL_thigh_joint": 0.8,
    "RL_calf_joint": -1.5,
}

FOOT_NAMES = ("FR_foot", "FL_foot", "RR_foot", "RL_foot")


def restore_inertials(urdf_root: ET.Element, mjcf_root: ET.Element) -> None:
    for link in urdf_root.findall("link"):
        link_name = link.attrib["name"]
        urdf_inertial = link.find("inertial")
        body = mjcf_root.find(f".//body[@name='{link_name}']")
        if urdf_inertial is None or body is None:
            raise ValueError(f"Missing inertial or MJCF body for link: {link_name}")

        origin = urdf_inertial.find("origin")
        mass = urdf_inertial.find("mass")
        inertia = urdf_inertial.find("inertia")
        if mass is None or inertia is None:
            raise ValueError(f"Incomplete URDF inertial for link: {link_name}")

        existing = body.find("inertial")
        insert_at = list(body).index(existing) if existing is not None else 0
        if existing is not None:
            body.remove(existing)

        inertial_rpy = origin.attrib.get("rpy", "0 0 0") if origin is not None else "0 0 0"
        if any(abs(float(value)) > 1e-12 for value in inertial_rpy.split()):
            raise ValueError(
                f"Link {link_name} uses a rotated, non-diagonal inertia tensor; rotate it into the link frame first"
            )

        attrs = {
            "pos": origin.attrib.get("xyz", "0 0 0") if origin is not None else "0 0 0",
            "mass": mass.attrib["value"],
            "fullinertia": " ".join(
                inertia.attrib[name] for name in ("ixx", "iyy", "izz", "ixy", "ixz", "iyz")
            ),
        }
        body.insert(insert_at, ET.Element("inertial", attrs))


def configure_simulation(mjcf_root: ET.Element) -> None:
    mjcf_root.attrib["model"] = "custom_dog"
    compiler = mjcf_root.find("compiler")
    if compiler is None:
        compiler = ET.Element("compiler")
        mjcf_root.insert(0, compiler)
    compiler.attrib.update({"angle": "radian", "autolimits": "true", "balanceinertia": "false"})

    option = mjcf_root.find("option")
    if option is None:
        option = ET.SubElement(mjcf_root, "option")
    option.attrib.update({"timestep": "0.005", "gravity": "0 0 -9.81", "integrator": "implicitfast"})

    collision = mjcf_root.find(".//default[@class='collision']/geom")
    if collision is None:
        raise ValueError("MJCF is missing the collision geom defaults")
    collision.attrib.update(
        {
            "condim": "3",
            "contype": "0",
            "conaffinity": "1",
            "friction": "1.0 0.01 0.01",
            "solimp": "0.99 0.999 0.00001",
            "solref": "0.005 1.0",
        }
    )


def add_observation_sensors(mjcf_root: ET.Element) -> None:
    base = mjcf_root.find(".//body[@name='base']")
    if base is None:
        raise ValueError("MJCF is missing the base body")

    site_names = {"base_imu_site", *(f"{name}_site" for name in FOOT_NAMES)}
    for body in mjcf_root.findall(".//body"):
        for site in list(body.findall("site")):
            if site.attrib.get("name") in site_names:
                body.remove(site)

    base.append(
        ET.Element(
            "site",
            {
                "name": "base_imu_site",
                "type": "sphere",
                "size": "0.005",
                "pos": "0 0 0",
                "rgba": "1 0.2 0.1 0.7",
                "group": "4",
            },
        )
    )
    for foot_name in FOOT_NAMES:
        foot = mjcf_root.find(f".//body[@name='{foot_name}']")
        if foot is None:
            raise ValueError(f"MJCF is missing foot body: {foot_name}")
        foot.append(
            ET.Element(
                "site",
                {
                    "name": f"{foot_name}_site",
                    "type": "sphere",
                    "size": "0.015",
                    "pos": "0 0 0",
                    "rgba": "0.1 0.8 0.2 0.5",
                    "group": "4",
                },
            )
        )

    existing = mjcf_root.find("sensor")
    if existing is not None:
        mjcf_root.remove(existing)
    sensors = ET.SubElement(mjcf_root, "sensor")
    ET.SubElement(sensors, "accelerometer", {"name": "base_acc", "site": "base_imu_site"})
    ET.SubElement(sensors, "gyro", {"name": "base_gyro", "site": "base_imu_site"})
    ET.SubElement(
        sensors,
        "framequat",
        {"name": "base_quat", "objtype": "site", "objname": "base_imu_site"},
    )
    for foot_name in FOOT_NAMES:
        ET.SubElement(
            sensors,
            "touch",
            {"name": f"{foot_name}_touch", "site": f"{foot_name}_site"},
        )


def add_home_keyframe(mjcf_root: ET.Element) -> None:
    joint_names = [joint.attrib["name"] for joint in mjcf_root.findall(".//joint")]
    missing = set(HOME_POSITION).difference(joint_names)
    if missing:
        raise ValueError(f"MJCF is missing policy joints: {sorted(missing)}")

    qpos = [0.0, 0.0, 0.324, 1.0, 0.0, 0.0, 0.0]
    qpos.extend(HOME_POSITION[name] for name in joint_names)

    keyframe = mjcf_root.find("keyframe")
    if keyframe is None:
        keyframe = ET.SubElement(mjcf_root, "keyframe")
    for key in list(keyframe):
        if key.attrib.get("name") == "home":
            keyframe.remove(key)
    ET.SubElement(keyframe, "key", {"name": "home", "qpos": " ".join(str(value) for value in qpos)})


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--urdf", type=Path, required=True)
    parser.add_argument("--mjcf", type=Path, required=True)
    args = parser.parse_args()

    urdf_tree = ET.parse(args.urdf)
    mjcf_tree = ET.parse(args.mjcf)
    root = mjcf_tree.getroot()
    restore_inertials(urdf_tree.getroot(), root)
    configure_simulation(root)
    add_observation_sensors(root)
    add_home_keyframe(root)
    ET.indent(mjcf_tree, space="  ")
    mjcf_tree.write(args.mjcf, encoding="utf-8", xml_declaration=True)


if __name__ == "__main__":
    main()

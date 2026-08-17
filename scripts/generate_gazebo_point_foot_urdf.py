#!/usr/bin/env python3
"""Generate the Gazebo point-foot model without thigh/calf ground colliders."""

from __future__ import annotations

import argparse
import xml.etree.ElementTree as ET
from pathlib import Path


LEGS = ("FR", "FL", "RR", "RL")
REMOVED_SUFFIXES = ("thigh", "calf")


def remove_leg_ground_colliders(root: ET.Element) -> int:
    """Remove exactly one thigh and calf collision from every leg."""

    removed = 0
    for leg in LEGS:
        for suffix in REMOVED_SUFFIXES:
            link_name = f"{leg}_{suffix}"
            link = root.find(f"./link[@name='{link_name}']")
            if link is None:
                raise ValueError(f"URDF is missing {link_name}")
            collisions = link.findall("./collision")
            if len(collisions) != 1:
                raise ValueError(
                    f"Expected one {link_name} collision, found {len(collisions)}"
                )
            link.remove(collisions[0])
            removed += 1
    return removed


def main() -> None:
    project_root = Path(__file__).resolve().parents[1]
    description_urdf = project_root / "ros2/src/custom_dog_description/urdf"
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        type=Path,
        default=description_urdf / "custom_dog.urdf",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=description_urdf / "custom_dog_gazebo_point_foot.urdf",
    )
    args = parser.parse_args()

    tree = ET.parse(args.input)
    count = remove_leg_ground_colliders(tree.getroot())
    args.output.parent.mkdir(parents=True, exist_ok=True)
    ET.indent(tree, space="  ")
    tree.write(args.output, encoding="utf-8", xml_declaration=True)
    print(f"Generated {args.output} without {count} thigh/calf colliders")


if __name__ == "__main__":
    main()

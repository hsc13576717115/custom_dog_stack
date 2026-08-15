#!/usr/bin/env python3
"""Generate the cleaned URDF used only by selective self-collision tasks."""

from __future__ import annotations

import argparse
import importlib.util
import xml.etree.ElementTree as ET
from pathlib import Path


def load_contract(path: Path):
    spec = importlib.util.spec_from_file_location("custom_dog_collision_contract", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load collision contract: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def apply_cleaned_collision_proxies(
    root: ET.Element, proxies: dict[str, dict[str, float]]
) -> int:
    updated = 0
    for suffix, proxy in proxies.items():
        for leg in ("FR", "FL", "RR", "RL"):
            link = root.find(f"./link[@name='{leg}_{suffix}']")
            if link is None:
                raise ValueError(f"URDF is missing {leg}_{suffix}")
            collision = link.find("./collision")
            if collision is None:
                raise ValueError(f"URDF is missing the {leg}_{suffix} collision")
            origin = collision.find("./origin")
            if origin is None:
                origin = ET.SubElement(collision, "origin")
            origin.attrib.update(
                {"xyz": f"0 0 {proxy['center_z']:.6g}", "rpy": "0 0 0"}
            )
            geometry = collision.find("./geometry")
            if geometry is None:
                geometry = ET.SubElement(collision, "geometry")
            geometry.clear()
            ET.SubElement(
                geometry,
                "cylinder",
                {
                    "radius": f"{proxy['radius']:.6g}",
                    "length": f"{proxy['length']:.6g}",
                },
            )
            updated += 1
    return updated


def main() -> None:
    project_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        type=Path,
        default=project_root / "ros2/src/custom_dog_description/urdf/custom_dog.urdf",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=project_root
        / "ros2/src/custom_dog_description/urdf/custom_dog_selective_collision.urdf",
    )
    args = parser.parse_args()

    contract = load_contract(
        project_root / "rl/src/custom_dog_rl/assets/collision_contract.py"
    )
    tree = ET.parse(args.input)
    proxies = {
        suffix: dict(proxy) for suffix, proxy in contract.LEG_COLLISION_PROXIES.items()
    }
    count = apply_cleaned_collision_proxies(tree.getroot(), proxies)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    ET.indent(tree, space="  ")
    tree.write(args.output, encoding="utf-8", xml_declaration=True)
    print(f"Generated {args.output} with {count} cleaned leg collision proxies")


if __name__ == "__main__":
    main()

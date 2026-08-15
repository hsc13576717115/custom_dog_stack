#!/usr/bin/env python3
"""Generate a MuJoCo model with ground contact and cross-leg self-collision."""

from __future__ import annotations

import argparse
import importlib.util
import xml.etree.ElementTree as ET
from pathlib import Path


def load_postprocessor(path: Path):
    spec = importlib.util.spec_from_file_location("custom_dog_postprocess_mjcf", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def apply_cleaned_leg_proxies(
    root: ET.Element, proxies: dict[str, dict[str, float]]
) -> int:
    updated = 0
    for suffix, proxy in proxies.items():
        start_z = proxy["center_z"] + 0.5 * proxy["length"]
        end_z = proxy["center_z"] - 0.5 * proxy["length"]
        for leg in ("FR", "FL", "RR", "RL"):
            geom = root.find(
                f".//body[@name='{leg}_{suffix}']/geom[@class='collision']"
            )
            if geom is None:
                raise ValueError(f"MJCF is missing the {leg}_{suffix} collision geom")
            for attribute in ("mesh", "pos", "quat", "euler", "axisangle"):
                geom.attrib.pop(attribute, None)
            geom.attrib.update(
                {
                    "type": "capsule",
                    "size": f"{proxy['radius']:.6g}",
                    "fromto": f"0 0 {start_z:.6g} 0 0 {end_z:.6g}",
                }
            )
            updated += 1
    return updated


def main() -> None:
    project_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        type=Path,
        default=project_root / "sim2sim/custom_dog/custom_dog.xml",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=project_root / "sim2sim/custom_dog/custom_dog_selective.xml",
    )
    args = parser.parse_args()

    tree = ET.parse(args.input)
    root = tree.getroot()
    compiler = root.find("compiler")
    if compiler is None:
        raise ValueError("MJCF must define compiler settings")
    meshdir = Path(compiler.attrib.get("meshdir", "."))
    if not meshdir.is_absolute():
        meshdir = (args.input.resolve().parent / meshdir).resolve()
    compiler.attrib["meshdir"] = str(meshdir)

    postprocessor = load_postprocessor(
        project_root / "sim2sim/custom_dog/postprocess_mjcf.py"
    )
    contract = load_postprocessor(
        project_root / "rl/src/custom_dog_rl/assets/collision_contract.py"
    )
    proxies = {
        suffix: dict(proxy) for suffix, proxy in contract.LEG_COLLISION_PROXIES.items()
    }
    apply_cleaned_leg_proxies(root, proxies)
    postprocessor.configure_selective_self_collisions(root)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    ET.indent(tree, space="  ")
    tree.write(args.output, encoding="utf-8", xml_declaration=True)
    print(f"Generated selective-collision MJCF: {args.output}")


if __name__ == "__main__":
    main()

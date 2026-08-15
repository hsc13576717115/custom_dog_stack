#!/usr/bin/env python3
"""Generate a diagnostic MJCF variant with explicit ground-contact parameters."""

from __future__ import annotations

import argparse
import xml.etree.ElementTree as ET
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--sliding-friction", type=float, required=True)
    parser.add_argument("--time-constant", type=float, required=True)
    parser.add_argument("--damping-ratio", type=float, default=1.0)
    args = parser.parse_args()
    if args.sliding_friction <= 0.0:
        parser.error("--sliding-friction must be positive")
    if args.time_constant <= 0.0 or args.damping_ratio <= 0.0:
        parser.error("--time-constant and --damping-ratio must be positive")

    tree = ET.parse(args.input)
    root = tree.getroot()
    compiler = root.find("compiler")
    if compiler is None:
        raise ValueError("MJCF must define compiler settings")
    source_meshdir = Path(compiler.attrib.get("meshdir", "."))
    if not source_meshdir.is_absolute():
        source_meshdir = (args.input.resolve().parent / source_meshdir).resolve()
    compiler.attrib["meshdir"] = str(source_meshdir)

    collision = root.find(".//default[@class='collision']/geom")
    floor = root.find(".//default[@class='floor']/geom")
    if collision is None or floor is None:
        raise ValueError("MJCF must define collision and floor geom defaults")

    for geom in (collision, floor):
        friction = geom.attrib.get("friction", "1 0.005 0.0001").split()
        friction[0] = str(args.sliding_friction)
        geom.attrib["friction"] = " ".join(friction)
        geom.attrib["solref"] = f"{args.time_constant} {args.damping_ratio}"

    args.output.parent.mkdir(parents=True, exist_ok=True)
    ET.indent(tree, space="  ")
    tree.write(args.output, encoding="utf-8", xml_declaration=True)
    print(
        f"wrote {args.output}: friction={args.sliding_friction}, "
        f"solref={args.time_constant} {args.damping_ratio}"
    )


if __name__ == "__main__":
    main()

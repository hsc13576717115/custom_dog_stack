"""Engine-independent selective self-collision contract for the custom dog."""

from __future__ import annotations

from itertools import combinations


LEG_NAMES = ("FR", "FL", "RR", "RL")
LEG_LINK_SUFFIXES = ("hip", "thigh", "calf", "foot")

# Raw CAD convex hulls extend into neighboring joints and near the foot
# contact surface.  These axis-aligned link proxies deliberately leave a gap
# at both joints.  URDF uses cylinders and MuJoCo uses equal-sized capsules;
# visuals, inertia and the dedicated foot collider remain unchanged.
LEG_COLLISION_PROXIES = {
    "thigh": {"radius": 0.035, "length": 0.170, "center_z": -0.090},
    "calf": {"radius": 0.022, "length": 0.150, "center_z": -0.090},
}


def filtered_body_pairs() -> tuple[tuple[str, str], ...]:
    """Return pairs that must not collide when cross-leg contact is enabled."""

    leg_links = {
        leg: tuple(f"{leg}_{suffix}" for suffix in LEG_LINK_SUFFIXES) for leg in LEG_NAMES
    }
    pairs = [("base", link) for links in leg_links.values() for link in links]
    for links in leg_links.values():
        pairs.extend(combinations(links, 2))
    return tuple(pairs)

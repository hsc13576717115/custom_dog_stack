from __future__ import annotations

import importlib.util
from itertools import combinations
from pathlib import Path
import ast


ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = ROOT / "rl/src/custom_dog_rl/assets/collision_contract.py"
SPEC = importlib.util.spec_from_file_location("custom_dog_collision_contract", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)

LEG_LINK_SUFFIXES = MODULE.LEG_LINK_SUFFIXES
LEG_NAMES = MODULE.LEG_NAMES
filtered_body_pairs = MODULE.filtered_body_pairs


def test_filters_base_and_same_leg_but_keeps_every_cross_leg_pair() -> None:
    filtered = {frozenset(pair) for pair in filtered_body_pairs()}
    assert len(filtered) == 40

    for leg in LEG_NAMES:
        links = tuple(f"{leg}_{suffix}" for suffix in LEG_LINK_SUFFIXES)
        for link in links:
            assert frozenset(("base", link)) in filtered
        for pair in combinations(links, 2):
            assert frozenset(pair) in filtered

    for leg_a, leg_b in combinations(LEG_NAMES, 2):
        for suffix_a in LEG_LINK_SUFFIXES:
            for suffix_b in LEG_LINK_SUFFIXES:
                pair = frozenset((f"{leg_a}_{suffix_a}", f"{leg_b}_{suffix_b}"))
                assert pair not in filtered


def test_filter_contract_has_no_duplicate_or_reflexive_pairs() -> None:
    pairs = filtered_body_pairs()
    assert len(pairs) == len({frozenset(pair) for pair in pairs})
    assert all(body_a != body_b for body_a, body_b in pairs)


def test_cleaned_leg_proxies_keep_visual_and_inertial_contract_unchanged() -> None:
    assert MODULE.LEG_COLLISION_PROXIES == {
        "thigh": {"radius": 0.035, "length": 0.17, "center_z": -0.09},
        "calf": {"radius": 0.022, "length": 0.15, "center_z": -0.09},
    }
    assert all(
        0.0 < value < 1.0
        for proxy in MODULE.LEG_COLLISION_PROXIES.values()
        for value in proxy.values()
        if value != proxy["center_z"]
    )


def test_isaac_spawner_authors_every_filter_on_both_endpoints() -> None:
    asset_path = ROOT / "rl/src/custom_dog_rl/assets/custom_dog.py"
    module = ast.parse(asset_path.read_text(encoding="utf-8"))
    function = next(
        node
        for node in module.body
        if isinstance(node, ast.FunctionDef)
        and node.name == "_spawn_custom_dog_with_selective_self_collisions"
    )
    source = ast.unparse(function)
    assert "((prim_a, prim_b), (prim_b, prim_a))" in source
    assert "AddTarget(Sdf.Path(target.GetPath()))" in source

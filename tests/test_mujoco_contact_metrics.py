from __future__ import annotations

import ast
from pathlib import Path
from types import SimpleNamespace

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RUNNER = PROJECT_ROOT / "sim2sim/custom_dog/run_sim2sim.py"


def _load_function(name: str):
    tree = ast.parse(RUNNER.read_text(encoding="utf-8"))
    function = next(
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == name
    )
    module = ast.Module(
        body=[
            ast.ImportFrom(
                module="__future__",
                names=[ast.alias(name="annotations")],
                level=0,
            ),
            function,
        ],
        type_ignores=[],
    )
    namespace = {"np": np}
    exec(compile(ast.fix_missing_locations(module), str(RUNNER), "exec"), namespace)
    return namespace[name]


def _contact(geom1: int, geom2: int, efc_address: int = 0) -> SimpleNamespace:
    return SimpleNamespace(geom1=geom1, geom2=geom2, efc_address=efc_address)


def test_ground_contacts_exclude_robot_self_collision() -> None:
    contacts = _load_function("ground_foot_contacts")
    # geom 0 is ground (world body 0), geoms 1..4 are four feet, and
    # geom 5 is another robot link.  The FR-foot/link contact must not count.
    geom_body_ids = np.asarray([0, 11, 12, 13, 14, 21], dtype=np.int32)
    foot_body_ids = np.asarray([11, 12, 13, 14], dtype=np.int32)
    result = contacts(
        [
            _contact(0, 1),
            _contact(2, 0),
            _contact(1, 5),
            _contact(0, 3, efc_address=-1),
        ],
        geom_body_ids,
        foot_body_ids,
    )

    np.testing.assert_array_equal(result, [True, True, False, False])


def test_self_contact_counter_excludes_world_same_body_and_inactive_contacts() -> None:
    count_pairs = _load_function("robot_self_contact_pair_count")
    geom_body_ids = np.asarray([0, 11, 11, 12, 13], dtype=np.int32)
    contacts = [
        _contact(0, 1),
        _contact(1, 2),
        _contact(2, 3),
        _contact(3, 4, efc_address=-1),
    ]
    assert count_pairs(contacts, geom_body_ids) == 1


def test_illegal_ground_counter_excludes_feet_and_self_contact() -> None:
    count_pairs = _load_function("illegal_ground_contact_pair_count")
    geom_body_ids = np.asarray([0, 11, 12, 21, 22], dtype=np.int32)
    foot_body_ids = np.asarray([11, 12], dtype=np.int32)
    contacts = [
        _contact(0, 1),
        _contact(2, 0),
        _contact(0, 3),
        _contact(3, 4),
        _contact(0, 4, efc_address=-1),
    ]
    assert count_pairs(contacts, geom_body_ids, foot_body_ids) == 1

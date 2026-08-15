#!/usr/bin/env python3
"""Compare a dumped Isaac PhysX contract with the compiled MuJoCo model."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import mujoco
import numpy as np


def object_name(model: mujoco.MjModel, object_type, index: int) -> str:
    name = mujoco.mj_id2name(model, object_type, index)
    if name is None:
        raise ValueError(f"Unnamed MuJoCo object: type={object_type}, index={index}")
    return name


def quaternion_matrix_wxyz(quaternion: np.ndarray) -> np.ndarray:
    w, x, y, z = quaternion / np.linalg.norm(quaternion)
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ],
        dtype=np.float64,
    )


def maximum_error(left: np.ndarray, right: np.ndarray) -> float:
    return float(np.max(np.abs(left - right)))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--isaac", type=Path, required=True)
    parser.add_argument("--mjcf", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    isaac = json.loads(args.isaac.read_text(encoding="utf-8"))
    model = mujoco.MjModel.from_xml_path(str(args.mjcf.resolve()))

    isaac_body_names = isaac["body_names"]
    isaac_body_ids = {name: index for index, name in enumerate(isaac_body_names)}
    mujoco_body_ids = {
        object_name(model, mujoco.mjtObj.mjOBJ_BODY, index): index
        for index in range(1, model.nbody)
    }
    if set(isaac_body_ids) != set(mujoco_body_ids):
        raise ValueError(
            f"Body-name mismatch: Isaac-only={set(isaac_body_ids) - set(mujoco_body_ids)}, "
            f"MuJoCo-only={set(mujoco_body_ids) - set(isaac_body_ids)}"
        )

    mass_errors = []
    com_errors = []
    inertia_errors = []
    inertia_relative_errors = []
    for name in isaac_body_names:
        isaac_id = isaac_body_ids[name]
        mujoco_id = mujoco_body_ids[name]
        isaac_mass = float(isaac["body"]["mass_kg"][isaac_id])
        mujoco_mass = float(model.body_mass[mujoco_id])
        mass_errors.append(abs(isaac_mass - mujoco_mass))

        isaac_com = np.asarray(isaac["body"]["com_pose_xyzw"][isaac_id][:3])
        mujoco_com = model.body_ipos[mujoco_id]
        com_errors.append(maximum_error(isaac_com, mujoco_com))

        isaac_inertia = np.asarray(isaac["body"]["inertia_matrix"][isaac_id]).reshape(3, 3)
        rotation = quaternion_matrix_wxyz(model.body_iquat[mujoco_id])
        mujoco_inertia = rotation @ np.diag(model.body_inertia[mujoco_id]) @ rotation.T
        inertia_error = maximum_error(isaac_inertia, mujoco_inertia)
        inertia_errors.append(inertia_error)
        inertia_relative_errors.append(inertia_error / max(float(np.max(np.abs(isaac_inertia))), 1e-12))

    isaac_joint_names = isaac["joint_names"]
    isaac_joint_ids = {name: index for index, name in enumerate(isaac_joint_names)}
    mujoco_joint_ids = {
        object_name(model, mujoco.mjtObj.mjOBJ_JOINT, index): index
        for index in range(model.njnt)
        if object_name(model, mujoco.mjtObj.mjOBJ_JOINT, index) != "floating_base"
    }
    if set(isaac_joint_ids) != set(mujoco_joint_ids):
        raise ValueError("Actuated joint names differ between Isaac and MuJoCo")

    mujoco_actuator_ids = {
        object_name(model, mujoco.mjtObj.mjOBJ_ACTUATOR, index): index
        for index in range(model.nu)
    }
    isaac_kp: dict[str, float] = {}
    isaac_kd: dict[str, float] = {}
    for actuator in isaac["actuator_models"].values():
        for index, name in enumerate(actuator["joint_names"]):
            isaac_kp[name] = float(actuator["stiffness"][index])
            isaac_kd[name] = float(actuator["damping"][index])

    limit_errors = []
    friction_errors = []
    kp_errors = []
    kd_errors = []
    for name in isaac_joint_names:
        isaac_id = isaac_joint_ids[name]
        mujoco_id = mujoco_joint_ids[name]
        actuator_id = mujoco_actuator_ids[name]
        limit_errors.append(
            maximum_error(
                np.asarray(isaac["joint"]["limits"][isaac_id]),
                model.jnt_range[mujoco_id],
            )
        )
        friction_errors.append(
            abs(float(isaac["joint"]["friction_properties"][isaac_id][0]) - model.dof_frictionloss[model.jnt_dofadr[mujoco_id]])
        )
        kp_errors.append(abs(isaac_kp[name] - model.actuator_gainprm[actuator_id, 0]))
        kd_errors.append(abs(isaac_kd[name] + model.actuator_biasprm[actuator_id, 2]))

    result = {
        "isaac_task": isaac["task"],
        "mjcf": str(args.mjcf.resolve()),
        "counts": {"bodies": len(isaac_body_names), "joints": len(isaac_joint_names)},
        "timestep_error_s": abs(float(isaac["sim"]["physics_dt_s"]) - float(model.opt.timestep)),
        "gravity_max_error_m_s2": maximum_error(np.asarray(isaac["sim"]["gravity"]), model.opt.gravity),
        "total_mass": {
            "isaac_kg": float(sum(isaac["body"]["mass_kg"])),
            "mujoco_kg": float(np.sum(model.body_mass[1:])),
        },
        "max_body_mass_error_kg": max(mass_errors),
        "max_com_component_error_m": max(com_errors),
        "max_inertia_component_error_kg_m2": max(inertia_errors),
        "max_inertia_relative_error": max(inertia_relative_errors),
        "max_joint_limit_error_rad": max(limit_errors),
        "max_joint_friction_error_nm": max(friction_errors),
        "max_position_gain_error_nm_rad": max(kp_errors),
        "max_velocity_gain_error_nm_s_rad": max(kd_errors),
        "contact": {
            "isaac_body_static_friction_range": [
                min(row[0] for row in isaac["body"]["material_properties"]),
                max(row[0] for row in isaac["body"]["material_properties"]),
            ],
            "isaac_body_dynamic_friction_range": [
                min(row[1] for row in isaac["body"]["material_properties"]),
                max(row[1] for row in isaac["body"]["material_properties"]),
            ],
            "isaac_contact_offset_range_m": [
                min(isaac["body"]["contact_offset_m"]),
                max(isaac["body"]["contact_offset_m"]),
            ],
            "mujoco_sliding_friction_range": [
                float(np.min(model.geom_friction[:, 0])),
                float(np.max(model.geom_friction[:, 0])),
            ],
            "mujoco_solref_ranges": {
                "time_constant_s": [float(np.min(model.geom_solref[:, 0])), float(np.max(model.geom_solref[:, 0]))],
                "damping_ratio": [float(np.min(model.geom_solref[:, 1])), float(np.max(model.geom_solref[:, 1]))],
            },
        },
    }
    payload = json.dumps(result, indent=2, sort_keys=True)
    print(payload)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(payload + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Validate the generated MJCF contract without opening a viewer."""

from __future__ import annotations

import argparse
from pathlib import Path

import mujoco
import numpy as np


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

HOME_POSITION = np.array(
    [-0.1, 0.8, -1.5, 0.1, 0.8, -1.5, -0.1, 0.8, -1.5, 0.1, 0.8, -1.5],
    dtype=np.float64,
)


def names(model: mujoco.MjModel, object_type: mujoco.mjtObj, count: int) -> list[str]:
    return [mujoco.mj_id2name(model, object_type, index) for index in range(count)]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("mjcf", type=Path)
    args = parser.parse_args()

    model = mujoco.MjModel.from_xml_path(str(args.mjcf.resolve()))
    joint_names = names(model, mujoco.mjtObj.mjOBJ_JOINT, model.njnt)
    actuator_names = names(model, mujoco.mjtObj.mjOBJ_ACTUATOR, model.nu)
    body_names = names(model, mujoco.mjtObj.mjOBJ_BODY, model.nbody)
    sensor_names = names(model, mujoco.mjtObj.mjOBJ_SENSOR, model.nsensor)

    assert model.nq == 19, f"expected nq=19, got {model.nq}"
    assert model.nv == 18, f"expected nv=18, got {model.nv}"
    assert model.nu == 12, f"expected nu=12, got {model.nu}"
    assert joint_names[0] == "floating_base", joint_names
    assert actuator_names == SDK_JOINT_ORDER, actuator_names
    assert set(SDK_JOINT_ORDER).issubset(joint_names), joint_names
    assert {"base", "FR_foot", "FL_foot", "RR_foot", "RL_foot"}.issubset(body_names)
    assert sensor_names == [
        "base_acc",
        "base_gyro",
        "base_quat",
        "FR_foot_touch",
        "FL_foot_touch",
        "RR_foot_touch",
        "RL_foot_touch",
    ], sensor_names
    assert model.nsensordata == 14, model.nsensordata
    assert abs(model.opt.timestep - 0.005) < 1e-12
    assert abs(float(model.body_mass.sum()) - 13.84916) < 1e-5, model.body_mass.sum()

    data = mujoco.MjData(model)
    home_key = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_KEY, "home")
    assert home_key >= 0
    mujoco.mj_resetDataKeyframe(model, data, home_key)
    data.ctrl[:] = HOME_POSITION
    for _ in range(200):
        mujoco.mj_step(model, data)
    assert np.isfinite(data.qpos).all()
    assert np.isfinite(data.qvel).all()

    print(f"MJCF OK: nq={model.nq}, nv={model.nv}, nu={model.nu}, bodies={model.nbody}")
    print(f"Total mass: {model.body_mass.sum():.5f} kg")
    print(f"Base height after 1 s standing smoke: {data.qpos[2]:.4f} m")


if __name__ == "__main__":
    main()

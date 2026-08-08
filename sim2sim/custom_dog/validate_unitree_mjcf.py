#!/usr/bin/env python3
"""Validate the raw-address contract required by UnitreeSDK2Bridge."""

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
    actuator_names = names(model, mujoco.mjtObj.mjOBJ_ACTUATOR, model.nu)
    sensor_names = names(model, mujoco.mjtObj.mjOBJ_SENSOR, model.nsensor)
    expected_motor_names = [name.removesuffix("_joint") for name in SDK_JOINT_ORDER]
    expected_motor_sensors = (
        [f"{name}_pos" for name in expected_motor_names]
        + [f"{name}_vel" for name in expected_motor_names]
        + [f"{name}_torque" for name in expected_motor_names]
    )
    expected_imu_sensors = ["imu_quat", "imu_gyro", "imu_acc", "frame_pos", "frame_vel"]

    assert model.nq == 19, model.nq
    assert model.nv == 18, model.nv
    assert model.nu == 12, model.nu
    assert actuator_names == expected_motor_names, actuator_names
    assert sensor_names == expected_motor_sensors + expected_imu_sensors, sensor_names
    assert model.sensor_adr[:36].tolist() == list(range(36)), model.sensor_adr[:36]
    assert model.sensor_adr[36:].tolist() == [36, 40, 43, 46, 49], model.sensor_adr[36:]
    assert model.nsensordata == 52, model.nsensordata
    assert np.all(model.actuator_biastype == mujoco.mjtBias.mjBIAS_NONE)
    assert np.all(model.actuator_gaintype == mujoco.mjtGain.mjGAIN_FIXED)

    data = mujoco.MjData(model)
    home_key = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_KEY, "home")
    mujoco.mj_resetDataKeyframe(model, data, home_key)
    mujoco.mj_forward(model, data)
    for _ in range(2000):
        position = data.sensordata[:12]
        velocity = data.sensordata[12:24]
        torque = 25.0 * (HOME_POSITION - position) - 0.5 * velocity
        data.ctrl[:] = np.clip(torque, model.actuator_ctrlrange[:, 0], model.actuator_ctrlrange[:, 1])
        mujoco.mj_step(model, data)

    assert np.isfinite(data.qpos).all()
    assert np.isfinite(data.qvel).all()
    print("Unitree bridge MJCF OK: 12 torque motors, 36 address-ordered motor states, standard IMU")
    print(f"Total mass: {model.body_mass.sum():.5f} kg")
    print(f"Base height after 10 s SDK2-equivalent PD control: {data.qpos[2]:.4f} m")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Validate the custom-dog selective self-collision MJCF contract."""

from __future__ import annotations

import argparse
from pathlib import Path

import mujoco
import numpy as np


SDK_JOINT_ORDER = (
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
)
HOME_SDK = np.asarray([-0.1, 0.8, -1.5, 0.1, 0.8, -1.5] * 2)


def active_self_contacts(model: mujoco.MjModel, data: mujoco.MjData) -> list[tuple[str, str]]:
    pairs = []
    for contact in data.contact:
        if contact.efc_address < 0:
            continue
        body1 = int(model.geom_bodyid[contact.geom1])
        body2 = int(model.geom_bodyid[contact.geom2])
        if body1 == 0 or body2 == 0:
            continue
        pairs.append(
            (
                mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, contact.geom1),
                mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, contact.geom2),
            )
        )
    return pairs


def set_joint(model: mujoco.MjModel, data: mujoco.MjData, name: str, value: float) -> None:
    joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
    if joint_id < 0:
        raise ValueError(f"MJCF is missing joint: {name}")
    data.qpos[model.jnt_qposadr[joint_id]] = value


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mjcf", type=Path)
    parser.add_argument("--settle-seconds", type=float, default=2.0)
    args = parser.parse_args()

    model = mujoco.MjModel.from_xml_path(str(args.mjcf.resolve()))
    data = mujoco.MjData(model)
    home_key = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_KEY, "home")
    if home_key < 0:
        raise SystemExit("MJCF is missing the home keyframe")
    mujoco.mj_resetDataKeyframe(model, data, home_key)
    data.ctrl[:] = HOME_SDK

    self_contact_steps = 0
    steps = round(args.settle_seconds / model.opt.timestep)
    for _ in range(steps):
        mujoco.mj_step(model, data)
        self_contact_steps += bool(active_self_contacts(model, data))
    if self_contact_steps:
        raise SystemExit(
            f"Nominal stance produced self-contact during {self_contact_steps}/{steps} steps"
        )

    mujoco.mj_resetDataKeyframe(model, data, home_key)
    set_joint(model, data, "FR_hip_joint", 0.8)
    set_joint(model, data, "FL_hip_joint", -0.8)
    mujoco.mj_forward(model, data)
    forced_pairs = active_self_contacts(model, data)
    if not forced_pairs:
        raise SystemExit("Forced front-leg crossing did not produce any self-contact")

    print(
        f"selective collision OK: nominal_self_contact_steps=0/{steps}, "
        f"forced_cross_leg_pairs={len(forced_pairs)}"
    )


if __name__ == "__main__":
    main()

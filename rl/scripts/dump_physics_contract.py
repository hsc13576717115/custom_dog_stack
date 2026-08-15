#!/usr/bin/env python3
"""Dump the compiled PhysX articulation contract for cross-simulator comparison."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from isaaclab.app import AppLauncher


parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument(
    "--task",
    default="CustomDog-Velocity-OmniTrot-ClosedLoopFoundation-v1",
)
parser.add_argument("--output", type=Path, required=True)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()

app_launcher = AppLauncher(args)
simulation_app = app_launcher.app


import gymnasium as gym
import omni.usd
import torch

import custom_dog_rl.tasks  # noqa: F401, E402
import isaaclab_tasks  # noqa: F401, E402
import unitree_rl_lab.tasks  # noqa: F401, E402
from unitree_rl_lab.utils.parser_cfg import parse_env_cfg  # noqa: E402


def tensor_values(value: torch.Tensor) -> list:
    return value.detach().cpu().tolist()


def first_environment(value: torch.Tensor) -> list:
    return tensor_values(value[0])


def serializable_actuator_value(value):
    """Convert resolved actuator values without assuming a scalar config."""
    if isinstance(value, torch.Tensor):
        return first_environment(value)
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


def actuator_contract(robot) -> dict[str, dict[str, object]]:
    result: dict[str, dict[str, object]] = {}
    for name, actuator in robot.actuators.items():
        contract = {
            "class": actuator.__class__.__name__,
            "joint_names": list(actuator.joint_names),
        }
        for field in (
            "stiffness",
            "damping",
            "armature",
            "friction",
            "dynamic_friction",
            "viscous_friction",
            "effort_limit",
            "effort_limit_sim",
            "velocity_limit",
            "velocity_limit_sim",
        ):
            if hasattr(actuator, field):
                contract[field] = serializable_actuator_value(getattr(actuator, field))
        for field in ("min_delay", "max_delay", "X1", "X2", "Y1", "Y2", "Fs", "Fd", "Va"):
            if hasattr(actuator.cfg, field):
                contract[field] = serializable_actuator_value(getattr(actuator.cfg, field))
        result[name] = contract
    return result


def disable_startup_randomization(env_cfg) -> None:
    for name in (
        "physics_material",
        "add_base_mass",
        "scale_body_mass",
        "randomize_base_com",
        "randomize_actuator_gains",
        "randomize_joint_friction",
    ):
        if hasattr(env_cfg.events, name):
            setattr(env_cfg.events, name, None)


def usd_joint_contract() -> dict[str, dict[str, object]]:
    stage = omni.usd.get_context().get_stage()
    joints: dict[str, dict[str, object]] = {}
    for prim in stage.Traverse():
        if prim.GetTypeName() != "PhysicsRevoluteJoint":
            continue
        name = prim.GetName()
        if not name.endswith("_joint"):
            continue
        attributes: dict[str, object] = {"prim_path": str(prim.GetPath())}
        for attribute_name in (
            "physics:axis",
            "physics:localPos0",
            "physics:localPos1",
            "physics:localRot0",
            "physics:localRot1",
            "physics:lowerLimit",
            "physics:upperLimit",
        ):
            attribute = prim.GetAttribute(attribute_name)
            if attribute.IsValid() and attribute.HasAuthoredValueOpinion():
                value = attribute.Get()
                if hasattr(value, "GetReal") and hasattr(value, "GetImaginary"):
                    imaginary = value.GetImaginary()
                    value = [float(value.GetReal()), *(float(item) for item in imaginary)]
                elif not isinstance(value, (str, int, float, bool)) and value is not None:
                    try:
                        value = [float(item) for item in value]
                    except TypeError:
                        value = str(value)
                attributes[attribute_name.removeprefix("physics:")] = value
        joints[name] = attributes
    return joints


def main() -> None:
    env_cfg = parse_env_cfg(
        args.task,
        device=args.device,
        num_envs=1,
        use_fabric=False,
        entry_point_key="play_env_cfg_entry_point",
    )
    disable_startup_randomization(env_cfg)
    env_cfg.scene.terrain.terrain_generator.num_rows = 1
    env_cfg.scene.terrain.terrain_generator.num_cols = 1
    env = gym.make(args.task, cfg=env_cfg)
    try:
        robot = env.unwrapped.scene["robot"]
        view = robot.root_physx_view
        friction_properties = view.get_dof_friction_properties()
        material_properties = view.get_material_properties()
        result = {
            "task": args.task,
            "sim": {
                "physics_dt_s": float(env.unwrapped.physics_dt),
                "step_dt_s": float(env.unwrapped.step_dt),
                "decimation": int(round(env.unwrapped.step_dt / env.unwrapped.physics_dt)),
                "gravity": list(env_cfg.sim.gravity),
            },
            "body_names": list(robot.body_names),
            "joint_names": list(robot.joint_names),
            "body": {
                "mass_kg": first_environment(view.get_masses()),
                "com_pose_xyzw": first_environment(view.get_coms()),
                "inertia_matrix": first_environment(view.get_inertias()),
                "material_properties": first_environment(material_properties),
                "rest_offset_m": first_environment(view.get_rest_offsets()),
                "contact_offset_m": first_environment(view.get_contact_offsets()),
            },
            "joint": {
                "stiffness": first_environment(view.get_dof_stiffnesses()),
                "damping": first_environment(view.get_dof_dampings()),
                "armature": first_environment(view.get_dof_armatures()),
                "max_force": first_environment(view.get_dof_max_forces()),
                "max_velocity": first_environment(view.get_dof_max_velocities()),
                "limits": first_environment(view.get_dof_limits()),
                "friction_properties": first_environment(friction_properties),
            },
            "actuator_models": actuator_contract(robot),
            "usd_revolute_joints": usd_joint_contract(),
        }
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=2, sort_keys=True), encoding="utf-8")
        print(f"physics contract written: {args.output.resolve()}")
        print(f"total mass: {sum(result['body']['mass_kg']):.8f} kg")
        print(f"bodies={len(robot.body_names)}, joints={len(robot.joint_names)}")
    finally:
        env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()

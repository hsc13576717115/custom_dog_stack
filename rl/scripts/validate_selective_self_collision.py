#!/usr/bin/env python3
"""Validate the selective self-collision contract in a live Isaac simulation."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

from isaaclab.app import AppLauncher


parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument(
    "--task",
    default="CustomDog-Velocity-OmniTrot-ClosedLoopSelectiveCollision-v1",
)
parser.add_argument("--output", type=Path, required=True)
parser.add_argument("--settle-seconds", type=float, default=2.0)
parser.add_argument("--force-threshold", type=float, default=1.0)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()

app_launcher = AppLauncher(args)
simulation_app = app_launcher.app


import gymnasium as gym
import omni.usd
import torch
from pxr import UsdPhysics

import custom_dog_rl.tasks  # noqa: F401, E402
import isaaclab_tasks  # noqa: F401, E402
import unitree_rl_lab.tasks  # noqa: F401, E402
from custom_dog_rl.assets.collision_contract import filtered_body_pairs  # noqa: E402
from unitree_rl_lab.utils.parser_cfg import parse_env_cfg  # noqa: E402


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


def validate_filtered_pairs(robot_prim_path: str) -> int:
    stage = omni.usd.get_context().get_stage()
    count = 0
    for body_a, body_b in filtered_body_pairs():
        prim_a = stage.GetPrimAtPath(f"{robot_prim_path}/{body_a}")
        prim_b = stage.GetPrimAtPath(f"{robot_prim_path}/{body_b}")
        for source_name, source, target_name, target in (
            (body_a, prim_a, body_b, prim_b),
            (body_b, prim_b, body_a, prim_a),
        ):
            api = UsdPhysics.FilteredPairsAPI.Get(stage, source.GetPath())
            targets = api.GetFilteredPairsRel().GetTargets() if api else []
            if target.GetPath() not in targets:
                raise RuntimeError(
                    f"Missing reciprocal filtered pair: {source_name} -> {target_name}"
                )
        count += 1
    return count


def step_scene(
    env,
    joint_targets: torch.Tensor,
    steps: int,
    root_pose: torch.Tensor | None = None,
    root_velocity: torch.Tensor | None = None,
) -> list[torch.Tensor]:
    scene = env.unwrapped.scene
    robot = scene["robot"]
    sensor = scene["contact_forces"]
    samples = []
    for _ in range(steps):
        if root_pose is None:
            robot.set_joint_position_target(joint_targets)
        else:
            if root_velocity is None:
                raise ValueError("root_velocity is required when root_pose is held")
            robot.write_root_pose_to_sim(root_pose)
            robot.write_root_velocity_to_sim(root_velocity)
            robot.write_joint_state_to_sim(
                joint_targets,
                torch.zeros_like(robot.data.default_joint_vel),
            )
        scene.write_data_to_sim()
        env.unwrapped.sim.step()
        scene.update(env.unwrapped.physics_dt)
        samples.append(sensor.data.net_forces_w[0].clone())
    return samples


def main() -> int:
    if args.settle_seconds <= 0.0:
        raise ValueError("--settle-seconds must be positive")
    if args.force_threshold <= 0.0:
        raise ValueError("--force-threshold must be positive")

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
    asset_path = Path(env_cfg.scene.robot.spawn.asset_path).resolve()
    asset_sha256 = hashlib.sha256(asset_path.read_bytes()).hexdigest()
    env = gym.make(args.task, cfg=env_cfg)
    try:
        env.reset()
        scene = env.unwrapped.scene
        robot = scene["robot"]
        sensor = scene["contact_forces"]
        filtered_count = validate_filtered_pairs("/World/envs/env_0/Robot")

        steps = round(args.settle_seconds / env.unwrapped.physics_dt)
        nominal_targets = robot.data.default_joint_pos.clone()
        nominal_root_pose = robot.data.default_root_state[:, :7].clone()
        nominal_root_pose[:, :3] += scene.env_origins
        nominal_root_velocity = torch.zeros_like(robot.data.default_root_state[:, 7:13])
        nominal_vectors = step_scene(
            env,
            nominal_targets,
            steps,
            root_pose=nominal_root_pose,
            root_velocity=nominal_root_velocity,
        )
        nonfoot_ids = [
            index for index, name in enumerate(sensor.body_names) if not name.endswith("_foot")
        ]
        evaluation_vectors = nominal_vectors[len(nominal_vectors) // 2 :]
        evaluation_norms = [
            torch.linalg.vector_norm(sample, dim=-1) for sample in evaluation_vectors
        ]
        per_body = {}
        for body_id in nonfoot_ids:
            body_force_vectors = [sample[body_id] for sample in evaluation_vectors]
            body_forces = [float(torch.linalg.vector_norm(vector).item()) for vector in body_force_vectors]
            peak_index = max(range(len(body_forces)), key=body_forces.__getitem__)
            mean_vector = torch.mean(torch.stack(body_force_vectors), dim=0)
            per_body[sensor.body_names[body_id]] = {
                "contact_steps": sum(force > args.force_threshold for force in body_forces),
                "max_force_n": max(body_forces),
                "peak_net_force_w_n": [
                    float(value) for value in body_force_vectors[peak_index].tolist()
                ],
                "mean_net_force_w_n": [float(value) for value in mean_vector.tolist()],
            }
        nominal_nonfoot_max = max(
            float(sample[nonfoot_ids].max().item()) for sample in evaluation_norms
        )
        nominal_nonfoot_steps = sum(
            bool((sample[nonfoot_ids] > args.force_threshold).any().item())
            for sample in evaluation_norms
        )

        root_pose = robot.data.default_root_state[:, :7].clone()
        root_pose[:, :3] += scene.env_origins
        root_pose[:, 2] = scene.env_origins[:, 2] + 0.60
        root_velocity = torch.zeros_like(robot.data.default_root_state[:, 7:13])
        crossed_targets = robot.data.default_joint_pos.clone()
        fr_hip_ids, _ = robot.find_joints("FR_hip_joint")
        fl_hip_ids, _ = robot.find_joints("FL_hip_joint")
        crossed_targets[:, fr_hip_ids] = 0.8
        crossed_targets[:, fl_hip_ids] = -0.8

        forced_max = 0.0
        forced_contact_steps = 0
        forced_steps = max(20, round(0.25 / env.unwrapped.physics_dt))
        for _ in range(forced_steps):
            sample = step_scene(
                env,
                crossed_targets,
                1,
                root_pose=root_pose,
                root_velocity=root_velocity,
            )[0]
            sample_norm = torch.linalg.vector_norm(sample, dim=-1)
            forced_max = max(forced_max, float(sample_norm.max().item()))
            forced_contact_steps += bool((sample_norm > args.force_threshold).any().item())

        failures = []
        if nominal_nonfoot_steps:
            failures.append(
                "nominal stance produced persistent non-foot contact: "
                f"{nominal_nonfoot_steps}/{len(evaluation_norms)} steps, "
                f"max={nominal_nonfoot_max:.3f} N"
            )
        if not forced_contact_steps:
            failures.append("forced front-leg crossing did not produce contact")

        result = {
            "task": args.task,
            "asset_path": str(asset_path),
            "asset_sha256": asset_sha256,
            "filtered_pair_count": filtered_count,
            "expected_filtered_pair_count": len(filtered_body_pairs()),
            "nominal_evaluation_steps": len(evaluation_norms),
            "nominal_pose_held": True,
            "nominal_root_height_m": float(nominal_root_pose[0, 2].item()),
            "nominal_nonfoot_contact_steps": nominal_nonfoot_steps,
            "nominal_nonfoot_max_force_n": nominal_nonfoot_max,
            "nominal_nonfoot_by_body": per_body,
            "forced_cross_leg_steps": forced_steps,
            "forced_cross_leg_contact_steps": forced_contact_steps,
            "forced_cross_leg_max_force_n": forced_max,
            "force_threshold_n": args.force_threshold,
            "accepted": not failures,
            "failures": failures,
        }
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=2, sort_keys=True), encoding="utf-8")

        if failures:
            print("selective Isaac collision FAILED: " + "; ".join(failures), file=sys.stderr)
            return 1
        print(
            "selective Isaac collision OK: "
            f"filtered_pairs={filtered_count}, nominal_nonfoot_steps=0/"
            f"{len(evaluation_norms)}, forced_contact_steps={forced_contact_steps}/"
            f"{forced_steps}, forced_max={forced_max:.3f} N"
        )
        return 0
    finally:
        env.close()


if __name__ == "__main__":
    exit_code = 1
    try:
        exit_code = main()
    finally:
        simulation_app.close()
    raise SystemExit(exit_code)

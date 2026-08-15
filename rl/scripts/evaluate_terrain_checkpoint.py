#!/usr/bin/env python3
"""Evaluate one terrain checkpoint on grouped fixed commands in one Isaac run."""

from __future__ import annotations

import argparse
import json
from importlib.metadata import version
from pathlib import Path

import custom_dog_rl.tasks  # noqa: F401
from isaaclab.app import AppLauncher

import cli_args  # isort: skip


COMMANDS = {
    "T0": (
        (0.10, 0.0, 0.0),
        (0.50, 0.0, 0.0),
        (-0.50, 0.0, 0.0),
        (0.0, 0.15, 0.0),
        (0.0, 0.0, 0.35),
        (0.40, 0.10, 0.20),
    ),
    "T1": (
        (0.10, 0.0, 0.0),
        (1.00, 0.0, 0.0),
        (-1.00, 0.0, 0.0),
        (0.0, 0.30, 0.0),
        (0.0, 0.0, 0.80),
        (0.80, 0.25, 0.50),
    ),
}


parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--stage", choices=tuple(COMMANDS), required=True)
parser.add_argument("--task", required=True)
parser.add_argument("--output", type=Path, required=True)
parser.add_argument("--num_envs", type=int, default=128)
parser.add_argument("--max_steps", type=int, default=600)
parser.add_argument("--warmup_steps", type=int, default=100)
parser.add_argument("--rl_device", default="cuda:0")
cli_args.add_rsl_rl_args(parser)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
if args.checkpoint is None:
    parser.error("--checkpoint is required")
if args.num_envs < len(COMMANDS[args.stage]):
    parser.error("--num_envs must be at least the number of fixed commands")
if not 0 <= args.warmup_steps < args.max_steps:
    parser.error("warmup steps must be non-negative and less than max steps")

app_launcher = AppLauncher(args)
simulation_app = app_launcher.app


import gymnasium as gym
import torch
from rsl_rl.runners import DistillationRunner, OnPolicyRunner

import isaaclab_tasks  # noqa: F401, E402
import unitree_rl_lab.tasks  # noqa: F401, E402
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper
from unitree_rl_lab.utils.parser_cfg import parse_env_cfg


def absolute_gates(row: dict[str, object]) -> dict[str, bool]:
    command = row["command"]
    moving = sum(abs(float(value)) for value in command) > 1.0e-6
    return {
        "success_rate": float(row["success_rate"]) >= 0.95,
        "vx": float(row["error_vx_m_s"]) <= max(0.15, 0.25 * abs(float(command[0]))),
        "vy": float(row["error_vy_m_s"]) <= 0.12,
        "wz": float(row["error_wz_rad_s"]) <= 0.15,
        "height_p05": float(row["height_p05_m"]) >= 0.20,
        "tilt_p95": float(row["tilt_p95_deg"]) <= 15.0,
        "gait_transitions": (
            int(row["min_contact_transitions"]) >= 2 if moving else True
        ),
    }


def terrain_family_names(generator_cfg) -> tuple[str, ...]:
    """Mirror Isaac Lab's curriculum column-to-sub-terrain assignment."""

    weighted_names = [
        (name, float(cfg.proportion))
        for name, cfg in generator_cfg.sub_terrains.items()
    ]
    total = sum(weight for _, weight in weighted_names)
    if total <= 0.0:
        raise ValueError("terrain proportions must sum to a positive value")
    cumulative = []
    running = 0.0
    for name, weight in weighted_names:
        running += weight / total
        cumulative.append((name, running))

    names = []
    for column in range(generator_cfg.num_cols):
        position = column / generator_cfg.num_cols + 0.001
        names.append(next(name for name, upper in cumulative if position < upper))
    return tuple(names)


def main() -> None:
    env_cfg = parse_env_cfg(
        args.task,
        device=args.device,
        num_envs=args.num_envs,
        use_fabric=True,
        entry_point_key="env_cfg_entry_point",
    )
    terrain_generator_cfg = env_cfg.scene.terrain.terrain_generator
    if terrain_generator_cfg is None or not terrain_generator_cfg.curriculum:
        raise ValueError("terrain evaluation requires a curriculum terrain generator")
    env_cfg.scene.terrain.max_init_terrain_level = terrain_generator_cfg.num_rows - 1
    family_by_type = terrain_family_names(terrain_generator_cfg)
    agent_cfg = cli_args.parse_rsl_rl_cfg(args.task, args)
    agent_cfg.device = args.rl_device
    env = gym.make(args.task, cfg=env_cfg)
    env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)
    runner_class = (
        DistillationRunner
        if getattr(agent_cfg, "class_name", "OnPolicyRunner") == "DistillationRunner"
        else OnPolicyRunner
    )
    runner = runner_class(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    checkpoint = Path(args.checkpoint).resolve()
    runner.load(str(checkpoint), map_location=agent_cfg.device)
    policy = runner.get_inference_policy(device=env.unwrapped.device)

    commands = torch.tensor(COMMANDS[args.stage], device=env.unwrapped.device)
    terrain_types = env.unwrapped.scene.terrain.terrain_types
    family_names = tuple(dict.fromkeys(family_by_type))
    family_ids = torch.full(
        (env.num_envs,), -1, dtype=torch.long, device=env.unwrapped.device
    )
    group_ids = torch.empty(
        env.num_envs, dtype=torch.long, device=env.unwrapped.device
    )
    for family_index, family in enumerate(family_names):
        type_ids = [
            index
            for index, type_family in enumerate(family_by_type)
            if type_family == family
        ]
        family_mask = torch.zeros(
            env.num_envs, dtype=torch.bool, device=env.unwrapped.device
        )
        for type_id in type_ids:
            family_mask |= terrain_types == type_id
        family_env_ids = torch.nonzero(family_mask, as_tuple=False).squeeze(-1)
        if len(family_env_ids) < len(commands):
            raise RuntimeError(
                f"terrain family {family!r} has {len(family_env_ids)} environments; "
                f"need at least {len(commands)} for a command-stratified evaluation"
            )
        family_ids[family_env_ids] = family_index
        group_ids[family_env_ids] = (
            torch.arange(len(family_env_ids), device=env.unwrapped.device)
            % len(commands)
        )
    if torch.any(family_ids < 0):
        raise RuntimeError("some evaluation environments were not assigned to a terrain family")
    command_term = env.unwrapped.command_manager.get_term("base_velocity")

    def apply_commands() -> None:
        command_term.vel_command_b[:] = commands[group_ids]
        if hasattr(command_term, "_sampled_vel_command_b"):
            command_term._sampled_vel_command_b[:] = commands[group_ids]
        command_term.is_standing_env[:] = False

    apply_commands()
    obs = env.get_observations()
    if version("rsl-rl-lib").startswith("2.3."):
        obs, _ = obs

    robot = env.unwrapped.scene["robot"]
    contact = env.unwrapped.scene["contact_forces"]
    foot_ids = [
        contact.body_names.index(name)
        for name in ("FR_foot", "FL_foot", "RR_foot", "RL_foot")
    ]
    ever_done = torch.zeros(env.num_envs, dtype=torch.bool, device=env.unwrapped.device)
    previous_contact = None
    transitions = torch.zeros(
        env.num_envs, 4, dtype=torch.int64, device=env.unwrapped.device
    )
    velocity_samples: list[torch.Tensor] = []
    height_samples: list[torch.Tensor] = []
    tilt_samples: list[torch.Tensor] = []

    for step in range(args.max_steps):
        with torch.inference_mode():
            actions = policy(obs)
            obs, _, dones, _ = env.step(actions)
            ever_done |= dones.bool()
            apply_commands()
            obs = env.get_observations()
            if version("rsl-rl-lib").startswith("2.3."):
                obs, _ = obs
            foot_force = torch.linalg.vector_norm(
                contact.data.net_forces_w[:, foot_ids], dim=-1
            )
            current_contact = foot_force > 1.0
            if previous_contact is not None and step >= args.warmup_steps:
                transitions += current_contact != previous_contact
            previous_contact = current_contact
            if step >= args.warmup_steps:
                velocity_samples.append(
                    torch.cat((robot.data.root_lin_vel_b[:, :2], robot.data.root_ang_vel_b[:, 2:3]), dim=1)
                )
                height_samples.append(robot.data.root_pos_w[:, 2].clone())
                tilt_samples.append(
                    torch.rad2deg(
                        torch.acos(
                            torch.clamp(-robot.data.projected_gravity_b[:, 2], -1.0, 1.0)
                        )
                    )
                )

    velocity = torch.stack(velocity_samples)
    height = torch.stack(height_samples)
    tilt = torch.stack(tilt_samples)
    def command_metrics(
        command_index: int,
        mask: torch.Tensor,
        family: str | None = None,
    ) -> dict[str, object]:
        if not mask.any():
            raise RuntimeError(
                f"no environments for command {command_index}"
                + (f" on terrain family {family}" if family is not None else "")
            )
        measured = velocity[:, mask].mean(dim=(0, 1))
        command_tensor = commands[command_index]
        row = {
            "command_index": command_index,
            "command": list(COMMANDS[args.stage][command_index]),
            "num_envs": int(mask.sum().item()),
            "measured_vx_m_s": float(measured[0].item()),
            "measured_vy_m_s": float(measured[1].item()),
            "measured_wz_rad_s": float(measured[2].item()),
            "error_vx_m_s": float(torch.abs(measured[0] - command_tensor[0]).item()),
            "error_vy_m_s": float(torch.abs(measured[1] - command_tensor[1]).item()),
            "error_wz_rad_s": float(torch.abs(measured[2] - command_tensor[2]).item()),
            "height_p05_m": float(torch.quantile(height[:, mask], 0.05).item()),
            "tilt_p95_deg": float(torch.quantile(tilt[:, mask], 0.95).item()),
            "environments_terminated": int(ever_done[mask].sum().item()),
            "success_rate": float((~ever_done[mask]).float().mean().item()),
            "min_contact_transitions": int(transitions[mask].min().item()),
        }
        if family is not None:
            row["family"] = family
        row["gates"] = absolute_gates(row)
        row["passed"] = all(row["gates"].values())
        return row

    rows = [
        command_metrics(command_index, group_ids == command_index)
        for command_index in range(len(commands))
    ]

    family_rows = []
    family_command_rows = []
    for family_index, family in enumerate(family_names):
        type_ids = [index for index, value in enumerate(family_by_type) if value == family]
        family_mask = family_ids == family_index
        if not family_mask.any():
            raise RuntimeError(f"terrain family {family!r} has no evaluation environments")
        family_row = {
            "family": family,
            "terrain_type_ids": type_ids,
            "num_envs": int(family_mask.sum().item()),
            "success_rate": float((~ever_done[family_mask]).float().mean().item()),
            "height_p05_m": float(torch.quantile(height[:, family_mask], 0.05).item()),
            "tilt_p95_deg": float(torch.quantile(tilt[:, family_mask], 0.95).item()),
            "min_contact_transitions": int(transitions[family_mask].min().item()),
        }
        family_row["gates"] = {
            "success_rate": family_row["success_rate"] >= 0.95,
            "height_p05": family_row["height_p05_m"] >= 0.20,
            "tilt_p95": family_row["tilt_p95_deg"] <= 15.0,
            "gait_transitions": family_row["min_contact_transitions"] >= 2,
        }
        family_row["passed"] = all(family_row["gates"].values())
        family_rows.append(family_row)
        for command_index in range(len(commands)):
            family_command_rows.append(
                command_metrics(
                    command_index,
                    family_mask & (group_ids == command_index),
                    family=family,
                )
            )

    result = {
        "stage": args.stage,
        "task": args.task,
        "checkpoint": str(checkpoint),
        "commands_passed": sum(bool(row["passed"]) for row in rows),
        "total_commands": len(rows),
        "terrain_families_passed": sum(bool(row["passed"]) for row in family_rows),
        "total_terrain_families": len(family_rows),
        "family_commands_passed": sum(
            bool(row["passed"]) for row in family_command_rows
        ),
        "total_family_commands": len(family_command_rows),
        "passes_all": all(bool(row["passed"]) for row in rows)
        and all(bool(row["passed"]) for row in family_rows)
        and all(bool(row["passed"]) for row in family_command_rows),
        "rows": rows,
        "terrain_families": family_rows,
        "family_commands": family_command_rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))
    env.close()
    if not result["passes_all"]:
        raise SystemExit(1)


if __name__ == "__main__":
    try:
        main()
    finally:
        simulation_app.close()

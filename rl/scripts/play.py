# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Script to play a checkpoint if an RL agent from RSL-RL."""

"""Launch Isaac Sim Simulator first."""

import argparse
import csv
import json
from importlib.metadata import version

import custom_dog_rl.tasks  # noqa: F401

from isaaclab.app import AppLauncher

# local imports
import cli_args  # isort: skip

# add argparse arguments
parser = argparse.ArgumentParser(description="Train an RL agent with RSL-RL.")
parser.add_argument("--video", action="store_true", default=False, help="Record videos during training.")
parser.add_argument("--video_length", type=int, default=200, help="Length of the recorded video (in steps).")
parser.add_argument(
    "--disable_fabric", action="store_true", default=False, help="Disable fabric and use USD I/O operations."
)
parser.add_argument("--num_envs", type=int, default=None, help="Number of environments to simulate.")
parser.add_argument("--max_steps", type=int, default=None, help="Stop after this many simulation steps.")
parser.add_argument(
    "--fixed_command",
    nargs=3,
    type=float,
    metavar=("VX", "VY", "YAW"),
    help="Override base_velocity with a fixed body-frame command during playback.",
)
parser.add_argument(
    "--metric_warmup_steps",
    type=int,
    default=100,
    help="Playback steps excluded from fixed-command metrics.",
)
parser.add_argument(
    "--trace_csv",
    type=str,
    default=None,
    help="Write one-environment policy I/O and robot-state samples to CSV.",
)
parser.add_argument(
    "--metrics_json",
    type=str,
    default=None,
    help="Write fixed-command aggregate metrics and termination rate as JSON.",
)
parser.add_argument(
    "--nominal_conditions",
    action="store_true",
    help="Disable observation noise and domain/reset randomization for simulator alignment.",
)
parser.add_argument("--task", type=str, default=None, help="Name of the task.")
parser.add_argument(
    "--rl_device",
    type=str,
    default=None,
    help="Device used by the RSL-RL policy runner, for example cuda:0 or cpu.",
)
parser.add_argument(
    "--use_pretrained_checkpoint",
    action="store_true",
    help="Use the pre-trained checkpoint from Nucleus.",
)
parser.add_argument("--real-time", action="store_true", default=False, help="Run in real-time, if possible.")
# append RSL-RL cli arguments
cli_args.add_rsl_rl_args(parser)
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
if args_cli.metrics_json is not None and args_cli.fixed_command is None:
    parser.error("--metrics_json requires --fixed_command")
# always enable cameras to record video
if args_cli.video:
    args_cli.enable_cameras = True

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import gymnasium as gym
import os
import time
import torch

from rsl_rl.runners import OnPolicyRunner

import isaaclab_tasks  # noqa: F401
import unitree_rl_lab.tasks  # noqa: F401
from isaaclab.envs import DirectMARLEnv, multi_agent_to_single_agent
from isaaclab.utils.assets import retrieve_file_path
from isaaclab.utils.dict import print_dict
from isaaclab.utils.pretrained_checkpoint import get_published_pretrained_checkpoint
from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlVecEnvWrapper, export_policy_as_jit, export_policy_as_onnx
from isaaclab_tasks.utils import get_checkpoint_path

from unitree_rl_lab.utils.parser_cfg import parse_env_cfg


def main():
    """Play with RSL-RL agent."""
    # parse configuration
    env_cfg = parse_env_cfg(
        args_cli.task,
        device=args_cli.device,
        num_envs=args_cli.num_envs,
        use_fabric=not args_cli.disable_fabric,
        entry_point_key="play_env_cfg_entry_point",
    )
    if args_cli.nominal_conditions:
        env_cfg.observations.policy.enable_corruption = False
        for event_name in (
            "physics_material",
            "add_base_mass",
            "scale_body_mass",
            "randomize_base_com",
            "randomize_actuator_gains",
            "randomize_joint_friction",
        ):
            if hasattr(env_cfg.events, event_name):
                setattr(env_cfg.events, event_name, None)
        env_cfg.events.reset_base.params["pose_range"] = {
            axis: (0.0, 0.0) for axis in ("x", "y", "z", "roll", "pitch", "yaw")
        }
        env_cfg.events.reset_base.params["velocity_range"] = {
            axis: (0.0, 0.0) for axis in ("x", "y", "z", "roll", "pitch", "yaw")
        }
        env_cfg.events.reset_robot_joints.params["position_range"] = (1.0, 1.0)
        env_cfg.events.reset_robot_joints.params["velocity_range"] = (0.0, 0.0)
    agent_cfg: RslRlOnPolicyRunnerCfg = cli_args.parse_rsl_rl_cfg(args_cli.task, args_cli)
    if args_cli.rl_device is not None:
        agent_cfg.device = args_cli.rl_device

    # specify directory for logging experiments
    log_root_path = os.path.join("logs", "rsl_rl", agent_cfg.experiment_name)
    log_root_path = os.path.abspath(log_root_path)
    print(f"[INFO] Loading experiment from directory: {log_root_path}")
    if args_cli.use_pretrained_checkpoint:
        resume_path = get_published_pretrained_checkpoint("rsl_rl", args_cli.task)
        if not resume_path:
            print("[INFO] Unfortunately a pre-trained checkpoint is currently unavailable for this task.")
            return
    elif args_cli.checkpoint:
        resume_path = retrieve_file_path(args_cli.checkpoint)
    else:
        resume_path = get_checkpoint_path(log_root_path, agent_cfg.load_run, agent_cfg.load_checkpoint)

    log_dir = os.path.dirname(resume_path)

    # create isaac environment
    env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None)

    # convert to single-agent instance if required by the RL algorithm
    if isinstance(env.unwrapped, DirectMARLEnv):
        env = multi_agent_to_single_agent(env)

    # wrap for video recording
    if args_cli.video:
        video_kwargs = {
            "video_folder": os.path.join(log_dir, "videos", "play"),
            "step_trigger": lambda step: step == 0,
            "video_length": args_cli.video_length,
            "disable_logger": True,
        }
        print("[INFO] Recording videos during training.")
        print_dict(video_kwargs, nesting=4)
        env = gym.wrappers.RecordVideo(env, **video_kwargs)

    # wrap around environment for rsl-rl
    env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)

    print(f"[INFO]: Loading model checkpoint from: {resume_path}")
    # load previously trained model
    if not hasattr(agent_cfg, "class_name") or agent_cfg.class_name == "OnPolicyRunner":
        runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    elif agent_cfg.class_name == "DistillationRunner":
        from rsl_rl.runners import DistillationRunner

        runner = DistillationRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    else:
        raise ValueError(f"Unsupported runner class: {agent_cfg.class_name}")
    runner.load(resume_path, map_location=agent_cfg.device)

    # obtain the trained policy for inference
    policy = runner.get_inference_policy(device=env.unwrapped.device)

    # extract the neural network module
    # we do this in a try-except to maintain backwards compatibility.
    try:
        # version 2.3 onwards
        policy_nn = runner.alg.policy
    except AttributeError:
        # version 2.2 and below
        policy_nn = runner.alg.actor_critic

    # extract the normalizer
    if hasattr(policy_nn, "actor_obs_normalizer"):
        normalizer = policy_nn.actor_obs_normalizer
    elif hasattr(policy_nn, "student_obs_normalizer"):
        normalizer = policy_nn.student_obs_normalizer
    else:
        normalizer = None

    # export policy to onnx/jit
    export_model_dir = os.path.join(os.path.dirname(resume_path), "exported")
    export_policy_as_jit(policy_nn, normalizer=normalizer, path=export_model_dir, filename="policy.pt")
    export_policy_as_onnx(policy_nn, normalizer=normalizer, path=export_model_dir, filename="policy.onnx")

    dt = env.unwrapped.step_dt

    def apply_fixed_command():
        if args_cli.fixed_command is None:
            return
        command_term = env.unwrapped.command_manager.get_term("base_velocity")
        command = torch.tensor(
            args_cli.fixed_command,
            dtype=command_term.vel_command_b.dtype,
            device=command_term.vel_command_b.device,
        )
        command_term.vel_command_b[:] = command
        if hasattr(command_term, "_sampled_vel_command_b"):
            command_term._sampled_vel_command_b[:] = command
        command_term.is_standing_env[:] = False

    # reset environment
    apply_fixed_command()
    obs = env.get_observations()
    if version("rsl-rl-lib").startswith("2.3."):
        obs, _ = env.get_observations()
    timestep = 0
    trace_stream = None
    trace_writer = None
    if args_cli.trace_csv is not None:
        if env.num_envs != 1:
            raise ValueError("--trace_csv requires --num_envs 1")
        trace_path = os.path.abspath(args_cli.trace_csv)
        os.makedirs(os.path.dirname(trace_path), exist_ok=True)
        trace_stream = open(trace_path, "w", encoding="utf-8", newline="")
        trace_writer = csv.writer(trace_stream)
        policy_obs = obs["policy"]
        header = ["step", "time_s"]
        for prefix, count in (
            ("obs", policy_obs.shape[-1]),
            ("action", 12),
            ("target_q", 12),
            ("joint_q", 12),
            ("joint_dq", 12),
            ("ang_vel", 3),
            ("projected_gravity", 3),
            ("base_lin_vel", 3),
            ("base_pos", 3),
            ("base_quat", 4),
        ):
            header.extend(f"{prefix}_{index}" for index in range(count))
        trace_writer.writerow(header)
    velocity_samples = []
    yaw_rate_samples = []
    height_samples = []
    tilt_samples = []
    world_position_samples = []
    hip_outward_samples = []
    foot_contact_samples = []
    done_count = 0
    environments_done = None
    fixed_robot = None
    fixed_contact_sensor = None
    hip_joint_ids = None
    foot_sensor_body_ids = None
    if args_cli.fixed_command is not None:
        fixed_robot = env.unwrapped.scene["robot"]
        fixed_contact_sensor = env.unwrapped.scene["contact_forces"]
        hip_joint_ids = [
            fixed_robot.joint_names.index(name)
            for name in ("FR_hip_joint", "FL_hip_joint", "RR_hip_joint", "RL_hip_joint")
        ]
        foot_sensor_body_ids = [
            fixed_contact_sensor.body_names.index(name)
            for name in ("FR_foot", "FL_foot", "RR_foot", "RL_foot")
        ]
        hip_outward_sign = torch.tensor(
            (-1.0, 1.0, -1.0, 1.0),
            device=env.unwrapped.device,
        )
    # simulate environment
    while simulation_app.is_running():
        start_time = time.time()
        # run everything in inference mode
        with torch.inference_mode():
            # agent stepping
            actions = policy(obs)
            if trace_writer is not None:
                robot = env.unwrapped.scene["robot"]
                action_term = env.unwrapped.action_manager.get_term("JointPositionAction")
                target = actions * action_term._scale + action_term._offset
                if action_term.cfg.clip is not None:
                    target = torch.clamp(
                        target,
                        min=action_term._clip[:, :, 0],
                        max=action_term._clip[:, :, 1],
                    )
                joint_ids = action_term._joint_ids
                row = [timestep, timestep * dt]
                for values in (
                    obs["policy"][0],
                    actions[0],
                    target[0],
                    robot.data.joint_pos[0, joint_ids],
                    robot.data.joint_vel[0, joint_ids],
                    robot.data.root_ang_vel_b[0],
                    robot.data.projected_gravity_b[0],
                    robot.data.root_lin_vel_b[0],
                    robot.data.root_pos_w[0],
                    robot.data.root_quat_w[0],
                ):
                    row.extend(values.detach().cpu().tolist())
                trace_writer.writerow(row)
                trace_stream.flush()
            # env stepping
            obs, _, dones, _ = env.step(actions)
            if args_cli.fixed_command is not None:
                done_mask = dones.bool()
                done_count += int(done_mask.sum().item())
                if environments_done is None:
                    environments_done = done_mask.clone()
                else:
                    environments_done |= done_mask
            apply_fixed_command()
            if args_cli.fixed_command is not None:
                obs = env.get_observations()
                if version("rsl-rl-lib").startswith("2.3."):
                    obs, _ = obs
                if timestep >= args_cli.metric_warmup_steps:
                    robot = env.unwrapped.scene["robot"]
                    velocity_samples.append(robot.data.root_lin_vel_b[:, :2].clone())
                    yaw_rate_samples.append(robot.data.root_ang_vel_b[:, 2].clone())
                    height_samples.append(robot.data.root_pos_w[:, 2].clone())
                    tilt_samples.append(
                        torch.acos(torch.clamp(-robot.data.projected_gravity_b[:, 2], -1.0, 1.0))
                    )
                    assert fixed_robot is not None
                    assert fixed_contact_sensor is not None
                    assert hip_joint_ids is not None
                    assert foot_sensor_body_ids is not None
                    world_position_samples.append(fixed_robot.data.root_pos_w[:, :2].clone())
                    hip_outward_samples.append(
                        fixed_robot.data.joint_pos[:, hip_joint_ids].clone() * hip_outward_sign
                    )
                    foot_forces = torch.linalg.vector_norm(
                        fixed_contact_sensor.data.net_forces_w[:, foot_sensor_body_ids],
                        dim=-1,
                    )
                    foot_contact_samples.append(foot_forces > 1.0)
        if args_cli.video:
            timestep += 1
            # Exit the play loop after recording one video
            if timestep == args_cli.video_length:
                break
        elif args_cli.max_steps is not None:
            timestep += 1
            if timestep >= args_cli.max_steps:
                break

        # time delay for real-time evaluation
        sleep_time = dt - (time.time() - start_time)
        if args_cli.real_time and sleep_time > 0:
            time.sleep(sleep_time)

    if velocity_samples:
        velocity = torch.cat(velocity_samples)
        yaw_rate = torch.cat(yaw_rate_samples)
        height = torch.cat(height_samples)
        tilt = torch.cat(tilt_samples)
        print(
            "fixed-command metrics: "
            f"command={args_cli.fixed_command}, "
            f"mean_vx={velocity[:, 0].mean().item():.3f} m/s, "
            f"mean_vy={velocity[:, 1].mean().item():.3f} m/s, "
            f"mean_yaw_rate={yaw_rate.mean().item():.3f} rad/s, "
            f"min_height={height.min().item():.3f} m, "
            f"mean_height={height.mean().item():.3f} m, "
            f"max_tilt={torch.rad2deg(tilt.max()).item():.2f} deg"
        )
        metrics = {
            "command": [float(value) for value in args_cli.fixed_command],
            "num_envs": int(env.num_envs),
            "sample_steps": len(velocity_samples),
            "mean_vx_m_s": float(velocity[:, 0].mean().item()),
            "mean_vy_m_s": float(velocity[:, 1].mean().item()),
            "mean_wz_rad_s": float(yaw_rate.mean().item()),
            "error_vx_m_s": float(abs(velocity[:, 0].mean().item() - args_cli.fixed_command[0])),
            "error_vy_m_s": float(abs(velocity[:, 1].mean().item() - args_cli.fixed_command[1])),
            "error_wz_rad_s": float(abs(yaw_rate.mean().item() - args_cli.fixed_command[2])),
            "min_height_m": float(height.min().item()),
            "height_p05_m": float(torch.quantile(height, 0.05).item()),
            "mean_height_m": float(height.mean().item()),
            "max_tilt_deg": float(torch.rad2deg(tilt.max()).item()),
            "tilt_p95_deg": float(torch.rad2deg(torch.quantile(tilt, 0.95)).item()),
            "termination_count": done_count,
            "environments_terminated": (
                int(environments_done.sum().item()) if environments_done is not None else 0
            ),
            "success_rate": (
                1.0 - float(environments_done.float().mean().item())
                if environments_done is not None
                else 1.0
            ),
        }
        if len(world_position_samples) >= 2:
            positions = torch.stack(world_position_samples)
            displacements = positions[-1] - positions[0]
            mean_displacement = displacements.mean(dim=0)
            max_displacement = torch.linalg.vector_norm(displacements, dim=-1).max()
            yaw_rates = torch.stack(yaw_rate_samples)
            measured_yaw_integral = torch.trapezoid(yaw_rates, dx=dt, dim=0)
            metric_duration = (len(yaw_rate_samples) - 1) * dt
            requested_yaw_integral = args_cli.fixed_command[2] * metric_duration
            yaw_bias = measured_yaw_integral - requested_yaw_integral
            print(
                "fixed-command yaw integral: "
                f"measured_mean={measured_yaw_integral.mean().item():.3f} rad, "
                f"requested={requested_yaw_integral:.3f} rad, "
                f"bias_mean={yaw_bias.mean().item():.3f} rad, "
                f"bias_abs_max={yaw_bias.abs().max().item():.3f} rad"
            )
            metrics.update(
                {
                    "yaw_integral_bias_abs_max_rad": float(yaw_bias.abs().max().item()),
                    "world_displacement_max_m": float(max_displacement.item()),
                }
            )
            print(
                "fixed-command world displacement: "
                f"mean=[{mean_displacement[0].item():.3f}, "
                f"{mean_displacement[1].item():.3f}] m, "
                f"max_norm={max_displacement.item():.3f} m"
            )
            metrics.update(
                {
                    "mean_hip_outward_deg": [float(value) for value in hip_mean.cpu().tolist()],
                    "max_hip_outward_deg": float(hip_outward_deg.max().item()),
                }
            )
        if hip_outward_samples:
            hip_outward_deg = torch.rad2deg(torch.stack(hip_outward_samples))
            hip_mean = hip_outward_deg.mean(dim=(0, 1))
            print(
                "fixed-command posture: mean_hip_outward=[{}] deg, "
                "max_hip_outward={:.2f} deg".format(
                    ", ".join(f"{value:.2f}" for value in hip_mean.cpu().tolist()),
                    hip_outward_deg.max().item(),
                )
            )
            metrics.update(
                {
                    "contact_duty_per_leg": [float(value) for value in duty.cpu().tolist()],
                    "contact_transitions_mean_per_leg": [
                        float(value) for value in transitions_mean.cpu().tolist()
                    ],
                    "contact_transitions_min_per_leg": [
                        int(value) for value in transitions_min.cpu().tolist()
                    ],
                }
            )
        if args_cli.metrics_json is not None:
            metrics_path = os.path.abspath(args_cli.metrics_json)
            os.makedirs(os.path.dirname(metrics_path), exist_ok=True)
            with open(metrics_path, "w", encoding="utf-8") as stream:
                json.dump(metrics, stream, indent=2)
                stream.write("\n")
            print(f"Fixed-command metrics written: {metrics_path}")
        if foot_contact_samples:
            contacts = torch.stack(foot_contact_samples)
            duty = contacts.float().mean(dim=(0, 1))
            transitions = torch.count_nonzero(contacts[1:] != contacts[:-1], dim=0)
            transitions_mean = transitions.float().mean(dim=0)
            transitions_min = transitions.min(dim=0).values
            print(
                "fixed-command contacts: duty=[{}], transitions_mean=[{}], "
                "transitions_min=[{}]".format(
                    ", ".join(f"{value:.2f}" for value in duty.cpu().tolist()),
                    ", ".join(f"{value:.1f}" for value in transitions_mean.cpu().tolist()),
                    ", ".join(str(int(value)) for value in transitions_min.cpu().tolist()),
                )
            )

    if trace_stream is not None:
        trace_stream.close()
        print(f"Isaac policy trace written: {os.path.abspath(args_cli.trace_csv)}")

    # close the simulator
    env.close()


if __name__ == "__main__":
    # run the main function
    main()
    # WSL's Isaac Sim Vulkan teardown can segfault after finite export runs.
    # Let the process reclaim the simulator naturally for smoke/export commands.
    if args_cli.max_steps is None and not args_cli.video:
        simulation_app.close()

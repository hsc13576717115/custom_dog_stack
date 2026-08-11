# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Script to play a checkpoint if an RL agent from RSL-RL."""

"""Launch Isaac Sim Simulator first."""

import argparse
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
    velocity_samples = []
    yaw_rate_samples = []
    height_samples = []
    tilt_samples = []
    # simulate environment
    while simulation_app.is_running():
        start_time = time.time()
        # run everything in inference mode
        with torch.inference_mode():
            # agent stepping
            actions = policy(obs)
            # env stepping
            obs, _, _, _ = env.step(actions)
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

    # close the simulator
    env.close()


if __name__ == "__main__":
    # run the main function
    main()
    # WSL's Isaac Sim Vulkan teardown can segfault after finite export runs.
    # Let the process reclaim the simulator naturally for smoke/export commands.
    if args_cli.max_steps is None and not args_cli.video:
        simulation_app.close()

# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Script to train RL agent with RSL-RL."""

"""Launch Isaac Sim Simulator first."""


import gymnasium as gym
import sys

import custom_dog_rl.tasks  # noqa: F401

tasks = []
for task_spec in gym.registry.values():
    if task_spec.id.startswith("CustomDog-"):
        tasks.append(task_spec.id)

import argparse

import argcomplete

from isaaclab.app import AppLauncher

# local imports
import cli_args  # isort: skip

# add argparse arguments
parser = argparse.ArgumentParser(description="Train an RL agent with RSL-RL.")
parser.add_argument("--video", action="store_true", default=False, help="Record videos during training.")
parser.add_argument("--video_length", type=int, default=200, help="Length of the recorded video (in steps).")
parser.add_argument("--video_interval", type=int, default=2000, help="Interval between video recordings (in steps).")
parser.add_argument("--num_envs", type=int, default=None, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, default=None, choices=tasks, help="Name of the task.")
parser.add_argument("--seed", type=int, default=None, help="Seed used for the environment")
parser.add_argument("--max_iterations", type=int, default=None, help="RL Policy training iterations.")
parser.add_argument(
    "--rl_device",
    type=str,
    default=None,
    help="Device used by the RSL-RL policy and PPO optimizer, for example cuda:0 or cpu.",
)
parser.add_argument(
    "--distributed", action="store_true", default=False, help="Run training with multiple GPUs or nodes."
)
# append RSL-RL cli arguments
cli_args.add_rsl_rl_args(parser)
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
argcomplete.autocomplete(parser)
args_cli, hydra_args = parser.parse_known_args()

# always enable cameras to record video
if args_cli.video:
    args_cli.enable_cameras = True

# clear out sys.argv for Hydra
sys.argv = [sys.argv[0]] + hydra_args

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Check for minimum supported RSL-RL version."""

import importlib.metadata as metadata
import platform

from packaging import version

# for distributed training, check minimum supported rsl-rl version
RSL_RL_VERSION = "2.3.1"
installed_version = metadata.version("rsl-rl-lib")
if args_cli.distributed and version.parse(installed_version) < version.parse(RSL_RL_VERSION):
    if platform.system() == "Windows":
        cmd = [r".\isaaclab.bat", "-p", "-m", "pip", "install", f"rsl-rl-lib=={RSL_RL_VERSION}"]
    else:
        cmd = ["./isaaclab.sh", "-p", "-m", "pip", "install", f"rsl-rl-lib=={RSL_RL_VERSION}"]
    print(
        f"Please install the correct version of RSL-RL.\nExisting version is: '{installed_version}'"
        f" and required version is: '{RSL_RL_VERSION}'.\nTo install the correct version, run:"
        f"\n\n\t{' '.join(cmd)}\n"
    )
    exit(1)

"""Rest everything follows."""

import gymnasium as gym
import inspect
import math
import os
import shutil
import torch
from datetime import datetime

from rsl_rl.runners import DistillationRunner, OnPolicyRunner

from custom_dog_rl.agents.routed_teacher import (
    configure_privileged_omni_teacher,
    configure_routed_teacher,
)

import isaaclab_tasks  # noqa: F401
from isaaclab.envs import (
    DirectMARLEnv,
    DirectMARLEnvCfg,
    DirectRLEnvCfg,
    ManagerBasedRLEnvCfg,
    multi_agent_to_single_agent,
)
from isaaclab.utils.dict import print_dict
from isaaclab.utils.io import dump_yaml
from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlVecEnvWrapper
from isaaclab_tasks.utils import get_checkpoint_path
from isaaclab_tasks.utils.hydra import hydra_task_config

from unitree_rl_lab.utils.export_deploy_cfg import export_deploy_cfg

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cudnn.deterministic = False
torch.backends.cudnn.benchmark = False


def configure_actor_input_adapter(policy, new_columns: int) -> None:
    """Freeze the actor except for its newest first-layer input columns."""

    if new_columns <= 0:
        raise ValueError("CUSTOM_DOG_ACTOR_INPUT_ADAPTER_COLUMNS must be positive")
    actor_parameters = dict(policy.actor.named_parameters())
    first_weight = actor_parameters.get("0.weight")
    if first_weight is None or first_weight.ndim != 2 or new_columns >= first_weight.shape[1]:
        raise ValueError("Could not resolve a valid actor first-layer input adapter")

    for parameter in policy.actor.parameters():
        parameter.requires_grad_(False)
    first_weight.requires_grad_(True)

    old_columns = first_weight.shape[1] - new_columns

    def keep_adapter_gradient(gradient: torch.Tensor) -> torch.Tensor:
        masked = torch.zeros_like(gradient)
        masked[:, old_columns:] = gradient[:, old_columns:]
        return masked

    first_weight.register_hook(keep_adapter_gradient)
    for name in ("std", "log_std"):
        parameter = getattr(policy, name, None)
        if isinstance(parameter, torch.Tensor):
            parameter.requires_grad_(False)
    print(
        f"[INFO]: Frozen actor; training only the final {new_columns} input columns "
        f"of actor.0.weight"
    )


def load_distillation_student(policy, checkpoint_path: str) -> None:
    """Initialize a distillation student from an actor checkpoint."""

    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    state_dict = checkpoint.get("model_state_dict")
    if not isinstance(state_dict, dict):
        raise ValueError("Student checkpoint does not contain model_state_dict")
    student_state = {
        key.removeprefix("actor."): value
        for key, value in state_dict.items()
        if key.startswith("actor.")
    }
    if not student_state:
        raise ValueError("Student checkpoint does not contain actor parameters")
    policy.student.load_state_dict(student_state, strict=True)
    print(f"[INFO]: Initialized distillation student from: {checkpoint_path}")


@hydra_task_config(args_cli.task, "rsl_rl_cfg_entry_point")
def main(env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg | DirectMARLEnvCfg, agent_cfg: RslRlOnPolicyRunnerCfg):
    """Train with RSL-RL agent."""
    # override configurations with non-hydra CLI arguments
    agent_cfg = cli_args.update_rsl_rl_cfg(agent_cfg, args_cli)
    if args_cli.rl_device is not None:
        agent_cfg.device = args_cli.rl_device
    env_cfg.scene.num_envs = args_cli.num_envs if args_cli.num_envs is not None else env_cfg.scene.num_envs
    agent_cfg.max_iterations = (
        args_cli.max_iterations if args_cli.max_iterations is not None else agent_cfg.max_iterations
    )

    # set the environment seed
    # note: certain randomizations occur in the environment initialization so we set the seed here
    env_cfg.seed = agent_cfg.seed
    env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device

    # multi-gpu training configuration
    if args_cli.distributed:
        env_cfg.sim.device = f"cuda:{app_launcher.local_rank}"
        agent_cfg.device = f"cuda:{app_launcher.local_rank}"

        # set seed to have diversity in different threads
        seed = agent_cfg.seed + app_launcher.local_rank
        env_cfg.seed = seed
        agent_cfg.seed = seed

    # specify directory for logging experiments
    log_root_path = os.path.join("logs", "rsl_rl", agent_cfg.experiment_name)
    log_root_path = os.path.abspath(log_root_path)
    print(f"[INFO] Logging experiment in directory: {log_root_path}")
    # specify directory for logging runs: {time-stamp}_{run_name}
    log_dir = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    # This way, the Ray Tune workflow can extract experiment name.
    print(f"Exact experiment name requested from command line: {log_dir}")
    if agent_cfg.run_name:
        log_dir += f"_{agent_cfg.run_name}"
    log_dir = os.path.join(log_root_path, log_dir)

    # create isaac environment
    env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None)

    # convert to single-agent instance if required by the RL algorithm
    if isinstance(env.unwrapped, DirectMARLEnv):
        env = multi_agent_to_single_agent(env)

    # save resume path before creating a new log_dir
    if agent_cfg.resume or agent_cfg.algorithm.class_name == "Distillation":
        resume_path = get_checkpoint_path(log_root_path, agent_cfg.load_run, agent_cfg.load_checkpoint)

    # wrap for video recording
    if args_cli.video:
        video_kwargs = {
            "video_folder": os.path.join(log_dir, "videos", "train"),
            "step_trigger": lambda step: step % args_cli.video_interval == 0,
            "video_length": args_cli.video_length,
            "disable_logger": True,
        }
        print("[INFO] Recording videos during training.")
        print_dict(video_kwargs, nesting=4)
        env = gym.wrappers.RecordVideo(env, **video_kwargs)

    # wrap around environment for rsl-rl
    env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)

    # create runner from rsl-rl
    runner_class = DistillationRunner if agent_cfg.class_name == "DistillationRunner" else OnPolicyRunner
    runner = runner_class(env, agent_cfg.to_dict(), log_dir=log_dir, device=agent_cfg.device)
    # write git state to logs
    runner.add_git_repo_to_log(__file__)
    # load the checkpoint
    if agent_cfg.resume or agent_cfg.algorithm.class_name == "Distillation":
        print(f"[INFO]: Loading model checkpoint from: {resume_path}")
        # load previously trained model
        load_optimizer = os.environ.get("CUSTOM_DOG_LOAD_OPTIMIZER", "1").lower() not in {"0", "false", "no"}
        if load_optimizer:
            checkpoint_metadata = torch.load(resume_path, map_location="cpu", weights_only=True)
            expansion = checkpoint_metadata.get("actor_observation_expansion")
            if isinstance(expansion, dict) and expansion.get("optimizer_reset_required"):
                raise ValueError(
                    "Expanded actor-observation checkpoints require "
                    "CUSTOM_DOG_LOAD_OPTIMIZER=0"
                )
        runner.load(resume_path, load_optimizer=load_optimizer, map_location=agent_cfg.device)
        if not load_optimizer:
            print("[INFO]: Optimizer state reset; using the configured fine-tune learning rate")
            reset_policy_std = os.environ.get("CUSTOM_DOG_RESET_POLICY_STD")
            if reset_policy_std is not None:
                policy_std = float(reset_policy_std)
                if not math.isfinite(policy_std) or policy_std <= 0.0:
                    raise ValueError("CUSTOM_DOG_RESET_POLICY_STD must be a finite positive number")
                policy = runner.alg.policy
                with torch.no_grad():
                    if hasattr(policy, "std"):
                        policy.std.fill_(policy_std)
                    elif hasattr(policy, "log_std"):
                        policy.log_std.fill_(math.log(policy_std))
                    else:
                        raise AttributeError("Loaded policy does not expose std or log_std")
                print(f"[INFO]: Policy exploration standard deviation reset to {policy_std:.3f}")

    student_checkpoint = os.environ.get("CUSTOM_DOG_DISTILL_STUDENT_CHECKPOINT")
    if student_checkpoint is not None:
        if agent_cfg.class_name != "DistillationRunner":
            raise ValueError("CUSTOM_DOG_DISTILL_STUDENT_CHECKPOINT requires DistillationRunner")
        load_distillation_student(runner.alg.policy, student_checkpoint)

    reverse_teacher_checkpoint = os.environ.get("CUSTOM_DOG_ROUTED_TEACHER_REVERSE_CHECKPOINT")
    if reverse_teacher_checkpoint is not None:
        if agent_cfg.class_name != "DistillationRunner":
            raise ValueError("CUSTOM_DOG_ROUTED_TEACHER_REVERSE_CHECKPOINT requires DistillationRunner")
        command_index = int(os.environ.get("CUSTOM_DOG_ROUTED_TEACHER_COMMAND_INDEX", "6"))
        reverse_threshold = float(
            os.environ.get("CUSTOM_DOG_ROUTED_TEACHER_REVERSE_THRESHOLD", "-0.05")
        )
        blend_width = float(os.environ.get("CUSTOM_DOG_ROUTED_TEACHER_BLEND_WIDTH", "0.05"))
        configure_routed_teacher(
            runner.alg.policy,
            reverse_teacher_checkpoint,
            command_index=command_index,
            reverse_threshold=reverse_threshold,
            blend_width=blend_width,
        )
        print(
            "[INFO]: Configured command-routed distillation teacher: "
            f"reverse={reverse_teacher_checkpoint}, command_index={command_index}, "
            f"threshold={reverse_threshold:.3f}, blend_width={blend_width:.3f}"
        )

    privileged_forward_checkpoint = os.environ.get(
        "CUSTOM_DOG_PRIVILEGED_ROUTED_FORWARD_CHECKPOINT"
    )
    if privileged_forward_checkpoint is not None:
        if agent_cfg.class_name != "DistillationRunner":
            raise ValueError(
                "CUSTOM_DOG_PRIVILEGED_ROUTED_FORWARD_CHECKPOINT requires DistillationRunner"
            )
        if reverse_teacher_checkpoint is not None:
            raise ValueError("The reverse and privileged omni routers cannot be enabled together")
        configure_privileged_omni_teacher(
            runner.alg.policy,
            privileged_forward_checkpoint,
            forward_observation_dim=int(
                os.environ.get("CUSTOM_DOG_PRIVILEGED_FORWARD_OBSERVATION_DIM", "45")
            ),
            reverse_threshold=float(
                os.environ.get("CUSTOM_DOG_PRIVILEGED_REVERSE_THRESHOLD", "0.05")
            ),
            lateral_threshold=float(
                os.environ.get("CUSTOM_DOG_PRIVILEGED_LATERAL_THRESHOLD", "0.025")
            ),
            yaw_threshold=float(
                os.environ.get("CUSTOM_DOG_PRIVILEGED_YAW_THRESHOLD", "0.025")
            ),
            pure_yaw_forward_threshold=float(
                os.environ.get("CUSTOM_DOG_PRIVILEGED_PURE_YAW_VX_THRESHOLD", "0.05")
            ),
            blend_width=float(
                os.environ.get("CUSTOM_DOG_PRIVILEGED_ROUTE_BLEND_WIDTH", "0.025")
            ),
        )
        print(
            "[INFO]: Configured privileged omni routed teacher: "
            f"forward={privileged_forward_checkpoint}"
        )

    adapter_columns = os.environ.get("CUSTOM_DOG_ACTOR_INPUT_ADAPTER_COLUMNS")
    if adapter_columns is not None:
        if not (agent_cfg.resume or agent_cfg.algorithm.class_name == "Distillation"):
            raise ValueError("Actor input-adapter training requires a resumed checkpoint")
        configure_actor_input_adapter(runner.alg.policy, int(adapter_columns))

    # dump the configuration into log-directory
    dump_yaml(os.path.join(log_dir, "params", "env.yaml"), env_cfg)
    dump_yaml(os.path.join(log_dir, "params", "agent.yaml"), agent_cfg)
    export_deploy_cfg(env.unwrapped, log_dir)
    # copy the environment configuration file to the log directory
    shutil.copy(
        inspect.getfile(env_cfg.__class__),
        os.path.join(log_dir, "params", os.path.basename(inspect.getfile(env_cfg.__class__))),
    )

    # run training
    runner.learn(num_learning_iterations=agent_cfg.max_iterations, init_at_random_ep_len=True)

    # close the simulator
    env.close()


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()

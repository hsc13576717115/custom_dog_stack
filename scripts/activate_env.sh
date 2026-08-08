#!/usr/bin/env bash

CUSTOM_DOG_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ISAACLAB_ROOT="${ISAACLAB_ROOT:-/home/hsc/IsaacLab}"
UNITREE_RL_LAB_ROOT="${UNITREE_RL_LAB_ROOT:-/home/hsc/unitree_rl_lab}"
UNITREE_MUJOCO_ROOT="${UNITREE_MUJOCO_ROOT:-/home/hsc/unitree_mujoco}"
CUSTOM_DOG_CONDA_ENV="${CUSTOM_DOG_CONDA_ENV:-env_isaaclab}"

source /home/hsc/miniconda3/etc/profile.d/conda.sh
conda activate "${CUSTOM_DOG_CONDA_ENV}"

export CUSTOM_DOG_ROOT
export ISAACLAB_ROOT
export UNITREE_RL_LAB_ROOT
export UNITREE_MUJOCO_ROOT
export CUSTOM_DOG_DESCRIPTION_DIR="${CUSTOM_DOG_DESCRIPTION_DIR:-${CUSTOM_DOG_ROOT}/ros2/src/custom_dog_description}"
export PYTHONPATH="${CUSTOM_DOG_ROOT}/rl/src:${UNITREE_RL_LAB_ROOT}/source/unitree_rl_lab:${PYTHONPATH:-}"
export ROS_PACKAGE_PATH="${CUSTOM_DOG_ROOT}/ros2/src:/home/hsc/unitree_ros/robots:${ROS_PACKAGE_PATH:-}"
export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib:/usr/lib/wsl/lib:${LD_LIBRARY_PATH:-}"
export TERM="${TERM:-xterm-256color}"

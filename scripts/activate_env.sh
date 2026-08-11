#!/usr/bin/env bash

CUSTOM_DOG_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
WORKSPACE_ROOT="${CUSTOM_DOG_WORKSPACE_ROOT:-$(cd "${CUSTOM_DOG_ROOT}/.." && pwd)}"
ISAACLAB_ROOT="${ISAACLAB_ROOT:-${CUSTOM_DOG_ROOT}/third_party/IsaacLab}"
UNITREE_RL_LAB_ROOT="${UNITREE_RL_LAB_ROOT:-${CUSTOM_DOG_ROOT}/third_party/unitree_rl_lab}"
UNITREE_MUJOCO_ROOT="${UNITREE_MUJOCO_ROOT:-${CUSTOM_DOG_ROOT}/third_party/unitree_mujoco}"
CUSTOM_DOG_CONDA_ENV="${CUSTOM_DOG_CONDA_ENV:-env_isaaclab}"

# NVIDIA Omniverse EULA was accepted for this workstation. Keep an explicit
# override available for CI or a different machine.
OMNI_KIT_ACCEPT_EULA="${OMNI_KIT_ACCEPT_EULA:-YES}"

if ! declare -F conda >/dev/null 2>&1; then
    conda_exe="${CONDA_EXE:-$(command -v conda 2>/dev/null || true)}"
    conda_candidates=()
    if [[ -n "${conda_exe}" ]]; then
        conda_candidates+=("$(cd "$(dirname "${conda_exe}")/.." && pwd)/etc/profile.d/conda.sh")
    fi
    conda_candidates+=(
        "${HOME}/miniconda3/etc/profile.d/conda.sh"
        "${HOME}/anaconda3/etc/profile.d/conda.sh"
        "/opt/conda/etc/profile.d/conda.sh"
    )
    for conda_candidate in "${conda_candidates[@]}"; do
        if [[ -f "${conda_candidate}" ]]; then
            source "${conda_candidate}"
            break
        fi
    done
fi

if ! declare -F conda >/dev/null 2>&1; then
    echo "Conda was not found. Install Miniconda/Anaconda or set CONDA_EXE." >&2
    return 1 2>/dev/null || exit 1
fi

conda activate "${CUSTOM_DOG_CONDA_ENV}"

export CUSTOM_DOG_ROOT
export WORKSPACE_ROOT
export OMNI_KIT_ACCEPT_EULA
export ISAACLAB_ROOT
export UNITREE_RL_LAB_ROOT
export UNITREE_MUJOCO_ROOT
export CUSTOM_DOG_DESCRIPTION_DIR="${CUSTOM_DOG_DESCRIPTION_DIR:-${CUSTOM_DOG_ROOT}/ros2/src/custom_dog_description}"
export PYTHONPATH="${CUSTOM_DOG_ROOT}/rl/src:${UNITREE_RL_LAB_ROOT}/source/unitree_rl_lab:${PYTHONPATH:-}"
export ROS_PACKAGE_PATH="${CUSTOM_DOG_ROOT}/ros2/src${UNITREE_ROS_ROOT:+:${UNITREE_ROS_ROOT}/robots}${ROS_PACKAGE_PATH:+:${ROS_PACKAGE_PATH}}"
export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
if [[ -d /usr/lib/wsl/lib ]]; then
    export LD_LIBRARY_PATH="/usr/lib/wsl/lib:${LD_LIBRARY_PATH}"
fi
export TERM="${TERM:-xterm-256color}"

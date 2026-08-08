#!/usr/bin/env bash
set -euo pipefail

check_repo() {
    local name="$1"
    local path="$2"
    local expected="$3"
    local actual

    if [[ ! -d "${path}/.git" ]]; then
        echo "${name}: missing repository at ${path}" >&2
        return 1
    fi
    actual="$(git -C "${path}" rev-parse HEAD)"
    if [[ "${actual}" != "${expected}" ]]; then
        echo "${name}: expected ${expected}, found ${actual}" >&2
        return 1
    fi
    echo "${name}: ${actual}"
}

isaaclab_root="${ISAACLAB_ROOT:-/home/hsc/IsaacLab}"
unitree_rl_lab_root="${UNITREE_RL_LAB_ROOT:-/home/hsc/unitree_rl_lab}"
unitree_mujoco_root="${UNITREE_MUJOCO_ROOT:-/home/hsc/unitree_mujoco}"

check_repo "IsaacLab" "${isaaclab_root}" "3c6e67bb5c7ada942a6d1884ab69338f57596f77"
check_repo "unitree_rl_lab" "${unitree_rl_lab_root}" "4960b84732b0c2ec593dccbfe963fda1bcd7b1e3"
check_repo "unitree_mujoco" "${unitree_mujoco_root}" "ae6a8403e272733e9996ef59990880330496177f"

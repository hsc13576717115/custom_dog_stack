#!/usr/bin/env bash
set -euo pipefail

check_repo() {
    local name="$1"
    local path="$2"
    local expected="$3"
    local actual

    if ! git -C "${path}" rev-parse --git-dir >/dev/null 2>&1; then
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

project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
isaaclab_root="${ISAACLAB_ROOT:-${project_root}/third_party/IsaacLab}"
unitree_rl_lab_root="${UNITREE_RL_LAB_ROOT:-${project_root}/third_party/unitree_rl_lab}"
unitree_mujoco_root="${UNITREE_MUJOCO_ROOT:-${project_root}/third_party/unitree_mujoco}"
dog_control_root="${DOG_CONTROL_ROOT:-${project_root}/third_party/Dog-control}"
unitree_sdk2_root="${UNITREE_SDK2_ROOT:-${project_root}/third_party/unitree_sdk2}"

check_repo "IsaacLab" "${isaaclab_root}" "3c6e67bb5c7ada942a6d1884ab69338f57596f77"
check_repo "unitree_rl_lab" "${unitree_rl_lab_root}" "4960b84732b0c2ec593dccbfe963fda1bcd7b1e3"
check_repo "unitree_mujoco" "${unitree_mujoco_root}" "ae6a8403e272733e9996ef59990880330496177f"
check_repo "Dog-control" "${dog_control_root}" "d4adbbafecd8e60fceb5f820d444090bf3a3e9d5"
check_repo "unitree_sdk2" "${unitree_sdk2_root}" "21d0a3b2c46ee48c8fdf2783becb6be3beb0a59b"

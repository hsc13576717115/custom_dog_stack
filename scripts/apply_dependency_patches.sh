#!/usr/bin/env bash
set -euo pipefail

project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

apply_patch_once() {
    local repository="$1"
    local patch_file="$2"

    if [[ ! -d "${repository}/.git" && ! -f "${repository}/.git" ]]; then
        echo "Missing dependency checkout: ${repository}" >&2
        return 1
    fi
    if git -C "${repository}" apply --check "${patch_file}"; then
        git -C "${repository}" apply "${patch_file}"
        echo "Applied $(basename "${patch_file}")"
    elif git -C "${repository}" apply --reverse --check "${patch_file}"; then
        echo "Already applied: $(basename "${patch_file}")"
    else
        echo "Patch does not match the pinned dependency: ${patch_file}" >&2
        return 1
    fi
}

apply_patch_once \
    "${project_root}/third_party/unitree_rl_lab" \
    "${project_root}/third_party/patches/unitree_rl_lab.patch"
apply_patch_once \
    "${project_root}/third_party/unitree_mujoco" \
    "${project_root}/third_party/patches/unitree_mujoco.patch"

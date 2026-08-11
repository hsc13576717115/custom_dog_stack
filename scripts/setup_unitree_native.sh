#!/usr/bin/env bash
set -euo pipefail

project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
sdk_prefix="${UNITREE_SDK_PREFIX:-${HOME}/.local/unitree_robotics}"
mujoco_version="3.3.6"
mujoco_root="${MUJOCO_NATIVE_ROOT:-${HOME}/.mujoco/mujoco-${mujoco_version}}"
mujoco_link="${project_root}/third_party/unitree_mujoco/simulate/mujoco"

if [[ "$(uname -m)" != "x86_64" ]]; then
    echo "Native MuJoCo setup currently pins the x86_64 release archive." >&2
    exit 1
fi

if [[ ! -d "${mujoco_root}" ]]; then
    archive_url="https://github.com/google-deepmind/mujoco/releases/download/${mujoco_version}/mujoco-${mujoco_version}-linux-x86_64.tar.gz"
    temp_dir="$(mktemp -d)"
    trap 'rm -rf "${temp_dir}"' EXIT
    curl --fail --location "${archive_url}" --output "${temp_dir}/mujoco.tar.gz"
    mkdir -p "$(dirname "${mujoco_root}")"
    tar -xzf "${temp_dir}/mujoco.tar.gz" -C "$(dirname "${mujoco_root}")"
fi

if [[ -L "${mujoco_link}" ]]; then
    if [[ "$(readlink -f "${mujoco_link}")" != "$(readlink -f "${mujoco_root}")" ]]; then
        echo "Existing MuJoCo link points to another release: ${mujoco_link}" >&2
        exit 1
    fi
elif [[ -e "${mujoco_link}" ]]; then
    echo "Refusing to replace existing path: ${mujoco_link}" >&2
    exit 1
else
    ln -s "${mujoco_root}" "${mujoco_link}"
fi

cmake -S "${project_root}/third_party/unitree_mujoco/simulate" \
    -B "${project_root}/third_party/unitree_mujoco/simulate/build" \
    -DCMAKE_BUILD_TYPE=Release \
    -DUNITREE_SDK_PREFIX="${sdk_prefix}"
cmake --build "${project_root}/third_party/unitree_mujoco/simulate/build" \
    --parallel "${CMAKE_BUILD_PARALLEL_LEVEL:-$(nproc)}"

cmake -S "${project_root}/third_party/unitree_rl_lab/deploy/robots/go2" \
    -B "${project_root}/third_party/unitree_rl_lab/deploy/robots/go2/build" \
    -DCMAKE_BUILD_TYPE=Release \
    -DUNITREE_SDK_PREFIX="${sdk_prefix}"
cmake --build "${project_root}/third_party/unitree_rl_lab/deploy/robots/go2/build" \
    --parallel "${CMAKE_BUILD_PARALLEL_LEVEL:-$(nproc)}"

#!/usr/bin/env bash
set -euo pipefail

project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
unitree_root="${UNITREE_MUJOCO_ROOT:-${project_root}/third_party/unitree_mujoco}"
sdk_prefix="${UNITREE_SDK_PREFIX:-${HOME}/.local/unitree_robotics}"
isaac_conda_lib="${ISAAC_CONDA_LIB:-${CONDA_PREFIX:-}/lib}"
binary="${unitree_root}/simulate/build/unitree_mujoco"
model="${project_root}/sim2sim/custom_dog/custom_dog_unitree.xml"

if [[ ! -x "${binary}" ]]; then
    echo "Unitree MuJoCo binary not found: ${binary}" >&2
    exit 1
fi
if [[ ! -f "${model}" ]]; then
    echo "Unitree-compatible MJCF not found. Run ./scripts/generate_mjcf.sh first." >&2
    exit 1
fi

export LD_LIBRARY_PATH="${sdk_prefix}/lib:${unitree_root}/simulate/mujoco/lib:${isaac_conda_lib}${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
exec "${binary}" \
    -i "${CUSTOM_DOG_DDS_DOMAIN:-1}" \
    -n "${CUSTOM_DOG_DDS_INTERFACE:-lo}" \
    -r custom_dog \
    -s "${model}"

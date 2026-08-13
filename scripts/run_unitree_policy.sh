#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 1 ]]; then
    echo "Usage: $0 deploy/candidates/model_N" >&2
    exit 2
fi

project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
unitree_rl_lab_root="${UNITREE_RL_LAB_ROOT:-${project_root}/third_party/unitree_rl_lab}"
sdk_prefix="${UNITREE_SDK_PREFIX:-${HOME}/.local/unitree_robotics}"
env_name="${CUSTOM_DOG_MUJOCO_ENV:-custom_dog_mujoco}"
conda_exe="${CONDA_EXE:-$(command -v conda 2>/dev/null || true)}"
mujoco_python="${CUSTOM_DOG_MUJOCO_PYTHON:-}"
policy_dir="$(realpath "$1")"
runtime_dir="${project_root}/sim2sim/unitree_deploy/build/runtime"
source_binary="${unitree_rl_lab_root}/deploy/robots/go2/build/go2_ctrl"

if [[ -z "${conda_exe}" ]]; then
    for candidate in "${HOME}/miniconda3/bin/conda" "${HOME}/anaconda3/bin/conda" /opt/conda/bin/conda; do
        if [[ -x "${candidate}" ]]; then
            conda_exe="${candidate}"
            break
        fi
    done
fi

if [[ -z "${mujoco_python}" && -n "${conda_exe}" ]]; then
    mujoco_python="$("${conda_exe}" run -n "${env_name}" python -c 'import sys; print(sys.executable)')"
fi

if [[ -z "${mujoco_python}" || ! -x "${mujoco_python}" ]]; then
    echo "MuJoCo environment Python not found: ${mujoco_python}" >&2
    exit 1
fi

controller_binary="$("${mujoco_python}" \
    "${project_root}/sim2sim/unitree_deploy/prepare_controller.py" \
    --template "${project_root}/sim2sim/unitree_deploy/config.template.yaml" \
    --policy-dir "${policy_dir}" \
    --source-binary "${source_binary}" \
    --runtime-dir "${runtime_dir}")"

export LD_LIBRARY_PATH="${unitree_rl_lab_root}/deploy/thirdparty/onnxruntime-linux-x64-1.22.0/lib:${sdk_prefix}/lib${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
exec "${controller_binary}" \
    --domain "${CUSTOM_DOG_DDS_DOMAIN:-1}" \
    --network "${CUSTOM_DOG_DDS_INTERFACE:-lo}" \
    --auto-start

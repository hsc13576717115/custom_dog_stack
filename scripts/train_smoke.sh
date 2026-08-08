#!/usr/bin/env bash
set -euo pipefail

project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source "${project_root}/scripts/activate_env.sh"

if [[ "${OMNI_KIT_ACCEPT_EULA:-}" != "YES" ]]; then
    echo "Set OMNI_KIT_ACCEPT_EULA=YES only after accepting the NVIDIA Omniverse EULA." >&2
    exit 2
fi

sim_device="${CUSTOM_DOG_SIM_DEVICE:-cpu}"
rl_device="${CUSTOM_DOG_RL_DEVICE:-cuda:0}"
num_envs="${CUSTOM_DOG_NUM_ENVS:-64}"
max_iterations="${CUSTOM_DOG_MAX_ITERATIONS:-1}"
task="${CUSTOM_DOG_TASK:-CustomDog-Velocity-v0}"
kit_args="${CUSTOM_DOG_KIT_ARGS:---/renderer/enabled=pxr --/renderer/active=pxr --/renderer/multiGpu/enabled=false}"

cd "${project_root}"
python rl/scripts/train.py \
    --headless \
    --device "${sim_device}" \
    --rl_device "${rl_device}" \
    --task "${task}" \
    --num_envs "${num_envs}" \
    --max_iterations "${max_iterations}" \
    --seed "${CUSTOM_DOG_SEED:-42}" \
    --kit_args "${kit_args}" \
    "$@"

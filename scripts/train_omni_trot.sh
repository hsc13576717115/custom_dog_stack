#!/usr/bin/env bash
set -euo pipefail

project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

export OMNI_KIT_ACCEPT_EULA="${OMNI_KIT_ACCEPT_EULA:-YES}"
export CUSTOM_DOG_TASK="CustomDog-Velocity-OmniTrot-v1"
export CUSTOM_DOG_SIM_DEVICE="${CUSTOM_DOG_SIM_DEVICE:-cuda:0}"
export CUSTOM_DOG_RL_DEVICE="${CUSTOM_DOG_RL_DEVICE:-cuda:0}"
export CUSTOM_DOG_NUM_ENVS="${CUSTOM_DOG_NUM_ENVS:-4096}"
export CUSTOM_DOG_MAX_ITERATIONS="${CUSTOM_DOG_MAX_ITERATIONS:-5000}"

exec "${project_root}/scripts/train.sh" \
    --run_name "${CUSTOM_DOG_RUN_NAME:-omni_trot_v1_main}" \
    "$@"

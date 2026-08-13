#!/usr/bin/env bash
set -euo pipefail

project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source_run="${CUSTOM_DOG_SOURCE_RUN:-2026-08-12_23-22-07_omni45_v2_main}"
source_checkpoint="${CUSTOM_DOG_SOURCE_CHECKPOINT:-model_4960.pt}"

export OMNI_KIT_ACCEPT_EULA="${OMNI_KIT_ACCEPT_EULA:-YES}"
export CUSTOM_DOG_TASK="CustomDog-Velocity-Omni45-HighSpeed-v1"
export CUSTOM_DOG_SIM_DEVICE="${CUSTOM_DOG_SIM_DEVICE:-cuda:0}"
export CUSTOM_DOG_RL_DEVICE="${CUSTOM_DOG_RL_DEVICE:-cuda:0}"
export CUSTOM_DOG_NUM_ENVS="${CUSTOM_DOG_NUM_ENVS:-4096}"
export CUSTOM_DOG_MAX_ITERATIONS="${CUSTOM_DOG_MAX_ITERATIONS:-5000}"
# The new task has a deliberately different optimizer configuration.
export CUSTOM_DOG_LOAD_OPTIMIZER="${CUSTOM_DOG_LOAD_OPTIMIZER:-0}"

exec "${project_root}/scripts/train.sh" \
    --run_name "${CUSTOM_DOG_RUN_NAME:-omni45_high_speed_v1}" \
    --resume \
    --load_run "${source_run}" \
    --checkpoint "${source_checkpoint}" \
    "$@"

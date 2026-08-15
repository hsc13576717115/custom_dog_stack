#!/usr/bin/env bash
set -euo pipefail

: "${CUSTOM_DOG_SOURCE_RUN:?Set CUSTOM_DOG_SOURCE_RUN to the accepted 51-D teacher run}"
: "${CUSTOM_DOG_SOURCE_CHECKPOINT:?Set CUSTOM_DOG_SOURCE_CHECKPOINT to its accepted model_N.pt}"

project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export OMNI_KIT_ACCEPT_EULA="${OMNI_KIT_ACCEPT_EULA:-YES}"
export CUSTOM_DOG_TASK="CustomDog-Velocity-ClosedLoop-History213-Distill-v1"
export CUSTOM_DOG_SIM_DEVICE="${CUSTOM_DOG_SIM_DEVICE:-cuda:0}"
export CUSTOM_DOG_RL_DEVICE="${CUSTOM_DOG_RL_DEVICE:-cuda:0}"
export CUSTOM_DOG_NUM_ENVS="${CUSTOM_DOG_NUM_ENVS:-4096}"
export CUSTOM_DOG_MAX_ITERATIONS="${CUSTOM_DOG_MAX_ITERATIONS:-800}"

exec "${project_root}/scripts/train.sh" \
    --run_name "${CUSTOM_DOG_RUN_NAME:-closed_loop_history213_distill_seed42}" \
    --load_run "${CUSTOM_DOG_SOURCE_RUN}" \
    --checkpoint "${CUSTOM_DOG_SOURCE_CHECKPOINT}" \
    "$@"

#!/usr/bin/env bash
set -euo pipefail

project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source_run="${CUSTOM_DOG_SOURCE_RUN:-2026-08-14_17-10-48_closed_loop_a1_yaw_height_from975}"
source_checkpoint="${CUSTOM_DOG_SOURCE_CHECKPOINT:-model_1150.pt}"

export OMNI_KIT_ACCEPT_EULA="${OMNI_KIT_ACCEPT_EULA:-YES}"
export CUSTOM_DOG_TASK="CustomDog-Velocity-OmniTrot-ClosedLoopPolishA2-v1"
export CUSTOM_DOG_SIM_DEVICE="${CUSTOM_DOG_SIM_DEVICE:-cuda:0}"
export CUSTOM_DOG_RL_DEVICE="${CUSTOM_DOG_RL_DEVICE:-cuda:0}"
export CUSTOM_DOG_NUM_ENVS="${CUSTOM_DOG_NUM_ENVS:-4096}"
export CUSTOM_DOG_MAX_ITERATIONS="${CUSTOM_DOG_MAX_ITERATIONS:-300}"
export CUSTOM_DOG_LOAD_OPTIMIZER="${CUSTOM_DOG_LOAD_OPTIMIZER:-0}"
export CUSTOM_DOG_RESET_POLICY_STD="${CUSTOM_DOG_RESET_POLICY_STD:-0.18}"

exec "${project_root}/scripts/train.sh" \
    --run_name "${CUSTOM_DOG_RUN_NAME:-closed_loop_a2_yaw_boundary_standing_band}" \
    --resume \
    --load_run "${source_run}" \
    --checkpoint "${source_checkpoint}" \
    "$@"

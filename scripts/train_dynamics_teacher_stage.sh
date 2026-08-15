#!/usr/bin/env bash
set -euo pipefail

: "${CUSTOM_DOG_SOURCE_RUN:?Set CUSTOM_DOG_SOURCE_RUN to an accepted gait run}"
: "${CUSTOM_DOG_SOURCE_CHECKPOINT:?Set CUSTOM_DOG_SOURCE_CHECKPOINT to its accepted model_N.pt}"

project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
log_root="${project_root}/logs/rsl_rl/custom_dog_velocity"
source_run_dir="${log_root}/${CUSTOM_DOG_SOURCE_RUN}"
source_checkpoint="${source_run_dir}/${CUSTOM_DOG_SOURCE_CHECKPOINT}"
expanded_checkpoint="${source_run_dir}/${CUSTOM_DOG_SOURCE_CHECKPOINT%.pt}_dyn62.pt"

if [[ ! -f "${expanded_checkpoint}" ]]; then
    python3 "${project_root}/scripts/expand_rsl_rl_actor_observation.py" \
        "${source_checkpoint}" "${expanded_checkpoint}" \
        --source-dim 51 --target-dim 62
fi

export OMNI_KIT_ACCEPT_EULA="${OMNI_KIT_ACCEPT_EULA:-YES}"
export CUSTOM_DOG_TASK="CustomDog-Velocity-OmniTrot-DynamicsTeacher-v1"
export CUSTOM_DOG_SIM_DEVICE="${CUSTOM_DOG_SIM_DEVICE:-cuda:0}"
export CUSTOM_DOG_RL_DEVICE="${CUSTOM_DOG_RL_DEVICE:-cuda:0}"
export CUSTOM_DOG_NUM_ENVS="${CUSTOM_DOG_NUM_ENVS:-4096}"
export CUSTOM_DOG_MAX_ITERATIONS="${CUSTOM_DOG_MAX_ITERATIONS:-500}"
export CUSTOM_DOG_LOAD_OPTIMIZER=0
export CUSTOM_DOG_RESET_POLICY_STD="${CUSTOM_DOG_RESET_POLICY_STD:-0.25}"

exec "${project_root}/scripts/train.sh" \
    --run_name "${CUSTOM_DOG_RUN_NAME:-dynamics_teacher_seed42}" \
    --resume \
    --load_run "${CUSTOM_DOG_SOURCE_RUN}" \
    --checkpoint "$(basename "${expanded_checkpoint}")" \
    "$@"

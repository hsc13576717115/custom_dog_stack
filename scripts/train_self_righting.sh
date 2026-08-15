#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 1 ]]; then
    echo "Usage: $0 R0|R1|R2 [train.py arguments]" >&2
    exit 2
fi

stage="${1^^}"
shift
case "${stage}" in
    R0)
        default_iterations=1500
        default_run_name=self_righting_r0_seed42
        ;;
    R1)
        default_iterations=1000
        default_run_name=self_righting_r1_seed42
        ;;
    R2)
        default_iterations=1000
        default_run_name=self_righting_r2_seed42
        ;;
    *)
        echo "Unknown self-righting stage: ${stage}; expected R0, R1, or R2" >&2
        exit 2
        ;;
esac

project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export OMNI_KIT_ACCEPT_EULA="${OMNI_KIT_ACCEPT_EULA:-YES}"
export CUSTOM_DOG_TASK="CustomDog-SelfRighting-${stage}-v2"
export CUSTOM_DOG_SIM_DEVICE="${CUSTOM_DOG_SIM_DEVICE:-cuda:0}"
export CUSTOM_DOG_RL_DEVICE="${CUSTOM_DOG_RL_DEVICE:-cuda:0}"
export CUSTOM_DOG_NUM_ENVS="${CUSTOM_DOG_NUM_ENVS:-4096}"
export CUSTOM_DOG_MAX_ITERATIONS="${CUSTOM_DOG_MAX_ITERATIONS:-${default_iterations}}"

resume_args=()
if [[ "${stage}" != "R0" ]]; then
    : "${CUSTOM_DOG_SOURCE_RUN:?Set CUSTOM_DOG_SOURCE_RUN to the accepted previous-stage run}"
    : "${CUSTOM_DOG_SOURCE_CHECKPOINT:?Set CUSTOM_DOG_SOURCE_CHECKPOINT to its accepted model_N.pt}"
    export CUSTOM_DOG_LOAD_OPTIMIZER="${CUSTOM_DOG_LOAD_OPTIMIZER:-0}"
    export CUSTOM_DOG_RESET_POLICY_STD="${CUSTOM_DOG_RESET_POLICY_STD:-0.35}"
    resume_args=(
        --resume
        --load_run "${CUSTOM_DOG_SOURCE_RUN}"
        --checkpoint "${CUSTOM_DOG_SOURCE_CHECKPOINT}"
    )
fi

exec "${project_root}/scripts/train.sh" \
    --run_name "${CUSTOM_DOG_RUN_NAME:-${default_run_name}}" \
    "${resume_args[@]}" \
    "$@"

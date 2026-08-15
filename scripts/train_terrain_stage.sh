#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 1 ]]; then
    echo "Usage: $0 T0|T1 [train.py arguments]" >&2
    exit 2
fi

stage="${1^^}"
shift
case "${stage}" in
    T0)
        default_iterations=800
        ;;
    T1)
        default_iterations=1000
        ;;
    *)
        echo "Unknown terrain stage: ${stage}; expected T0 or T1" >&2
        exit 2
        ;;
esac

: "${CUSTOM_DOG_SOURCE_RUN:?Set CUSTOM_DOG_SOURCE_RUN to an accepted source run}"
: "${CUSTOM_DOG_SOURCE_CHECKPOINT:?Set CUSTOM_DOG_SOURCE_CHECKPOINT to its accepted model_N.pt}"

project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export OMNI_KIT_ACCEPT_EULA="${OMNI_KIT_ACCEPT_EULA:-YES}"
export CUSTOM_DOG_TASK="CustomDog-Velocity-OmniTrot-Terrain${stage}-v1"
export CUSTOM_DOG_SIM_DEVICE="${CUSTOM_DOG_SIM_DEVICE:-cuda:0}"
export CUSTOM_DOG_RL_DEVICE="${CUSTOM_DOG_RL_DEVICE:-cuda:0}"
export CUSTOM_DOG_NUM_ENVS="${CUSTOM_DOG_NUM_ENVS:-4096}"
export CUSTOM_DOG_MAX_ITERATIONS="${CUSTOM_DOG_MAX_ITERATIONS:-${default_iterations}}"
export CUSTOM_DOG_LOAD_OPTIMIZER="${CUSTOM_DOG_LOAD_OPTIMIZER:-0}"
export CUSTOM_DOG_RESET_POLICY_STD="${CUSTOM_DOG_RESET_POLICY_STD:-0.35}"

exec "${project_root}/scripts/train.sh" \
    --run_name "${CUSTOM_DOG_RUN_NAME:-terrain_${stage,,}_seed42}" \
    --resume \
    --load_run "${CUSTOM_DOG_SOURCE_RUN}" \
    --checkpoint "${CUSTOM_DOG_SOURCE_CHECKPOINT}" \
    "$@"

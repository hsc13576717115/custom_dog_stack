#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 1 ]]; then
    echo "Usage: $0 /absolute/path/to/model_N.pt [play.py arguments]" >&2
    exit 2
fi

project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source "${project_root}/scripts/activate_env.sh"

if [[ "${OMNI_KIT_ACCEPT_EULA:-}" != "YES" ]]; then
    echo "Set OMNI_KIT_ACCEPT_EULA=YES only after accepting the NVIDIA Omniverse EULA." >&2
    exit 2
fi

checkpoint="$(realpath "$1")"
shift
task="${CUSTOM_DOG_TASK:-CustomDog-Velocity-v0}"
kit_args="${CUSTOM_DOG_KIT_ARGS:---/renderer/enabled=pxr --/renderer/active=pxr --/renderer/multiGpu/enabled=false}"

cd "${project_root}"
python rl/scripts/play.py \
    --headless \
    --device "${CUSTOM_DOG_SIM_DEVICE:-cpu}" \
    --rl_device "${CUSTOM_DOG_RL_DEVICE:-${CUSTOM_DOG_SIM_DEVICE:-cpu}}" \
    --task "${task}" \
    --num_envs 1 \
    --checkpoint "${checkpoint}" \
    --max_steps "${CUSTOM_DOG_PLAY_STEPS:-2}" \
    --kit_args "${kit_args}" \
    "$@"

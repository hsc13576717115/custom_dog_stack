#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 1 ]]; then
    echo "Usage: $0 /absolute/path/to/model_N.pt" >&2
    exit 2
fi

project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source "${project_root}/scripts/activate_env.sh"

if [[ "${OMNI_KIT_ACCEPT_EULA:-}" != "YES" ]]; then
    echo "Set OMNI_KIT_ACCEPT_EULA=YES after accepting the NVIDIA Omniverse EULA." >&2
    exit 2
fi

checkpoint="$(realpath "$1")"
kit_args="${CUSTOM_DOG_KIT_ARGS:---/renderer/enabled=pxr --/renderer/active=pxr --/renderer/multiGpu/enabled=false}"

echo "Starting Isaac Lab GUI playback for ${checkpoint}"
echo "Close the Isaac Sim window to stop playback."
cd "${project_root}"
exec python rl/scripts/play.py \
    --device "${CUSTOM_DOG_SIM_DEVICE:-cpu}" \
    --task CustomDog-Velocity-v0 \
    --num_envs 1 \
    --checkpoint "${checkpoint}" \
    --real-time \
    --kit_args "${kit_args}"

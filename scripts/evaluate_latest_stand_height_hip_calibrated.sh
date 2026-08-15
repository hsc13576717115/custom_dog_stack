#!/usr/bin/env bash
set -euo pipefail

project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
log_root="${project_root}/logs/rsl_rl/custom_dog_velocity"
mapfile -t runs < <(
    find "${log_root}" -mindepth 1 -maxdepth 1 -type d \
        -name '*_stand_height_hip_calibrated_from319' -printf '%T@ %p\n' \
        | sort -n
)
if [[ ${#runs[@]} -eq 0 ]]; then
    echo "No completed v3 stand run was found." >&2
    exit 1
fi

CUSTOM_DOG_RUN_DIR="${runs[-1]#* }" \
exec "${project_root}/scripts/evaluate_stand_height_hip_calibrated.sh"

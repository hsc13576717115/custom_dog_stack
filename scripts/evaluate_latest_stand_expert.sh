#!/usr/bin/env bash
set -euo pipefail

project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
log_root="${project_root}/logs/rsl_rl/custom_dog_velocity"
mapfile -t runs < <(
    find "${log_root}" -mindepth 1 -maxdepth 1 -type d \
        -name '*_stand_expert_seed42' -printf '%T@ %p\n' \
        | sort -n
)
if [[ ${#runs[@]} -eq 0 ]]; then
    echo "No completed stand-expert run was found." >&2
    exit 1
fi

run_dir="${runs[-1]#* }"
CUSTOM_DOG_RUN_DIR="${run_dir}" exec "${project_root}/scripts/evaluate_stand_expert.sh"

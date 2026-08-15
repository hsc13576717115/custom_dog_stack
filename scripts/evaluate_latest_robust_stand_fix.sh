#!/usr/bin/env bash
set -euo pipefail

project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
log_root="${project_root}/logs/rsl_rl/custom_dog_velocity"
mapfile -t runs < <(
    find "${log_root}" -mindepth 1 -maxdepth 1 -type d \
        -name '*_robust_stand_fix_from_rf700' -printf '%T@ %p\n' \
        | sort -n
)
if [[ ${#runs[@]} -eq 0 ]]; then
    echo "No completed RobustStandFix run was found." >&2
    exit 1
fi

run_dir="${runs[-1]#* }"
CUSTOM_DOG_RUN_DIR="${run_dir}" \
    exec "${project_root}/scripts/evaluate_robust_stand_fix.sh"

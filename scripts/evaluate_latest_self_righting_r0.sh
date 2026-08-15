#!/usr/bin/env bash
set -euo pipefail

project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
log_root="${project_root}/logs/rsl_rl/custom_dog_velocity"
mapfile -t runs < <(
    find "${log_root}" -mindepth 1 -maxdepth 1 -type d -name '*_self_righting_r0_seed42' \
        -printf '%T@ %p\n' | sort -n
)
if [[ ${#runs[@]} -eq 0 ]]; then
    echo "No formal R0 run was produced; skipping R0 evaluation."
    exit 0
fi
"${project_root}/scripts/evaluate_self_righting_run.sh" R0 "${runs[-1]#* }"

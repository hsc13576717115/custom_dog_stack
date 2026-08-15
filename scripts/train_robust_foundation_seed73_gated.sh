#!/usr/bin/env bash
set -euo pipefail

project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
log_root="${project_root}/logs/rsl_rl/custom_dog_velocity"
mapfile -t stand_selections < <(
    find "${log_root}" -mindepth 3 -maxdepth 3 -type f \
        -path '*/evaluation/stand_selection.json' \
        -printf '%T@ %p\n' | sort -n
)

if [[ ${#stand_selections[@]} -eq 0 ]]; then
    echo "An evaluated seed-42 stand expert is missing; seed 73 is blocked." >&2
    exit 1
fi
stand_selection="${stand_selections[-1]#* }"
routed_selection="$(dirname "${stand_selection}")/routed_stage_a_selection.json"
if [[ ! -f "${routed_selection}" ]] || ! jq -e \
    '.accepted == true and (.selected | startswith("ROUTED_"))' \
    "${routed_selection}" >/dev/null; then
    echo "The seed-42 routed Stage A candidate did not pass; blocking seed 73." >&2
    exit 1
fi
if ! jq -e '.accepted == true and .selected_candidate != null' \
    "${stand_selection}" >/dev/null; then
    echo "The seed-42 stand expert did not pass; blocking seed 73." >&2
    exit 1
fi
stand_candidate="$(jq -r '.selected_candidate' "${stand_selection}")"

CUSTOM_DOG_SEED=73 \
CUSTOM_DOG_RUN_NAME=closed_loop_robust_foundation_seed73 \
"${project_root}/scripts/train_omni_trot_closed_loop_robust_foundation.sh"

mapfile -t runs < <(
    find "${log_root}" -mindepth 1 -maxdepth 1 -type d \
        -name '*_closed_loop_robust_foundation_seed73' -printf '%T@ %p\n' \
        | sort -n
)
if [[ ${#runs[@]} -eq 0 ]]; then
    echo "Could not locate the completed seed-73 run." >&2
    exit 1
fi
seed73_run="${runs[-1]#* }"
CUSTOM_DOG_RUN_DIR="${seed73_run}" \
CUSTOM_DOG_STAND_CANDIDATE="${stand_candidate}" \
"${project_root}/scripts/evaluate_routed_robust_foundation.sh"

#!/usr/bin/env bash
set -euo pipefail

project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
log_root="${project_root}/logs/rsl_rl/custom_dog_velocity"
mapfile -t terrain_selections < <(
    find "${log_root}" -mindepth 3 -maxdepth 3 -type f \
        -path '*_terrain_t1_seed42/evaluation/terrain_t1_selection.json' \
        -printf '%T@ %p\n' | sort -n
)
mapfile -t stand_selections < <(
    find "${log_root}" -mindepth 3 -maxdepth 3 -type f \
        -path '*/evaluation/stand_selection.json' -printf '%T@ %p\n' | sort -n
)
if [[ ${#terrain_selections[@]} -eq 0 || ${#stand_selections[@]} -eq 0 ]]; then
    echo "Accepted T1 and stand selections are required; blocking final distillation." >&2
    exit 1
fi
selection="${terrain_selections[-1]#* }"
stand_selection="${stand_selections[-1]#* }"
if ! jq -e '.accepted == true and (.selected | startswith("T1_"))' "${selection}" >/dev/null \
    || ! jq -e '.accepted == true and .selected_candidate != null' "${stand_selection}" >/dev/null; then
    echo "T1 or stand expert did not pass; blocking final distillation." >&2
    exit 1
fi
python3 "${project_root}/scripts/check_selection_training_asset.py" \
    "${selection}" --expected custom_dog_selective_collision.urdf

source_candidate="$(jq -r '.selected_candidate' "${selection}")"
stand_candidate="$(jq -r '.selected_candidate' "${stand_selection}")"
source_checkpoint="$(basename "${source_candidate}").pt"
source_run_dir="$(dirname "$(dirname "$(dirname "${source_candidate}")")")"

CUSTOM_DOG_SOURCE_RUN="$(basename "${source_run_dir}")" \
CUSTOM_DOG_SOURCE_CHECKPOINT="${source_checkpoint}" \
CUSTOM_DOG_RUN_NAME=closed_loop_history213_final_seed42 \
"${project_root}/scripts/distill_closed_loop_history213.sh"

mapfile -t runs < <(
    find "${log_root}" -mindepth 1 -maxdepth 1 -type d \
        -name '*_closed_loop_history213_final_seed42' -printf '%T@ %p\n' | sort -n
)
if [[ ${#runs[@]} -eq 0 ]]; then
    echo "Could not locate completed final history-distillation run." >&2
    exit 1
fi
CUSTOM_DOG_STAND_CANDIDATE="${stand_candidate}" \
"${project_root}/scripts/evaluate_history213_distillation.sh" \
    "${runs[-1]#* }" "${source_candidate}"

#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 1 ]]; then
    echo "Usage: $0 B|C|D" >&2
    exit 2
fi
stage="${1^^}"
case "${stage}" in B) previous=collision ;; C) previous=b ;; D) previous=c ;; *) echo "Expected B, C, or D" >&2; exit 2 ;; esac

project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
log_root="${project_root}/logs/rsl_rl/custom_dog_velocity"
mapfile -t stand_selections < <(
    find "${log_root}" -mindepth 3 -maxdepth 3 -type f \
        -path '*/evaluation/stand_selection.json' \
        -printf '%T@ %p\n' | sort -n
)
if [[ ${#stand_selections[@]} -eq 0 ]]; then
    echo "No accepted stand expert is available for Stage ${stage}; blocking." >&2
    exit 1
fi
stand_selection="${stand_selections[-1]#* }"
if ! jq -e '.accepted == true and .selected_candidate != null' \
    "${stand_selection}" >/dev/null; then
    echo "The stand expert did not pass; blocking Stage ${stage}." >&2
    exit 1
fi
stand_candidate="$(jq -r '.selected_candidate' "${stand_selection}")"
if [[ "${previous}" == collision ]]; then
    python3 "${project_root}/scripts/check_selective_collision_report.py" \
        "${project_root}/reports/selective_collision_isaac_runtime.json" \
        --asset "${project_root}/ros2/src/custom_dog_description/urdf/custom_dog_selective_collision.urdf"
    run_pattern='*_closed_loop_selective_collision_stage_a'
    selection_name=selective_collision_selection.json
else
    run_pattern="*_closed_loop_stage_${previous}_seed42"
    selection_name="closed_loop_stage_${previous}_selection.json"
fi
mapfile -t selections < <(
    find "${log_root}" -mindepth 3 -maxdepth 3 -type f \
        -path "${run_pattern}/evaluation/${selection_name}" -printf '%T@ %p\n' \
        | sort -n
)
if [[ ${#selections[@]} -eq 0 ]]; then
    echo "No accepted source selection exists for Stage ${stage}; blocking." >&2
    exit 1
fi
selection="${selections[-1]#* }"
if ! jq -e '.accepted == true and .selected_candidate != null' "${selection}" >/dev/null; then
    echo "Previous stage did not pass; blocking Stage ${stage}." >&2
    exit 1
fi
python3 "${project_root}/scripts/check_selection_training_asset.py" \
    "${selection}" --expected custom_dog_selective_collision.urdf
if [[ "${previous}" == collision ]] \
    && ! jq -e '.selected | startswith("SC_")' "${selection}" >/dev/null; then
    echo "The collision-trained candidates did not win Stage A; blocking Stage B." >&2
    exit 1
fi
source_candidate="$(jq -r '.selected_candidate' "${selection}")"
source_label="$(basename "${source_candidate}")"
source_checkpoint="${source_label}.pt"
source_run_dir="$(dirname "$(dirname "$(dirname "${source_candidate}")")")"

CUSTOM_DOG_SOURCE_RUN="$(basename "${source_run_dir}")" \
CUSTOM_DOG_SOURCE_CHECKPOINT="${source_checkpoint}" \
CUSTOM_DOG_RUN_NAME="closed_loop_stage_${stage,,}_seed42" \
"${project_root}/scripts/train_closed_loop_stage.sh" "${stage}"

mapfile -t runs < <(
    find "${log_root}" -mindepth 1 -maxdepth 1 -type d \
        -name "*_closed_loop_stage_${stage,,}_seed42" -printf '%T@ %p\n' | sort -n
)
if [[ ${#runs[@]} -eq 0 ]]; then
    echo "Could not locate completed Stage ${stage} run." >&2
    exit 1
fi
CUSTOM_DOG_STAND_CANDIDATE="${stand_candidate}" \
"${project_root}/scripts/evaluate_closed_loop_stage.sh" \
    "${stage}" "${runs[-1]#* }" "${source_candidate}"

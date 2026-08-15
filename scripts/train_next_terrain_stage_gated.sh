#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 1 ]]; then
    echo "Usage: $0 T0|T1" >&2
    exit 2
fi
stage="${1^^}"
case "${stage}" in T0|T1) ;; *) echo "Expected T0 or T1" >&2; exit 2 ;; esac
project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
log_root="${project_root}/logs/rsl_rl/custom_dog_velocity"

mapfile -t stand_selections < <(
    find "${log_root}" -mindepth 3 -maxdepth 3 -type f \
        -path '*/evaluation/stand_selection.json' -printf '%T@ %p\n' | sort -n
)
if [[ ${#stand_selections[@]} -eq 0 ]]; then
    echo "No accepted stand expert; blocking ${stage}." >&2
    exit 1
fi
stand_selection="${stand_selections[-1]#* }"
if ! jq -e '.accepted == true and .selected_candidate != null' "${stand_selection}" >/dev/null; then
    echo "Stand expert did not pass; blocking ${stage}." >&2
    exit 1
fi
stand_candidate="$(jq -r '.selected_candidate' "${stand_selection}")"

if [[ "${stage}" == T0 ]]; then
    pattern='*_dynamics_teacher_seed42/evaluation/dynamics_teacher_selection.json'
    prefix='DYN_'
else
    pattern='*_terrain_t0_seed42/evaluation/terrain_t0_selection.json'
    prefix='T0_'
fi
mapfile -t selections < <(
    find "${log_root}" -mindepth 3 -maxdepth 3 -type f \
        -path "${pattern}" -printf '%T@ %p\n' | sort -n
)
if [[ ${#selections[@]} -eq 0 ]]; then
    echo "No accepted source selection for ${stage}; blocking." >&2
    exit 1
fi
selection="${selections[-1]#* }"
if ! jq -e --arg prefix "${prefix}" \
    '.accepted == true and (.selected | startswith($prefix)) and .selected_candidate != null' \
    "${selection}" >/dev/null; then
    echo "Previous terrain admission gate did not pass; blocking ${stage}." >&2
    exit 1
fi
python3 "${project_root}/scripts/check_selection_training_asset.py" \
    "${selection}" --expected custom_dog_selective_collision.urdf
source_candidate="$(jq -r '.selected_candidate' "${selection}")"
source_checkpoint="$(basename "${source_candidate}").pt"
source_run_dir="$(dirname "$(dirname "$(dirname "${source_candidate}")")")"

CUSTOM_DOG_SOURCE_RUN="$(basename "${source_run_dir}")" \
CUSTOM_DOG_SOURCE_CHECKPOINT="${source_checkpoint}" \
CUSTOM_DOG_RUN_NAME="terrain_${stage,,}_seed42" \
"${project_root}/scripts/train_terrain_stage.sh" "${stage}"

mapfile -t runs < <(
    find "${log_root}" -mindepth 1 -maxdepth 1 -type d \
        -name "*_terrain_${stage,,}_seed42" -printf '%T@ %p\n' | sort -n
)
if [[ ${#runs[@]} -eq 0 ]]; then
    echo "Could not locate completed ${stage} run." >&2
    exit 1
fi
CUSTOM_DOG_STAND_CANDIDATE="${stand_candidate}" \
"${project_root}/scripts/evaluate_terrain_stage.sh" \
    "${stage}" "${runs[-1]#* }" "${source_candidate}"

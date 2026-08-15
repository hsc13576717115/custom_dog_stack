#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 1 ]]; then
    echo "Usage: $0 R1|R2" >&2
    exit 2
fi
stage="${1^^}"
case "${stage}" in R1) previous=R0 ;; R2) previous=R1 ;; *) echo "Expected R1 or R2" >&2; exit 2 ;; esac

project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
log_root="${project_root}/logs/rsl_rl/custom_dog_velocity"
mapfile -t selections < <(
    find "${log_root}" -mindepth 3 -maxdepth 3 -type f \
        -path "*_self_righting_${previous,,}_seed42/evaluation/self_righting_${previous,,}_selection.json" \
        -printf '%T@ %p\n' | sort -n
)
if [[ ${#selections[@]} -eq 0 ]]; then
    echo "No ${previous} selection exists; blocking ${stage}." >&2
    exit 1
fi
selection="${selections[-1]#* }"
if ! jq -e '.accepted == true' "${selection}" >/dev/null; then
    echo "${previous} did not pass all canonical recovery poses; blocking ${stage}." >&2
    exit 1
fi
python3 "${project_root}/scripts/check_selection_training_asset.py" \
    "${selection}" --expected custom_dog_selective_collision.urdf
source_run_dir="$(dirname "$(dirname "${selection}")")"
source_checkpoint="$(jq -r '.selected' "${selection}").pt"

CUSTOM_DOG_SOURCE_RUN="$(basename "${source_run_dir}")" \
CUSTOM_DOG_SOURCE_CHECKPOINT="${source_checkpoint}" \
CUSTOM_DOG_RUN_NAME="self_righting_${stage,,}_seed42" \
"${project_root}/scripts/train_self_righting.sh" "${stage}"

mapfile -t runs < <(
    find "${log_root}" -mindepth 1 -maxdepth 1 -type d -name "*_self_righting_${stage,,}_seed42" \
        -printf '%T@ %p\n' | sort -n
)
if [[ ${#runs[@]} -eq 0 ]]; then
    echo "Could not locate the completed ${stage} run." >&2
    exit 1
fi
"${project_root}/scripts/evaluate_self_righting_run.sh" "${stage}" "${runs[-1]#* }"

if [[ "${stage}" == R2 ]]; then
    "${project_root}/scripts/evaluate_recovery_handoff_gated.sh" "${runs[-1]#* }"
fi

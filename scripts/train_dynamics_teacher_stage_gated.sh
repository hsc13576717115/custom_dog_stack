#!/usr/bin/env bash
set -euo pipefail

project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
log_root="${project_root}/logs/rsl_rl/custom_dog_velocity"
mapfile -t gait_selections < <(
    find "${log_root}" -mindepth 3 -maxdepth 3 -type f \
        -path '*_closed_loop_gait_robust_seed42/evaluation/gait_robust_selection.json' \
        -printf '%T@ %p\n' | sort -n
)
mapfile -t stand_selections < <(
    find "${log_root}" -mindepth 3 -maxdepth 3 -type f \
        -path '*/evaluation/stand_selection.json' -printf '%T@ %p\n' | sort -n
)
if [[ ${#gait_selections[@]} -eq 0 || ${#stand_selections[@]} -eq 0 ]]; then
    echo "Accepted gait and stand selections are required; blocking dynamics teacher." >&2
    exit 1
fi
selection="${gait_selections[-1]#* }"
stand_selection="${stand_selections[-1]#* }"
if ! jq -e '.accepted == true and (.selected | startswith("GR_"))' "${selection}" >/dev/null \
    || ! jq -e '.accepted == true and .selected_candidate != null' "${stand_selection}" >/dev/null; then
    echo "Gait or stand admission gate failed; blocking dynamics teacher." >&2
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
"${project_root}/scripts/train_dynamics_teacher_stage.sh"

mapfile -t runs < <(
    find "${log_root}" -mindepth 1 -maxdepth 1 -type d \
        -name '*_dynamics_teacher_seed42' -printf '%T@ %p\n' | sort -n
)
if [[ ${#runs[@]} -eq 0 ]]; then
    echo "Could not locate the completed dynamics-teacher run." >&2
    exit 1
fi
CUSTOM_DOG_STAND_CANDIDATE="${stand_candidate}" \
CUSTOM_DOG_EVAL_TASK="CustomDog-Velocity-OmniTrot-DynamicsTeacher-v1" \
CUSTOM_DOG_EVAL_LABEL_PREFIX="DYN_" \
CUSTOM_DOG_EVAL_OUTPUT_STEM="dynamics_teacher" \
"${project_root}/scripts/evaluate_gait_robust_stage.sh" \
    "${runs[-1]#* }" "${source_candidate}"

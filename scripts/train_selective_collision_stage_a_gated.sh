#!/usr/bin/env bash
set -euo pipefail

project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
log_root="${project_root}/logs/rsl_rl/custom_dog_velocity"
seed42_run="${log_root}/2026-08-14_19-59-12_closed_loop_robust_foundation_seed42"
mapfile -t stand_selections < <(
    find "${log_root}" -mindepth 3 -maxdepth 3 -type f \
        -path '*/evaluation/stand_selection.json' \
        -printf '%T@ %p\n' | sort -n
)

mapfile -t seed73_selections < <(
    find "${log_root}" -mindepth 3 -maxdepth 3 -type f \
        -path '*_closed_loop_robust_foundation_seed73/evaluation/routed_robust_foundation_selection.json' \
        -printf '%T@ %p\n' | sort -n
)
if [[ ${#stand_selections[@]} -eq 0 || ${#seed73_selections[@]} -eq 0 ]]; then
    echo "Two evaluated routed Robust Foundation seeds are not available; blocking collision adaptation." >&2
    exit 1
fi
stand_selection="${stand_selections[-1]#* }"
seed42_selection="$(dirname "${stand_selection}")/routed_stage_a_selection.json"
seed73_selection="${seed73_selections[-1]#* }"
if [[ ! -f "${seed42_selection}" ]] \
    || ! jq -e '.accepted == true and (.selected | startswith("ROUTED_"))' \
        "${seed42_selection}" >/dev/null \
    || ! jq -e '.accepted == true and (.selected | startswith("RFR_"))' \
        "${seed73_selection}" >/dev/null \
    || ! jq -e '.accepted == true and .selected_candidate != null' \
        "${stand_selection}" >/dev/null; then
    echo "Both routed Robust Foundation seeds and the stand expert did not pass; blocking collision adaptation." >&2
    exit 1
fi
source_checkpoint="model_700.pt"
source_candidate="$(jq -r '.selected_candidate' "${seed42_selection}")"
stand_candidate="$(jq -r '.selected_candidate' "${stand_selection}")"

# Revalidate the live PhysX filtering contract against the current asset before
# spending a training run on it.  The report is also the admission artifact for
# the independent self-righting branch.
source "${project_root}/scripts/activate_env.sh"
OMNI_KIT_ACCEPT_EULA="${OMNI_KIT_ACCEPT_EULA:-YES}" \
python "${project_root}/rl/scripts/validate_selective_self_collision.py" \
    --headless \
    --device "${CUSTOM_DOG_SIM_DEVICE:-cuda:0}" \
    --output "${project_root}/reports/selective_collision_isaac_runtime.json"
python3 "${project_root}/scripts/check_selective_collision_report.py" \
    "${project_root}/reports/selective_collision_isaac_runtime.json" \
    --asset "${project_root}/ros2/src/custom_dog_description/urdf/custom_dog_selective_collision.urdf"

CUSTOM_DOG_SOURCE_RUN="$(basename "${seed42_run}")" \
CUSTOM_DOG_SOURCE_CHECKPOINT="${source_checkpoint}" \
CUSTOM_DOG_RUN_NAME=closed_loop_selective_collision_stage_a \
"${project_root}/scripts/train_selective_collision_stage_a.sh"

mapfile -t collision_runs < <(
    find "${log_root}" -mindepth 1 -maxdepth 1 -type d \
        -name '*_closed_loop_selective_collision_stage_a' -printf '%T@ %p\n' \
        | sort -n
)
if [[ ${#collision_runs[@]} -eq 0 ]]; then
    echo "Could not locate the completed selective-collision run." >&2
    exit 1
fi
collision_run="${collision_runs[-1]#* }"
CUSTOM_DOG_RUN_DIR="${collision_run}" \
CUSTOM_DOG_SOURCE_CANDIDATE="${source_candidate}" \
CUSTOM_DOG_STAND_CANDIDATE="${stand_candidate}" \
"${project_root}/scripts/evaluate_selective_collision_stage_a.sh"

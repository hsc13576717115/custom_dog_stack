#!/usr/bin/env bash
set -euo pipefail

: "${CUSTOM_DOG_RUN_DIR:?Set CUSTOM_DOG_RUN_DIR to the completed Robust Foundation run}"
: "${CUSTOM_DOG_STAND_CANDIDATE:?Set CUSTOM_DOG_STAND_CANDIDATE to the accepted stand export}"

project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
run_dir="$(realpath "${CUSTOM_DOG_RUN_DIR}")"
stand_candidate="$(realpath "${CUSTOM_DOG_STAND_CANDIDATE}")"
task="CustomDog-Velocity-OmniTrot-ClosedLoopRobustFoundation-v1"

mapfile -t available < <(
    find "${run_dir}" -mindepth 1 -maxdepth 1 -type f -name 'model_*.pt' -printf '%f\n' \
        | sort -V
)
if [[ ${#available[@]} -lt 5 ]]; then
    echo "Expected at least five Robust Foundation checkpoints, found ${#available[@]}" >&2
    exit 1
fi

declare -a checkpoints=()
declare -A selected_names=()
for numerator in 1 2 3 4 5 6; do
    index=$((numerator * (${#available[@]} - 1) / 6))
    checkpoint="${available[${index}]}"
    if [[ -z "${selected_names[${checkpoint}]:-}" ]]; then
        checkpoints+=("${checkpoint}")
        selected_names["${checkpoint}"]=1
    fi
done

"${project_root}/scripts/export_checkpoint_candidates.sh" \
    "${run_dir}" "${task}" "${checkpoints[@]}"

candidate_args=()
baseline_label=""
for checkpoint in "${checkpoints[@]}"; do
    iteration="${checkpoint#model_}"
    iteration="${iteration%.pt}"
    label="RFR_${iteration}"
    motion="${run_dir}/evaluation/candidates/model_${iteration}"
    routed="${run_dir}/evaluation/routed_candidates/model_${iteration}"
    python3 "${project_root}/scripts/build_routed_candidate.py" \
        "${motion}" "${stand_candidate}" "${routed}"
    candidate_args+=(--candidate "${label}=${routed}")
    if [[ -z "${baseline_label}" ]]; then
        baseline_label="${label}"
    fi
done

python="${CUSTOM_DOG_MUJOCO_PYTHON:-${HOME}/miniconda3/envs/custom_dog_mujoco/bin/python}"
"${python}" "${project_root}/scripts/evaluate_mujoco_grid.py" \
    "${candidate_args[@]}" \
    --baseline-label "${baseline_label}" \
    --absolute-only \
    --duration 10 \
    --warmup 2 \
    --output-csv "${run_dir}/evaluation/routed_robust_foundation_grid_10s.csv" \
    --output-json "${run_dir}/evaluation/routed_robust_foundation_grid_10s.json"

python3 "${project_root}/scripts/select_mujoco_candidate.py" \
    "${run_dir}/evaluation/routed_robust_foundation_grid_10s.json" \
    --output "${run_dir}/evaluation/routed_robust_foundation_selection.json"

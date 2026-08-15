#!/usr/bin/env bash
set -euo pipefail

: "${CUSTOM_DOG_RUN_DIR:?Set CUSTOM_DOG_RUN_DIR to the completed v3 stand run}"

project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
run_dir="$(realpath "${CUSTOM_DOG_RUN_DIR}")"
task="CustomDog-Stand-HeightHipCalibrated-v3"
source_candidate="${project_root}/logs/rsl_rl/custom_dog_velocity/2026-08-14_22-19-41_stand_height_calibrated_from200/evaluation/candidates/model_319"

mapfile -t available < <(
    find "${run_dir}" -mindepth 1 -maxdepth 1 -type f -name 'model_*.pt' -printf '%f\n' \
        | sort -V
)
if [[ ${#available[@]} -lt 5 ]]; then
    echo "Expected at least five v3 stand checkpoints, found ${#available[@]}" >&2
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

candidate_args=(--candidate "STAND_CAL2_319_SOURCE=${source_candidate}")
for checkpoint in "${checkpoints[@]}"; do
    iteration="${checkpoint#model_}"
    iteration="${iteration%.pt}"
    candidate_args+=(
        --candidate "STAND_CAL3_${iteration}=${run_dir}/evaluation/candidates/model_${iteration}"
    )
done

python="${CUSTOM_DOG_MUJOCO_PYTHON:-${HOME}/miniconda3/envs/custom_dog_mujoco/bin/python}"
"${python}" "${project_root}/scripts/evaluate_mujoco_grid.py" \
    "${candidate_args[@]}" \
    --baseline-label STAND_CAL2_319_SOURCE \
    --absolute-only \
    --command 0 0 0 \
    --duration 15 \
    --warmup 3 \
    --output-csv "${run_dir}/evaluation/stand_grid_15s.csv" \
    --output-json "${run_dir}/evaluation/stand_grid_15s.json"

python3 "${project_root}/scripts/select_mujoco_candidate.py" \
    "${run_dir}/evaluation/stand_grid_15s.json" \
    --output "${run_dir}/evaluation/stand_selection.json"

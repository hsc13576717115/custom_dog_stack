#!/usr/bin/env bash
set -euo pipefail

: "${CUSTOM_DOG_RUN_DIR:?Set CUSTOM_DOG_RUN_DIR to the completed RobustStandFix run}"

project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
run_dir="$(realpath "${CUSTOM_DOG_RUN_DIR}")"
source_run="${project_root}/logs/rsl_rl/custom_dog_velocity/2026-08-14_19-59-12_closed_loop_robust_foundation_seed42"
source_candidate="${source_run}/evaluation/candidates/model_700"
task="CustomDog-Velocity-OmniTrot-RobustStandFix-v1"

mapfile -t available < <(
    find "${run_dir}" -mindepth 1 -maxdepth 1 -type f -name 'model_*.pt' -printf '%f\n' \
        | sort -V
)
if [[ ${#available[@]} -lt 5 ]]; then
    echo "Expected at least five RobustStandFix checkpoints, found ${#available[@]}" >&2
    exit 1
fi

declare -a checkpoints=()
declare -A selected_names=()
for numerator in 0 1 2 3 4 5 6; do
    index=$((numerator * (${#available[@]} - 1) / 6))
    checkpoint="${available[${index}]}"
    if [[ -z "${selected_names[${checkpoint}]:-}" ]]; then
        checkpoints+=("${checkpoint}")
        selected_names["${checkpoint}"]=1
    fi
done

"${project_root}/scripts/export_checkpoint_candidates.sh" \
    "${run_dir}" "${task}" "${checkpoints[@]}"

candidate_args=(--candidate "RF_700=${source_candidate}")
for checkpoint in "${checkpoints[@]}"; do
    iteration="${checkpoint#model_}"
    iteration="${iteration%.pt}"
    candidate_args+=(
        --candidate "RSF_${iteration}=${run_dir}/evaluation/candidates/model_${iteration}"
    )
done

python="${CUSTOM_DOG_MUJOCO_PYTHON:-${HOME}/miniconda3/envs/custom_dog_mujoco/bin/python}"
"${python}" "${project_root}/scripts/evaluate_mujoco_grid.py" \
    "${candidate_args[@]}" \
    --baseline-label RF_700 \
    --absolute-only \
    --duration 10 \
    --warmup 2 \
    --output-csv "${run_dir}/evaluation/robust_stand_fix_grid_10s.csv" \
    --output-json "${run_dir}/evaluation/robust_stand_fix_grid_10s.json"

python3 "${project_root}/scripts/select_mujoco_candidate.py" \
    "${run_dir}/evaluation/robust_stand_fix_grid_10s.json" \
    --output "${run_dir}/evaluation/robust_stand_fix_selection.json"

#!/usr/bin/env bash
set -euo pipefail

: "${CUSTOM_DOG_RUN_DIR:?Set CUSTOM_DOG_RUN_DIR to the completed selective-collision run}"
: "${CUSTOM_DOG_SOURCE_CANDIDATE:?Set CUSTOM_DOG_SOURCE_CANDIDATE to the accepted source export}"
: "${CUSTOM_DOG_STAND_CANDIDATE:?Set CUSTOM_DOG_STAND_CANDIDATE to the accepted stand export}"

project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
run_dir="$(realpath "${CUSTOM_DOG_RUN_DIR}")"
source_candidate="$(realpath "${CUSTOM_DOG_SOURCE_CANDIDATE}")"
stand_candidate="$(realpath "${CUSTOM_DOG_STAND_CANDIDATE}")"
task="CustomDog-Velocity-OmniTrot-ClosedLoopSelectiveCollision-v1"
mapfile -t available < <(
    find "${run_dir}" -mindepth 1 -maxdepth 1 -type f -name 'model_*.pt' -printf '%f\n' \
        | sort -V
)
if [[ ${#available[@]} -lt 5 ]]; then
    echo "Expected at least five collision-adaptation checkpoints, found ${#available[@]}" >&2
    exit 1
fi
declare -A selected_names=()
for numerator in 1 2 3 4 5; do
    index=$((numerator * (${#available[@]} - 1) / 5))
    checkpoint="${available[${index}]}"
    if [[ -z "${selected_names[${checkpoint}]:-}" ]]; then
        checkpoints+=("${checkpoint}")
        selected_names["${checkpoint}"]=1
    fi
done

"${project_root}/scripts/export_checkpoint_candidates.sh" \
    "${run_dir}" "${task}" "${checkpoints[@]}"

selective_mjcf="${run_dir}/evaluation/custom_dog_selective.xml"
python3 "${project_root}/scripts/generate_selective_mujoco.py" \
    --output "${selective_mjcf}"
"${CUSTOM_DOG_MUJOCO_PYTHON:-${HOME}/miniconda3/envs/custom_dog_mujoco/bin/python}" \
    "${project_root}/scripts/validate_mujoco_self_collision.py" "${selective_mjcf}"

candidate_args=(--candidate "source=${source_candidate}")
for checkpoint in "${checkpoints[@]}"; do
    iteration="${checkpoint#model_}"
    iteration="${iteration%.pt}"
    motion="${run_dir}/evaluation/candidates/model_${iteration}"
    routed="${run_dir}/evaluation/routed_candidates/model_${iteration}"
    python3 "${project_root}/scripts/build_routed_candidate.py" \
        "${motion}" "${stand_candidate}" "${routed}"
    candidate_args+=(
        --candidate "SC_${iteration}=${routed}"
    )
done

python="${CUSTOM_DOG_MUJOCO_PYTHON:-${HOME}/miniconda3/envs/custom_dog_mujoco/bin/python}"
"${python}" "${project_root}/scripts/evaluate_mujoco_grid.py" \
    "${candidate_args[@]}" \
    --baseline-label source \
    --mjcf "${selective_mjcf}" \
    --duration 10 \
    --warmup 2 \
    --output-csv "${run_dir}/evaluation/selective_collision_grid_10s.csv" \
    --output-json "${run_dir}/evaluation/selective_collision_grid_10s.json"

python3 "${project_root}/scripts/select_mujoco_candidate.py" \
    "${run_dir}/evaluation/selective_collision_grid_10s.json" \
    --label-prefix SC_ \
    --output "${run_dir}/evaluation/selective_collision_selection.json"

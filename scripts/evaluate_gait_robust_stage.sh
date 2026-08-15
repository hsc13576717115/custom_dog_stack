#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 2 ]]; then
    echo "Usage: $0 RUN_DIR SOURCE_CANDIDATE" >&2
    exit 2
fi
: "${CUSTOM_DOG_STAND_CANDIDATE:?Set CUSTOM_DOG_STAND_CANDIDATE to the accepted stand export}"

project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
run_dir="$(realpath "$1")"
source_candidate="$(realpath "$2")"
stand_candidate="$(realpath "${CUSTOM_DOG_STAND_CANDIDATE}")"
task="${CUSTOM_DOG_EVAL_TASK:-CustomDog-Velocity-OmniTrot-ClosedLoopGaitRobust-v1}"
label_prefix="${CUSTOM_DOG_EVAL_LABEL_PREFIX:-GR_}"
output_stem="${CUSTOM_DOG_EVAL_OUTPUT_STEM:-gait_robust}"
mapfile -t available < <(
    find "${run_dir}" -mindepth 1 -maxdepth 1 -type f -name 'model_*.pt' \
        -printf '%f\n' | sort -V
)
if [[ ${#available[@]} -lt 5 ]]; then
    echo "Expected at least five gait-robust checkpoints, found ${#available[@]}" >&2
    exit 1
fi
declare -a checkpoints=()
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
python3 "${project_root}/scripts/generate_selective_mujoco.py" --output "${selective_mjcf}"

candidate_args=(--candidate "source=${source_candidate}")
for checkpoint in "${checkpoints[@]}"; do
    iteration="${checkpoint#model_}"
    iteration="${iteration%.pt}"
    motion="${run_dir}/evaluation/candidates/model_${iteration}"
    routed="${run_dir}/evaluation/routed_candidates/model_${iteration}"
    python3 "${project_root}/scripts/build_routed_candidate.py" \
        "${motion}" "${stand_candidate}" "${routed}"
    candidate_args+=(--candidate "${label_prefix}${iteration}=${routed}")
done

python="${CUSTOM_DOG_MUJOCO_PYTHON:-${HOME}/miniconda3/envs/custom_dog_mujoco/bin/python}"
"${python}" "${project_root}/scripts/evaluate_mujoco_grid.py" \
    "${candidate_args[@]}" \
    --baseline-label source \
    --absolute-only \
    --stage D \
    --mjcf "${selective_mjcf}" \
    --duration "${CUSTOM_DOG_EVAL_DURATION:-15}" \
    --warmup "${CUSTOM_DOG_EVAL_WARMUP:-3}" \
    --output-csv "${run_dir}/evaluation/${output_stem}_grid.csv" \
    --output-json "${run_dir}/evaluation/${output_stem}_grid.json"

python3 "${project_root}/scripts/select_mujoco_candidate.py" \
    "${run_dir}/evaluation/${output_stem}_grid.json" \
    --label-prefix "${label_prefix}" \
    --output "${run_dir}/evaluation/${output_stem}_selection.json"

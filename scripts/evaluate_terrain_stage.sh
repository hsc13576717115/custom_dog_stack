#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 3 ]]; then
    echo "Usage: $0 T0|T1 RUN_DIR SOURCE_CANDIDATE" >&2
    exit 2
fi
: "${CUSTOM_DOG_STAND_CANDIDATE:?Set CUSTOM_DOG_STAND_CANDIDATE to the accepted stand export}"
stage="${1^^}"
case "${stage}" in T0|T1) ;; *) echo "Expected T0 or T1" >&2; exit 2 ;; esac
run_dir="$(realpath "$2")"
source_candidate="$(realpath "$3")"
stand_candidate="$(realpath "${CUSTOM_DOG_STAND_CANDIDATE}")"
project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source "${project_root}/scripts/activate_env.sh"
task="CustomDog-Velocity-OmniTrot-Terrain${stage}-v1"

mapfile -t available < <(
    find "${run_dir}" -mindepth 1 -maxdepth 1 -type f -name 'model_*.pt' \
        -printf '%f\n' | sort -V
)
if [[ ${#available[@]} -lt 5 ]]; then
    echo "Expected at least five terrain checkpoints, found ${#available[@]}" >&2
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
terrain_args=()
for checkpoint in "${checkpoints[@]}"; do
    iteration="${checkpoint#model_}"
    iteration="${iteration%.pt}"
    label="${stage}_${iteration}"
    motion="${run_dir}/evaluation/candidates/model_${iteration}"
    routed="${run_dir}/evaluation/routed_candidates/model_${iteration}"
    python3 "${project_root}/scripts/build_routed_candidate.py" \
        "${motion}" "${stand_candidate}" "${routed}"
    candidate_args+=(--candidate "${label}=${routed}")

    terrain_result="${run_dir}/evaluation/${label}_terrain.json"
    if OMNI_KIT_ACCEPT_EULA=YES \
        python \
        "${project_root}/rl/scripts/evaluate_terrain_checkpoint.py" \
        --task "${task}" \
        --stage "${stage}" \
        --checkpoint "${run_dir}/${checkpoint}" \
        --output "${terrain_result}" \
        --num_envs "${CUSTOM_DOG_TERRAIN_EVAL_ENVS:-128}" \
        --max_steps "${CUSTOM_DOG_TERRAIN_EVAL_STEPS:-600}" \
        --warmup_steps "${CUSTOM_DOG_TERRAIN_EVAL_WARMUP:-100}" \
        --headless \
        --device "${CUSTOM_DOG_SIM_DEVICE:-cuda:0}" \
        --rl_device "${CUSTOM_DOG_RL_DEVICE:-cuda:0}"; then
        echo "${label}: Isaac terrain gate passed"
    else
        echo "${label}: Isaac terrain gate failed"
    fi
    terrain_args+=(--terrain-result "${label}=${terrain_result}")
done

mujoco_python="${CUSTOM_DOG_MUJOCO_PYTHON:-${HOME}/miniconda3/envs/custom_dog_mujoco/bin/python}"
flat_grid="${run_dir}/evaluation/terrain_${stage,,}_flat_stage_d_grid.json"
"${mujoco_python}" "${project_root}/scripts/evaluate_mujoco_grid.py" \
    "${candidate_args[@]}" \
    --baseline-label source \
    --absolute-only \
    --stage D \
    --mjcf "${selective_mjcf}" \
    --duration "${CUSTOM_DOG_EVAL_DURATION:-15}" \
    --warmup "${CUSTOM_DOG_EVAL_WARMUP:-3}" \
    --output-csv "${run_dir}/evaluation/terrain_${stage,,}_flat_stage_d_grid.csv" \
    --output-json "${flat_grid}"

python3 "${project_root}/scripts/select_terrain_candidate.py" \
    --flat-grid "${flat_grid}" \
    "${terrain_args[@]}" \
    --output "${run_dir}/evaluation/terrain_${stage,,}_selection.json"

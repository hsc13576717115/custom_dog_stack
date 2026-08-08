#!/usr/bin/env bash
set -euo pipefail

project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
conda_exe="${CONDA_EXE:-/home/hsc/miniconda3/bin/conda}"
env_name="${CUSTOM_DOG_MUJOCO_ENV:-custom_dog_mujoco}"
model_dir="${project_root}/sim2sim/custom_dog"
urdf="${project_root}/ros2/src/custom_dog_description/urdf/custom_dog.urdf"
mjcf="${model_dir}/custom_dog.xml"

if ! "${conda_exe}" env list | awk '{print $1}' | grep -Fx "${env_name}" >/dev/null; then
    echo "Missing conda environment '${env_name}'. Run ./scripts/setup_mujoco.sh first." >&2
    exit 1
fi

mkdir -p "${model_dir}"
export URDF2MJCF_MODEL_PATH="${project_root}/ros2/src${URDF2MJCF_MODEL_PATH:+:${URDF2MJCF_MODEL_PATH}}"

nice -n "${CUSTOM_DOG_CONVERT_NICE:-10}" \
    "${conda_exe}" run -n "${env_name}" urdf-to-mjcf \
    "${urdf}" \
    --output "${mjcf}" \
    --metadata "${model_dir}/config/conversion.json" \
    --actuator-metadata "${model_dir}/config/actuators.json" \
    --collision-type convex_hull \
    --max-vertices 50000

"${conda_exe}" run -n "${env_name}" python \
    "${model_dir}/postprocess_mjcf.py" \
    --urdf "${urdf}" \
    --mjcf "${mjcf}"

"${conda_exe}" run -n "${env_name}" python \
    "${model_dir}/validate_mjcf.py" \
    "${mjcf}"

echo "Generated and validated: ${mjcf}"

#!/usr/bin/env bash
set -euo pipefail

project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
conda_exe="${CONDA_EXE:-$(command -v conda 2>/dev/null || true)}"
env_name="${CUSTOM_DOG_MUJOCO_ENV:-custom_dog_mujoco}"
model_dir="${project_root}/sim2sim/custom_dog"
urdf="${project_root}/ros2/src/custom_dog_description/urdf/custom_dog.urdf"
mjcf="${model_dir}/custom_dog.xml"
unitree_mjcf="${model_dir}/custom_dog_unitree.xml"

if [[ -z "${conda_exe}" ]]; then
    for candidate in "${HOME}/miniconda3/bin/conda" "${HOME}/anaconda3/bin/conda" /opt/conda/bin/conda; do
        if [[ -x "${candidate}" ]]; then
            conda_exe="${candidate}"
            break
        fi
    done
fi

if [[ -z "${conda_exe}" ]]; then
    echo "Conda was not found. Set CONDA_EXE or run setup_ubuntu2204.sh." >&2
    exit 1
fi

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

"${conda_exe}" run -n "${env_name}" python \
    "${model_dir}/create_unitree_mjcf.py" \
    --input "${mjcf}" \
    --output "${unitree_mjcf}"

"${conda_exe}" run -n "${env_name}" python \
    "${model_dir}/validate_unitree_mjcf.py" \
    "${unitree_mjcf}"

echo "Generated and validated: ${mjcf}"
echo "Generated and validated for Unitree SDK2 bridge: ${unitree_mjcf}"

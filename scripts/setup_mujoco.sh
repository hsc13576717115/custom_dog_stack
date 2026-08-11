#!/usr/bin/env bash
set -euo pipefail

project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
conda_exe="${CONDA_EXE:-$(command -v conda 2>/dev/null || true)}"
env_name="${CUSTOM_DOG_MUJOCO_ENV:-custom_dog_mujoco}"

if [[ -z "${conda_exe}" ]]; then
    for candidate in "${HOME}/miniconda3/bin/conda" "${HOME}/anaconda3/bin/conda" /opt/conda/bin/conda; do
        if [[ -x "${candidate}" ]]; then
            conda_exe="${candidate}"
            break
        fi
    done
fi

if [[ -z "${conda_exe}" || ! -x "${conda_exe}" ]]; then
    echo "Conda executable not found. Set CONDA_EXE or install Miniconda." >&2
    exit 1
fi

if ! "${conda_exe}" env list | awk '{print $1}' | grep -Fx "${env_name}" >/dev/null; then
    "${conda_exe}" create -y -n "${env_name}" python=3.10 pip
fi

"${conda_exe}" run -n "${env_name}" python -m pip install \
    --requirement "${project_root}/requirements/mujoco.txt"

"${conda_exe}" run --no-capture-output -n "${env_name}" python -c \
    'import mujoco, numpy, onnxruntime, yaml; print(f"MuJoCo: {mujoco.__version__}"); print(f"NumPy: {numpy.__version__}"); print(f"ONNX Runtime: {onnxruntime.__version__}"); print(f"PyYAML: {yaml.__version__}")'

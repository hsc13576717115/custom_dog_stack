#!/usr/bin/env bash
set -euo pipefail

project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
env_name="${CUSTOM_DOG_CONDA_ENV:-env_isaaclab}"
install_system_deps=0

usage() {
    cat <<'EOF'
Usage: ./scripts/setup_ubuntu2204.sh [--install-system-deps]

Reconstructs the pinned x86_64 Ubuntu 22.04 training and sim2sim environment.
The optional flag installs build, graphics, Git LFS and ROS build prerequisites
with apt. NVIDIA driver and ROS 2 Humble installation remain host-level steps.
EOF
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --install-system-deps) install_system_deps=1 ;;
        -h|--help) usage; exit 0 ;;
        *) echo "Unknown argument: $1" >&2; usage >&2; exit 2 ;;
    esac
    shift
done

if [[ -r /etc/os-release ]]; then
    # shellcheck disable=SC1091
    source /etc/os-release
    if [[ "${ID:-}" != "ubuntu" || "${VERSION_ID:-}" != "22.04" ]]; then
        echo "WARNING: validated for Ubuntu 22.04, detected ${PRETTY_NAME:-unknown}." >&2
    fi
fi

if [[ "$(uname -m)" != "x86_64" ]]; then
    echo "This installer is for the x86_64 training workstation, not Jetson/Orin." >&2
    exit 1
fi

if [[ ${install_system_deps} -eq 1 ]]; then
    sudo apt-get update
    sudo apt-get install -y --no-install-recommends \
        build-essential cmake git git-lfs curl ca-certificates pkg-config \
        libegl1 libgl1 libglib2.0-0 libx11-6 libxcursor1 libxi6 \
        libxinerama1 libxkbcommon0 libxrandr2 libxxf86vm1 libglfw3-dev \
        libyaml-cpp-dev libeigen3-dev libboost-all-dev libspdlog-dev libfmt-dev
    git lfs install
fi

git -C "${project_root}" submodule update --init --recursive
if command -v git-lfs >/dev/null 2>&1; then
    git -C "${project_root}" lfs pull
fi
"${project_root}/scripts/apply_dependency_patches.sh"

if [[ "${OMNI_KIT_ACCEPT_EULA:-}" != "YES" ]]; then
    echo "Set OMNI_KIT_ACCEPT_EULA=YES only after accepting the NVIDIA Omniverse EULA." >&2
    exit 1
fi
export OMNI_KIT_ACCEPT_EULA

conda_exe="${CONDA_EXE:-$(command -v conda 2>/dev/null || true)}"
if [[ -z "${conda_exe}" ]]; then
    for candidate in "${HOME}/miniconda3/bin/conda" "${HOME}/anaconda3/bin/conda" /opt/conda/bin/conda; do
        if [[ -x "${candidate}" ]]; then
            conda_exe="${candidate}"
            break
        fi
    done
fi
if [[ -z "${conda_exe}" ]]; then
    echo "Conda is required. Install Miniconda, then rerun this script." >&2
    exit 1
fi

conda_sh="$(cd "$(dirname "${conda_exe}")/.." && pwd)/etc/profile.d/conda.sh"
# shellcheck disable=SC1090
source "${conda_sh}"

if ! conda env list | awk '{print $1}' | grep -Fx "${env_name}" >/dev/null; then
    conda create -y -n "${env_name}" --override-channels -c conda-forge python=3.11 pip glfw
fi
conda activate "${env_name}"

python -m pip install --upgrade pip
python -m pip install "isaacsim[all,extscache]==5.1.0" --extra-index-url https://pypi.nvidia.com
python -m pip install --upgrade \
    torch==2.7.0 torchvision==0.22.0 \
    --index-url https://download.pytorch.org/whl/cu128
"${project_root}/third_party/IsaacLab/isaaclab.sh" --install
python -m pip install --editable "${project_root}/third_party/unitree_rl_lab/source/unitree_rl_lab"
python -m pip install --requirement "${project_root}/requirements/training-extra.txt"
python -m pip install --no-deps --editable "${project_root}"

sdk_prefix="${UNITREE_SDK_PREFIX:-${HOME}/.local/unitree_robotics}"
cmake -S "${project_root}/third_party/unitree_sdk2" \
    -B "${project_root}/build/unitree_sdk2" \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_INSTALL_PREFIX="${sdk_prefix}"
cmake --build "${project_root}/build/unitree_sdk2" --parallel "${CMAKE_BUILD_PARALLEL_LEVEL:-$(nproc)}"
cmake --install "${project_root}/build/unitree_sdk2"
"${project_root}/scripts/setup_unitree_native.sh"

CUSTOM_DOG_CONDA_ENV="${env_name}" "${project_root}/scripts/setup_mujoco.sh"
CUSTOM_DOG_CONDA_ENV="${env_name}" "${project_root}/scripts/check_dependency_versions.sh"
CUSTOM_DOG_CONDA_ENV="${env_name}" "${project_root}/scripts/check_system.sh"

cat <<EOF

Environment ready. In a new shell run:
  cd ${project_root}
  source scripts/activate_env.sh
EOF

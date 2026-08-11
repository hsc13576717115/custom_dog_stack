#!/usr/bin/env bash
set -euo pipefail

project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source "${project_root}/scripts/activate_env.sh"

python -m pytest -q "${project_root}/tests"
python -m py_compile \
    "${project_root}/rl/src/custom_dog_rl/assets/custom_dog.py" \
    "${project_root}/rl/src/custom_dog_rl/tasks/locomotion/velocity_env_cfg.py" \
    "${project_root}/rl/src/custom_dog_rl/tasks/locomotion/mdp/commands.py" \
    "${project_root}/rl/src/custom_dog_rl/tasks/locomotion/mdp/rewards.py" \
    "${project_root}/rl/src/custom_dog_rl/agents/ppo_cfg.py" \
    "${project_root}/rl/scripts/train.py" \
    "${project_root}/rl/scripts/play.py" \
    "${project_root}/sim2sim/custom_dog/postprocess_mjcf.py" \
    "${project_root}/sim2sim/custom_dog/validate_mjcf.py" \
    "${project_root}/sim2sim/custom_dog/create_unitree_mjcf.py" \
    "${project_root}/sim2sim/custom_dog/validate_unitree_mjcf.py" \
    "${project_root}/sim2sim/custom_dog/run_sim2sim.py" \
    "${project_root}/sim2sim/unitree_deploy/prepare_controller.py" \
    "${project_root}/scripts/analyze_policy_trace.py"

state_machine_test="$(mktemp /tmp/custom_dog_state_machine_test.XXXXXX)"
trap 'rm -f "${state_machine_test}"' EXIT
"${CXX:-c++}" -std=c++17 -Wall -Wextra -Wpedantic -Werror \
    -I"${project_root}/ros2/src/custom_dog_controller/include" \
    "${project_root}/ros2/src/custom_dog_controller/test/deployment_state_machine_test.cpp" \
    -o "${state_machine_test}"
"${state_machine_test}"

echo "Static validation passed."

#!/usr/bin/env bash
set -euo pipefail

project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
collision_unit="${CUSTOM_DOG_COLLISION_UNIT:-custom-dog-selective-collision-v4.service}"
unit_suffix="${CUSTOM_DOG_PIPELINE_SUFFIX:-v4}"

queue_after() {
    local unit="$1"
    local dependency="$2"
    shift 2
    if systemctl --user list-unit-files "${unit}" --no-legend 2>/dev/null \
        | grep -q .; then
        echo "Unit already exists: ${unit}"
        return
    fi
    systemd-run --user --unit="${unit%.service}" --collect \
        "${project_root}/scripts/run_after_successful_user_service.sh" \
        "${dependency}" "$@"
}

# The two curricula are logically independent after collision admission, but a
# single GPU should not run them concurrently.  Interleaving keeps both moving
# while preserving an unambiguous dependency and evaluation chain.
stage_b="custom-dog-stage-b-${unit_suffix}.service"
recovery_r0="custom-dog-recovery-r0-${unit_suffix}.service"
stage_c="custom-dog-stage-c-${unit_suffix}.service"
recovery_r1="custom-dog-recovery-r1-${unit_suffix}.service"
stage_d="custom-dog-stage-d-${unit_suffix}.service"
recovery_r2="custom-dog-recovery-r2-${unit_suffix}.service"
gait_robust="custom-dog-gait-robust-${unit_suffix}.service"
dynamics_teacher="custom-dog-dynamics-teacher-${unit_suffix}.service"
terrain_t0="custom-dog-terrain-t0-${unit_suffix}.service"
terrain_t1="custom-dog-terrain-t1-${unit_suffix}.service"
history213="custom-dog-history213-${unit_suffix}.service"

queue_after "${stage_b}" "${collision_unit}" \
    "${project_root}/scripts/train_next_closed_loop_stage_gated.sh" B
queue_after "${recovery_r0}" "${stage_b}" \
    "${project_root}/scripts/train_self_righting_r0_gated.sh"
queue_after "${stage_c}" "${recovery_r0}" \
    "${project_root}/scripts/train_next_closed_loop_stage_gated.sh" C
queue_after "${recovery_r1}" "${stage_c}" \
    "${project_root}/scripts/train_next_self_righting_stage_gated.sh" R1
queue_after "${stage_d}" "${recovery_r1}" \
    "${project_root}/scripts/train_next_closed_loop_stage_gated.sh" D
queue_after "${recovery_r2}" "${stage_d}" \
    "${project_root}/scripts/train_next_self_righting_stage_gated.sh" R2
queue_after "${gait_robust}" "${recovery_r2}" \
    "${project_root}/scripts/train_gait_robust_stage_gated.sh"
queue_after "${dynamics_teacher}" "${gait_robust}" \
    "${project_root}/scripts/train_dynamics_teacher_stage_gated.sh"
queue_after "${terrain_t0}" "${dynamics_teacher}" \
    "${project_root}/scripts/train_next_terrain_stage_gated.sh" T0
queue_after "${terrain_t1}" "${terrain_t0}" \
    "${project_root}/scripts/train_next_terrain_stage_gated.sh" T1
queue_after "${history213}" "${terrain_t1}" \
    "${project_root}/scripts/train_history213_distillation_gated.sh"

echo "Queued gated pipeline after ${collision_unit}: B -> R0 -> C -> R1 -> D -> R2 -> gait -> dynamics -> T0 -> T1 -> H213"

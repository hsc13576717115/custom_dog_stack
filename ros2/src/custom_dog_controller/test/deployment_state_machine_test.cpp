#include "custom_dog_controller/deployment_state_machine.hpp"

#include <cstdlib>
#include <cmath>

namespace {

bool close(float left, float right) {
  return std::abs(left - right) < 1.0e-6F;
}

void expect(bool condition) {
  if (!condition) {
    std::abort();
  }
}

}  // namespace

int main() {
  using custom_dog_controller::DeploymentMode;
  using custom_dog_controller::DeploymentStateMachine;
  using custom_dog_controller::JointPositions;
  using custom_dog_controller::VelocityCommand;

  DeploymentStateMachine state_machine;
  const VelocityCommand request{0.4F, 0.2F, 0.3F};
  auto output = state_machine.update(0.0, request);
  expect(output.mode == DeploymentMode::kPassive);
  expect(!output.motors_enabled);

  JointPositions measured{};
  JointPositions home{};
  measured.fill(-1.0F);
  home.fill(1.0F);
  state_machine.begin_fix_stand(measured, home, 10.0);

  output = state_machine.update(10.0, request);
  expect(output.mode == DeploymentMode::kFixStand);
  expect(output.motors_enabled);
  expect(!output.use_policy);
  expect(close(output.stand_target[0], -1.0F));

  output = state_machine.update(11.0, request);
  expect(output.mode == DeploymentMode::kFixStand);
  expect(close(output.stand_target[0], 0.0F));

  output = state_machine.update(12.0, request);
  expect(output.mode == DeploymentMode::kPolicyHold);
  expect(output.use_policy);
  expect(output.reset_policy_history);
  expect(close(output.velocity_command[0], 0.0F));

  output = state_machine.update(12.02, request);
  expect(output.mode == DeploymentMode::kPolicyHold);
  expect(!output.reset_policy_history);
  expect(close(output.velocity_command[0], 0.0F));

  output = state_machine.update(13.0, request);
  expect(output.mode == DeploymentMode::kVelocity);
  expect(output.use_policy);
  expect(close(output.velocity_command[0], 0.4F));
  expect(close(output.velocity_command[1], 0.2F));
  expect(close(output.velocity_command[2], 0.3F));

  state_machine.enter_passive();
  output = state_machine.update(13.02, request);
  expect(output.mode == DeploymentMode::kPassive);
  expect(!output.motors_enabled);

  DeploymentStateMachine recovery_machine;
  custom_dog_controller::RecoveryTelemetry upright;
  upright.valid = true;
  upright.height_m = 0.28;
  upright.tilt_rad = 0.10;
  upright.angular_speed_rad_s = 0.10;
  upright.foot_contacts.fill(true);
  recovery_machine.begin_recovery(0.0);
  output = recovery_machine.update(0.0, request, upright);
  expect(output.mode == DeploymentMode::kUprightDwell);
  expect(output.use_policy);
  expect(output.use_recovery_policy == false);
  expect(close(output.velocity_command[0], 0.0F));

  output = recovery_machine.update(0.2, request, upright);
  expect(output.mode == DeploymentMode::kUprightDwell);
  expect(!output.reset_policy_history);

  output = recovery_machine.update(0.4, request, upright);
  expect(output.mode == DeploymentMode::kPolicyHold);
  expect(output.reset_policy_history);
  expect(close(output.velocity_command[0], 0.0F));

  output = recovery_machine.update(0.41, request, upright);
  expect(output.mode == DeploymentMode::kPolicyHold);
  expect(!output.reset_policy_history);
  output = recovery_machine.update(1.41, request, upright);
  expect(output.mode == DeploymentMode::kVelocity);
  expect(close(output.velocity_command[0], request[0]));

  DeploymentStateMachine timeout_machine;
  timeout_machine.begin_recovery(0.0);
  output = timeout_machine.update(7.0, request, custom_dog_controller::RecoveryTelemetry{});
  expect(output.mode == DeploymentMode::kPassive);
  expect(!output.motors_enabled);

  DeploymentStateMachine dropout_machine;
  dropout_machine.begin_recovery(0.0);
  output = dropout_machine.update(0.0, request, upright);
  expect(output.mode == DeploymentMode::kUprightDwell);
  auto lost_contact = upright;
  lost_contact.foot_contacts[2] = false;
  output = dropout_machine.update(0.1, request, lost_contact);
  expect(output.mode == DeploymentMode::kRecoveryPolicy);
  expect(output.use_recovery_policy);
  expect(output.reset_policy_history);
  return 0;
}

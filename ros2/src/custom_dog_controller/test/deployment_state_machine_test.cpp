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
  return 0;
}

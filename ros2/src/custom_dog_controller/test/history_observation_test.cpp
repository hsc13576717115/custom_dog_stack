#include "custom_dog_controller/history_observation.hpp"

#include <cmath>
#include <cstdlib>

namespace {

bool close(float left, float right) {
  return std::abs(left - right) < 1.0e-6F;
}

void expect(bool condition) {
  if (!condition) {
    std::abort();
  }
}

custom_dog_controller::BaseObservation frame(float value) {
  custom_dog_controller::BaseObservation result{};
  result.fill(value);
  result[custom_dog_controller::kVelocityCommand] = value + 100.0F;
  return result;
}

}  // namespace

int main() {
  using custom_dog_controller::HistoryObservationBuilder;
  using custom_dog_controller::kHistoryBaseAngularVelocity;
  using custom_dog_controller::kHistoryJointPositionRelative;
  using custom_dog_controller::kHistoryVelocityCommand;

  HistoryObservationBuilder builder;
  auto observation = builder.push(frame(1.0F));
  for (std::size_t history = 0; history < 5; ++history) {
    expect(close(observation[kHistoryBaseAngularVelocity + history * 3], 1.0F));
    expect(close(observation[kHistoryJointPositionRelative + history * 12], 1.0F));
  }
  expect(close(observation[kHistoryVelocityCommand], 101.0F));

  observation = builder.push(frame(2.0F));
  expect(close(observation[kHistoryBaseAngularVelocity], 1.0F));
  expect(close(observation[kHistoryBaseAngularVelocity + 12], 2.0F));
  expect(close(observation[kHistoryVelocityCommand], 102.0F));

  builder.reset();
  observation = builder.push(frame(3.0F));
  expect(close(observation[kHistoryBaseAngularVelocity], 3.0F));
  expect(close(observation[kHistoryBaseAngularVelocity + 12], 3.0F));
  expect(close(observation[kHistoryVelocityCommand], 103.0F));
  return 0;
}

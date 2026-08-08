#pragma once

#include <array>
#include <cstddef>

namespace custom_dog_controller {

constexpr std::size_t kJointCount = 12;
constexpr std::size_t kObservationDim = 45;
constexpr std::size_t kActionDim = 12;
constexpr float kControlPeriodSeconds = 0.02F;
constexpr float kActionScale = 0.25F;

enum ObservationOffset : std::size_t {
  kBaseAngularVelocity = 0,
  kProjectedGravity = 3,
  kVelocityCommand = 6,
  kJointPositionRelative = 9,
  kJointVelocity = 21,
  kLastAction = 33,
};

using Observation = std::array<float, kObservationDim>;
using Action = std::array<float, kActionDim>;

}  // namespace custom_dog_controller

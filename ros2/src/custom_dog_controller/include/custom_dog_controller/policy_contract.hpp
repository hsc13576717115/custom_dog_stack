#pragma once

#include <array>
#include <cstddef>

namespace custom_dog_controller {

constexpr char kPolicyContractVersion[] = "2.3";
constexpr std::size_t kJointCount = 12;
constexpr std::size_t kBaseObservationDim = 45;
constexpr std::size_t kPhaseObservationDim = 47;
constexpr std::size_t kVelocityFeedbackObservationDim = 47;
constexpr std::size_t kHistoryLength = 5;
constexpr std::size_t kHistoryObservationDim = 213;
constexpr std::size_t kObservationDim = kVelocityFeedbackObservationDim;
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
  kGaitPhase = 45,
  kBaseLinearVelocityXY = 45,
};

using BaseObservation = std::array<float, kBaseObservationDim>;
using PhaseObservation = std::array<float, kPhaseObservationDim>;
using VelocityFeedbackObservation =
    std::array<float, kVelocityFeedbackObservationDim>;
using HistoryObservation = std::array<float, kHistoryObservationDim>;
using Observation = VelocityFeedbackObservation;
using Action = std::array<float, kActionDim>;

enum HistoryObservationOffset : std::size_t {
  kHistoryBaseAngularVelocity = 0,
  kHistoryProjectedGravity = 15,
  kHistoryVelocityCommand = 30,
  kHistoryJointPositionRelative = 33,
  kHistoryJointVelocity = 93,
  kHistoryLastAction = 153,
};

}  // namespace custom_dog_controller

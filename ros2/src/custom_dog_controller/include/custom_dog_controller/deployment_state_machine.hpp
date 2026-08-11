#pragma once

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <stdexcept>

namespace custom_dog_controller {

constexpr std::size_t kDeploymentJointCount = 12;

using JointPositions = std::array<float, kDeploymentJointCount>;
using VelocityCommand = std::array<float, 3>;

enum class DeploymentMode {
  kPassive,
  kFixStand,
  kPolicyHold,
  kVelocity,
};

struct DeploymentStateMachineConfig {
  double stand_duration_s{2.0};
  double policy_hold_duration_s{1.0};
};

struct DeploymentOutput {
  DeploymentMode mode{DeploymentMode::kPassive};
  bool motors_enabled{false};
  bool use_policy{false};
  bool reset_policy_history{false};
  JointPositions stand_target{};
  VelocityCommand velocity_command{};
};

class DeploymentStateMachine {
 public:
  explicit DeploymentStateMachine(
      DeploymentStateMachineConfig config = DeploymentStateMachineConfig{})
      : config_(config) {
    if (!std::isfinite(config_.stand_duration_s) || config_.stand_duration_s <= 0.0 ||
        !std::isfinite(config_.policy_hold_duration_s) ||
        config_.policy_hold_duration_s < 0.0) {
      throw std::invalid_argument("Deployment state-machine durations are invalid");
    }
  }

  void enter_passive() noexcept {
    mode_ = DeploymentMode::kPassive;
    policy_reset_pending_ = false;
  }

  void begin_fix_stand(
      const JointPositions& measured_position,
      const JointPositions& home_position,
      double now_s) {
    require_finite(measured_position, "measured_position");
    require_finite(home_position, "home_position");
    require_finite(now_s, "now_s");
    stand_start_position_ = measured_position;
    home_position_ = home_position;
    stand_start_time_s_ = now_s;
    mode_ = DeploymentMode::kFixStand;
    policy_reset_pending_ = false;
  }

  DeploymentOutput update(double now_s, const VelocityCommand& requested_velocity) {
    require_finite(now_s, "now_s");
    require_finite(requested_velocity, "requested_velocity");

    DeploymentOutput output;
    if (mode_ == DeploymentMode::kPassive) {
      return output;
    }

    const double elapsed_s = std::max(0.0, now_s - stand_start_time_s_);
    if (mode_ == DeploymentMode::kFixStand && elapsed_s >= config_.stand_duration_s) {
      mode_ = DeploymentMode::kPolicyHold;
      policy_reset_pending_ = true;
    }
    if (mode_ == DeploymentMode::kPolicyHold && !policy_reset_pending_ &&
        elapsed_s >= config_.stand_duration_s + config_.policy_hold_duration_s) {
      mode_ = DeploymentMode::kVelocity;
    }

    output.mode = mode_;
    output.motors_enabled = true;
    if (mode_ == DeploymentMode::kFixStand) {
      const double progress = elapsed_s / config_.stand_duration_s;
      const float blend = static_cast<float>(quintic_smoothstep(progress));
      for (std::size_t index = 0; index < output.stand_target.size(); ++index) {
        output.stand_target[index] =
            (1.0F - blend) * stand_start_position_[index] + blend * home_position_[index];
      }
      return output;
    }

    output.use_policy = true;
    output.reset_policy_history = policy_reset_pending_;
    output.stand_target = home_position_;
    if (mode_ == DeploymentMode::kVelocity) {
      output.velocity_command = requested_velocity;
    }
    policy_reset_pending_ = false;
    return output;
  }

  DeploymentMode mode() const noexcept { return mode_; }

  static double quintic_smoothstep(double progress) noexcept {
    const double value = std::clamp(progress, 0.0, 1.0);
    return value * value * value * (10.0 + value * (-15.0 + 6.0 * value));
  }

 private:
  template <std::size_t Size>
  static void require_finite(const std::array<float, Size>& values, const char* label) {
    if (!std::all_of(values.begin(), values.end(), [](float value) {
          return std::isfinite(value);
        })) {
      throw std::invalid_argument(label);
    }
  }

  static void require_finite(double value, const char* label) {
    if (!std::isfinite(value)) {
      throw std::invalid_argument(label);
    }
  }

  DeploymentStateMachineConfig config_;
  DeploymentMode mode_{DeploymentMode::kPassive};
  JointPositions stand_start_position_{};
  JointPositions home_position_{};
  double stand_start_time_s_{0.0};
  bool policy_reset_pending_{false};
};

}  // namespace custom_dog_controller

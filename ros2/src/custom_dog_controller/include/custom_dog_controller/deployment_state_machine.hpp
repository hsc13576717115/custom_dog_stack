#pragma once

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <stdexcept>

namespace custom_dog_controller {

constexpr std::size_t kDeploymentJointCount = 12;
constexpr double kPi = 3.14159265358979323846;

using JointPositions = std::array<float, kDeploymentJointCount>;
using VelocityCommand = std::array<float, 3>;

enum class DeploymentMode {
  kPassive,
  kFixStand,
  kRecoveryPolicy,
  kUprightDwell,
  kPolicyHold,
  kVelocity,
};

struct DeploymentStateMachineConfig {
  double stand_duration_s{2.0};
  double policy_hold_duration_s{1.0};
  double recovery_timeout_s{7.0};
  double recovery_upright_dwell_s{0.4};
  double recovery_min_height_m{0.27};
  double recovery_max_tilt_rad{15.0 * kPi / 180.0};
  double recovery_max_angular_speed_rad_s{0.50};
};

struct RecoveryTelemetry {
  bool valid{false};
  double height_m{0.0};
  double tilt_rad{0.0};
  double angular_speed_rad_s{0.0};
  std::array<bool, 4> foot_contacts{};
};

struct DeploymentOutput {
  DeploymentMode mode{DeploymentMode::kPassive};
  bool motors_enabled{false};
  bool use_policy{false};
  bool use_recovery_policy{false};
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
        config_.policy_hold_duration_s < 0.0 ||
        !std::isfinite(config_.recovery_timeout_s) || config_.recovery_timeout_s <= 0.0 ||
        !std::isfinite(config_.recovery_upright_dwell_s) ||
        config_.recovery_upright_dwell_s <= 0.0 ||
        !std::isfinite(config_.recovery_min_height_m) || config_.recovery_min_height_m <= 0.0 ||
        !std::isfinite(config_.recovery_max_tilt_rad) || config_.recovery_max_tilt_rad <= 0.0 ||
        !std::isfinite(config_.recovery_max_angular_speed_rad_s) ||
        config_.recovery_max_angular_speed_rad_s <= 0.0) {
      throw std::invalid_argument("Deployment state-machine durations are invalid");
    }
  }

  void enter_passive() noexcept {
    mode_ = DeploymentMode::kPassive;
    policy_reset_pending_ = false;
    recovery_dwell_start_time_s_ = -1.0;
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
    recovery_dwell_start_time_s_ = -1.0;
  }

  DeploymentOutput update(double now_s, const VelocityCommand& requested_velocity) {
    return update(now_s, requested_velocity, RecoveryTelemetry{});
  }

  void begin_recovery(double now_s) {
    require_finite(now_s, "now_s");
    recovery_start_time_s_ = now_s;
    recovery_dwell_start_time_s_ = -1.0;
    mode_ = DeploymentMode::kRecoveryPolicy;
    policy_reset_pending_ = true;
  }

  DeploymentOutput update(
      double now_s,
      const VelocityCommand& requested_velocity,
      const RecoveryTelemetry& telemetry) {
    require_finite(now_s, "now_s");
    require_finite(requested_velocity, "requested_velocity");
    require_finite(telemetry.height_m, "recovery height");
    require_finite(telemetry.tilt_rad, "recovery tilt");
    require_finite(telemetry.angular_speed_rad_s, "recovery angular speed");

    DeploymentOutput output;
    if (mode_ == DeploymentMode::kPassive) {
      return output;
    }

    const double elapsed_s = std::max(0.0, now_s - stand_start_time_s_);
    if (mode_ == DeploymentMode::kRecoveryPolicy) {
      output.mode = mode_;
      output.motors_enabled = true;
      output.use_policy = true;
      output.use_recovery_policy = true;
      output.reset_policy_history = policy_reset_pending_;
      output.velocity_command = {0.0F, 0.0F, 0.0F};
      policy_reset_pending_ = false;
      const double recovery_elapsed = std::max(0.0, now_s - recovery_start_time_s_);
      if (recovery_elapsed >= config_.recovery_timeout_s) {
        enter_passive();
        return DeploymentOutput{};
      }
      if (recovery_ready(telemetry)) {
        if (recovery_dwell_start_time_s_ < 0.0) {
          recovery_dwell_start_time_s_ = now_s;
        }
        mode_ = DeploymentMode::kUprightDwell;
        stand_start_time_s_ = now_s;
        return update(now_s, requested_velocity, telemetry);
      } else {
        recovery_dwell_start_time_s_ = -1.0;
      }
      return output;
    }

    if (mode_ == DeploymentMode::kUprightDwell) {
      output.mode = mode_;
      output.motors_enabled = true;
      output.use_policy = true;
      output.velocity_command = {0.0F, 0.0F, 0.0F};
      output.stand_target = home_position_;
      if (!recovery_ready(telemetry)) {
        mode_ = DeploymentMode::kRecoveryPolicy;
        recovery_dwell_start_time_s_ = -1.0;
        policy_reset_pending_ = true;
        return update(now_s, requested_velocity, telemetry);
      }
      if (now_s - recovery_dwell_start_time_s_ >= config_.recovery_upright_dwell_s) {
        mode_ = DeploymentMode::kPolicyHold;
        policy_reset_pending_ = true;
        policy_hold_start_time_s_ = now_s;
      }
      output.mode = mode_;
      output.reset_policy_history = policy_reset_pending_;
      policy_reset_pending_ = false;
      return output;
    }

    if (mode_ == DeploymentMode::kFixStand && elapsed_s >= config_.stand_duration_s) {
      mode_ = DeploymentMode::kPolicyHold;
      policy_reset_pending_ = true;
      policy_hold_start_time_s_ = now_s;
    }
    if (mode_ == DeploymentMode::kPolicyHold && !policy_reset_pending_ &&
        now_s - policy_hold_start_time_s_ >= config_.policy_hold_duration_s) {
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

  bool recovery_ready(const RecoveryTelemetry& telemetry) const noexcept {
    return telemetry.valid && telemetry.height_m >= config_.recovery_min_height_m &&
           telemetry.tilt_rad <= config_.recovery_max_tilt_rad &&
           telemetry.angular_speed_rad_s <= config_.recovery_max_angular_speed_rad_s &&
           std::all_of(telemetry.foot_contacts.begin(), telemetry.foot_contacts.end(),
                       [](bool contact) { return contact; });
  }

  DeploymentStateMachineConfig config_;
  DeploymentMode mode_{DeploymentMode::kPassive};
  JointPositions stand_start_position_{};
  JointPositions home_position_{};
  double stand_start_time_s_{0.0};
  double policy_hold_start_time_s_{0.0};
  double recovery_start_time_s_{0.0};
  double recovery_dwell_start_time_s_{-1.0};
  bool policy_reset_pending_{false};
};

}  // namespace custom_dog_controller

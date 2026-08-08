#pragma once

#include <array>
#include <cstdint>
#include <string>

namespace custom_dog_hardware {

constexpr std::size_t kMotorCount = 12;

struct MotorCommand {
  std::uint8_t motor_id{0};
  float position{0.0F};
  float velocity{0.0F};
  float feedforward_torque{0.0F};
  float stiffness{0.0F};
  float damping{0.0F};
};

struct MotorState {
  std::uint8_t motor_id{0};
  float position{0.0F};
  float velocity{0.0F};
  float estimated_torque{0.0F};
  float temperature_c{0.0F};
  bool valid{false};
};

class MotorBus {
 public:
  virtual ~MotorBus() = default;

  virtual void open(const std::string& device, std::uint32_t baud_rate) = 0;
  virtual void close() noexcept = 0;
  virtual bool exchange(const MotorCommand& command, MotorState& state) = 0;
  virtual void emergency_stop() noexcept = 0;
};

using MotorCommands = std::array<MotorCommand, kMotorCount>;
using MotorStates = std::array<MotorState, kMotorCount>;

}  // namespace custom_dog_hardware

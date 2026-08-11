#pragma once

#include "custom_dog_controller/policy_contract.hpp"

#include <algorithm>
#include <array>
#include <cstddef>

namespace custom_dog_controller {

class HistoryObservationBuilder {
 public:
  HistoryObservationBuilder() { reset(); }

  void reset() {
    for (auto& frame : frames_) {
      frame.fill(0.0F);
    }
    initialized_ = false;
  }

  HistoryObservation push(const BaseObservation& current) {
    if (!initialized_) {
      frames_.fill(current);
      initialized_ = true;
    } else {
      std::move(frames_.begin() + 1, frames_.end(), frames_.begin());
      frames_.back() = current;
    }
    return build();
  }

 private:
  template <std::size_t Width>
  void copy_history_term(
      HistoryObservation& result,
      std::size_t source_offset,
      std::size_t destination_offset) const {
    for (std::size_t frame = 0; frame < kHistoryLength; ++frame) {
      std::copy_n(
          frames_[frame].begin() + source_offset,
          Width,
          result.begin() + destination_offset + frame * Width);
    }
  }

  HistoryObservation build() const {
    HistoryObservation result{};
    copy_history_term<3>(result, kBaseAngularVelocity, kHistoryBaseAngularVelocity);
    copy_history_term<3>(result, kProjectedGravity, kHistoryProjectedGravity);
    std::copy_n(
        frames_.back().begin() + kVelocityCommand,
        3,
        result.begin() + kHistoryVelocityCommand);
    copy_history_term<12>(
        result, kJointPositionRelative, kHistoryJointPositionRelative);
    copy_history_term<12>(result, kJointVelocity, kHistoryJointVelocity);
    copy_history_term<12>(result, kLastAction, kHistoryLastAction);
    return result;
  }

  std::array<BaseObservation, kHistoryLength> frames_{};
  bool initialized_{false};
};

}  // namespace custom_dog_controller

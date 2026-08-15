# custom_dog_controller

This package owns observation construction, ONNX inference, action scaling and
safety limiting. `DeploymentStateMachine` implements the hardware-independent
control sequence:

```text
Passive -> 2 s measured-position quintic FixStand
        -> 1 s zero-command PolicyHold -> Velocity
```

For a separately validated recovery actor, the state machine also supports:

```text
RecoveryPolicy -> UprightDwell(0.4 s) -> 1 s zero-command PolicyHold -> Velocity
```

`begin_recovery()` starts the recovery policy with zero velocity. The overload of `update()` that
takes `RecoveryTelemetry` requires height `>=0.27 m`, tilt `<=15 deg`, angular speed `<=0.50 rad/s`,
and all four foot contacts throughout the dwell. A timeout or loss of any condition returns to
`RecoveryPolicy` or Passive; velocity commands are never forwarded during recovery or dwell.
The output flag `use_recovery_policy` lets the eventual ONNX executor select the recovery actor
without coupling this package to a specific inference implementation.

The state machine takes the HOME position from the selected candidate contract,
requests a policy-history reset exactly once at takeover, and suppresses external
`vx/vy/wz` until the hold completes. It does not yet contain the ONNX Runtime node
or the GO-M8010-6 RS485 protocol implementation.

`policy_contract.hpp` defines the 45-D base observation and two explicit 47-D
variants: gait phase, or body-frame x/y velocity feedback. The latter requires a
time-synchronized IMU plus leg-kinematics velocity estimator before hardware
deployment; raw IMU data alone does not provide linear velocity.

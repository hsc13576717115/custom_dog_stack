# custom_dog_controller

This package owns observation construction, ONNX inference, action scaling and
safety limiting. `DeploymentStateMachine` implements the hardware-independent
control sequence:

```text
Passive -> 2 s measured-position quintic FixStand
        -> 1 s zero-command PolicyHold -> Velocity
```

The state machine takes the HOME position from the selected candidate contract,
requests a policy-history reset exactly once at takeover, and suppresses external
`vx/vy/wz` until the hold completes. It does not yet contain the ONNX Runtime node
or the GO-M8010-6 RS485 protocol implementation.

`policy_contract.hpp` defines the 45-D base observation and two explicit 47-D
variants: gait phase, or body-frame x/y velocity feedback. The latter requires a
time-synchronized IMU plus leg-kinematics velocity estimator before hardware
deployment; raw IMU data alone does not provide linear velocity.

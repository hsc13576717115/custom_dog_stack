# Omni45 v3 Polish Report

Date: 2026-08-11

## Scope

This experiment kept the deployed policy contract unchanged: 45 policy observations,
12 joint-position actions, and a 50 Hz control period. It resumed the V2 baseline
`2026-08-11_11-31-14_omni45_v2_vx1_omni_main/model_4999.pt` rather than retraining
from scratch.

## Experiments

`omni45_v3_polish_from4999` resumed 1,000 requested PPO iterations. It added a high
pure-lateral command share, low-speed signed-x oversampling, component-relative
tracking `-2.0`, inactive-axis drift `-1.0`, pre-contact soft landing, and mirror
loss. It was stopped after `model_5820.pt`: MuJoCo evaluation showed persistent
velocity overshoot. For example, `model_5400.pt` produced `vx=1 -> 1.694 m/s` and
`vy=+/-0.4 -> +0.785/-0.751 m/s`. The final `model_5820.pt` passed only one of nine
fixed-grid points, with maximum errors of `0.750 m/s` in vx and `0.263 m/s` in vy.

`omni45_v3_conservative_from5200` resumed the best early V3 checkpoint
`model_5200.pt` for 500 requested PPO iterations. It reduced the component tracking
weight to `-0.75`, reduced style weights, and used a training-only lateral band of
`[-0.25, 0.25]`. Its final checkpoint is `model_5699.pt`. Training remained stable:
the final 30-iteration mean timeout rate was `0.9561` and bad-orientation rate was
`0.0439`. Stability alone was insufficient: `model_5300.pt` tracked
`vy=+/-0.4` as `+0.404/-0.406 m/s`, but simultaneously produced `vx=1 -> 2.149 m/s`.
At `model_5600.pt`, `vx=1 -> 2.369 m/s`.

## Fixed MuJoCo Grid

All checks used the canonical MJCF, each exported ONNX, `8 s` duration and `2 s`
warm-up. The grid was zero command, `vx={0.3,0.5,1,-1}`, `vy={+/-0.4}`, and
`wz={+/-1}`.

| Candidate | Best observed improvement | Rejection reason |
| --- | --- | --- |
| V2 `model_4999` | Stable forward/reverse/yaw baseline | `vx=0.3` dead zone and asymmetric pure lateral motion |
| V3 `model_5200` | `vx=0.3 -> 0.209`, `vy=+0.4 -> 0.325` | `vy=-0.4 -> -0.161`; not balanced |
| V3 `model_5400` | `vx=0.3 -> 0.316` | high-speed and lateral overshoot; hip outward max `21.21 deg` |
| Conservative `model_5300` | `vy=+/-0.4 -> +0.404/-0.406` | `vx=1 -> 2.149 m/s`, yaw error `0.154 rad/s` |
| Conservative `model_5600` | lower tilt than baseline | `vx=1 -> 2.369 m/s` |

No V3 checkpoint is a deployment candidate. Keep V2 `model_4999` as the only
forward-motion reference from this branch. The next training direction must preserve
the accepted forward controller while learning lateral/yaw in separate teachers and
then distilling or gating them; do not continue the same full-axis reward fine-tune.

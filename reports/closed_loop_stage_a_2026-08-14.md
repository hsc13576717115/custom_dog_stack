# Closed-Loop Stage A Evaluation

## Foundation Run

- Run: `2026-08-14_15-13-55_omni_trot_closed_loop_foundation_seed42`
- Checkpoint interval: 25 iterations; final checkpoint: `model_999.pt`
- Training completed: 1000 iterations, 131,072,000 simulation steps
- Actor contract: 51 observations (45 base + 4 trot clock + 2 body-frame planar velocity)

## Fixed MuJoCo Grid

Six checkpoints were exported and evaluated for 10 seconds per command with a 2 second warmup.
The grid contains standing, signed low/medium forward commands, signed lateral commands, signed
low/high yaw commands, and signed combined commands. Raw results are stored under the run's
`evaluation/stage_a_grid_10s.{csv,json}` files.

| Candidate | Absolute passes | Max vx error | Max vy error | Max wz error | Standing height | Max tilt | Mean hip outward |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| model_625 | 11/15 | 0.061 | 0.009 | 0.127 | 0.301 m | 6.90 deg | 4.69 deg |
| model_775 | 11/15 | 0.070 | 0.031 | 0.139 | 0.301 m | 6.67 deg | 4.53 deg |
| model_875 | 11/15 | 0.067 | 0.029 | 0.126 | 0.303 m | 6.70 deg | 4.27 deg |
| model_900 | 11/15 | 0.041 | 0.013 | 0.115 | 0.302 m | 6.37 deg | 4.37 deg |
| model_975 | 11/15 | 0.030 | 0.003 | 0.109 | 0.300 m | 6.62 deg | 3.87 deg |
| model_999 | 12/15 | 0.043 | 0.013 | 0.116 | 0.300 m | 7.05 deg | 4.05 deg |

The pass counts above were rescored from the saved raw measurements after adding the missing
pure-yaw integral-bias gate (`abs(yaw bias) / measured duration <= 0.05 rad/s`), tightening
pure-yaw XY drift to 0.05 m/s, applying the 3 deg standing-tilt gate, and requiring every leg to
show at least two ground-contact transitions even at the 0.05 m/s command. The overall maximum
tilt column spans all commands and is therefore not the standing-tilt measurement.

All candidates passed planar tracking, pure-axis decoupling, low-speed contact transitions,
stability, and pure-yaw XY drift. The remaining common failures were:

- zero-command height settled at 0.300-0.303 m instead of the 0.310-0.335 m gate;
- low and high pure-yaw commands accumulated excessive yaw bias, while `wz=+/-0.25 rad/s`
  overshot to approximately `+/-0.35...0.39 rad/s`;
- pure-yaw XY drift remained low at approximately 0.002-0.007 m/s, so drift penalties should
  not be increased in the next run.

The retained interactive candidates are `model_975` (best aggregate tracking and hip posture),
`model_999` (best standing tilt), and `model_900` (13/15 absolute gates and good tracking).

```bash
./scripts/teleop_mujoco_policy.sh logs/rsl_rl/custom_dog_velocity/2026-08-14_15-13-55_omni_trot_closed_loop_foundation_seed42/evaluation/candidates/model_975
./scripts/teleop_mujoco_policy.sh logs/rsl_rl/custom_dog_velocity/2026-08-14_15-13-55_omni_trot_closed_loop_foundation_seed42/evaluation/candidates/model_999
./scripts/teleop_mujoco_policy.sh logs/rsl_rl/custom_dog_velocity/2026-08-14_15-13-55_omni_trot_closed_loop_foundation_seed42/evaluation/candidates/model_900
```

## Stage A.1 Decision

`model_975` is the continuation source. `ClosedLoopPolishA1` keeps the Stage-A command envelope,
extends command holds to 8-12 seconds, increases the pure-yaw bucket and low-yaw sampling,
strengthens yaw-rate error, and raises the zero-command height objective. It deliberately leaves
the already-passing pure-axis drift and locomotion range unchanged.

The formal run is `2026-08-14_17-10-48_closed_loop_a1_yaw_height_from975`, configured for 400
additional iterations. It must pass the same fixed grid before a second random seed or Stage B is
started.

The run completed at `model_1374`. A strict 120-run grid compared the foundation source and seven
A1 checkpoints. No candidate passed Stage A. `model_1150` and `model_1200` were the best A1
checkpoints at 12/15 absolute gates, but their high pure-yaw responses still overshot:

| Candidate | `wz=+0.25` | `wz=-0.25` | Standing height | Pure-yaw XY drift | Min contacts |
| --- | ---: | ---: | ---: | ---: | ---: |
| foundation `model_975` | 0.354 | -0.359 | 0.300 m | 0.004 m/s | 23 |
| A1 `model_1150` | 0.355 | -0.371 | 0.301 m | 0.006 m/s | 23 |
| A1 `model_1200` | 0.362 | -0.373 | 0.303 m | 0.007 m/s | 23 |

The same A1 `model_1150` under a fixed Isaac command produced `0.285 rad/s` for a requested
`0.25 rad/s`. This establishes two contributors: the policy itself overshoots in its training
simulator, and MuJoCo further amplifies that response. A1 improved the logged aggregate yaw error
but over-allocated yaw-bearing samples to the 0.05-0.18 low-speed band, leaving the 0.20-0.25
boundary underrepresented.

## Stage A.2 Decision

`ClosedLoopPolishA2` keeps the command envelope and 8-12 second holds unchanged. It explicitly
assigns 40% of yaw-bearing samples to 0.20-0.25 rad/s, retains 50% low-yaw samples, adds a pure-yaw
relative overspeed loss, and adds a normalized 0.31-0.33 m zero-command height-band loss. Standing
joint and foot-placement constraints are strengthened to reverse A1's transient hip-abduction
regression. The continuation source is A1 `model_1150`, and the run is limited to 300 additional
iterations. Stage B and the independent seed remain blocked until the same MuJoCo grid passes.

The A2 run `2026-08-14_17-53-06_closed_loop_a2_yaw_boundary_standing_band` completed 300
additional iterations at `model_1449`. Six checkpoints plus the A1 source were exported for a
105-run grid. A2 reduced high-yaw error substantially and raised early-checkpoint standing height,
but all candidates still failed the same three commands: zero stand and pure yaw at +/-0.25 rad/s.

| Candidate | Max yaw error | Stand height | Stand tilt | Front hip outward | Min contacts |
| --- | ---: | ---: | ---: | ---: | ---: |
| A1 `model_1150` | 0.121 | 0.301 m | 0.74 deg | 11.4/12.1 deg | 23 |
| A2 `model_1300` | 0.083 | 0.316 m | 3.30 deg | 17.3/18.3 deg | 23 |
| A2 `model_1350` | **0.078** | **0.313 m** | 3.42 deg | 17.5/18.4 deg | 23 |
| A2 `model_1400` | 0.089 | 0.309 m | **1.79 deg** | 16.1/17.0 deg | 23 |
| A2 `model_1449` | 0.086 | 0.305 m | **1.68 deg** | 16.1/16.9 deg | 23 |

A fixed `wz=0.25` diagnostic on A2 `model_1350` measured 0.239 rad/s in one Isaac environment
and 0.230 rad/s averaged over 32 randomized Isaac environments, versus 0.321 rad/s in MuJoCo.
The policy is therefore accurate to slightly conservative in its training simulator; its remaining
yaw failure is a physics-transfer gap, not a reason to keep increasing yaw penalties. A MuJoCo
sensitivity sweep over contact friction 0.70, 0.85, 1.10 and joint friction 0.01 left the maximum
yaw error at 0.077-0.092 rad/s, so no friction value was adopted as an unsupported calibration.

A separate `ClosedLoopStandFix` continuation starts from A2 `model_1350`. It removes the A2
overspeed-only yaw loss, increases exact standing samples, and adds zero-command-only normalized
hip-pose and trunk-orientation losses. Thigh and calf joints remain free to achieve the height band.
The transfer gap will be addressed with explicit cross-physics/dynamics robustness rather than a
hidden joystick calibration.

That StandFix run completed 120 additional iterations at `model_1469`, but it regressed rather
than repaired standing. The source A2 `model_1350` remained at 12 absolute and 12 all-gate passes;
StandFix checkpoints fell from 10 all-gate passes at `model_1380` to zero at `model_1420` and later.
Standing height declined from 0.313 m to 0.297 m, tilt rose from 3.42 deg to 8.06 deg, and maximum
hip outward angle rose from 23.37 deg to 32.06 deg. The entire StandFix branch is rejected.

A deterministic stand-hold experiment was also rejected. The symmetric default pose settled at
only 0.235 m. Holding the mean zero-command joint targets extracted from A2 `model_1350` removed
drift but settled at 0.263 m with 19.37 deg maximum tilt. The acceptable A2 standing result depends
on continuous policy feedback, so fixed joint targets must not replace it. A dedicated stand expert
or a unified policy with a better feedback/state objective is required for materially better idle
posture.

## Stage A.3 Cross-Physics Adaptation

The retained source remains A2 `model_1350`. A3 does not expand the command envelope or add more
yaw bias. It removes the A2 overspeed-only loss because fixed Isaac tests already measure
0.230-0.239 rad/s for a 0.25 rad/s request, then randomizes the physical parameters most likely to
explain the MuJoCo response gap:

- contact friction and restitution;
- base payload plus 0.95-1.05 all-link mass scaling;
- base center of mass by up to 15 mm in X/Y and 10 mm in Z;
- stiffness 0.85-1.15 and damping 0.75-1.35 scaling;
- joint friction from 0.00 to 0.03;
- zero to two physics steps of actuator delay (0-10 ms at 200 Hz physics).

The one-iteration, 64-environment GPU smoke run activated all six startup event groups and restored
the 51-D A2 checkpoint successfully. The formal 300-iteration continuation is
`2026-08-14_18-57-35_closed_loop_cross_physics_from_a2_1350`. It must improve the same fixed MuJoCo
grid without regressing the fixed Isaac response before a second seed is started.

The formal run completed at `model_1649`; the randomized-physics training yaw metric fell from
approximately 0.155 after startup to 0.106. Six checkpoints were then compared with the A2 source
over the same 105-run MuJoCo grid:

| Candidate | Absolute passes | Max yaw error | Stand height | Stand tilt | Max hip outward |
| --- | ---: | ---: | ---: | ---: | ---: |
| A2 `model_1350` | 12/15 | 0.078 | 0.313 m | 3.42 deg | 23.37 deg |
| A3 `model_1400` | 12/15 | **0.067** | 0.304 m | 4.67 deg | 28.91 deg |
| A3 `model_1460` | 12/15 | 0.071 | 0.301 m | 6.08 deg | 31.59 deg |
| A3 `model_1520` | 12/15 | 0.079 | 0.298 m | 7.89 deg | 35.27 deg |
| A3 `model_1580` | 12/15 | 0.085 | 0.294 m | 11.07 deg | 39.57 deg |
| A3 `model_1620` | 12/15 | 0.088 | 0.295 m | 9.74 deg | 37.62 deg |
| A3 `model_1649` | 12/15 | 0.095 | 0.292 m | 7.63 deg | 35.70 deg |

A3 therefore fails Stage A. Its early checkpoint slightly reduces the MuJoCo yaw response
(`+0.312/-0.317` instead of A2's `+0.321/-0.328` for requested `+/-0.25`), but standing posture
regresses immediately and continues to deteriorate. Broad unobserved dynamics randomization makes
the policy optimize an average physical system; it does not identify which physical system is
currently active. A2 `model_1350` remains the retained baseline.

The next action is physics alignment rather than another reward continuation: compare mass, center
of mass, inertia, imported joint axes, free-space actuator step response, grounded contact response,
and solver timing between Isaac and MuJoCo. If a documented model correction cannot close the yaw
gap, the next policy architecture must expose privileged dynamics to a teacher and infer them from
history in a deployable student (RMA/HIM style). An arbitrary hidden joystick scale is not accepted
as a training fix.

## Gated Training Roadmap

The following stages are intentionally sequential. A later stage must not hide a failure in an
earlier one by increasing command diversity or terrain difficulty.

1. **Stage A.1: closed-loop flat-ground correction.** Export representative checkpoints from the
   400-iteration continuation and repeat the same 15-command MuJoCo grid. Require standing height
   0.310-0.335 m, standing tilt below 3 deg, pure-yaw XY drift below 0.05 m/s, pure-axis inactive
   velocity below 0.05 m/s (or yaw below 0.08 rad/s), yaw error below 0.10 rad/s, and low-speed
   contact transitions on every leg. Retain two or three checkpoints for keyboard review.
2. **Independent seed confirmation.** Train the final Stage-A configuration from scratch with a
   second seed. It must pass the same numerical gates. This distinguishes a repeatable objective
   from a lucky continuation checkpoint.
3. **Collision calibration before speed expansion.** Build a separate collision asset variant with
   simplified convex collision geometry. Enable only non-adjacent cross-leg contacts in both Isaac
   Lab and MuJoCo; keep overlapping parent-child joint housings filtered. Validate the nominal pose
   without persistent contacts, verify intentional left-right leg contact, and add a collision count
   to the fixed grid. The current assets disable robot self-contact in both simulators.
4. **Stage B:** continue to `vx +/-0.8`, `vy +/-0.2`, `wz +/-0.5`. Re-run signed pure-axis,
   combined-command, standing, and keyboard tests. Do not advance on a total-reward improvement
   alone.
5. **Stage C:** continue to `vx +/-1.5`, `vy +/-0.4`, `wz +/-1.0`, preserving explicit low-speed
   buckets and 8-12 second command holds. Add push-recovery and longer 30 second drift trials.
6. **Stage D:** continue to the requested final envelope `vx +/-3.0`, `vy +/-0.6`,
   `wz +/-2.0`. Enforce the 0.33-to-0.28 m speed-adaptive height target, tilt below 10 deg at the
   envelope edge, no fall, motor torque-speed limits, and thermal/energy proxy limits.
7. **Separate self-righting expert.** Train a recovery policy only after collision behavior agrees
   between simulators. Its reset distribution must cover prone, supine, both sides, randomized limb
   poses, and low-energy dynamic falls. Deployment uses a state machine to enter recovery, require
   an upright/height/low-angular-rate dwell, then switch to the locomotion policy. The existing
   `RecoveryOmni` prototype and the scripted two-second joint interpolation are not this capability.
8. **Robustness and gait controls.** Add actuator delay, gain/strength, mass/COM, friction, payload,
   sensor noise, and external pushes using HIM/RMA-style latent adaptation. Add Walk These Ways
   parameters such as step frequency, swing height, stance width, body height, and pitch only after
   the base velocity policy remains closed-loop stable; do not import an unrelated checkpoint.
9. **Random terrain.** Start with 60-70% mild terrain and retain 30-40% flat samples so flat-ground
   tracking is not forgotten. Progress slope, roughness, steps, and gaps separately, with regression
   checks on the complete flat-ground grid after every terrain stage.
10. **Deployable student.** Distill the privileged 51-D teacher into a history policy that estimates
    motion from IMU, encoder, command, and action history. Update the existing 213-D path, which is
    currently tied to an older 47-D teacher contract, then validate observation ordering and reset
    history identically in Isaac Lab, Python MuJoCo, C++, and ROS 2.

The self-righting expert and random-terrain policy are therefore included in the plan, but neither
should be introduced before flat-ground tracking and collision consistency pass their gates.

R2 now has a second, stricter integration gate: `scripts/evaluate_recovery_handoff_gated.sh`
loads the accepted R2 recovery export together with the accepted Stage-D locomotion route and stand
expert, and runs all four canonical fall orientations through the state machine. A passing artifact
must contain both the 0.4 s recovery success and the later locomotion-command-release marker. Until
that JSON exists, R2 means only “the recovery actor stood up in isolation.”

### Self-Righting V2 Contract (Implemented, Not Yet Trained)

The old `RecoveryOmni-v1` task remains rejected: it mixes recovery and velocity tracking, covers
only a level folded-belly reset, releases nonzero commands after one second, and retains an
orientation termination that immediately rejects side and back states. A new independent contract
is now implemented behind the selective-collision asset:

- `CustomDog-SelfRighting-R0-v2` uses only the measured folded-belly distribution;
- `CustomDog-SelfRighting-R1-v2` samples belly, back, left side, and right side equally;
- `CustomDog-SelfRighting-R2-v2` adds 35% arbitrary SO(3) orientations, asymmetric limb poses,
  up to `0.5 m/s` initial linear velocity, and up to `1.5 rad/s` initial angular velocity;
- every stage holds the velocity command at exactly zero and uses a separate 45-D actor;
- base contact and bad-orientation termination are disabled during recovery;
- success requires height at least `0.27 m`, tilt at most 15 deg, angular speed at most
  `0.50 rad/s`, and all four feet above the contact-force threshold continuously for 0.40 seconds;
- the reward supplies smooth orientation progress even from a fully supine reset, height progress,
  stable four-foot support, time cost, and conservative torque/action/limit regularization.

The reset, reward, success-dwell, staged-distribution, PPO, and task-registration contracts have
focused tests. No recovery training is allowed until the live Isaac selective-collision validator
proves both zero persistent non-foot contacts in nominal stance and detectable contact under forced
front-leg crossing. R0 must then pass belly recovery before R1 or R2 is started.

## Physics Alignment And Robust Foundation

The nominal Isaac Lab runtime contract was dumped from a live one-environment simulation and
compared against the compiled MuJoCo model. The comparison is stored in
`reports/physics_contract_comparison.json`. After adding the missing `0.01 N m` joint friction loss
to both generated MJCF files, the engines agree on all model quantities that can be compared
directly:

- total robot mass is `13.849160 kg`;
- maximum per-link COM error is below `2.5e-9 m`;
- maximum inertia-component error is below `4.2e-9 kg m^2`;
- maximum joint-limit error is below `2.9e-7 rad`;
- explicit actuator gains are `Kp=25`, `Kd=0.5` and torque-speed limits agree;
- simulation step is `0.005 s`, policy step is `0.020 s`, and gravity agrees exactly.

A structured MuJoCo contact sweep varied friction from 0.60 to 1.40 and solver time constants from
0.0025 to 0.0200 seconds. None closed the yaw gap without degrading standing height, tilt, or hip
posture. The nominal contact parameters are therefore retained. An explicit-PD MuJoCo diagnostic
also reduced joint-speed mismatch but worsened yaw tracking, so the normal position-actuator path
remains the deployment default.

Matched policy traces establish the remaining gap more precisely. With randomization and
observation corruption disabled, A2 `model_1350` reaches approximately `0.233 rad/s` in Isaac and
`0.322 rad/s` in MuJoCo for a `0.25 rad/s` yaw command. Both engines reproduce the exported ONNX
observations, actions, and targets to numerical precision. The foot collision meshes are also
identical to their generated convex hulls. This rules out an observation-order bug, gain mismatch,
or accidental collision-mesh substitution as the main cause; engine contact integration remains a
real transfer difference that the policy must tolerate.

The next candidate is consequently trained from scratch instead of continuing the locally
over-fitted A2 branch. `ClosedLoopRobustFoundation-v1` keeps the 51-D teacher observation and Stage
A command envelope, enables the documented dynamics randomization from the first iteration,
increases the policy's yaw-gyro scale from `0.2` to `1.0`, applies bounded gyro noise, restores a
relative yaw-overspeed loss, and increases exact standing exposure. It deliberately does not
reintroduce the rejected forced stand joint-pose continuation.

The formal seed-42 run is
`2026-08-14_19-59-12_closed_loop_robust_foundation_seed42`, configured for 1000 iterations and
4096 environments. At iteration 23 it was healthy at approximately 28,300 simulation steps per
second, with no base-contact failures, no numerical errors, and an ETA near 21:15 CST. Checkpoints
are saved every 25 iterations. On completion, representative checkpoints will be exported and
ranked by the same 15-command MuJoCo grid plus clean fixed-command Isaac traces. A second seed is
allowed only if one checkpoint passes or materially improves the explicit Stage-A gates without a
standing regression.

The first meaningful convergence comparison at iteration 250 supports the observation change.
Under randomized training physics, the robust foundation's aggregate yaw error is `0.126 rad/s`,
versus `0.224 rad/s` for the original foundation at the same iteration, a reduction of approximately
44%. Aggregate planar error is temporarily worse (`0.093 m/s` versus `0.073 m/s`), so the result is
not yet a pass. Episodes reach the full 1000 steps, bad-orientation termination is zero, the
standing-height-band loss is near zero, and hip-band loss is lower than the original run. Training
therefore continues to the predeclared checkpoints; no selection is made from total reward alone.

## Evaluation And Collision Contract Updates

The evaluator now treats pure-yaw integral bias as a first-class gate, requires at least two
ground-contact transitions from every leg at the 0.05 m/s command, uses a 3 deg zero-command tilt
limit, and applies command-aware hip-abduction limits: 12 deg standing, 18 deg pure fore-aft, and
25 deg when lateral or yaw motion needs additional stance width. Foot contact statistics now count
only contacts against world-owned terrain, so future leg-to-leg contacts cannot masquerade as gait
transitions, foot slip, or touchdown impact.

An opt-in MuJoCo selective-collision generator is implemented but is not yet the default. It assigns
one collision bit per leg, preserves ground contact, filters all links within the same leg, and
allows contact across different legs. The generated full robot MJCF compiled successfully and the
automated validator reported:

```text
selective collision OK: nominal_self_contact_steps=0/400, forced_cross_leg_pairs=4
```

Isaac Lab must still receive equivalent filtered pairs and pass a one-environment contact smoke
test before either simulator switches to this collision model. This prevents a one-sided physics
change from invalidating sim2sim comparisons.

The focused contract tests pass. The full suite reports 90 passing tests and seven setup errors
caused by two pre-existing missing legacy candidate artifact directories:
`model_700_compact_omni_balanced` and `model_800_omni_stability_calibrated`.

## Executable Self-Righting Curriculum

The independent R0/R1/R2 entry point is now `scripts/train_self_righting.sh`. R0 starts from
scratch. R1 and R2 require an explicit accepted source, so an arbitrary latest checkpoint cannot
silently enter the next curriculum stage:

```bash
./scripts/train_self_righting.sh R0

CUSTOM_DOG_SOURCE_RUN=<accepted-r0-run> \
CUSTOM_DOG_SOURCE_CHECKPOINT=model_N.pt \
./scripts/train_self_righting.sh R1

CUSTOM_DOG_SOURCE_RUN=<accepted-r1-run> \
CUSTOM_DOG_SOURCE_CHECKPOINT=model_N.pt \
./scripts/train_self_righting.sh R2
```

The MuJoCo runner now provides direct `recovery-belly`, `recovery-back`, `recovery-left`, and
`recovery-right` initial states. Unlike the legacy `prone` option, these states do not execute the
scripted two-second joint interpolation. Exported candidates are checked with:

```bash
python scripts/evaluate_self_righting_mujoco.py \
  <candidate-directory> --stage R0 --output <candidate-directory>/recovery_r0.json
```

R0 requires belly recovery; R1 and R2 require all four canonical poses. Merely crossing upright
does not pass: the height, tilt, angular-rate, and four-foot support gate must remain true for 0.40
seconds. R2's arbitrary-orientation and dynamic-fall distribution also needs a batched Isaac test,
because the deterministic MuJoCo gate intentionally covers only reproducible canonical poses.

## Terrain Admission Gate

Random terrain will be a separate curriculum branch, not an immediate edit to the flat-ground
task. It starts only after two independent flat-ground seeds pass the command grid, selective
self-collision agrees across both simulators, and Stage B passes 30-second drift plus push-recovery
tests. A frozen flat-ground candidate remains the regression baseline.

Terrain T0 will use 70% flat samples and 30% mild slopes/roughness. T1 may use 40% flat samples and
add low steps; gaps and discrete obstacles remain T2. Advancement requires terrain success without
regressing the unchanged flat command grid. Self-righting samples are never mixed into normal
locomotion episodes: a deployment supervisor switches between the separate recovery and locomotion
actors.

The gated implementations now exist as `ClosedLoopStageB/C/D-v1`. Stage B inherits the robust
51-D observation, bounded dynamics randomization, and selective cross-leg collision contract;
Stages C and D inherit it transitively. Their cumulative MuJoCo grids retain all Stage-A commands
and add boundaries at `(+/-0.8, +/-0.2, +/-0.5)`, then `(+/-1.5, +/-0.4, +/-1.0)`, and finally
`(+/-3.0, +/-0.6, +/-2.0)` for `(vx, vy, wz)`. The evaluator additionally checks the commanded
speed-dependent 0.33-to-0.28 m mean body-height target. Training is explicit and gated:

```bash
CUSTOM_DOG_SOURCE_RUN=<accepted-previous-run> \
CUSTOM_DOG_SOURCE_CHECKPOINT=model_N.pt \
./scripts/train_closed_loop_stage.sh B
```

The dormant terrain tasks are `TerrainT0-v1` (70% flat, at most 2.5 cm random roughness and mild
slopes) and `TerrainT1-v1` (40% flat, at most 5 cm roughness, steeper slopes, and at most 8 cm low
stairs). The policy observation remains proprioceptive; only the critic receives the height scan.
They are launched by `scripts/train_terrain_stage.sh` only after the terrain admission gate passes.

Finally, `CustomDog-Velocity-ClosedLoop-History213-Distill-v1` provides a new, non-destructive
51-to-213 contract. The teacher observes the exact 45-D base state followed by four trot-clock
values and true `vx/vy`; the deployable student receives five frames of IMU, gravity, joint state,
and last action plus the current command. This is the history-based adaptation step inspired by
RMA/HIM, but it is not yet claimed to be a learned explicit dynamics latent. Its entry point is
`scripts/distill_closed_loop_history213.sh` and remains blocked until a final 51-D teacher passes.

## Accepted Routed Stage A Baseline

The seed-42 robust-foundation run completed and the fixed MuJoCo grid selected `model_700` as the
locomotion expert. It passed all 14 moving commands with maximum tracking errors of `0.021 m/s`,
`0.022 m/s`, and `0.025 rad/s` for vx, vy, and wz. Its only failed command was zero stand, so the
motion policy was frozen instead of being distorted by another mixed standing/locomotion reward
continuation.

A separate stand expert was trained and calibrated. The smallest symmetric stand-only hip target
bias that passed every zero-command gate was `0.04 rad`; larger passing biases were deliberately
rejected as unnecessary. Over a 15-second MuJoCo trial it produced:

- mean height `0.321 m` and minimum height `0.319 m`;
- maximum tilt `2.63 deg`;
- mean/max hip outward angles `9.22/11.49 deg`;
- all four feet continuously grounded, no fall, and no self-collision count.

The accepted candidate `deploy/candidates/closed_loop_stage_a_routed_seed42` routes exact and near
zero commands to the stand expert and moving commands to robust-foundation `model_700`. It uses
hysteresis (`0.015/0.025 m/s` enter/exit for planar speed and `0.025/0.04 rad/s` for yaw) plus a
`0.30 s` quintic action blend. The complete routed policy passed all 15/15 Stage-A commands. Its
maximum tracking errors remained `0.021/0.022/0.025`, maximum pure-yaw XY drift was
`0.0186 m/s`, maximum pure-yaw integral-bias rate was `0.0253 rad/s`, and the minimum low-speed
per-leg contact-transition count was 22.

A separate stand-to-`vx=0.05 m/s`-to-stand trace also passed: segment speed errors were
`0.007/0.011/0.007 m/s`, maximum hip outward angle was `11.92 deg`, dynamic maximum tilt was
`5.30 deg`, minimum height was `0.296 m`, and there was no fall. This validates the router around
the physical low-speed transition instead of only validating steady commands.

Seed 73 is now training the identical robust-foundation configuration from scratch. Its completion
automatically triggers the same routed 15-command MuJoCo evaluation using the frozen accepted stand
expert. Selective-collision adaptation is queued behind that gate and cannot start if seed 73 fails.

## Live Selective-Collision Correction

The first live Isaac selective-collision report was not a pass: although all 40 configured filter
pairs were present, nominal standing produced non-foot contact above `1 N` during `200/200`
evaluation steps, with a maximum net force of `10.41 N`. MuJoCo's selective model had zero nominal
self-contact, so the two engines were not yet equivalent and collision adaptation remained blocked.

Inspection of Isaac Sim's own `addPairFilter` implementation found that it authors a filtered-pair
relationship on both endpoints. The custom spawner had authored only `A -> B`; that relationship was
visible in USD inspection but did not reliably suppress articulation-link contact. The spawner and
validator now require reciprocal `A -> B` and `B -> A` relationships for every base/same-leg pair.
The validator additionally records per-body force and contact-step diagnostics. This correction has
passed static contract tests but is deliberately not marked accepted yet. The queued collision
service will regenerate `reports/selective_collision_isaac_runtime.json` in a live one-environment
PhysX run after seed 73 completes; training proceeds only if nominal non-foot contact is zero and the
forced front-leg crossing still produces contact.

## Executable Post-D Pipeline

The post-D gait stage is now `CustomDog-Velocity-OmniTrot-ClosedLoopGaitRobust-v1`. It preserves the
51-D teacher contract and the accepted diagonal-trot clock while making commanded swing clearance
increase from `0.045 m` to `0.080 m` over the speed envelope. Lateral and yaw commands receive modest
landing-width allowances. This is the first Walk-These-Ways-inspired step: deterministic,
command-conditioned gait shaping without importing an incompatible checkpoint or adding unestimated
hardware inputs. Every candidate must still pass the complete Stage-D selective-collision MuJoCo
grid before it can seed terrain training.

T0 and T1 now use deterministic curriculum terrain columns. `curriculum=True` is required because
the stratified command sampler assigns the first `14/20` T0 columns and `8/20` T1 columns to flat
rehearsal at the full Stage-D command envelope. Remaining T0 terrain is clamped to Stage B and
remaining T1 terrain to Stage C. Terrain level advancement is re-enabled after the flat-ground
foundation disabled it. The terrain evaluator loads the full training configuration, samples every
difficulty row, and gates every terrain family separately in addition to the six command groups.
This prevents a good flat average from hiding a failure on roughness, either slope direction, or
either stair direction.

The final `ClosedLoop-History213-Distill-v1` student is trained on the accepted T1 mixed distribution:
flat environments continue rehearsing the full Stage-D envelope while rough environments retain the
Stage-C safety clamp. Five exported student checkpoints are evaluated with the full Stage-D flat
MuJoCo grid and the grouped T1 Isaac terrain gate. The locomotion student is still routed with the
accepted stand expert for sim2sim evaluation; distillation does not silently claim to have absorbed
the Python-only stand hip bias.

The queue script defines the full dependency chain
`B -> R0 -> C -> R1 -> D -> R2 -> gait -> T0 -> T1 -> H213`. At the time of this report the host has
accepted units through R2. Adding the last four user-systemd units was not performed because the
command-approval backend returned an unsupported-model 404; the scripts and static gates are ready,
but those units must not be reported as queued until a later successful systemd operation confirms
them.

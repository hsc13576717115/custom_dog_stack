# HimLoco Terrain Refinement Evaluation

Evaluation date: 2026-08-17

## Current best model

**`model_12000` remains the best overall model for current deployment.** Use:

`/home/hsc/Dog_RL/custom_dog_stack/deploy/candidates/himloco_model_12000`

The terrain-refinement run started from this checkpoint and produced `model_14100`,
`model_14900`, and `model_14999`. The refinement improved late Isaac Lab terrain
survival and velocity tracking, but it did not improve the complete MuJoCo gate set.

## Test scope

- 9 fixed commands: stand, forward/backward, lateral, pure yaw, and combined motion
- 10 s per command with 2 s warm-up
- Flat MuJoCo plus 20 mm, 35 mm, and 50 mm rough terrain
- 4 candidates: `model_12000`, `model_14100`, `model_14900`, `model_14999`

## Summary

Across 36 test cases, `model_12000` passed 16 complete absolute gates. The best
refined checkpoint, `model_14999`, passed 7. All candidates remained upright in
these tests, and `model_14999` had better raw velocity tracking (32/36 versus
27/36), but it was generally too high for the configured body-height target and
had worse pure-yaw integral bias. The refinement also does not add self-righting
after a fall.

Raw results are stored in the accompanying `flat.json`, `rough20.json`,
`rough35.json`, and `rough50.json` files.

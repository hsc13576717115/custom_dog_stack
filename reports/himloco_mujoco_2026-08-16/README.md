# HimLoco MuJoCo Baseline Evaluation

Evaluation date: 2026-08-16

This directory records the first reproducible MuJoCo comparison of HimLoco
checkpoints `model_11300`, `model_12000`, and `model_14999` on a supported
velocity grid and three generated rough terrains.

## Result

`model_12000` was selected as the overall baseline. It passed 14 of 31 commands
on the broad supported grid, compared with 9 for `model_11300` and 10 for
`model_14999`. None of the candidates passed the complete grid, so this is a
relative selection result rather than a deployment certification.

The three rough-terrain files use deterministic seeds and nominal height scales
of 20 mm, 35 mm, and 50 mm. CSV files contain flattened measurements; JSON files
preserve commands, individual gates, and summaries. The `terrains` directory
contains the exact MJCF inputs used by the evaluator.

Later terrain-refinement results are recorded in
`../himloco_terrain_refine_2026-08-17`.

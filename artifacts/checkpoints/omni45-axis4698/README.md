# omni45-axis4698 Resume Bundle

`model_4698.pt` is the training checkpoint used to export the migration
baseline in `deploy/releases/omni45-axis4698/`. The parameter snapshots are
kept with it so a new Ubuntu workstation can resume the same task. Validate a
resumed policy in MuJoCo before replacing the release ONNX.

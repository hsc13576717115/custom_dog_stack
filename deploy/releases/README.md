# Policy Releases

Do not copy raw training checkpoints into this directory. A reviewed release has:

```text
vX.Y.Z/
  policy.onnx
  deploy.yaml
  metadata.yaml
  sha256.txt
```

The ONNX file is tracked with Git LFS. `validated_for_hardware` remains false
until sim2sim and the staged hardware safety checks are complete.

Current migration baseline:

```text
omni45-axis4698/
```

It is a 45-observation/12-action policy reviewed in Python MuJoCo. It is not
validated on hardware and does not support reverse velocity.

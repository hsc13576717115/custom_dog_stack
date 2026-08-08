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

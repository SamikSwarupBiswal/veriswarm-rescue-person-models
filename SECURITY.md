# Security and artifact handling

- The repository contains no credentials, datasets, environments, model
  weights, ONNX files, TensorRT engines, camera configuration, or generated
  evidence in Git history.
- Checkpoints are distributed only as hash-bound GitHub Release assets.
- Setup scripts refuse to overwrite an existing checkpoint and verify SHA-256
  plus byte count before inference.
- Do not bind a demonstration server to a non-loopback interface without a
  separate threat review and authorization.
- Do not commit `.env` files, GitHub tokens, RunPod credentials, Jetson machine
  configuration, camera indices, or fleet network configuration.
- Report suspected credential or model-integrity issues privately to the
  repository owner.

This repository does not change Suyash's frozen rescue event schema,
five-camera evidence, security evidence, branch, or network configuration.

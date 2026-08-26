# Release assets

Tag: `candidate-2026-08-23`

Assets:

- `yolov8n-640-best.pt`
- `yolov8n-960-best.pt`
- `SHA256SUMS`
- `manifest.json`

The two `.pt` files are release candidates for integration testing. Publishing
them as release assets keeps binary weights outside Git history while allowing
hash-bound transfer.

Do not add training datasets, environments, raw results, UI evidence, ONNX
exports, TensorRT engines, secrets, or machine configuration to a release.

# Jetson integration and qualification boundary

The target Jetson is production hardware. A Windows or RTX demonstration does
not qualify edge deployment.

## 1. Transfer and verify

Download both `.pt` release assets without renaming their contents and verify:

```bash
sha256sum yolov8n-640-best.pt yolov8n-960-best.pt
```

Expected values are in `SHA256SUMS` and `models/manifest.json`. Stop immediately
if a hash or byte count differs.

## 2. Runtime setup

Use the Python, PyTorch, Torchvision, CUDA, cuDNN, TensorRT, JetPack, and L4T
versions approved for the actual target Jetson. Install NVIDIA's Jetson-specific
PyTorch build before installing this package. Do not replace it with a generic
x86 or PyPI CUDA wheel.

Record every installed runtime version in the generated benchmark evidence.

## 3. Static ONNX

Export batch 1 with a fixed shape:

- `yolov8n-640`: `1x3x640x640`
- `yolov8n-960`: `1x3x960x960`

Keep the `.pt`, ONNX, and export log outside Git. Recompute and record SHA-256
for the ONNX file.

## 4. TensorRT FP16 on the target

Build each engine on the actual target Jetson under its pinned runtime. Never
transfer an RTX-built TensorRT engine to the Jetson.

Example static-shape commands, with paths adjusted outside the checkout:

```bash
/usr/src/tensorrt/bin/trtexec \
  --onnx=/opt/veriswarm/models/yolov8n-640-best.onnx \
  --saveEngine=/opt/veriswarm/models/yolov8n-640-fp16.engine \
  --fp16 --shapes=images:1x3x640x640

/usr/src/tensorrt/bin/trtexec \
  --onnx=/opt/veriswarm/models/yolov8n-960-best.onnx \
  --saveEngine=/opt/veriswarm/models/yolov8n-960-fp16.engine \
  --fp16 --shapes=images:1x3x960x960
```

Record the exact executed engine hash, not only its filename.

## 5. Frozen acceptance boundary

For each candidate, the actual Jetson Orin Nano 8GB at 15 W must demonstrate:

- at least 15 continuous minutes and 4,500 completed frames after 50 warmup
  frames;
- throughput `>= 5.0 FPS`;
- TensorRT inference p95 `<= 180 ms`;
- capture-to-event p95 `<= 250 ms`;
- dropped or invalid frames `<= 1%`;
- zero inference/pipeline errors, OOMs, or restarts;
- at least 1 GiB available memory, no swap growth, and no thermal throttling;
- `.pt` versus FP16 loss no greater than 0.01 absolute for recall and mAP50;
- offline restart, camera-timeout, and corrupt-frame fail-closed tests.

Capture p50/p95/p99/max latency, temperatures, clocks, power, memory,
`tegrastats`, exact runtime versions, and model/ONNX/engine hashes. Keep all
benchmark evidence outside Git.

## Semantic requirement

Both candidates emit only `person_candidate`. They do not confirm survivors or
invent GPS/NED. Until a separate state classifier is trained and hash-bound,
every display box remains red and `UNVERIFIED`.

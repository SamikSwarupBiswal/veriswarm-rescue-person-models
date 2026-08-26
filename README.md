# VeriSwarm Rescue Person Models

Two hash-bound YOLOv8n rescue-person detector candidates prepared for local
testing and physical Jetson integration.

| Candidate | Input | Bytes | SHA-256 |
|---|---:|---:|---|
| `yolov8n-640` | `640x640`, batch 1 | 6,224,554 | `b794911b3a29f05e28c32717ce1e29074b5621e02267f6cef54ff82ede857b8e` |
| `yolov8n-960` | `960x960`, batch 1 | 6,263,146 | `41ff406e02b28d0e05acbf567bbcefbbc1809fea6de5ddd24b5170d2aa735f2b` |

The `.pt` files are GitHub Release assets, not Git objects. Datasets,
environments, weights, ONNX files, TensorRT engines, and generated evidence are
excluded from Git by design.

## Important semantic boundary

These checkpoints emit one RGB detector class only: `person_candidate`. Their
embedded Ultralytics class name is `item` because the completed training runs
used `single_cls=true`; the hash-bound compatibility mapping in this repository
maps class 0 to `person_candidate`.

The detectors do **not** confirm survivors and do not infer distress, safe
walking, hazards, thermal evidence, GPS, range, or NED position. No separate
person-state model is included. Every standalone box is therefore red and
marked `UNVERIFIED`.

Neither checkpoint is release-qualified. See [MODEL_CARD.md](MODEL_CARD.md) and
[docs/JETSON.md](docs/JETSON.md) before deployment work.

## Windows quick start

Requirements: Python 3.10-3.12, Git, and GitHub CLI for release-asset download.

```powershell
git clone https://github.com/SamikSwarupBiswal/veriswarm-rescue-person-models.git
cd veriswarm-rescue-person-models
.\scripts\setup-local.ps1
```

The setup script creates `.venv`, installs the pinned local dependencies,
downloads only missing release assets, and verifies both hashes and sizes.

To verify again:

```powershell
.\.venv\Scripts\python.exe -m veriswarm_rescue_models.cli verify
```

## Detect an image or video

```powershell
.\.venv\Scripts\python.exe -m veriswarm_rescue_models.cli detect `
  --model yolov8n-640 `
  --source C:\path\to\input.mp4 `
  --output outputs\annotated-640.mp4
```

Use `--model yolov8n-960` for the 960 candidate. Each run writes annotated
media and a neighboring `.jsonl` file containing the exact model-owned pixel
and normalized boxes. That standalone JSONL is candidate evidence; it is not a
replacement for Suyash's frozen `veriswarm.rescue.event.v1` envelope.

## Detect a webcam or Iriun camera

Use the Windows DirectShow/OpenCV index reported by your machine:

```powershell
.\.venv\Scripts\python.exe -m veriswarm_rescue_models.cli detect `
  --model yolov8n-640 --source 0 --show
```

Try `1`, `2`, and higher indices for another built-in, USB, DroidCam, OBS, or
Iriun feed. Press `q` to stop. Camera indices are machine-local and must not be
committed as fleet configuration.

## Static ONNX export

Export outside Git with the candidate's fixed input size and batch 1:

```powershell
.\.venv\Scripts\python.exe -m veriswarm_rescue_models.cli export `
  --model yolov8n-640 --output-dir outputs\onnx

.\.venv\Scripts\python.exe -m veriswarm_rescue_models.cli export `
  --model yolov8n-960 --output-dir outputs\onnx
```

Build TensorRT FP16 engines on the actual target Jetson, never on an RTX host.
Follow [docs/JETSON.md](docs/JETSON.md).

## Repository status

Repository source, tests, manifests, and documentation are Codex-prepared
automated candidate artifacts for Samik's review. They must not be reported as
Samik human-completed, accepted, or release-qualified work without his explicit
review and acceptance.

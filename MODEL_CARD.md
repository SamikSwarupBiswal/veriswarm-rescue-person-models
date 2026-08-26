# Model card: `sar-rgb-person-v1` candidates

## Intended use

RGB person-candidate detection for VeriSwarm rescue-perception integration,
local video/camera demonstrations, and physical Jetson performance testing.

## Candidate results

The values below are the final rows recorded in each 40-epoch training run's
`results.csv`. They are not Jetson benchmarks and do not establish release
qualification.

| Candidate | Precision | Recall | mAP50 | mAP50-95 |
|---|---:|---:|---:|---:|
| YOLOv8n@640 | 0.62855 | 0.44069 | 0.48026 | 0.18716 |
| YOLOv8n@960 | 0.70662 | 0.53769 | 0.60449 | 0.26561 |

The frozen detector release minima include recall `>= 0.75` and mAP50
`>= 0.70`; both candidates are below those gates.

## Integrity and provenance

- Both release assets are the `best.pt` checkpoints, not `last.pt`.
- The release manifest binds each filename, byte count, SHA-256, static input
  size, batch size, output label, and final-epoch metrics.
- Both checkpoints embed `{0: "item"}` because training used
  `single_cls=true`. This repository rejects any other embedded class map and
  maps only class 0 to `person_candidate`.
- No formal `veriswarm_training_run.json` completion record is packaged for
  either candidate. The 640 run was stopped after its 40 epochs and checkpoint
  publication but before that record was written. At the captured handoff, the
  960 post-training audit had not produced a verified record either.

## Limitations

- RGB-only detector; no thermal, pose, depth, geolocation, or state inference.
- Does not identify a person, confirm a survivor, or determine whether a person
  is safe, stressed, injured, or in a disaster state.
- False positives and false negatives are expected, especially for small,
  occluded, low-contrast, unusual-angle, and disaster-scene subjects.
- Thresholds used for a local demo are not automatically valid for flight or
  production.
- Physical Jetson behavior, TensorRT accuracy, latency, thermals, and memory are
  unqualified until the frozen acceptance procedure passes.

## Display policy

Without a separately trained and hash-bound person-state classifier, all
detections remain red as `DISASTER / UNVERIFIED`. Detector output alone must
never produce a green box.

## Evidence status

The repository packaging, verification scripts, and automated tests are
Codex-generated candidate evidence. They are not Samik human-completed review,
Suyash acceptance, or production authorization.

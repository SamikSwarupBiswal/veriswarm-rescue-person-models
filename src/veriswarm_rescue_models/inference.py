"""Standalone image, video, camera, and static ONNX candidate tooling."""

from __future__ import annotations

import json
import math
import os
import shutil
import time
from pathlib import Path
from typing import Any, Iterable

from .manifest import (
    MODEL_ID,
    OUTPUT_CLASS,
    ModelRepositoryError,
    ModelSpec,
    sha256_file,
    verify_model_file,
)


IMAGE_SUFFIXES = frozenset({".jpg", ".jpeg", ".png", ".bmp", ".webp"})
VIDEO_SUFFIXES = frozenset({".mp4", ".mov", ".avi", ".mkv", ".webm"})
RED = (0, 0, 255)


def normalize_source(value: str) -> int | Path:
    stripped = value.strip()
    if stripped.isdecimal():
        return int(stripped)
    path = Path(stripped).expanduser().resolve()
    if not path.is_file():
        raise ModelRepositoryError(f"input source does not exist: {path}")
    if path.suffix.lower() not in IMAGE_SUFFIXES | VIDEO_SUFFIXES:
        raise ModelRepositoryError(f"unsupported input suffix: {path.suffix}")
    return path


def clamp_box(
    values: Iterable[float], width: int, height: int
) -> tuple[float, float, float, float]:
    if width <= 0 or height <= 0:
        raise ModelRepositoryError("frame dimensions must be positive")
    points = tuple(float(value) for value in values)
    if len(points) != 4 or not all(math.isfinite(value) for value in points):
        raise ModelRepositoryError("detector box must contain four finite coordinates")
    x1, y1, x2, y2 = points
    result = (
        min(float(width), max(0.0, x1)),
        min(float(height), max(0.0, y1)),
        min(float(width), max(0.0, x2)),
        min(float(height), max(0.0, y2)),
    )
    if result[2] <= result[0] or result[3] <= result[1]:
        raise ModelRepositoryError("detector box is empty after frame clamping")
    return result


def normalized_box(
    box: tuple[float, float, float, float], width: int, height: int
) -> tuple[float, float, float, float]:
    return (box[0] / width, box[1] / height, box[2] / width, box[3] / height)


def _runtime_imports() -> tuple[Any, Any]:
    try:
        import cv2
        from ultralytics import YOLO
    except ImportError as error:
        raise ModelRepositoryError(
            "local inference dependencies are missing; install with pip install -e '.[local]'"
        ) from error
    return cv2, YOLO


def load_bound_model(spec: ModelSpec, checkpoint: Path) -> Any:
    verify_model_file(spec, checkpoint)
    _, yolo = _runtime_imports()
    model = yolo(str(checkpoint), task="detect")
    names = model.names
    normalized = (
        {int(key): str(value) for key, value in names.items()}
        if isinstance(names, dict)
        else {index: str(value) for index, value in enumerate(names)}
    )
    if normalized != {0: "item"}:
        raise ModelRepositoryError(
            f"embedded class map must be exactly {{0: 'item'}}, got {normalized!r}"
        )
    return model


def _annotate(
    frame: Any,
    model: Any,
    spec: ModelSpec,
    *,
    confidence: float,
    device: str | None,
    frame_index: int,
    source_timestamp_ms: int,
) -> tuple[Any, list[dict[str, Any]]]:
    cv2, _ = _runtime_imports()
    height, width = frame.shape[:2]
    kwargs: dict[str, Any] = {
        "source": frame,
        "imgsz": spec.imgsz,
        "conf": confidence,
        "verbose": False,
    }
    if device:
        kwargs["device"] = device
    result = model.predict(**kwargs)[0]
    annotated = frame.copy()
    records: list[dict[str, Any]] = []
    for detection_index, detection in enumerate(result.boxes):
        class_id = int(detection.cls.item())
        if class_id != 0:
            raise ModelRepositoryError(f"unexpected detector class id: {class_id}")
        try:
            box = clamp_box(detection.xyxy[0].tolist(), width, height)
        except ModelRepositoryError as error:
            if "empty after frame clamping" in str(error):
                continue
            raise
        score = float(detection.conf.item())
        bbox_norm = normalized_box(box, width, height)
        record = {
            "schema": "veriswarm.rescue.standalone_detection.v1",
            "status": "codex_generated_candidate",
            "frame_index": frame_index,
            "detection_index": detection_index,
            "source_timestamp_ms": source_timestamp_ms,
            "class_id": OUTPUT_CLASS,
            "embedded_class_id": class_id,
            "confidence": score,
            "bbox_px": list(box),
            "bbox_norm": list(bbox_norm),
            "model_id": MODEL_ID,
            "model_candidate": spec.name,
            "model_sha256": spec.sha256,
            "display": {"color": "red", "status": "UNVERIFIED"},
        }
        records.append(record)
        x1, y1, x2, y2 = box
        p1 = (round(x1), round(y1))
        p2 = (round(x2), round(y2))
        cv2.rectangle(annotated, p1, p2, RED, 2)
        cv2.putText(
            annotated,
            f"person_candidate {score:.2f} UNVERIFIED",
            (p1[0], max(16, p1[1] - 6)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.48,
            RED,
            1,
            cv2.LINE_AA,
        )
    return annotated, records


def _exclusive_output(path: Path) -> Path:
    resolved = path.expanduser().resolve()
    if resolved.exists():
        raise ModelRepositoryError(f"refusing to overwrite existing output: {resolved}")
    resolved.parent.mkdir(parents=True, exist_ok=True)
    return resolved


def run_detection(
    spec: ModelSpec,
    checkpoint: Path,
    source_value: str,
    output: Path,
    *,
    confidence: float = 0.25,
    device: str | None = None,
    show: bool = False,
    max_frames: int | None = None,
) -> tuple[Path, Path, int, int]:
    if not 0.0 <= confidence <= 1.0 or not math.isfinite(confidence):
        raise ModelRepositoryError("confidence must be finite and inside [0, 1]")
    if max_frames is not None and max_frames < 1:
        raise ModelRepositoryError("max_frames must be positive")
    source = normalize_source(source_value)
    output = _exclusive_output(output)
    jsonl = _exclusive_output(output.with_suffix(".jsonl"))
    model = load_bound_model(spec, checkpoint)
    cv2, _ = _runtime_imports()

    if isinstance(source, Path) and source.suffix.lower() in IMAGE_SUFFIXES:
        frame = cv2.imread(str(source))
        if frame is None:
            raise ModelRepositoryError(f"could not decode image: {source}")
        annotated, records = _annotate(
            frame,
            model,
            spec,
            confidence=confidence,
            device=device,
            frame_index=0,
            source_timestamp_ms=0,
        )
        if not cv2.imwrite(str(output), annotated):
            raise ModelRepositoryError(f"could not write annotated image: {output}")
        with jsonl.open("x", encoding="utf-8", newline="\n") as handle:
            for record in records:
                handle.write(json.dumps(record, sort_keys=True, allow_nan=False) + "\n")
        return output, jsonl, 1, len(records)

    backend = cv2.CAP_DSHOW if isinstance(source, int) and os.name == "nt" else cv2.CAP_ANY
    capture = cv2.VideoCapture(source if isinstance(source, int) else str(source), backend)
    if not capture.isOpened():
        capture.release()
        raise ModelRepositoryError(f"could not open video/camera source: {source}")
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = float(capture.get(cv2.CAP_PROP_FPS))
    if width <= 0 or height <= 0:
        capture.release()
        raise ModelRepositoryError("source reported invalid frame dimensions")
    if not math.isfinite(fps) or fps <= 0.0:
        fps = 30.0
    writer = cv2.VideoWriter(
        str(output), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height)
    )
    if not writer.isOpened():
        capture.release()
        writer.release()
        raise ModelRepositoryError(f"could not open annotated video output: {output}")

    frame_index = 0
    detections = 0
    started = time.monotonic()
    try:
        with jsonl.open("x", encoding="utf-8", newline="\n") as handle:
            while max_frames is None or frame_index < max_frames:
                ok, frame = capture.read()
                if not ok or frame is None:
                    break
                timestamp = (
                    round((time.monotonic() - started) * 1000)
                    if isinstance(source, int)
                    else round(capture.get(cv2.CAP_PROP_POS_MSEC))
                )
                annotated, records = _annotate(
                    frame,
                    model,
                    spec,
                    confidence=confidence,
                    device=device,
                    frame_index=frame_index,
                    source_timestamp_ms=max(0, timestamp),
                )
                writer.write(annotated)
                for record in records:
                    handle.write(json.dumps(record, sort_keys=True, allow_nan=False) + "\n")
                detections += len(records)
                frame_index += 1
                if show:
                    cv2.imshow("VeriSwarm rescue person candidate", annotated)
                    if cv2.waitKey(1) & 0xFF == ord("q"):
                        break
    finally:
        capture.release()
        writer.release()
        if show:
            cv2.destroyAllWindows()
    return output, jsonl, frame_index, detections


def export_static_onnx(
    spec: ModelSpec,
    checkpoint: Path,
    output_dir: Path,
    *,
    device: str | None = None,
) -> tuple[Path, str]:
    model = load_bound_model(spec, checkpoint)
    destination_dir = output_dir.expanduser().resolve()
    destination_dir.mkdir(parents=True, exist_ok=True)
    destination = destination_dir / f"{spec.name}-best.onnx"
    if destination.exists():
        raise ModelRepositoryError(f"refusing to overwrite existing ONNX: {destination}")
    adjacent = checkpoint.with_suffix(".onnx")
    if adjacent.exists():
        raise ModelRepositoryError(
            f"refusing export because Ultralytics output already exists: {adjacent}"
        )
    kwargs: dict[str, Any] = {
        "format": "onnx",
        "imgsz": spec.imgsz,
        "batch": 1,
        "dynamic": False,
        "simplify": False,
        "opset": 17,
        "half": False,
    }
    if device:
        kwargs["device"] = device
    generated = Path(str(model.export(**kwargs))).resolve()
    if not generated.is_file():
        raise ModelRepositoryError(f"Ultralytics did not produce the expected ONNX: {generated}")
    shutil.move(str(generated), str(destination))
    return destination, sha256_file(destination)

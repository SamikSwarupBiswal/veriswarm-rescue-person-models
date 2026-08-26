"""Strict manifest loading and checkpoint integrity verification."""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping


MANIFEST_SCHEMA = "veriswarm.rescue.model_release.v1"
MODEL_ID = "sar-rgb-person-v1"
OUTPUT_CLASS = "person_candidate"
EMBEDDED_CLASS = "item"
CLASS_BINDING = "ultralytics_single_cls_item_to_person_candidate"
EXPECTED_MODELS = {"yolov8n-640": 640, "yolov8n-960": 960}
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")


class ModelRepositoryError(ValueError):
    """A model manifest, checkpoint, or inference input violated the contract."""


@dataclass(frozen=True, slots=True)
class ModelSpec:
    name: str
    asset: str
    sha256: str
    bytes: int
    imgsz: int
    batch: int
    epochs_recorded: int
    final_epoch_metrics: Mapping[str, float]
    formal_completion_record_packaged: bool

    def __post_init__(self) -> None:
        expected_size = EXPECTED_MODELS.get(self.name)
        if expected_size is None:
            raise ModelRepositoryError(f"unsupported model candidate: {self.name!r}")
        if self.imgsz != expected_size:
            raise ModelRepositoryError(
                f"{self.name} requires imgsz={expected_size}, not {self.imgsz}"
            )
        if self.asset != f"{self.name}-best.pt":
            raise ModelRepositoryError(f"unexpected checkpoint asset: {self.asset!r}")
        if not _SHA256_RE.fullmatch(self.sha256):
            raise ModelRepositoryError("checkpoint SHA-256 must be lowercase hexadecimal")
        if self.bytes <= 0:
            raise ModelRepositoryError("checkpoint byte count must be positive")
        if self.batch != 1:
            raise ModelRepositoryError("only static batch 1 candidates are supported")
        if self.epochs_recorded != 40:
            raise ModelRepositoryError("candidate manifest must bind all 40 recorded epochs")
        if self.formal_completion_record_packaged:
            raise ModelRepositoryError(
                "these candidates do not include a formal completion record"
            )
        required_metrics = {"precision", "recall", "map50", "map50_95"}
        if set(self.final_epoch_metrics) != required_metrics:
            raise ModelRepositoryError("final epoch metrics are incomplete")
        for name, value in self.final_epoch_metrics.items():
            if not isinstance(value, (int, float)) or not 0.0 <= float(value) <= 1.0:
                raise ModelRepositoryError(f"invalid metric {name}: {value!r}")


@dataclass(frozen=True, slots=True)
class ReleaseManifest:
    path: Path
    release_tag: str
    models: Mapping[str, ModelSpec]

    def require(self, name: str) -> ModelSpec:
        try:
            return self.models[name]
        except KeyError as error:
            raise ModelRepositoryError(f"unknown model candidate: {name!r}") from error


def repository_root() -> Path:
    return Path(__file__).resolve().parents[2]


def default_manifest_path() -> Path:
    return repository_root() / "models" / "manifest.json"


def default_models_dir() -> Path:
    return repository_root() / "models"


def _read_json(path: Path) -> Mapping[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ModelRepositoryError(f"could not read manifest {path}: {error}") from error
    if not isinstance(value, Mapping):
        raise ModelRepositoryError("model manifest root must be an object")
    return value


def load_manifest(path: Path | None = None) -> ReleaseManifest:
    resolved = (path or default_manifest_path()).expanduser().resolve()
    raw = _read_json(resolved)
    expected_root = {
        "schema": MANIFEST_SCHEMA,
        "status": "codex_generated_candidate",
        "model_id": MODEL_ID,
        "output_class": OUTPUT_CLASS,
        "embedded_class": EMBEDDED_CLASS,
        "class_binding": CLASS_BINDING,
        "release_qualified": False,
    }
    for key, expected in expected_root.items():
        if raw.get(key) != expected:
            raise ModelRepositoryError(
                f"manifest {key!r} must be {expected!r}, got {raw.get(key)!r}"
            )
    release_tag = raw.get("release_tag")
    if not isinstance(release_tag, str) or not release_tag.strip():
        raise ModelRepositoryError("manifest release_tag must be a non-empty string")
    values = raw.get("models")
    if not isinstance(values, list):
        raise ModelRepositoryError("manifest models must be a list")
    models: dict[str, ModelSpec] = {}
    for value in values:
        if not isinstance(value, Mapping):
            raise ModelRepositoryError("each model entry must be an object")
        try:
            spec = ModelSpec(
                name=str(value["name"]),
                asset=str(value["asset"]),
                sha256=str(value["sha256"]),
                bytes=int(value["bytes"]),
                imgsz=int(value["imgsz"]),
                batch=int(value["batch"]),
                epochs_recorded=int(value["epochs_recorded"]),
                final_epoch_metrics=dict(value["final_epoch_metrics"]),
                formal_completion_record_packaged=bool(
                    value["formal_completion_record_packaged"]
                ),
            )
        except (KeyError, TypeError, ValueError) as error:
            raise ModelRepositoryError(f"invalid model entry: {error}") from error
        if spec.name in models:
            raise ModelRepositoryError(f"duplicate model candidate: {spec.name}")
        models[spec.name] = spec
    if set(models) != set(EXPECTED_MODELS):
        raise ModelRepositoryError(
            f"manifest model set must be {sorted(EXPECTED_MODELS)}, got {sorted(models)}"
        )
    return ReleaseManifest(path=resolved, release_tag=release_tag, models=models)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify_model_file(spec: ModelSpec, path: Path) -> str:
    resolved = path.expanduser().resolve()
    if not resolved.is_file():
        raise ModelRepositoryError(f"checkpoint is missing: {resolved}")
    actual_bytes = resolved.stat().st_size
    if actual_bytes != spec.bytes:
        raise ModelRepositoryError(
            f"checkpoint size mismatch for {spec.name}: "
            f"expected {spec.bytes}, got {actual_bytes}"
        )
    actual_hash = sha256_file(resolved)
    if actual_hash != spec.sha256:
        raise ModelRepositoryError(
            f"checkpoint hash mismatch for {spec.name}: "
            f"expected {spec.sha256}, got {actual_hash}"
        )
    return actual_hash

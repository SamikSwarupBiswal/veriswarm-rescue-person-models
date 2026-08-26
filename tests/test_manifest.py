from __future__ import annotations

import hashlib
from pathlib import Path

import pytest

from veriswarm_rescue_models.manifest import (
    CLASS_BINDING,
    EMBEDDED_CLASS,
    EXPECTED_MODELS,
    MODEL_ID,
    OUTPUT_CLASS,
    ModelRepositoryError,
    ModelSpec,
    load_manifest,
    repository_root,
    verify_model_file,
)


def test_repository_manifest_binds_exact_candidate_set() -> None:
    manifest = load_manifest()
    assert manifest.release_tag == "candidate-2026-08-23"
    assert set(manifest.models) == set(EXPECTED_MODELS)
    assert manifest.models["yolov8n-640"].sha256 == (
        "b794911b3a29f05e28c32717ce1e29074b5621e02267f6cef54ff82ede857b8e"
    )
    assert manifest.models["yolov8n-960"].sha256 == (
        "41ff406e02b28d0e05acbf567bbcefbbc1809fea6de5ddd24b5170d2aa735f2b"
    )


def test_frozen_semantic_binding_is_explicit() -> None:
    assert MODEL_ID == "sar-rgb-person-v1"
    assert EMBEDDED_CLASS == "item"
    assert OUTPUT_CLASS == "person_candidate"
    assert CLASS_BINDING == "ultralytics_single_cls_item_to_person_candidate"
    assert all(
        not spec.formal_completion_record_packaged
        for spec in load_manifest().models.values()
    )


def test_verify_model_file_checks_size_then_hash(tmp_path: Path) -> None:
    content = b"candidate-checkpoint"
    checkpoint = tmp_path / "yolov8n-640-best.pt"
    checkpoint.write_bytes(content)
    spec = ModelSpec(
        name="yolov8n-640",
        asset=checkpoint.name,
        sha256=hashlib.sha256(content).hexdigest(),
        bytes=len(content),
        imgsz=640,
        batch=1,
        epochs_recorded=40,
        final_epoch_metrics={
            "precision": 0.1,
            "recall": 0.2,
            "map50": 0.3,
            "map50_95": 0.4,
        },
        formal_completion_record_packaged=False,
    )
    assert verify_model_file(spec, checkpoint) == spec.sha256
    checkpoint.write_bytes(content + b"changed")
    with pytest.raises(ModelRepositoryError, match="size mismatch"):
        verify_model_file(spec, checkpoint)


def test_git_checkout_does_not_contain_binary_models() -> None:
    root = repository_root()
    forbidden = {".pt", ".pth", ".onnx", ".engine", ".plan"}
    tracked_source_candidates = [
        path
        for path in root.rglob("*")
        if path.is_file()
        and ".git" not in path.parts
        and ".venv" not in path.parts
        and path.suffix.lower() in forbidden
    ]
    assert tracked_source_candidates == []

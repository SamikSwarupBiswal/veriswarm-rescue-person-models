from __future__ import annotations

from pathlib import Path

import pytest

from veriswarm_rescue_models.inference import clamp_box, normalize_source, normalized_box
from veriswarm_rescue_models.manifest import ModelRepositoryError


def test_camera_source_uses_integer_index() -> None:
    assert normalize_source("0") == 0
    assert normalize_source(" 5 ") == 5


def test_media_source_requires_supported_existing_file(tmp_path: Path) -> None:
    video = tmp_path / "input.mp4"
    video.write_bytes(b"placeholder")
    assert normalize_source(str(video)) == video.resolve()
    unsupported = tmp_path / "input.txt"
    unsupported.write_text("not media", encoding="utf-8")
    with pytest.raises(ModelRepositoryError, match="unsupported input suffix"):
        normalize_source(str(unsupported))


def test_model_box_is_clamped_once_then_normalized() -> None:
    box = clamp_box((-5.0, 10.0, 220.0, 130.0), width=200, height=100)
    assert box == (0.0, 10.0, 200.0, 100.0)
    assert normalized_box(box, 200, 100) == (0.0, 0.1, 1.0, 1.0)


def test_empty_box_is_rejected() -> None:
    with pytest.raises(ModelRepositoryError, match="empty"):
        clamp_box((10, 10, 10, 20), width=100, height=100)

"""Command-line interface for the two hash-bound model candidates."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

from .inference import IMAGE_SUFFIXES, export_static_onnx, run_detection
from .manifest import (
    ModelRepositoryError,
    default_manifest_path,
    default_models_dir,
    load_manifest,
    verify_model_file,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Verify, run, or export VeriSwarm rescue-person candidates"
    )
    parser.add_argument("--manifest", type=Path, default=default_manifest_path())
    subparsers = parser.add_subparsers(dest="command", required=True)

    verify = subparsers.add_parser("verify", help="verify both release checkpoints")
    verify.add_argument("--models-dir", type=Path, default=default_models_dir())

    detect = subparsers.add_parser("detect", help="annotate an image, video, or camera")
    detect.add_argument("--model", required=True)
    detect.add_argument("--models-dir", type=Path, default=default_models_dir())
    detect.add_argument("--source", required=True, help="media path or OpenCV camera index")
    detect.add_argument("--output", type=Path)
    detect.add_argument("--confidence", type=float, default=0.25)
    detect.add_argument("--device")
    detect.add_argument("--show", action="store_true")
    detect.add_argument("--max-frames", type=int)

    export = subparsers.add_parser("export", help="export static batch-1 ONNX")
    export.add_argument("--model", required=True)
    export.add_argument("--models-dir", type=Path, default=default_models_dir())
    export.add_argument("--output-dir", type=Path, default=Path("outputs/onnx"))
    export.add_argument("--device")
    return parser


def _default_output(model: str, source: str) -> Path:
    source_path = Path(source)
    stamp = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
    if source_path.suffix.lower() in IMAGE_SUFFIXES:
        return Path("outputs") / f"{model}-{source_path.stem}-{stamp}.jpg"
    return Path("outputs") / f"{model}-{stamp}.mp4"


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        manifest = load_manifest(args.manifest)
        if args.command == "verify":
            results = []
            for name in sorted(manifest.models):
                spec = manifest.models[name]
                path = args.models_dir / spec.asset
                digest = verify_model_file(spec, path)
                result = {
                    "candidate": spec.name,
                    "asset": str(path.resolve()),
                    "bytes": spec.bytes,
                    "imgsz": spec.imgsz,
                    "sha256": digest,
                    "status": "PASS",
                }
                results.append(result)
                print(json.dumps(result, sort_keys=True))
            print(f"PASS verified_models={len(results)} release={manifest.release_tag}")
            return 0
        spec = manifest.require(args.model)
        checkpoint = args.models_dir / spec.asset
        if args.command == "detect":
            output = args.output or _default_output(spec.name, args.source)
            media, jsonl, frames, detections = run_detection(
                spec,
                checkpoint,
                args.source,
                output,
                confidence=args.confidence,
                device=args.device,
                show=args.show,
                max_frames=args.max_frames,
            )
            print(
                f"PASS candidate={spec.name} frames={frames} detections={detections} "
                f"media={media} jsonl={jsonl}"
            )
            return 0
        if args.command == "export":
            onnx, digest = export_static_onnx(
                spec, checkpoint, args.output_dir, device=args.device
            )
            print(
                f"PASS candidate={spec.name} batch=1 imgsz={spec.imgsz} "
                f"onnx={onnx} sha256={digest}"
            )
            return 0
        raise ModelRepositoryError(f"unsupported command: {args.command}")
    except ModelRepositoryError as error:
        print(f"ERROR: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())

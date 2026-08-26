"""Hash-bound VeriSwarm rescue-person candidate tooling."""

from .manifest import MANIFEST_SCHEMA, ModelSpec, load_manifest, verify_model_file

__all__ = ["MANIFEST_SCHEMA", "ModelSpec", "load_manifest", "verify_model_file"]
__version__ = "0.1.0"

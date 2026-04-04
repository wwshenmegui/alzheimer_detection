from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .data_quality import compute_sha256


DEFAULT_MODEL_VERSION = "v1"
DEFAULT_VERSIONED_MODEL_FILENAME = "model.pkl"
DEFAULT_MODEL_METADATA_FILENAME = "metadata.json"
DEFAULT_CURRENT_POINTER_FILENAME = "current.json"


def utc_timestamp() -> str:
    return datetime.now(timezone.utc).isoformat()


def derive_model_name(output_model: Path, model_name: str | None = None) -> str:
    return model_name or output_model.stem


def resolve_versioned_model_paths(
    output_model: Path,
    *,
    model_version: str = DEFAULT_MODEL_VERSION,
    model_name: str | None = None,
) -> dict[str, Path]:
    resolved_model_name = derive_model_name(output_model, model_name)
    model_root = output_model.parent / resolved_model_name
    version_dir = model_root / model_version
    return {
        "model_root": model_root,
        "version_dir": version_dir,
        "model_path": version_dir / DEFAULT_VERSIONED_MODEL_FILENAME,
        "metadata_path": version_dir / DEFAULT_MODEL_METADATA_FILENAME,
        "current_path": model_root / DEFAULT_CURRENT_POINTER_FILENAME,
    }


def write_json_file(payload: dict[str, Any], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)


def read_json_file(file_path: Path) -> dict[str, Any]:
    with file_path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def write_current_version_pointer(
    current_path: Path,
    *,
    model_name: str,
    model_version: str,
    model_path: Path,
    metadata_path: Path,
) -> None:
    write_json_file(
        {
            "model_name": model_name,
            "model_version": model_version,
            "model_path": str(model_path),
            "metadata_path": str(metadata_path),
            "updated_at": utc_timestamp(),
        },
        current_path,
    )


def resolve_model_artifacts(
    configured_model_path: Path,
    *,
    model_version: str | None = None,
    model_name: str | None = None,
) -> tuple[Path, Path | None]:
    paths = resolve_versioned_model_paths(
        configured_model_path,
        model_version=model_version or DEFAULT_MODEL_VERSION,
        model_name=model_name,
    )
    if model_version and paths["model_path"].exists():
        metadata_path = paths["metadata_path"] if paths["metadata_path"].exists() else None
        return paths["model_path"], metadata_path

    current_path = paths["current_path"]
    if current_path.exists():
        current_payload = read_json_file(current_path)
        model_path = Path(str(current_payload["model_path"]))
        metadata_path_value = current_payload.get("metadata_path")
        metadata_path = Path(str(metadata_path_value)) if metadata_path_value else None
        return model_path, metadata_path if metadata_path and metadata_path.exists() else None

    if configured_model_path.exists():
        legacy_metadata_path = configured_model_path.with_suffix(".metadata.json")
        return configured_model_path, legacy_metadata_path if legacy_metadata_path.exists() else None

    return paths["model_path"], paths["metadata_path"]


def load_model_metadata(metadata_path: Path | None) -> dict[str, Any] | None:
    if metadata_path is None or not metadata_path.exists():
        return None
    return read_json_file(metadata_path)


def update_model_metadata(metadata_path: Path, updates: dict[str, Any]) -> dict[str, Any]:
    existing = load_model_metadata(metadata_path) or {}
    existing.update(updates)
    write_json_file(existing, metadata_path)
    return existing


def build_artifact_lineage(*, artifact_path: Path, artifact_key: str) -> dict[str, str]:
    return {
        f"{artifact_key}_path": str(artifact_path),
        f"{artifact_key}_sha256": compute_sha256(artifact_path),
    }


def activate_model_version(
    configured_model_path: Path,
    *,
    model_version: str,
    model_name: str | None = None,
) -> dict[str, Any]:
    paths = resolve_versioned_model_paths(
        configured_model_path,
        model_version=model_version,
        model_name=model_name,
    )
    if not paths["model_path"].exists():
        raise FileNotFoundError(f"Model version does not exist: {paths['model_path']}")
    if not paths["metadata_path"].exists():
        raise FileNotFoundError(f"Metadata for model version does not exist: {paths['metadata_path']}")

    metadata = read_json_file(paths["metadata_path"])
    resolved_model_name = str(metadata.get("model_name", derive_model_name(configured_model_path, model_name)))
    resolved_version = str(metadata.get("model_version", model_version))
    write_current_version_pointer(
        paths["current_path"],
        model_name=resolved_model_name,
        model_version=resolved_version,
        model_path=paths["model_path"],
        metadata_path=paths["metadata_path"],
    )
    return metadata


def list_registered_models(
    configured_model_path: Path,
    *,
    model_name: str | None = None,
) -> list[dict[str, Any]]:
    paths = resolve_versioned_model_paths(configured_model_path, model_name=model_name)
    model_root = paths["model_root"]
    if not model_root.exists():
        return []

    discovered: list[dict[str, Any]] = []
    for version_dir in sorted(path for path in model_root.iterdir() if path.is_dir()):
        metadata_path = version_dir / DEFAULT_MODEL_METADATA_FILENAME
        if metadata_path.exists():
            discovered.append(read_json_file(metadata_path))
    return discovered
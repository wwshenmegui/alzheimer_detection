from __future__ import annotations

import argparse
import csv
import importlib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable


DEFAULT_LABEL_TO_ID = {
    "NonDemented": 0,
    "VeryMildDemented": 1,
    "MildDemented": 2,
    "ModerateDemented": 3,
}

DEFAULT_DATASET_HANDLE = "aryansinghal10/alzheimers-multiclass-dataset-equal-and-augmented"
DEFAULT_ALLOWED_EXTENSIONS = (".jpg", ".jpeg", ".png")
DEFAULT_CONFIG_PATH = Path("configs/training.yaml")
MANIFEST_COLUMNS = ("sample_id", "image_path", "label_name", "label_id")


@dataclass
class IngestionConfig:
    dataset_root: Path
    output_manifest: Path
    label_to_id: dict[str, int] = field(default_factory=lambda: dict(DEFAULT_LABEL_TO_ID))
    allowed_extensions: tuple[str, ...] = DEFAULT_ALLOWED_EXTENSIONS


def load_ingestion_settings(config_path: Path) -> dict[str, Any]:
    if not config_path.exists():
        return {}

    try:
        yaml = importlib.import_module("yaml")
    except ImportError as exc:
        raise RuntimeError(
            "PyYAML is required for config loading. Install it with 'pip install pyyaml'."
        ) from exc

    with config_path.open("r", encoding="utf-8") as handle:
        config_data = yaml.safe_load(handle) or {}

    ingestion_settings = config_data.get("ingestion", {})
    if not isinstance(ingestion_settings, dict):
        raise ValueError("The 'ingestion' section in the config file must be a mapping.")

    return ingestion_settings


def download_dataset(dataset_handle: str = DEFAULT_DATASET_HANDLE) -> Path:
    try:
        kagglehub = importlib.import_module("kagglehub")
    except ImportError as exc:
        raise RuntimeError(
            "kagglehub is required for dataset download. Install it with 'pip install kagglehub'."
        ) from exc

    return Path(kagglehub.dataset_download(dataset_handle))


def resolve_dataset_root(dataset_root: Path, label_to_id: dict[str, int]) -> Path:
    expected_labels = set(label_to_id)
    direct_subdirectories = {path.name for path in dataset_root.iterdir() if path.is_dir()}
    if expected_labels.issubset(direct_subdirectories):
        return dataset_root

    for child in sorted(path for path in dataset_root.iterdir() if path.is_dir()):
        child_subdirectories = {path.name for path in child.iterdir() if path.is_dir()}
        if expected_labels.issubset(child_subdirectories):
            return child

    return dataset_root


def discover_image_files(dataset_root: Path, allowed_extensions: tuple[str, ...]) -> Iterable[Path]:
    normalized_extensions = {suffix.lower() for suffix in allowed_extensions}
    for file_path in sorted(dataset_root.rglob("*")):
        if file_path.is_file() and file_path.suffix.lower() in normalized_extensions:
            yield file_path


def is_readable_image(file_path: Path) -> bool:
    try:
        with file_path.open("rb") as handle:
            header = handle.read(16)
            if header.startswith(b"\x89PNG\r\n\x1a\n"):
                return True
            if header[:3] == b"\xff\xd8\xff":
                return True
    except OSError:
        return False
    return False


def build_manifest(config: IngestionConfig) -> list[dict[str, str | int]]:
    manifest: list[dict[str, str | int]] = []
    sample_number = 1
    dataset_root = resolve_dataset_root(config.dataset_root, config.label_to_id)

    for image_path in discover_image_files(dataset_root, config.allowed_extensions):
        label_name = image_path.parent.name
        if label_name not in config.label_to_id:
            continue
        if not is_readable_image(image_path):
            continue

        manifest.append(
            {
                "sample_id": f"sample_{sample_number:05d}",
                "image_path": str(image_path.resolve()),
                "label_name": label_name,
                "label_id": config.label_to_id[label_name],
            }
        )
        sample_number += 1

    return manifest


def save_manifest(rows: list[dict[str, str | int]], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=MANIFEST_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)


def build_ingestion_config(
    *,
    dataset_root: Path | None,
    output_manifest: Path | None,
    label_to_id: dict[str, int] | None = None,
    allowed_extensions: tuple[str, ...] | None = None,
) -> IngestionConfig:
    if dataset_root is None:
        raise ValueError("A dataset root is required to build ingestion config.")
    if output_manifest is None:
        raise ValueError("An output manifest path is required to build ingestion config.")

    return IngestionConfig(
        dataset_root=dataset_root,
        output_manifest=output_manifest,
        label_to_id=label_to_id or dict(DEFAULT_LABEL_TO_ID),
        allowed_extensions=allowed_extensions or DEFAULT_ALLOWED_EXTENSIONS,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a simple image manifest for Alzheimer dataset ingestion.")
    parser.add_argument(
        "--config",
        default=str(DEFAULT_CONFIG_PATH),
        help="Path to the YAML config file.",
    )
    parser.add_argument("--dataset-root", help="Path to the raw dataset root.")
    parser.add_argument("--output-manifest", help="Path to the output CSV manifest.")
    parser.add_argument(
        "--download",
        action="store_true",
        help="Download the dataset using KaggleHub before building the manifest.",
    )
    parser.add_argument(
        "--dataset-handle",
        default=DEFAULT_DATASET_HANDLE,
        help="KaggleHub dataset handle to download when --download is provided.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config_settings = load_ingestion_settings(Path(args.config))

    download = args.download or bool(config_settings.get("download", False))
    dataset_handle = args.dataset_handle or config_settings.get("dataset_handle", DEFAULT_DATASET_HANDLE)
    dataset_root_value = args.dataset_root or config_settings.get("dataset_root")
    output_manifest_value = args.output_manifest or config_settings.get("output_manifest")
    allowed_extensions_value = tuple(config_settings.get("allowed_extensions", DEFAULT_ALLOWED_EXTENSIONS))
    label_to_id_value = config_settings.get("label_to_id", dict(DEFAULT_LABEL_TO_ID))

    dataset_root = Path(dataset_root_value) if dataset_root_value else None
    if download:
        dataset_root = download_dataset(str(dataset_handle))
    if dataset_root is None:
        raise ValueError("Provide --dataset-root or use --download.")
    if output_manifest_value is None:
        raise ValueError("Provide --output-manifest or configure it in the YAML file.")

    config = build_ingestion_config(
        dataset_root=dataset_root,
        output_manifest=Path(output_manifest_value),
        label_to_id={str(key): int(value) for key, value in label_to_id_value.items()},
        allowed_extensions=tuple(str(item) for item in allowed_extensions_value),
    )
    manifest = build_manifest(config)
    save_manifest(manifest, config.output_manifest)


if __name__ == "__main__":
    main()

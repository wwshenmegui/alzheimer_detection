from __future__ import annotations

import argparse
import csv
import importlib
import json
import logging
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable

from shared.data_quality import (
    DEFAULT_ASPECT_RATIO_RANGE,
    DEFAULT_DUPLICATE_HASH_DISTANCE,
    DEFAULT_MIN_CENTER_BORDER_DIFF,
    DEFAULT_MIN_IMAGE_SIZE,
    DEFAULT_MIN_STDDEV,
    assign_duplicate_groups,
    extract_patient_id,
    inspect_image_path,
    summarize_duplicates,
    validate_mri_image_metadata,
)


DEFAULT_LABEL_TO_ID = {
    "NonDemented": 0,
    "VeryMildDemented": 1,
    "MildDemented": 2,
    "ModerateDemented": 3,
}

DEFAULT_DATASET_HANDLE = "aryansinghal10/alzheimers-multiclass-dataset-equal-and-augmented"
DEFAULT_ALLOWED_EXTENSIONS = (".jpg", ".jpeg", ".png")
DEFAULT_CONFIG_PATH = Path("configs/training.yaml")
DEFAULT_INGESTION_REPORT_PATH = Path("data/reports/ingestion_summary.json")
DEFAULT_DUPLICATE_REPORT_PATH = Path("data/reports/duplicate_report.csv")
MANIFEST_COLUMNS = (
    "sample_id",
    "image_path",
    "label_name",
    "label_id",
    "patient_id",
    "group_id",
    "duplicate_group_id",
    "width",
    "height",
    "mode",
    "image_format",
    "file_size_bytes",
    "mean_intensity",
    "std_intensity",
    "center_mean_intensity",
    "border_mean_intensity",
    "sha256",
    "average_hash",
    "mri_is_valid",
    "mri_error_code",
    "mri_message",
)


LOGGER = logging.getLogger(__name__)


@dataclass
class IngestionConfig:
    dataset_root: Path
    output_manifest: Path
    label_to_id: dict[str, int] = field(default_factory=lambda: dict(DEFAULT_LABEL_TO_ID))
    allowed_extensions: tuple[str, ...] = DEFAULT_ALLOWED_EXTENSIONS
    output_report: Path = DEFAULT_INGESTION_REPORT_PATH
    duplicate_report: Path = DEFAULT_DUPLICATE_REPORT_PATH
    patient_id_regex: str | None = None
    min_image_size: tuple[int, int] = DEFAULT_MIN_IMAGE_SIZE
    aspect_ratio_range: tuple[float, float] = DEFAULT_ASPECT_RATIO_RANGE
    min_stddev: float = DEFAULT_MIN_STDDEV
    min_center_border_diff: float = DEFAULT_MIN_CENTER_BORDER_DIFF
    duplicate_hash_distance: int = DEFAULT_DUPLICATE_HASH_DISTANCE


def configure_logging(log_level: str) -> None:
    logging.basicConfig(
        level=getattr(logging, log_level.upper(), logging.INFO),
        format="%(asctime)s | %(levelname)s | %(message)s",
    )


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
    LOGGER.info("Downloading dataset from KaggleHub: %s", dataset_handle)
    try:
        kagglehub = importlib.import_module("kagglehub")
    except ImportError as exc:
        raise RuntimeError(
            "kagglehub is required for dataset download. Install it with 'pip install kagglehub'."
        ) from exc

    downloaded_path = Path(kagglehub.dataset_download(dataset_handle))
    LOGGER.info("Downloaded dataset to %s", downloaded_path)
    return downloaded_path


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


def build_manifest(config: IngestionConfig) -> list[dict[str, Any]]:
    manifest, _ = build_manifest_with_report(config)
    return manifest


def build_manifest_with_report(config: IngestionConfig) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    LOGGER.info("Starting ingestion from dataset root %s", config.dataset_root)
    manifest: list[dict[str, Any]] = []
    sample_number = 1
    dataset_root = resolve_dataset_root(config.dataset_root, config.label_to_id)
    discovered_files = 0
    skipped_unknown_label = 0
    unreadable_files = 0

    LOGGER.info("Resolved dataset root to %s", dataset_root)

    for image_path in discover_image_files(dataset_root, config.allowed_extensions):
        discovered_files += 1
        label_name = image_path.parent.name
        if label_name not in config.label_to_id:
            skipped_unknown_label += 1
            continue
        if not is_readable_image(image_path):
            unreadable_files += 1
            continue

        metadata = inspect_image_path(image_path)
        mri_feedback = validate_mri_image_metadata(
            metadata,
            min_image_size=config.min_image_size,
            aspect_ratio_range=config.aspect_ratio_range,
            min_stddev=config.min_stddev,
            min_center_border_diff=config.min_center_border_diff,
        )
        patient_id = extract_patient_id(image_path, config.patient_id_regex)

        manifest.append(
            {
                "sample_id": f"sample_{sample_number:05d}",
                "image_path": str(image_path.resolve()),
                "label_name": label_name,
                "label_id": config.label_to_id[label_name],
                "patient_id": patient_id or "",
                "group_id": "",
                "duplicate_group_id": "",
                "width": metadata["width"],
                "height": metadata["height"],
                "mode": metadata["mode"],
                "image_format": metadata["image_format"],
                "file_size_bytes": metadata["file_size_bytes"],
                "mean_intensity": metadata["mean_intensity"],
                "std_intensity": metadata["std_intensity"],
                "center_mean_intensity": metadata["center_mean_intensity"],
                "border_mean_intensity": metadata["border_mean_intensity"],
                "sha256": metadata["sha256"],
                "average_hash": metadata["average_hash"],
                "mri_is_valid": str(mri_feedback.passed),
                "mri_error_code": mri_feedback.error_code or "",
                "mri_message": mri_feedback.message or "",
            }
        )
        sample_number += 1

    LOGGER.info(
        "Scanned %s candidate files, kept %s rows, skipped %s unknown-label files, skipped %s unreadable files",
        discovered_files,
        len(manifest),
        skipped_unknown_label,
        unreadable_files,
    )

    manifest = assign_duplicate_groups(manifest, max_hash_distance=config.duplicate_hash_distance)
    mri_valid_count = sum(str(row.get("mri_is_valid", "")).lower() == "true" for row in manifest)
    report = {
        "passed": True,
        "dataset_root": str(dataset_root),
        "discovered_files": discovered_files,
        "ingested_rows": len(manifest),
        "unknown_label_files": skipped_unknown_label,
        "unreadable_files": unreadable_files,
        "class_distribution": dict(Counter(str(row["label_name"]) for row in manifest)),
        "mri_valid_rows": mri_valid_count,
        "mri_invalid_rows": len(manifest) - mri_valid_count,
        "duplicate_summary": summarize_duplicates(manifest),
        "patient_id_rows": sum(1 for row in manifest if row.get("patient_id")),
    }
    LOGGER.info(
        "Ingestion summary: %s valid MRI rows, %s invalid MRI rows, %s duplicate rows",
        report["mri_valid_rows"],
        report["mri_invalid_rows"],
        report["duplicate_summary"]["duplicate_rows"],
    )
    return manifest, report


def save_manifest(rows: list[dict[str, str | int]], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=MANIFEST_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)
    LOGGER.info("Saved manifest with %s rows to %s", len(rows), output_path)


def save_ingestion_report(report: dict[str, Any], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2, sort_keys=True)
    LOGGER.info("Saved ingestion report to %s", output_path)


def save_duplicate_report(rows: list[dict[str, Any]], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    duplicate_rows = [row for row in rows if row.get("duplicate_group_id")]
    fieldnames = (
        "sample_id",
        "image_path",
        "label_name",
        "patient_id",
        "duplicate_group_id",
        "group_id",
        "sha256",
        "average_hash",
    )
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(
            {
                fieldname: row.get(fieldname, "")
                for fieldname in fieldnames
            }
            for row in duplicate_rows
        )
    LOGGER.info("Saved duplicate report with %s rows to %s", len(duplicate_rows), output_path)


def build_ingestion_config(
    *,
    dataset_root: Path | None,
    output_manifest: Path | None,
    label_to_id: dict[str, int] | None = None,
    allowed_extensions: tuple[str, ...] | None = None,
    output_report: Path | None = DEFAULT_INGESTION_REPORT_PATH,
    duplicate_report: Path | None = DEFAULT_DUPLICATE_REPORT_PATH,
    patient_id_regex: str | None = None,
    min_image_size: tuple[int, int] = DEFAULT_MIN_IMAGE_SIZE,
    aspect_ratio_range: tuple[float, float] = DEFAULT_ASPECT_RATIO_RANGE,
    min_stddev: float = DEFAULT_MIN_STDDEV,
    min_center_border_diff: float = DEFAULT_MIN_CENTER_BORDER_DIFF,
    duplicate_hash_distance: int = DEFAULT_DUPLICATE_HASH_DISTANCE,
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
        output_report=output_report or DEFAULT_INGESTION_REPORT_PATH,
        duplicate_report=duplicate_report or DEFAULT_DUPLICATE_REPORT_PATH,
        patient_id_regex=patient_id_regex,
        min_image_size=min_image_size,
        aspect_ratio_range=aspect_ratio_range,
        min_stddev=min_stddev,
        min_center_border_diff=min_center_border_diff,
        duplicate_hash_distance=duplicate_hash_distance,
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
    parser.add_argument("--output-report", help="Path to the output JSON ingestion summary report.")
    parser.add_argument("--duplicate-report", help="Path to the output CSV duplicate report.")
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
    parser.add_argument("--log-level", default="INFO", help="Logging level, for example DEBUG, INFO, WARNING.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    configure_logging(args.log_level)
    config_settings = load_ingestion_settings(Path(args.config))

    download = args.download or bool(config_settings.get("download", False))
    dataset_handle = args.dataset_handle or config_settings.get("dataset_handle", DEFAULT_DATASET_HANDLE)
    dataset_root_value = args.dataset_root or config_settings.get("dataset_root")
    output_manifest_value = args.output_manifest or config_settings.get("output_manifest")
    output_report_value = args.output_report or config_settings.get("output_report") or str(DEFAULT_INGESTION_REPORT_PATH)
    duplicate_report_value = args.duplicate_report or config_settings.get("duplicate_report") or str(DEFAULT_DUPLICATE_REPORT_PATH)
    allowed_extensions_value = tuple(config_settings.get("allowed_extensions", DEFAULT_ALLOWED_EXTENSIONS))
    label_to_id_value = config_settings.get("label_to_id", dict(DEFAULT_LABEL_TO_ID))
    patient_id_regex_value = config_settings.get("patient_id_regex")
    min_image_size_value = tuple(config_settings.get("min_image_size", list(DEFAULT_MIN_IMAGE_SIZE)))
    aspect_ratio_range_value = tuple(config_settings.get("aspect_ratio_range", list(DEFAULT_ASPECT_RATIO_RANGE)))
    min_stddev_value = float(config_settings.get("min_stddev", DEFAULT_MIN_STDDEV))
    min_center_border_diff_value = float(config_settings.get("min_center_border_diff", DEFAULT_MIN_CENTER_BORDER_DIFF))
    duplicate_hash_distance_value = int(config_settings.get("duplicate_hash_distance", DEFAULT_DUPLICATE_HASH_DISTANCE))

    dataset_root = Path(dataset_root_value) if dataset_root_value else None
    if download:
        dataset_root = download_dataset(str(dataset_handle))
    if dataset_root is None:
        LOGGER.error("No dataset root provided and download disabled")
        raise ValueError("Provide --dataset-root or use --download.")
    if output_manifest_value is None:
        LOGGER.error("No output manifest path provided")
        raise ValueError("Provide --output-manifest or configure it in the YAML file.")

    config = build_ingestion_config(
        dataset_root=dataset_root,
        output_manifest=Path(output_manifest_value),
        label_to_id={str(key): int(value) for key, value in label_to_id_value.items()},
        allowed_extensions=tuple(str(item) for item in allowed_extensions_value),
        output_report=Path(output_report_value),
        duplicate_report=Path(duplicate_report_value),
        patient_id_regex=str(patient_id_regex_value) if patient_id_regex_value else None,
        min_image_size=(int(min_image_size_value[0]), int(min_image_size_value[1])),
        aspect_ratio_range=(float(aspect_ratio_range_value[0]), float(aspect_ratio_range_value[1])),
        min_stddev=min_stddev_value,
        min_center_border_diff=min_center_border_diff_value,
        duplicate_hash_distance=duplicate_hash_distance_value,
    )
    manifest, report = build_manifest_with_report(config)
    save_manifest(manifest, config.output_manifest)
    save_ingestion_report(report, config.output_report)
    save_duplicate_report(manifest, config.duplicate_report)
    LOGGER.info("Ingestion completed successfully")


if __name__ == "__main__":
    main()

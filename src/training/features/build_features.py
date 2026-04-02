from __future__ import annotations

import argparse
import csv
import importlib
import json
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image
from sklearn.model_selection import train_test_split

from training.ingestion.ingest import DEFAULT_CONFIG_PATH


DEFAULT_IMAGE_SIZE = (128, 128)
DEFAULT_FEATURES_OUTPUT_PATH = Path("data/processed/features.npz")
DEFAULT_FEATURES_REPORT_PATH = Path("data/reports/features_report.json")
DEFAULT_SPLIT_RATIOS = (0.6, 0.2, 0.2)
DEFAULT_SPLIT_RANDOM_STATE = 42
REQUIRED_MANIFEST_COLUMNS = ("sample_id", "image_path", "label_name", "label_id")


@dataclass
class FeaturesConfig:
    validated_manifest: Path
    output_features: Path
    output_report: Path
    image_size: tuple[int, int] = DEFAULT_IMAGE_SIZE
    split_ratios: tuple[float, float, float] = DEFAULT_SPLIT_RATIOS
    split_random_state: int = DEFAULT_SPLIT_RANDOM_STATE


def load_feature_settings(config_path: Path) -> dict[str, Any]:
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

    feature_settings = config_data.get("features", {})
    if not isinstance(feature_settings, dict):
        raise ValueError("The 'features' section in the config file must be a mapping.")

    return feature_settings


def build_features_config(
    *,
    validated_manifest: Path | None,
    output_features: Path | None,
    output_report: Path | None,
    image_size: tuple[int, int] = DEFAULT_IMAGE_SIZE,
    split_ratios: tuple[float, float, float] = DEFAULT_SPLIT_RATIOS,
    split_random_state: int = DEFAULT_SPLIT_RANDOM_STATE,
) -> FeaturesConfig:
    if validated_manifest is None:
        raise ValueError("A validated manifest path is required to build features config.")
    if output_features is None:
        raise ValueError("An output features path is required to build features config.")
    if output_report is None:
        raise ValueError("An output report path is required to build features config.")

    return FeaturesConfig(
        validated_manifest=validated_manifest,
        output_features=output_features,
        output_report=output_report,
        image_size=image_size,
        split_ratios=split_ratios,
        split_random_state=split_random_state,
    )


def load_manifest_rows(manifest_path: Path) -> list[dict[str, str]]:
    with manifest_path.open("r", newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def preprocess_image(image_path: Path, image_size: tuple[int, int]) -> np.ndarray:
    with Image.open(image_path) as image:
        grayscale_image = image.convert("L")
        resized_image = grayscale_image.resize(image_size)
        image_array = np.asarray(resized_image, dtype=np.float32) / 255.0
    return image_array[..., np.newaxis]


def build_feature_dataset(rows: list[dict[str, str]], image_size: tuple[int, int]) -> tuple[np.ndarray, np.ndarray, np.ndarray, Counter[str]]:
    images: list[np.ndarray] = []
    labels: list[int] = []
    sample_ids: list[str] = []
    class_distribution: Counter[str] = Counter()

    for row in rows:
        image_array = preprocess_image(Path(row["image_path"]), image_size)
        images.append(image_array)
        labels.append(int(row["label_id"]))
        sample_ids.append(row["sample_id"])
        class_distribution[row["label_name"]] += 1

    return (
        np.stack(images).astype(np.float32),
        np.asarray(labels, dtype=np.int64),
        np.asarray(sample_ids),
        class_distribution,
    )


def assign_splits(labels: np.ndarray, split_ratios: tuple[float, float, float], random_state: int) -> np.ndarray:
    train_ratio, validation_ratio, test_ratio = split_ratios
    ratio_sum = train_ratio + validation_ratio + test_ratio
    if not np.isclose(ratio_sum, 1.0):
        raise ValueError("Split ratios must sum to 1.0.")

    indices = np.arange(labels.shape[0])
    temp_indices, test_indices, temp_labels, _ = train_test_split(
        indices,
        labels,
        test_size=test_ratio,
        random_state=random_state,
        stratify=labels,
    )

    validation_share_of_temp = validation_ratio / (train_ratio + validation_ratio)
    train_indices, validation_indices = train_test_split(
        temp_indices,
        test_size=validation_share_of_temp,
        random_state=random_state,
        stratify=temp_labels,
    )

    splits = np.empty(labels.shape[0], dtype="<U10")
    splits[train_indices] = "train"
    splits[validation_indices] = "validation"
    splits[test_indices] = "test"
    return splits


def save_feature_dataset(
    images: np.ndarray,
    labels: np.ndarray,
    sample_ids: np.ndarray,
    splits: np.ndarray,
    output_features: Path,
) -> None:
    output_features.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output_features, images=images, labels=labels, sample_ids=sample_ids, splits=splits)


def write_features_report(report: dict[str, Any], output_report: Path) -> None:
    output_report.parent.mkdir(parents=True, exist_ok=True)
    with output_report.open("w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2, sort_keys=True)


def run_feature_build(config: FeaturesConfig) -> dict[str, Any]:
    if not config.validated_manifest.exists():
        report = {
            "passed": False,
            "errors": [f"Validated manifest file does not exist: {config.validated_manifest}"],
            "total_rows": 0,
            "image_shape": [],
            "class_distribution": {},
        }
        write_features_report(report, config.output_report)
        return report

    rows = load_manifest_rows(config.validated_manifest)
    missing_columns = [
        column for column in REQUIRED_MANIFEST_COLUMNS if column not in (rows[0].keys() if rows else [])
    ]
    if missing_columns:
        report = {
            "passed": False,
            "errors": [f"Missing required columns: {', '.join(missing_columns)}"],
            "total_rows": len(rows),
            "image_shape": [],
            "class_distribution": {},
        }
        write_features_report(report, config.output_report)
        return report

    if not rows:
        report = {
            "passed": False,
            "errors": ["Validated manifest is empty."],
            "total_rows": 0,
            "image_shape": [],
            "class_distribution": {},
        }
        write_features_report(report, config.output_report)
        return report

    try:
        images, labels, sample_ids, class_distribution = build_feature_dataset(rows, config.image_size)
        splits = assign_splits(labels, config.split_ratios, config.split_random_state)
    except (FileNotFoundError, OSError, ValueError) as exc:
        report = {
            "passed": False,
            "errors": [str(exc)],
            "total_rows": len(rows),
            "image_shape": [],
            "class_distribution": {},
            "split_distribution": {},
        }
        write_features_report(report, config.output_report)
        return report

    save_feature_dataset(images, labels, sample_ids, splits, config.output_features)

    split_distribution = Counter(splits.tolist())

    report = {
        "passed": True,
        "errors": [],
        "total_rows": len(rows),
        "image_shape": list(images.shape[1:]),
        "class_distribution": dict(class_distribution),
        "split_distribution": dict(split_distribution),
        "output_features": str(config.output_features),
    }
    write_features_report(report, config.output_report)
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build grayscale 1-channel feature tensors from the validated manifest.")
    parser.add_argument("--config", default=str(DEFAULT_CONFIG_PATH), help="Path to the YAML config file.")
    parser.add_argument("--validated-manifest", help="Path to the validated manifest CSV file.")
    parser.add_argument("--output-features", help="Path to the output NPZ feature artifact.")
    parser.add_argument("--output-report", help="Path to the output JSON feature report.")
    parser.add_argument("--image-width", type=int, help="Target image width.")
    parser.add_argument("--image-height", type=int, help="Target image height.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    settings = load_feature_settings(Path(args.config))

    validated_manifest_value = args.validated_manifest or settings.get("validated_manifest")
    output_features_value = args.output_features or settings.get("output_features") or str(DEFAULT_FEATURES_OUTPUT_PATH)
    output_report_value = args.output_report or settings.get("output_report") or str(DEFAULT_FEATURES_REPORT_PATH)
    image_size_value = settings.get("image_size", list(DEFAULT_IMAGE_SIZE))
    split_ratios_value = settings.get("split_ratios", list(DEFAULT_SPLIT_RATIOS))
    split_random_state_value = int(settings.get("split_random_state", DEFAULT_SPLIT_RANDOM_STATE))

    image_width = args.image_width if args.image_width is not None else int(image_size_value[0])
    image_height = args.image_height if args.image_height is not None else int(image_size_value[1])

    config = build_features_config(
        validated_manifest=Path(validated_manifest_value) if validated_manifest_value else None,
        output_features=Path(output_features_value) if output_features_value else None,
        output_report=Path(output_report_value) if output_report_value else None,
        image_size=(image_width, image_height),
        split_ratios=tuple(float(value) for value in split_ratios_value),
        split_random_state=split_random_state_value,
    )
    run_feature_build(config)


if __name__ == "__main__":
    main()

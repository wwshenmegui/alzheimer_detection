from __future__ import annotations

import argparse
import csv
import importlib
import json
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from shared.data_quality import summarize_duplicates
from training.ingestion.ingest import DEFAULT_CONFIG_PATH, DEFAULT_LABEL_TO_ID, MANIFEST_COLUMNS


REQUIRED_MANIFEST_COLUMNS = ("sample_id", "image_path", "label_name", "label_id")
DEFAULT_VALIDATION_REPORT_PATH = Path("data/reports/validation_report.json")
DEFAULT_VALIDATED_MANIFEST_PATH = Path("data/processed/validated_manifest.csv")


@dataclass
class ValidationConfig:
    manifest_path: Path
    output_report: Path
    label_to_id: dict[str, int] = field(default_factory=lambda: dict(DEFAULT_LABEL_TO_ID))
    approved_manifest: Path | None = DEFAULT_VALIDATED_MANIFEST_PATH


def load_validation_settings(config_path: Path) -> dict[str, Any]:
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

    validation_settings = config_data.get("validation", {})
    if not isinstance(validation_settings, dict):
        raise ValueError("The 'validation' section in the config file must be a mapping.")

    return validation_settings


def build_validation_config(
    *,
    manifest_path: Path | None,
    output_report: Path | None,
    label_to_id: dict[str, int] | None = None,
    approved_manifest: Path | None = DEFAULT_VALIDATED_MANIFEST_PATH,
) -> ValidationConfig:
    if manifest_path is None:
        raise ValueError("A manifest path is required to build validation config.")
    if output_report is None:
        raise ValueError("An output report path is required to build validation config.")

    return ValidationConfig(
        manifest_path=manifest_path,
        output_report=output_report,
        label_to_id=label_to_id or dict(DEFAULT_LABEL_TO_ID),
        approved_manifest=approved_manifest,
    )


def load_manifest_rows(manifest_path: Path) -> list[dict[str, str]]:
    with manifest_path.open("r", newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def validate_manifest_rows(rows: list[dict[str, str]], label_to_id: dict[str, int]) -> dict[str, Any]:
    errors: list[str] = []
    warnings: list[str] = []
    class_distribution: Counter[str] = Counter()
    sample_ids: set[str] = set()
    mri_invalid_rows = 0
    patient_id_rows = 0
    valid_rows: list[dict[str, str]] = []

    for index, row in enumerate(rows, start=2):
        row_has_error = False
        for column in REQUIRED_MANIFEST_COLUMNS:
            if column not in row:
                errors.append(f"Missing required column: {column}")
                return {
                    "passed": False,
                    "errors": sorted(set(errors)),
                    "warnings": warnings,
                    "total_rows": len(rows),
                    "class_distribution": {},
                    "valid_rows": [],
                }
            if not row[column]:
                errors.append(f"Row {index} has an empty value for '{column}'.")
                row_has_error = True

        sample_id = row.get("sample_id", "")
        image_path = row.get("image_path", "")
        label_name = row.get("label_name", "")
        label_id = row.get("label_id", "")

        if sample_id in sample_ids:
            errors.append(f"Duplicate sample_id found: {sample_id}")
            row_has_error = True
        sample_ids.add(sample_id)

        if image_path and not Path(image_path).exists():
            errors.append(f"Image path does not exist: {image_path}")
            row_has_error = True

        if label_name not in label_to_id:
            errors.append(f"Invalid label_name '{label_name}' at row {index}.")
            row_has_error = True
        else:
            expected_label_id = str(label_to_id[label_name])
            if label_id != expected_label_id:
                errors.append(
                    f"Label mismatch at row {index}: label_name '{label_name}' expects label_id '{expected_label_id}', got '{label_id}'."
                )
                row_has_error = True
        if row.get("patient_id"):
            patient_id_rows += 1

        if str(row.get("mri_is_valid", "True")).lower() != "true":
            mri_invalid_rows += 1
            warnings.append(
                f"Row {index} failed MRI validation with code '{row.get('mri_error_code', '') or 'unknown'}'."
            )
            continue

        if not row_has_error:
            valid_rows.append(row)
            class_distribution[label_name] += 1

    duplicate_summary = summarize_duplicates(rows)
    if duplicate_summary["exact_duplicate_groups"]:
        warnings.append(f"Detected {duplicate_summary['exact_duplicate_groups']} exact duplicate groups.")
    if duplicate_summary["near_duplicate_groups"]:
        warnings.append(f"Detected {duplicate_summary['near_duplicate_groups']} near-duplicate groups.")

    return {
        "passed": not errors,
        "errors": errors,
        "warnings": sorted(set(warnings)),
        "total_rows": len(rows),
        "class_distribution": dict(class_distribution),
        "duplicate_summary": duplicate_summary,
        "mri_invalid_rows": mri_invalid_rows,
        "patient_id_rows": patient_id_rows,
        "valid_rows": valid_rows,
    }


def write_validation_report(report: dict[str, Any], output_report: Path) -> None:
    output_report.parent.mkdir(parents=True, exist_ok=True)
    with output_report.open("w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2, sort_keys=True)


def save_validated_manifest(rows: list[dict[str, str]], approved_manifest: Path | None) -> None:
    if approved_manifest is None:
        return
    approved_manifest.parent.mkdir(parents=True, exist_ok=True)
    with approved_manifest.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=MANIFEST_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)


def run_validation(config: ValidationConfig) -> dict[str, Any]:
    if not config.manifest_path.exists():
        report = {
            "passed": False,
            "errors": [f"Manifest file does not exist: {config.manifest_path}"],
            "warnings": [],
            "total_rows": 0,
            "class_distribution": {},
            "valid_rows": [],
        }
        write_validation_report(report, config.output_report)
        return report

    rows = load_manifest_rows(config.manifest_path)
    missing_columns = [
        column for column in REQUIRED_MANIFEST_COLUMNS if column not in (rows[0].keys() if rows else [])
    ]

    if missing_columns:
        report = {
            "passed": False,
            "errors": [f"Missing required columns: {', '.join(missing_columns)}"],
            "warnings": [],
            "total_rows": len(rows),
            "class_distribution": {},
            "valid_rows": [],
        }
        write_validation_report(report, config.output_report)
        return report

    report = validate_manifest_rows(rows, config.label_to_id)
    write_validation_report(report, config.output_report)

    if report["passed"]:
        save_validated_manifest(report["valid_rows"], config.approved_manifest)

    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate the ingested Alzheimer dataset manifest.")
    parser.add_argument("--config", default=str(DEFAULT_CONFIG_PATH), help="Path to the YAML config file.")
    parser.add_argument("--manifest-path", help="Path to the manifest CSV file.")
    parser.add_argument("--output-report", help="Path to the output JSON validation report.")
    parser.add_argument(
        "--approved-manifest",
        help="Optional path to write a validated copy of the manifest when checks pass.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    settings = load_validation_settings(Path(args.config))

    manifest_path_value = args.manifest_path or settings.get("manifest_path")
    output_report_value = args.output_report or settings.get("output_report") or str(DEFAULT_VALIDATION_REPORT_PATH)
    approved_manifest_value = args.approved_manifest
    if approved_manifest_value is None:
        approved_manifest_value = settings.get("approved_manifest")
        if approved_manifest_value is None:
            approved_manifest_value = str(DEFAULT_VALIDATED_MANIFEST_PATH)

    label_to_id_value = settings.get("label_to_id", dict(DEFAULT_LABEL_TO_ID))

    config = build_validation_config(
        manifest_path=Path(manifest_path_value) if manifest_path_value else None,
        output_report=Path(output_report_value) if output_report_value else None,
        label_to_id={str(key): int(value) for key, value in label_to_id_value.items()},
        approved_manifest=Path(approved_manifest_value) if approved_manifest_value else None,
    )
    run_validation(config)


if __name__ == "__main__":
    main()

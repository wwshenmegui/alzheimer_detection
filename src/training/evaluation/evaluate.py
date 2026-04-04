from __future__ import annotations

import argparse
import importlib
import json
import logging
import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix, precision_recall_fscore_support

from shared.model_registry import build_artifact_lineage, resolve_model_artifacts, update_model_metadata
from training.ingestion.ingest import DEFAULT_CONFIG_PATH
from training.models.train import flatten_images, load_feature_artifact


DEFAULT_EVALUATION_REPORT_PATH = Path("data/reports/evaluation_report.json")


LOGGER = logging.getLogger(__name__)


@dataclass
class EvaluationConfig:
    input_features: Path
    input_model: Path
    output_report: Path


def configure_logging(log_level: str) -> None:
    logging.basicConfig(
        level=getattr(logging, log_level.upper(), logging.INFO),
        format="%(asctime)s | %(levelname)s | %(message)s",
    )


def load_evaluation_settings(config_path: Path) -> dict[str, Any]:
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

    evaluation_settings = config_data.get("evaluation", {})
    if not isinstance(evaluation_settings, dict):
        raise ValueError("The 'evaluation' section in the config file must be a mapping.")

    return evaluation_settings


def build_evaluation_config(
    *,
    input_features: Path | None,
    input_model: Path | None,
    output_report: Path | None,
) -> EvaluationConfig:
    if input_features is None:
        raise ValueError("An input features path is required to build evaluation config.")
    if input_model is None:
        raise ValueError("An input model path is required to build evaluation config.")
    if output_report is None:
        raise ValueError("An output report path is required to build evaluation config.")

    return EvaluationConfig(
        input_features=input_features,
        input_model=input_model,
        output_report=output_report,
    )


def load_model(input_model: Path):
    with input_model.open("rb") as handle:
        return pickle.load(handle)


def write_evaluation_report(report: dict[str, Any], output_report: Path) -> None:
    output_report.parent.mkdir(parents=True, exist_ok=True)
    with output_report.open("w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2, sort_keys=True)


def run_evaluation(config: EvaluationConfig) -> dict[str, Any]:
    LOGGER.info("Starting model evaluation")
    resolved_model_path, metadata_path = resolve_model_artifacts(config.input_model)

    if not config.input_features.exists():
        report = {
            "passed": False,
            "errors": [f"Feature artifact does not exist: {config.input_features}"],
            "validation_rows": 0,
        }
        LOGGER.error("Feature artifact does not exist: %s", config.input_features)
        write_evaluation_report(report, config.output_report)
        return report

    if not resolved_model_path.exists():
        report = {
            "passed": False,
            "errors": [f"Model file does not exist: {resolved_model_path}"],
            "validation_rows": 0,
        }
        LOGGER.error("Model file does not exist: %s", resolved_model_path)
        write_evaluation_report(report, config.output_report)
        return report

    try:
        images, labels, sample_ids, splits = load_feature_artifact(config.input_features)
        model = load_model(resolved_model_path)
    except (KeyError, ValueError, OSError, pickle.PickleError) as exc:
        report = {
            "passed": False,
            "errors": [str(exc)],
            "validation_rows": 0,
        }
        LOGGER.exception("Failed to load evaluation inputs")
        write_evaluation_report(report, config.output_report)
        return report

    if images.size == 0 or labels.size == 0:
        report = {
            "passed": False,
            "errors": ["Feature artifact is empty."],
            "validation_rows": 0,
        }
        LOGGER.error("Feature artifact is empty")
        write_evaluation_report(report, config.output_report)
        return report

    features = flatten_images(images)
    LOGGER.info("Loaded %s samples for evaluation", features.shape[0])

    test_mask = splits == "test"
    if not test_mask.any():
        report = {
            "passed": False,
            "errors": ["Feature artifact must contain a non-empty 'test' split."],
            "test_rows": 0,
        }
        LOGGER.error("Feature artifact must contain a non-empty test split")
        write_evaluation_report(report, config.output_report)
        return report

    test_features = features[test_mask]
    test_labels = labels[test_mask]
    test_ids = sample_ids[test_mask]

    LOGGER.info("Running inference on %s test rows", test_features.shape[0])
    predictions = model.predict(test_features)
    class_ids = [int(class_id) for class_id in getattr(model, "classes_", np.unique(labels))]

    accuracy = float(accuracy_score(test_labels, predictions))
    precision, recall, f1, support = precision_recall_fscore_support(
        test_labels,
        predictions,
        labels=class_ids,
        average=None,
        zero_division=0,
    )
    precision_macro, recall_macro, f1_macro, _ = precision_recall_fscore_support(
        test_labels,
        predictions,
        average="macro",
        zero_division=0,
    )
    matrix = confusion_matrix(test_labels, predictions, labels=class_ids)

    report = {
        "passed": True,
        "errors": [],
        "test_rows": int(test_features.shape[0]),
        "accuracy": accuracy,
        "precision_macro": float(precision_macro),
        "recall_macro": float(recall_macro),
        "f1_macro": float(f1_macro),
        "class_ids": class_ids,
        "confusion_matrix": matrix.tolist(),
        "per_class_support": {str(class_id): int(class_support) for class_id, class_support in zip(class_ids, support.tolist())},
        "per_class_precision": {str(class_id): float(class_precision) for class_id, class_precision in zip(class_ids, precision.tolist())},
        "per_class_recall": {str(class_id): float(class_recall) for class_id, class_recall in zip(class_ids, recall.tolist())},
        "per_class_f1": {str(class_id): float(class_f1) for class_id, class_f1 in zip(class_ids, f1.tolist())},
        "test_sample_ids": [str(sample_id) for sample_id in test_ids.tolist()],
    }
    write_evaluation_report(report, config.output_report)
    if metadata_path is not None and metadata_path.exists():
        existing_lineage = (update_model_metadata(metadata_path, {}).get("lineage") or {})
        update_model_metadata(
            metadata_path,
            {
                "evaluation_report_path": str(config.output_report),
                "lineage": {
                    **existing_lineage,
                    **build_artifact_lineage(artifact_path=config.output_report, artifact_key="evaluation_report"),
                },
            },
        )
    LOGGER.info("Wrote evaluation report to %s", config.output_report)
    LOGGER.info("Evaluation accuracy: %.4f", accuracy)
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate a trained baseline model on the validation split.")
    parser.add_argument("--config", default=str(DEFAULT_CONFIG_PATH), help="Path to the YAML config file.")
    parser.add_argument("--input-features", help="Path to the NPZ feature artifact.")
    parser.add_argument("--input-model", help="Path to the saved pickled model.")
    parser.add_argument("--output-report", help="Path to the output JSON evaluation report.")
    parser.add_argument("--log-level", default="INFO", help="Logging level, for example DEBUG, INFO, WARNING.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    configure_logging(args.log_level)
    settings = load_evaluation_settings(Path(args.config))

    input_features_value = args.input_features or settings.get("input_features")
    input_model_value = args.input_model or settings.get("input_model")
    output_report_value = args.output_report or settings.get("output_report") or str(DEFAULT_EVALUATION_REPORT_PATH)

    config = build_evaluation_config(
        input_features=Path(input_features_value) if input_features_value else None,
        input_model=Path(input_model_value) if input_model_value else None,
        output_report=Path(output_report_value) if output_report_value else None,
    )
    report = run_evaluation(config)
    if report["passed"]:
        LOGGER.info("Evaluation completed successfully")
    else:
        LOGGER.error("Evaluation failed: %s", "; ".join(report["errors"]))


if __name__ == "__main__":
    main()

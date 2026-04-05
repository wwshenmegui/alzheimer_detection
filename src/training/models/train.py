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
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

from shared.model_registry import (
    build_artifact_lineage,
    DEFAULT_MODEL_VERSION,
    derive_model_name,
    resolve_versioned_model_paths,
    utc_timestamp,
    update_model_metadata,
    write_current_version_pointer,
    write_json_file,
)
from shared.experiment_tracking import (
    ExperimentTrackingConfig,
    build_experiment_tracking_config,
    capture_configured_stage_outputs,
    capture_stage_file,
    finalize_run,
    initialize_experiment_run,
    load_experiment_tracking_settings,
    log_remote_training_run,
    record_stage,
    update_run_metadata,
)
from training.ingestion.ingest import DEFAULT_CONFIG_PATH


DEFAULT_MODEL_OUTPUT_PATH = Path("models/trained/logistic_regression.pkl")
DEFAULT_TRAINING_REPORT_PATH = Path("data/reports/training_report.json")


LOGGER = logging.getLogger(__name__)


@dataclass
class ModelTrainingConfig:
    input_features: Path
    output_model: Path
    output_report: Path
    max_iter: int = 500
    model_name: str | None = None
    model_version: str = DEFAULT_MODEL_VERSION
    experiment_tracking: ExperimentTrackingConfig | None = None
    config_path: Path | None = None


def configure_logging(log_level: str) -> None:
    logging.basicConfig(
        level=getattr(logging, log_level.upper(), logging.INFO),
        format="%(asctime)s | %(levelname)s | %(message)s",
    )


def load_training_settings(config_path: Path) -> dict[str, Any]:
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

    training_settings = config_data.get("model", {})
    if not isinstance(training_settings, dict):
        raise ValueError("The 'model' section in the config file must be a mapping.")

    return training_settings


def build_training_config(
    *,
    input_features: Path | None,
    output_model: Path | None,
    output_report: Path | None,
    max_iter: int = 500,
    model_name: str | None = None,
    model_version: str = DEFAULT_MODEL_VERSION,
    experiment_tracking: ExperimentTrackingConfig | None = None,
    config_path: Path | None = None,
) -> ModelTrainingConfig:
    if input_features is None:
        raise ValueError("An input features path is required to build training config.")
    if output_model is None:
        raise ValueError("An output model path is required to build training config.")
    if output_report is None:
        raise ValueError("An output report path is required to build training config.")

    return ModelTrainingConfig(
        input_features=input_features,
        output_model=output_model,
        output_report=output_report,
        max_iter=max_iter,
        model_name=model_name,
        model_version=model_version,
        experiment_tracking=experiment_tracking,
        config_path=config_path,
    )


def load_feature_artifact(input_features: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    artifact = np.load(input_features, allow_pickle=False)
    return artifact["images"], artifact["labels"], artifact["sample_ids"], artifact["splits"]


def flatten_images(images: np.ndarray) -> np.ndarray:
    return images.reshape(images.shape[0], -1)


def save_model(model: LogisticRegression, output_model: Path) -> None:
    output_model.parent.mkdir(parents=True, exist_ok=True)
    with output_model.open("wb") as handle:
        pickle.dump(model, handle)


def write_training_report(report: dict[str, Any], output_report: Path) -> None:
    output_report.parent.mkdir(parents=True, exist_ok=True)
    with output_report.open("w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2, sort_keys=True)


def write_model_metadata(metadata: dict[str, Any], output_path: Path) -> None:
    write_json_file(metadata, output_path)


def run_training(config: ModelTrainingConfig) -> dict[str, Any]:
    LOGGER.info("Starting model training")
    LOGGER.info("Loading feature artifact from %s", config.input_features)
    experiment_run = None
    if config.experiment_tracking and config.experiment_tracking.enabled:
        experiment_run = initialize_experiment_run(
            config.experiment_tracking,
            config_snapshot_source=config.config_path,
        )

    if not config.input_features.exists():
        report = {
            "passed": False,
            "errors": [f"Feature artifact does not exist: {config.input_features}"],
            "train_rows": 0,
            "validation_rows": 0,
            "validation_accuracy": 0.0,
        }
        LOGGER.error("Feature artifact does not exist: %s", config.input_features)
        write_training_report(report, config.output_report)
        if experiment_run:
            record_stage(experiment_run, stage_name="training", status="failed", summary=report)
            finalize_run(experiment_run, status="failed")
        return report

    try:
        images, labels, sample_ids, splits = load_feature_artifact(config.input_features)
    except (KeyError, ValueError, OSError) as exc:
        report = {
            "passed": False,
            "errors": [str(exc)],
            "train_rows": 0,
            "validation_rows": 0,
            "validation_accuracy": 0.0,
        }
        LOGGER.exception("Failed to load feature artifact")
        write_training_report(report, config.output_report)
        if experiment_run:
            record_stage(experiment_run, stage_name="training", status="failed", summary=report)
            finalize_run(experiment_run, status="failed")
        return report

    if images.size == 0 or labels.size == 0:
        report = {
            "passed": False,
            "errors": ["Feature artifact is empty."],
            "train_rows": 0,
            "validation_rows": 0,
            "validation_accuracy": 0.0,
        }
        LOGGER.error("Feature artifact is empty")
        write_training_report(report, config.output_report)
        if experiment_run:
            record_stage(experiment_run, stage_name="training", status="failed", summary=report)
            finalize_run(experiment_run, status="failed")
        return report

    features = flatten_images(images)
    LOGGER.info("Loaded %s samples with %s flattened features", features.shape[0], features.shape[1])

    train_mask = splits == "train"
    validation_mask = splits == "validation"
    if not train_mask.any() or not validation_mask.any():
        report = {
            "passed": False,
            "errors": ["Feature artifact must contain non-empty 'train' and 'validation' splits."],
            "train_rows": 0,
            "validation_rows": 0,
            "validation_accuracy": 0.0,
        }
        LOGGER.error("Feature artifact must contain non-empty train and validation splits")
        write_training_report(report, config.output_report)
        if experiment_run:
            record_stage(experiment_run, stage_name="training", status="failed", summary=report)
            finalize_run(experiment_run, status="failed")
        return report

    train_features = features[train_mask]
    validation_features = features[validation_mask]
    train_labels = labels[train_mask]
    validation_labels = labels[validation_mask]
    train_ids = sample_ids[train_mask]
    validation_ids = sample_ids[validation_mask]

    LOGGER.info(
        "Split dataset into %s training rows and %s validation rows",
        train_features.shape[0],
        validation_features.shape[0],
    )

    model = LogisticRegression(max_iter=config.max_iter)
    LOGGER.info("Fitting logistic regression with max_iter=%s", config.max_iter)
    model.fit(train_features, train_labels)

    validation_predictions = model.predict(validation_features)
    validation_accuracy = float(accuracy_score(validation_labels, validation_predictions))
    LOGGER.info("Validation accuracy: %.4f", validation_accuracy)

    resolved_model_name = derive_model_name(config.output_model, config.model_name)
    version_paths = resolve_versioned_model_paths(
        config.output_model,
        model_version=config.model_version,
        model_name=resolved_model_name,
    )

    save_model(model, version_paths["model_path"])
    LOGGER.info("Saved trained model to %s", version_paths["model_path"])

    metadata = {
        "model_name": resolved_model_name,
        "model_version": config.model_version,
        "model_type": type(model).__name__,
        "model_path": str(version_paths["model_path"]),
        "metadata_path": str(version_paths["metadata_path"]),
        "training_report_path": str(config.output_report),
        "feature_artifact_path": str(config.input_features),
        "created_at": utc_timestamp(),
        "validation_accuracy": validation_accuracy,
        "train_rows": int(train_features.shape[0]),
        "validation_rows": int(validation_features.shape[0]),
        "num_features": int(train_features.shape[1]),
        "classes": [int(class_id) for class_id in model.classes_],
        "max_iter": int(config.max_iter),
        "lineage": {
            **build_artifact_lineage(artifact_path=config.input_features, artifact_key="feature_artifact"),
            **build_artifact_lineage(artifact_path=version_paths["model_path"], artifact_key="model_artifact"),
            "training_report_path": str(config.output_report),
        },
    }
    write_model_metadata(metadata, version_paths["metadata_path"])
    write_current_version_pointer(
        version_paths["current_path"],
        model_name=resolved_model_name,
        model_version=config.model_version,
        model_path=version_paths["model_path"],
        metadata_path=version_paths["metadata_path"],
    )

    report = {
        "passed": True,
        "errors": [],
        "train_rows": int(train_features.shape[0]),
        "validation_rows": int(validation_features.shape[0]),
        "num_features": int(train_features.shape[1]),
        "validation_accuracy": validation_accuracy,
        "classes": [int(class_id) for class_id in model.classes_],
        "model_name": resolved_model_name,
        "model_version": config.model_version,
        "output_model": str(version_paths["model_path"]),
        "output_metadata": str(version_paths["metadata_path"]),
        "current_pointer": str(version_paths["current_path"]),
        "train_sample_ids": [str(sample_id) for sample_id in train_ids.tolist()],
        "validation_sample_ids": [str(sample_id) for sample_id in validation_ids.tolist()],
    }
    write_training_report(report, config.output_report)
    update_model_metadata(
        version_paths["metadata_path"],
        {
            "lineage": {
                **metadata["lineage"],
                **build_artifact_lineage(artifact_path=config.output_report, artifact_key="training_report"),
            }
        },
    )
    if experiment_run:
        if config.config_path and config.config_path.exists():
            capture_configured_stage_outputs(
                experiment_run,
                config.config_path,
                save_stage_artifacts=config.experiment_tracking.save_stage_artifacts,
            )
        copied_report = capture_stage_file(
            experiment_run,
            source_path=config.output_report,
            destination_group="reports",
            destination_name=config.output_report.name,
        )
        training_artifacts = {
            "model_path": capture_stage_file(
                experiment_run,
                source_path=version_paths["model_path"],
                destination_group="artifacts",
                destination_name=version_paths["model_path"].name,
            )
            or "",
            "model_metadata_path": capture_stage_file(
                experiment_run,
                source_path=version_paths["metadata_path"],
                destination_group="artifacts",
                destination_name=version_paths["metadata_path"].name,
            )
            or "",
        }
        stage_summary = {
            "model_name": resolved_model_name,
            "model_version": config.model_version,
            "validation_accuracy": validation_accuracy,
            "train_rows": int(train_features.shape[0]),
            "validation_rows": int(validation_features.shape[0]),
        }
        remote_tracking = {
            "enabled": config.experiment_tracking.remote.enabled,
            "backend": config.experiment_tracking.remote.backend if config.experiment_tracking.remote.enabled else None,
            "run_id": None,
            "url": config.experiment_tracking.remote.tracking_uri,
        }
        if config.experiment_tracking.remote.enabled:
            remote_run_id = log_remote_training_run(
                config.experiment_tracking.remote,
                run_id=experiment_run.run_id,
                params={
                    "model_name": resolved_model_name,
                    "model_version": config.model_version,
                    "max_iter": config.max_iter,
                },
                metrics={
                    "validation_accuracy": validation_accuracy,
                    "train_rows": float(train_features.shape[0]),
                    "validation_rows": float(validation_features.shape[0]),
                },
                artifact_paths=[config.output_report, version_paths["model_path"], version_paths["metadata_path"]],
                tags={"project": "alzheimer_detection", "stage": "training"},
            )
            remote_tracking["run_id"] = remote_run_id
        update_model_metadata(
            version_paths["metadata_path"],
            {
                "experiment_run_id": experiment_run.run_id,
                "experiment_run_metadata_path": str(experiment_run.metadata_path),
                "remote_tracking": remote_tracking,
            },
        )
        record_stage(
            experiment_run,
            stage_name="training",
            status="completed",
            summary=stage_summary,
            report_path=copied_report,
            artifacts=training_artifacts,
        )
        finalize_run(
            experiment_run,
            status="completed",
            model={"model_name": resolved_model_name, "model_version": config.model_version, "serving_candidate": True},
            dataset={
                "feature_artifact_path": str(config.input_features),
            },
        )
        update_run_metadata(experiment_run.metadata_path, {"remote_tracking": remote_tracking})
    LOGGER.info("Wrote training report to %s", config.output_report)
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a simple baseline model on grayscale feature tensors.")
    parser.add_argument("--config", default=str(DEFAULT_CONFIG_PATH), help="Path to the YAML config file.")
    parser.add_argument("--input-features", help="Path to the NPZ feature artifact.")
    parser.add_argument("--output-model", help="Path to the output pickled model.")
    parser.add_argument("--output-report", help="Path to the output JSON training report.")
    parser.add_argument("--max-iter", type=int, help="Maximum optimizer iterations for logistic regression.")
    parser.add_argument("--model-name", help="Logical model name used for the versioned artifact directory.")
    parser.add_argument("--model-version", help="Model version label, for example v1 or 2026-04-03.")
    parser.add_argument("--log-level", default="INFO", help="Logging level, for example DEBUG, INFO, WARNING.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    configure_logging(args.log_level)
    settings = load_training_settings(Path(args.config))
    experiment_settings = load_experiment_tracking_settings(Path(args.config))

    input_features_value = args.input_features or settings.get("input_features")
    output_model_value = args.output_model or settings.get("output_model") or str(DEFAULT_MODEL_OUTPUT_PATH)
    output_report_value = args.output_report or settings.get("output_report") or str(DEFAULT_TRAINING_REPORT_PATH)
    max_iter_value = args.max_iter if args.max_iter is not None else int(settings.get("max_iter", 500))
    model_name_value = args.model_name or settings.get("model_name")
    model_version_value = args.model_version or settings.get("model_version", DEFAULT_MODEL_VERSION)

    config = build_training_config(
        input_features=Path(input_features_value) if input_features_value else None,
        output_model=Path(output_model_value) if output_model_value else None,
        output_report=Path(output_report_value) if output_report_value else None,
        max_iter=max_iter_value,
        model_name=str(model_name_value) if model_name_value else None,
        model_version=str(model_version_value),
        experiment_tracking=build_experiment_tracking_config(experiment_settings),
        config_path=Path(args.config),
    )
    report = run_training(config)
    if report["passed"]:
        LOGGER.info("Training completed successfully")
    else:
        LOGGER.error("Training failed: %s", "; ".join(report["errors"]))


if __name__ == "__main__":
    main()

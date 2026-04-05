from __future__ import annotations

import argparse
import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

from training.evaluation.evaluate import (
    DEFAULT_EVALUATION_REPORT_PATH,
    build_evaluation_config,
    load_evaluation_settings,
    run_evaluation,
)
from training.features.build_features import (
    DEFAULT_FEATURES_OUTPUT_PATH,
    DEFAULT_FEATURES_REPORT_PATH,
    DEFAULT_IMAGE_SIZE,
    DEFAULT_SPLIT_RANDOM_STATE,
    DEFAULT_SPLIT_RATIOS,
    build_features_config,
    load_feature_settings,
    run_feature_build,
)
from training.ingestion.ingest import (
    DEFAULT_CONFIG_PATH,
    DEFAULT_DATASET_HANDLE,
    DEFAULT_DUPLICATE_REPORT_PATH,
    DEFAULT_INGESTION_REPORT_PATH,
    DEFAULT_LABEL_TO_ID,
    build_ingestion_config,
    build_manifest_with_report,
    download_dataset,
    load_ingestion_settings,
    persist_downloaded_dataset,
    resolve_input_dataset_root,
    save_duplicate_report,
    save_ingestion_report,
    save_manifest,
)
from training.models.train import (
    DEFAULT_MODEL_OUTPUT_PATH,
    DEFAULT_TRAINING_REPORT_PATH,
    build_training_config,
    load_training_settings,
    run_training,
)
from training.validation.validate import (
    DEFAULT_VALIDATED_MANIFEST_PATH,
    DEFAULT_VALIDATION_REPORT_PATH,
    build_validation_config,
    load_validation_settings,
    run_validation,
)
from shared.experiment_tracking import (
    build_experiment_tracking_config,
    load_experiment_tracking_settings,
)


LOGGER = logging.getLogger(__name__)


@dataclass
class PipelineConfig:
    config_path: Path = DEFAULT_CONFIG_PATH
    dataset_root: Path | None = None
    download: bool = False
    dataset_handle: str = DEFAULT_DATASET_HANDLE
    log_level: str = "INFO"


def configure_logging(log_level: str) -> None:
    logging.basicConfig(
        level=getattr(logging, log_level.upper(), logging.INFO),
        format="%(asctime)s | %(levelname)s | %(message)s",
    )


def build_pipeline_config(
    *,
    config_path: Path = DEFAULT_CONFIG_PATH,
    dataset_root: Path | None = None,
    download: bool = False,
    dataset_handle: str = DEFAULT_DATASET_HANDLE,
    log_level: str = "INFO",
) -> PipelineConfig:
    return PipelineConfig(
        config_path=config_path,
        dataset_root=dataset_root,
        download=download,
        dataset_handle=dataset_handle,
        log_level=log_level,
    )


def _path_from_setting(value: Any, fallback: Path | None = None) -> Path | None:
    if value is None:
        return fallback
    if isinstance(value, Path):
        return value
    value_str = str(value).strip()
    if not value_str:
        return fallback
    return Path(value_str)


def _run_stage(stage_name: str, runner: Callable[[], dict[str, Any]]) -> tuple[dict[str, Any], bool]:
    LOGGER.info("Starting %s stage", stage_name)
    try:
        report = runner()
    except Exception as exc:  # pragma: no cover - guarded by failure-path tests indirectly
        LOGGER.exception("%s stage failed with an unexpected error", stage_name)
        return {"passed": False, "errors": [str(exc)]}, False

    if report.get("passed"):
        LOGGER.info("Completed %s stage", stage_name)
        return report, True

    LOGGER.error("%s stage failed: %s", stage_name, "; ".join(report.get("errors", [])))
    return report, False


def run_training_pipeline(config: PipelineConfig) -> dict[str, Any]:
    ingestion_settings = load_ingestion_settings(config.config_path)
    validation_settings = load_validation_settings(config.config_path)
    feature_settings = load_feature_settings(config.config_path)
    training_settings = load_training_settings(config.config_path)
    evaluation_settings = load_evaluation_settings(config.config_path)
    experiment_settings = load_experiment_tracking_settings(config.config_path)
    experiment_tracking = build_experiment_tracking_config(experiment_settings)

    pipeline_report: dict[str, Any] = {
        "passed": False,
        "config_path": str(config.config_path),
        "stages": {},
        "failed_stage": None,
    }

    download_enabled = config.download or bool(ingestion_settings.get("download", False))
    dataset_root = config.dataset_root or _path_from_setting(ingestion_settings.get("dataset_root"))
    dataset_handle = str(ingestion_settings.get("dataset_handle", config.dataset_handle) or config.dataset_handle)
    label_to_id = {
        str(key): int(value)
        for key, value in ingestion_settings.get("label_to_id", dict(DEFAULT_LABEL_TO_ID)).items()
    }
    ingestion_output_report = _path_from_setting(
        ingestion_settings.get("output_report"),
        DEFAULT_INGESTION_REPORT_PATH,
    )

    if download_enabled:
        downloaded_root = download_dataset(dataset_handle)
        dataset_root = (
            persist_downloaded_dataset(downloaded_root, dataset_root, label_to_id)
            if dataset_root is not None
            else downloaded_root
        )

    ingestion_config = build_ingestion_config(
        dataset_root=dataset_root,
        output_manifest=_path_from_setting(ingestion_settings.get("output_manifest")),
        label_to_id=label_to_id,
        allowed_extensions=tuple(ingestion_settings.get("allowed_extensions", (".jpg", ".jpeg", ".png"))),
        output_report=ingestion_output_report,
        duplicate_report=_path_from_setting(
            ingestion_settings.get("duplicate_report"),
            DEFAULT_DUPLICATE_REPORT_PATH,
        ),
        patient_id_regex=(
            str(ingestion_settings.get("patient_id_regex"))
            if ingestion_settings.get("patient_id_regex")
            else None
        ),
        min_image_size=tuple(int(value) for value in ingestion_settings.get("min_image_size", DEFAULT_IMAGE_SIZE)),
        aspect_ratio_range=tuple(float(value) for value in ingestion_settings.get("aspect_ratio_range", (0.75, 1.4))),
        min_stddev=float(ingestion_settings.get("min_stddev", 0.03)),
        min_center_border_diff=float(ingestion_settings.get("min_center_border_diff", 0.02)),
        duplicate_hash_distance=int(ingestion_settings.get("duplicate_hash_distance", 6)),
    )

    def ingestion_runner() -> dict[str, Any]:
        if not download_enabled:
            ingestion_config.dataset_root = resolve_input_dataset_root(
                ingestion_config.dataset_root,
                output_report=ingestion_config.output_report,
                label_to_id=ingestion_config.label_to_id,
            )
        manifest, report = build_manifest_with_report(ingestion_config)
        save_manifest(manifest, ingestion_config.output_manifest)
        save_ingestion_report(report, ingestion_config.output_report)
        save_duplicate_report(manifest, ingestion_config.duplicate_report)
        report["output_manifest"] = str(ingestion_config.output_manifest)
        report["output_report"] = str(ingestion_config.output_report)
        report["duplicate_report"] = str(ingestion_config.duplicate_report)
        return report

    report, passed = _run_stage("ingestion", ingestion_runner)
    pipeline_report["stages"]["ingestion"] = report
    if not passed:
        pipeline_report["failed_stage"] = "ingestion"
        return pipeline_report

    validation_config = build_validation_config(
        manifest_path=_path_from_setting(
            validation_settings.get("manifest_path"),
            ingestion_config.output_manifest,
        ),
        output_report=_path_from_setting(
            validation_settings.get("output_report"),
            DEFAULT_VALIDATION_REPORT_PATH,
        ),
        label_to_id={
            str(key): int(value)
            for key, value in validation_settings.get("label_to_id", dict(DEFAULT_LABEL_TO_ID)).items()
        },
        approved_manifest=_path_from_setting(
            validation_settings.get("approved_manifest"),
            DEFAULT_VALIDATED_MANIFEST_PATH,
        ),
    )

    report, passed = _run_stage("validation", lambda: run_validation(validation_config))
    pipeline_report["stages"]["validation"] = report
    if not passed:
        pipeline_report["failed_stage"] = "validation"
        return pipeline_report

    configured_image_size = feature_settings.get("image_size", list(DEFAULT_IMAGE_SIZE))
    configured_split_ratios = feature_settings.get("split_ratios", list(DEFAULT_SPLIT_RATIOS))
    features_config = build_features_config(
        validated_manifest=_path_from_setting(
            feature_settings.get("validated_manifest"),
            validation_config.approved_manifest,
        ),
        output_features=_path_from_setting(
            feature_settings.get("output_features"),
            DEFAULT_FEATURES_OUTPUT_PATH,
        ),
        output_report=_path_from_setting(
            feature_settings.get("output_report"),
            DEFAULT_FEATURES_REPORT_PATH,
        ),
        image_size=(int(configured_image_size[0]), int(configured_image_size[1])),
        split_ratios=tuple(float(value) for value in configured_split_ratios),
        split_random_state=int(feature_settings.get("split_random_state", DEFAULT_SPLIT_RANDOM_STATE)),
    )

    report, passed = _run_stage("features", lambda: run_feature_build(features_config))
    pipeline_report["stages"]["features"] = report
    if not passed:
        pipeline_report["failed_stage"] = "features"
        return pipeline_report

    training_config = build_training_config(
        input_features=_path_from_setting(
            training_settings.get("input_features"),
            features_config.output_features,
        ),
        output_model=_path_from_setting(
            training_settings.get("output_model"),
            DEFAULT_MODEL_OUTPUT_PATH,
        ),
        output_report=_path_from_setting(
            training_settings.get("output_report"),
            DEFAULT_TRAINING_REPORT_PATH,
        ),
        max_iter=int(training_settings.get("max_iter", 500)),
        model_name=(str(training_settings.get("model_name")) if training_settings.get("model_name") else None),
        model_version=str(training_settings.get("model_version", "v1")),
        experiment_tracking=experiment_tracking,
        config_path=config.config_path,
    )

    report, passed = _run_stage("training", lambda: run_training(training_config))
    pipeline_report["stages"]["training"] = report
    if not passed:
        pipeline_report["failed_stage"] = "training"
        return pipeline_report

    trained_model_path = _path_from_setting(report.get("output_model"), training_config.output_model)

    evaluation_config = build_evaluation_config(
        input_features=_path_from_setting(
            evaluation_settings.get("input_features"),
            training_config.input_features,
        ),
        input_model=_path_from_setting(
            evaluation_settings.get("input_model"),
            trained_model_path,
        ),
        output_report=_path_from_setting(
            evaluation_settings.get("output_report"),
            DEFAULT_EVALUATION_REPORT_PATH,
        ),
        experiment_tracking=experiment_tracking,
    )

    report, passed = _run_stage("evaluation", lambda: run_evaluation(evaluation_config))
    pipeline_report["stages"]["evaluation"] = report
    if not passed:
        pipeline_report["failed_stage"] = "evaluation"
        return pipeline_report

    pipeline_report["passed"] = True
    return pipeline_report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the full training pipeline from data ingestion through model evaluation."
    )
    parser.add_argument("--config", default=str(DEFAULT_CONFIG_PATH), help="Path to the YAML config file.")
    parser.add_argument("--dataset-root", help="Optional dataset root override for ingestion.")
    parser.add_argument(
        "--download",
        action="store_true",
        help="Download the dataset using KaggleHub before ingestion.",
    )
    parser.add_argument(
        "--dataset-handle",
        default=DEFAULT_DATASET_HANDLE,
        help="KaggleHub dataset handle to use when --download is provided.",
    )
    parser.add_argument("--log-level", default="INFO", help="Logging level, for example DEBUG, INFO, WARNING.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    configure_logging(args.log_level)
    config = build_pipeline_config(
        config_path=Path(args.config),
        dataset_root=Path(args.dataset_root) if args.dataset_root else None,
        download=args.download,
        dataset_handle=args.dataset_handle,
        log_level=args.log_level,
    )
    report = run_training_pipeline(config)
    if report["passed"]:
        LOGGER.info("Training pipeline completed successfully")
        return

    LOGGER.error("Training pipeline failed at stage '%s'", report.get("failed_stage") or "unknown")
    LOGGER.error("Pipeline summary: %s", json.dumps(report, indent=2, sort_keys=True))
    raise SystemExit(1)


if __name__ == "__main__":
    main()
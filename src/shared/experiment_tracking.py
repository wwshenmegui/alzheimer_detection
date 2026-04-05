from __future__ import annotations

import json
import shutil
import subprocess
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from uuid import uuid4

from .model_registry import load_model_metadata, write_json_file


DEFAULT_EXPERIMENTS_DIR = Path("data/experiments")
DEFAULT_EXPERIMENT_NAME = "alzheimer_detection"
DEFAULT_REMOTE_BACKEND = "mlflow"


@dataclass
class RemoteTrackingConfig:
    enabled: bool = False
    backend: str = DEFAULT_REMOTE_BACKEND
    tracking_uri: str | None = None
    experiment_name: str = DEFAULT_EXPERIMENT_NAME
    artifact_location: str | None = None


@dataclass
class ExperimentTrackingConfig:
    enabled: bool = False
    run_id: str | None = None
    local_runs_dir: Path = DEFAULT_EXPERIMENTS_DIR
    update_latest_reports: bool = True
    save_config_snapshot: bool = True
    save_stage_artifacts: bool = True
    remote: RemoteTrackingConfig = field(default_factory=RemoteTrackingConfig)


@dataclass
class ExperimentRun:
    run_id: str
    run_dir: Path
    reports_dir: Path
    artifacts_dir: Path
    metadata_path: Path
    config_snapshot_path: Path
    index_path: Path
    remote_run_id: str | None = None


def load_experiment_tracking_settings(config_path: Path) -> dict[str, Any]:
    if not config_path.exists():
        return {}

    import importlib

    try:
        yaml = importlib.import_module("yaml")
    except ImportError as exc:
        raise RuntimeError(
            "PyYAML is required for config loading. Install it with 'pip install pyyaml'."
        ) from exc

    with config_path.open("r", encoding="utf-8") as handle:
        config_data = yaml.safe_load(handle) or {}

    experiment_settings = config_data.get("experiment_tracking", {})
    if not isinstance(experiment_settings, dict):
        raise ValueError("The 'experiment_tracking' section in the config file must be a mapping.")
    return experiment_settings


def build_experiment_tracking_config(
    settings: dict[str, Any] | None = None,
) -> ExperimentTrackingConfig:
    settings = settings or {}
    remote_settings = settings.get("remote", {}) or {}
    return ExperimentTrackingConfig(
        enabled=bool(settings.get("enabled", False)),
        run_id=str(settings.get("run_id")) if settings.get("run_id") else None,
        local_runs_dir=Path(str(settings.get("local_runs_dir", DEFAULT_EXPERIMENTS_DIR))),
        update_latest_reports=bool(settings.get("update_latest_reports", True)),
        save_config_snapshot=bool(settings.get("save_config_snapshot", True)),
        save_stage_artifacts=bool(settings.get("save_stage_artifacts", True)),
        remote=RemoteTrackingConfig(
            enabled=bool(remote_settings.get("enabled", False)),
            backend=str(remote_settings.get("backend", DEFAULT_REMOTE_BACKEND)),
            tracking_uri=str(remote_settings.get("tracking_uri")) if remote_settings.get("tracking_uri") else None,
            experiment_name=str(remote_settings.get("experiment_name", DEFAULT_EXPERIMENT_NAME)),
            artifact_location=str(remote_settings.get("artifact_location")) if remote_settings.get("artifact_location") else None,
        ),
    )


def generate_run_id() -> str:
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    return f"{timestamp}_{uuid4().hex[:6]}"


def initialize_experiment_run(
    config: ExperimentTrackingConfig,
    *,
    config_snapshot_source: Path | None = None,
    project_name: str = DEFAULT_EXPERIMENT_NAME,
) -> ExperimentRun:
    run_id = config.run_id or generate_run_id()
    run_dir = config.local_runs_dir / run_id
    reports_dir = run_dir / "reports"
    artifacts_dir = run_dir / "artifacts"
    metadata_path = run_dir / "run_metadata.json"
    config_snapshot_path = run_dir / "config_snapshot.yaml"
    index_path = config.local_runs_dir / "index.json"

    reports_dir.mkdir(parents=True, exist_ok=True)
    artifacts_dir.mkdir(parents=True, exist_ok=True)

    if config.save_config_snapshot and config_snapshot_source and config_snapshot_source.exists():
        shutil.copy2(config_snapshot_source, config_snapshot_path)

    metadata = {
        "run_id": run_id,
        "status": "running",
        "started_at": utc_now(),
        "finished_at": None,
        "duration_seconds": None,
        "project": project_name,
        "git": collect_git_metadata(config_snapshot_source.parent if config_snapshot_source else Path.cwd()),
        "config_snapshot_path": str(config_snapshot_path) if config_snapshot_path.exists() else "",
        "stages": {},
        "dataset": {},
        "model": {},
        "remote_tracking": {
            "enabled": config.remote.enabled,
            "backend": config.remote.backend if config.remote.enabled else None,
            "run_id": None,
            "url": config.remote.tracking_uri,
        },
        "tags": [],
        "notes": None,
    }
    write_json_file(metadata, metadata_path)
    update_run_index(index_path, metadata_path)
    return ExperimentRun(
        run_id=run_id,
        run_dir=run_dir,
        reports_dir=reports_dir,
        artifacts_dir=artifacts_dir,
        metadata_path=metadata_path,
        config_snapshot_path=config_snapshot_path,
        index_path=index_path,
    )


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def collect_git_metadata(repo_path: Path) -> dict[str, Any]:
    try:
        commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=repo_path, text=True).strip()
        branch = subprocess.check_output(["git", "rev-parse", "--abbrev-ref", "HEAD"], cwd=repo_path, text=True).strip()
        dirty = bool(subprocess.check_output(["git", "status", "--porcelain"], cwd=repo_path, text=True).strip())
        return {"commit": commit, "branch": branch, "dirty": dirty}
    except Exception:
        return {"commit": None, "branch": None, "dirty": None}


def read_run_metadata(metadata_path: Path) -> dict[str, Any]:
    with metadata_path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def update_run_metadata(metadata_path: Path, updates: dict[str, Any]) -> dict[str, Any]:
    metadata = read_run_metadata(metadata_path)
    _deep_update(metadata, updates)
    write_json_file(metadata, metadata_path)
    update_run_index(metadata_path.parent.parent / "index.json", metadata_path)
    return metadata


def copy_into_run(source_path: Path, destination_path: Path) -> Path:
    destination_path.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source_path, destination_path)
    return destination_path


def capture_stage_file(
    run: ExperimentRun,
    *,
    source_path: Path | None,
    destination_group: str,
    destination_name: str,
) -> str | None:
    if source_path is None or not source_path.exists():
        return None
    base_dir = run.reports_dir if destination_group == "reports" else run.artifacts_dir
    copied_path = copy_into_run(source_path, base_dir / destination_name)
    return str(copied_path)


def record_stage(
    run: ExperimentRun,
    *,
    stage_name: str,
    status: str,
    summary: dict[str, Any],
    report_path: str | None = None,
    artifacts: dict[str, str] | None = None,
) -> dict[str, Any]:
    return update_run_metadata(
        run.metadata_path,
        {
            "stages": {
                stage_name: {
                    "status": status,
                    "report_path": report_path or "",
                    "artifacts": artifacts or {},
                    "summary": summary,
                }
            }
        },
    )


def finalize_run(run: ExperimentRun, *, status: str, model: dict[str, Any] | None = None, dataset: dict[str, Any] | None = None) -> dict[str, Any]:
    metadata = read_run_metadata(run.metadata_path)
    started_at = metadata.get("started_at")
    duration_seconds = None
    if started_at:
        duration_seconds = (datetime.now(timezone.utc) - datetime.fromisoformat(str(started_at))).total_seconds()

    return update_run_metadata(
        run.metadata_path,
        {
            "status": status,
            "finished_at": utc_now(),
            "duration_seconds": duration_seconds,
            "model": model or metadata.get("model", {}),
            "dataset": dataset or metadata.get("dataset", {}),
        },
    )


def update_run_index(index_path: Path, metadata_path: Path) -> None:
    index_path.parent.mkdir(parents=True, exist_ok=True)
    metadata = read_run_metadata(metadata_path)
    entry = {
        "run_id": metadata.get("run_id"),
        "status": metadata.get("status"),
        "started_at": metadata.get("started_at"),
        "finished_at": metadata.get("finished_at"),
        "model_name": metadata.get("model", {}).get("model_name"),
        "model_version": metadata.get("model", {}).get("model_version"),
        "training_validation_accuracy": metadata.get("stages", {}).get("training", {}).get("summary", {}).get("validation_accuracy"),
        "evaluation_accuracy": metadata.get("stages", {}).get("evaluation", {}).get("summary", {}).get("accuracy"),
        "run_metadata_path": str(metadata_path),
    }
    if index_path.exists():
        with index_path.open("r", encoding="utf-8") as handle:
            index = json.load(handle)
    else:
        index = {"runs": []}
    runs = [run for run in index.get("runs", []) if run.get("run_id") != entry["run_id"]]
    runs.append(entry)
    runs.sort(key=lambda item: item.get("started_at") or "", reverse=True)
    write_json_file({"runs": runs}, index_path)


def capture_configured_stage_outputs(run: ExperimentRun, config_path: Path, *, save_stage_artifacts: bool) -> dict[str, Any]:
    settings = load_full_config(config_path)
    stage_specs = {
        "ingestion": {
            "report": settings.get("ingestion", {}).get("output_report"),
            "summary_keys": ["ingested_rows", "mri_invalid_rows", "duplicate_summary"],
            "artifacts": {
                "manifest_path": settings.get("ingestion", {}).get("output_manifest"),
                "duplicate_report_path": settings.get("ingestion", {}).get("duplicate_report"),
            },
        },
        "validation": {
            "report": settings.get("validation", {}).get("output_report"),
            "summary_keys": ["passed", "total_rows", "mri_invalid_rows", "duplicate_summary"],
            "artifacts": {
                "validated_manifest_path": settings.get("validation", {}).get("approved_manifest"),
            },
        },
        "features": {
            "report": settings.get("features", {}).get("output_report"),
            "summary_keys": ["passed", "total_rows", "image_shape", "split_distribution", "group_split_distribution"],
            "artifacts": {
                "features_path": settings.get("features", {}).get("output_features"),
            },
        },
    }
    captured: dict[str, Any] = {}
    for stage_name, spec in stage_specs.items():
        report_source = Path(str(spec["report"])) if spec.get("report") else None
        report_payload = load_json_if_exists(report_source)
        copied_report = capture_stage_file(
            run,
            source_path=report_source,
            destination_group="reports",
            destination_name=report_source.name if report_source else f"{stage_name}_report.json",
        )
        copied_artifacts: dict[str, str] = {}
        if save_stage_artifacts:
            for artifact_key, artifact_value in spec.get("artifacts", {}).items():
                artifact_source = Path(str(artifact_value)) if artifact_value else None
                copied = capture_stage_file(
                    run,
                    source_path=artifact_source,
                    destination_group="artifacts",
                    destination_name=artifact_source.name if artifact_source else artifact_key,
                )
                if copied:
                    copied_artifacts[artifact_key] = copied
        summary = summarize_report(report_payload, spec.get("summary_keys", []))
        if report_payload or copied_artifacts:
            record_stage(
                run,
                stage_name=stage_name,
                status="completed" if report_payload and report_payload.get("passed", True) else "captured",
                summary=summary,
                report_path=copied_report,
                artifacts=copied_artifacts,
            )
            captured[stage_name] = {"report_path": copied_report or "", "artifacts": copied_artifacts, "summary": summary}
    return captured


def log_remote_training_run(
    config: RemoteTrackingConfig,
    *,
    run_id: str,
    params: dict[str, Any],
    metrics: dict[str, float],
    artifact_paths: list[Path],
    tags: dict[str, str] | None = None,
) -> str:
    mlflow = _load_mlflow()
    if config.tracking_uri:
        mlflow.set_tracking_uri(config.tracking_uri)
    mlflow.set_experiment(config.experiment_name)
    with mlflow.start_run(run_name=run_id) as run:
        mlflow.log_params({key: value for key, value in params.items() if value is not None})
        mlflow.log_metrics(metrics)
        if tags:
            mlflow.set_tags(tags)
        for artifact_path in artifact_paths:
            if artifact_path.exists():
                mlflow.log_artifact(str(artifact_path))
        return str(run.info.run_id)


def log_remote_evaluation_run(
    config: RemoteTrackingConfig,
    *,
    remote_run_id: str,
    metrics: dict[str, float],
    artifact_paths: list[Path],
) -> None:
    mlflow = _load_mlflow()
    if config.tracking_uri:
        mlflow.set_tracking_uri(config.tracking_uri)
    with mlflow.start_run(run_id=remote_run_id):
        mlflow.log_metrics(metrics)
        for artifact_path in artifact_paths:
            if artifact_path.exists():
                mlflow.log_artifact(str(artifact_path))


def get_run_from_model_metadata(metadata_path: Path | None) -> tuple[str | None, Path | None, str | None]:
    metadata = load_model_metadata(metadata_path)
    if not metadata:
        return None, None, None
    return (
        metadata.get("experiment_run_id"),
        Path(str(metadata["experiment_run_metadata_path"])) if metadata.get("experiment_run_metadata_path") else None,
        metadata.get("remote_tracking", {}).get("run_id"),
    )


def load_full_config(config_path: Path) -> dict[str, Any]:
    import importlib

    try:
        yaml = importlib.import_module("yaml")
    except ImportError as exc:
        raise RuntimeError(
            "PyYAML is required for config loading. Install it with 'pip install pyyaml'."
        ) from exc

    with config_path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle) or {}


def load_json_if_exists(file_path: Path | None) -> dict[str, Any] | None:
    if file_path is None or not file_path.exists():
        return None
    with file_path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def summarize_report(report: dict[str, Any] | None, summary_keys: list[str]) -> dict[str, Any]:
    if not report:
        return {}
    return {key: report.get(key) for key in summary_keys if key in report}


def _load_mlflow():
    import importlib

    try:
        return importlib.import_module("mlflow")
    except ImportError as exc:
        raise RuntimeError(
            "MLflow is required for remote experiment tracking. Install it with 'pip install mlflow'."
        ) from exc


def _deep_update(target: dict[str, Any], updates: dict[str, Any]) -> None:
    for key, value in updates.items():
        if isinstance(value, dict) and isinstance(target.get(key), dict):
            _deep_update(target[key], value)
        else:
            target[key] = value
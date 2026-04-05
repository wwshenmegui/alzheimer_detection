from __future__ import annotations

import json
import pickle
import sys
from pathlib import Path

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from shared.experiment_tracking import ExperimentTrackingConfig
from training.models.train import (
    build_training_config,
    flatten_images,
    load_training_settings,
    run_training,
)


def create_feature_artifact(file_path: Path) -> None:
    file_path.parent.mkdir(parents=True, exist_ok=True)
    images = []
    labels = []
    sample_ids = []
    splits = []

    for class_id in range(4):
        for sample_index in range(4):
            base_value = class_id / 3.0
            image = np.full((8, 8, 1), fill_value=base_value + (sample_index * 0.01), dtype=np.float32)
            images.append(image)
            labels.append(class_id)
            sample_ids.append(f"sample_{class_id}_{sample_index}")
            if sample_index in (0, 1):
                splits.append("train")
            elif sample_index == 2:
                splits.append("validation")
            else:
                splits.append("test")

    np.savez_compressed(
        file_path,
        images=np.stack(images),
        labels=np.asarray(labels, dtype=np.int64),
        sample_ids=np.asarray(sample_ids),
        splits=np.asarray(splits),
    )


def test_flatten_images_returns_2d_matrix() -> None:
    images = np.zeros((3, 4, 4, 1), dtype=np.float32)

    flattened = flatten_images(images)

    assert flattened.shape == (3, 16)


def test_run_training_creates_model_and_report(tmp_path: Path) -> None:
    features_path = tmp_path / "features.npz"
    create_feature_artifact(features_path)
    config_path = tmp_path / "training.yaml"
    config_path.write_text(
        """
ingestion:
  output_manifest: data/processed/manifest.csv
  output_report: data/reports/ingestion_summary.json
  duplicate_report: data/reports/duplicate_report.csv
validation:
  approved_manifest: data/processed/validated_manifest.csv
  output_report: data/reports/validation_report.json
features:
  output_features: data/processed/features.npz
  output_report: data/reports/features_report.json
""".strip(),
        encoding="utf-8",
    )

    output_model = tmp_path / "logistic_regression.pkl"
    output_report = tmp_path / "training_report.json"
    config = build_training_config(
        input_features=features_path,
        output_model=output_model,
        output_report=output_report,
        max_iter=300,
        experiment_tracking=ExperimentTrackingConfig(enabled=True, local_runs_dir=tmp_path / "experiments"),
        config_path=config_path,
    )

    report = run_training(config)

    assert report["passed"] is True
    assert report["model_version"] == "v1"
    assert report["model_name"] == "logistic_regression"
    assert report["train_rows"] == 8
    assert report["validation_rows"] == 4
    assert report["num_features"] == 64
    saved_model_path = Path(report["output_model"])
    metadata_path = Path(report["output_metadata"])
    current_pointer = Path(report["current_pointer"])
    assert saved_model_path.exists()
    assert metadata_path.exists()
    assert current_pointer.exists()
    assert output_report.exists()

    with saved_model_path.open("rb") as handle:
        model = pickle.load(handle)
    assert sorted(model.classes_.tolist()) == [0, 1, 2, 3]

    saved_report = json.loads(output_report.read_text(encoding="utf-8"))
    assert saved_report["passed"] is True

    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    assert metadata["model_version"] == "v1"
    assert metadata["model_name"] == "logistic_regression"
    assert metadata["lineage"]["feature_artifact_path"] == str(features_path)
    assert metadata["lineage"]["feature_artifact_sha256"]
    assert metadata["lineage"]["model_artifact_sha256"]
    assert metadata["lineage"]["training_report_path"] == str(output_report)
    assert metadata["lineage"]["training_report_sha256"]
    assert metadata["experiment_run_id"]
    run_metadata_path = Path(metadata["experiment_run_metadata_path"])
    assert run_metadata_path.exists()
    run_metadata = json.loads(run_metadata_path.read_text(encoding="utf-8"))
    assert run_metadata["status"] == "completed"
    assert run_metadata["stages"]["training"]["summary"]["model_version"] == "v1"


def test_run_training_writes_experiment_index(tmp_path: Path) -> None:
    features_path = tmp_path / "features.npz"
    create_feature_artifact(features_path)
    config = build_training_config(
        input_features=features_path,
        output_model=tmp_path / "logistic_regression.pkl",
        output_report=tmp_path / "training_report.json",
        experiment_tracking=ExperimentTrackingConfig(enabled=True, local_runs_dir=tmp_path / "experiments"),
    )

    report = run_training(config)

    assert report["passed"] is True
    index_path = tmp_path / "experiments" / "index.json"
    assert index_path.exists()
    index = json.loads(index_path.read_text(encoding="utf-8"))
    assert len(index["runs"]) == 1
    assert index["runs"][0]["run_id"]


def test_run_training_fails_for_missing_features(tmp_path: Path) -> None:
    output_report = tmp_path / "training_report.json"
    config = build_training_config(
        input_features=tmp_path / "missing.npz",
        output_model=tmp_path / "model.pkl",
        output_report=output_report,
    )

    report = run_training(config)

    assert report["passed"] is False
    assert "Feature artifact does not exist" in report["errors"][0]


def test_load_training_settings_reads_yaml_config(tmp_path: Path, monkeypatch) -> None:
    config_path = tmp_path / "training.yaml"
    config_path.write_text(
        """
model:
  input_features: data/processed/features.npz
  output_model: models/trained/logistic_regression.pkl
  output_report: data/reports/training_report.json
  max_iter: 500
""".strip(),
        encoding="utf-8",
    )

    import yaml

    monkeypatch.setitem(sys.modules, "yaml", yaml)

    settings = load_training_settings(config_path)

    assert settings["input_features"] == "data/processed/features.npz"
    assert settings["output_model"] == "models/trained/logistic_regression.pkl"
    assert settings["output_report"] == "data/reports/training_report.json"
    assert settings["max_iter"] == 500

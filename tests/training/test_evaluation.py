from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from shared.experiment_tracking import ExperimentTrackingConfig
from training.evaluation.evaluate import (
    build_evaluation_config,
    load_evaluation_settings,
    run_evaluation,
)
from training.models.train import build_training_config, run_training


def create_feature_artifact(file_path: Path) -> None:
    file_path.parent.mkdir(parents=True, exist_ok=True)
    images = []
    labels = []
    sample_ids = []
    splits = []

    for class_id in range(4):
        for sample_index in range(5):
            base_value = class_id / 3.0
            image = np.full((8, 8, 1), fill_value=base_value + (sample_index * 0.01), dtype=np.float32)
            images.append(image)
            labels.append(class_id)
            sample_ids.append(f"sample_{class_id}_{sample_index}")
            if sample_index in (0, 1, 2):
                splits.append("train")
            elif sample_index == 3:
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


def train_test_model(tmp_path: Path) -> tuple[Path, Path]:
    features_path = tmp_path / "features.npz"
    create_feature_artifact(features_path)
    model_path = tmp_path / "model.pkl"
    training_report = tmp_path / "training_report.json"
    training_config = build_training_config(
        input_features=features_path,
        output_model=model_path,
        output_report=training_report,
        max_iter=300,
        experiment_tracking=ExperimentTrackingConfig(enabled=True, local_runs_dir=tmp_path / "experiments"),
    )
    report = run_training(training_config)
    assert report["passed"] is True
    return features_path, model_path


def test_run_evaluation_creates_report(tmp_path: Path) -> None:
    features_path, model_path = train_test_model(tmp_path)
    report_path = tmp_path / "evaluation_report.json"
    config = build_evaluation_config(
        input_features=features_path,
        input_model=model_path,
        output_report=report_path,
        experiment_tracking=ExperimentTrackingConfig(enabled=True, local_runs_dir=tmp_path / "experiments"),
    )

    report = run_evaluation(config)

    assert report["passed"] is True
    assert report["test_rows"] == 4
    assert set(report.keys()) >= {
        "accuracy",
        "precision_macro",
        "recall_macro",
        "f1_macro",
        "confusion_matrix",
        "class_ids",
    }
    assert report_path.exists()
    saved_report = json.loads(report_path.read_text(encoding="utf-8"))
    assert saved_report["passed"] is True
    assert len(saved_report["confusion_matrix"]) == len(saved_report["class_ids"])

    metadata_path = tmp_path / "model" / "v1" / "metadata.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    assert metadata["evaluation_report_path"] == str(report_path)
    assert metadata["lineage"]["evaluation_report_path"] == str(report_path)
    assert metadata["lineage"]["evaluation_report_sha256"]
    run_metadata = json.loads(Path(metadata["experiment_run_metadata_path"]).read_text(encoding="utf-8"))
    assert run_metadata["stages"]["evaluation"]["summary"]["accuracy"] == report["accuracy"]


def test_run_evaluation_fails_for_missing_model(tmp_path: Path) -> None:
    features_path = tmp_path / "features.npz"
    create_feature_artifact(features_path)
    report_path = tmp_path / "evaluation_report.json"
    config = build_evaluation_config(
        input_features=features_path,
        input_model=tmp_path / "missing.pkl",
        output_report=report_path,
    )

    report = run_evaluation(config)

    assert report["passed"] is False
    assert "Model file does not exist" in report["errors"][0]


def test_run_evaluation_fails_for_missing_features(tmp_path: Path) -> None:
    report_path = tmp_path / "evaluation_report.json"
    config = build_evaluation_config(
        input_features=tmp_path / "missing.npz",
        input_model=tmp_path / "model.pkl",
        output_report=report_path,
    )

    report = run_evaluation(config)

    assert report["passed"] is False
    assert "Feature artifact does not exist" in report["errors"][0]


def test_load_evaluation_settings_reads_yaml_config(tmp_path: Path, monkeypatch) -> None:
    config_path = tmp_path / "training.yaml"
    config_path.write_text(
        """
evaluation:
  input_features: data/processed/features.npz
  input_model: models/trained/logistic_regression.pkl
  output_report: data/reports/evaluation_report.json
""".strip(),
        encoding="utf-8",
    )

    import yaml

    monkeypatch.setitem(sys.modules, "yaml", yaml)

    settings = load_evaluation_settings(config_path)

    assert settings["input_features"] == "data/processed/features.npz"
    assert settings["input_model"] == "models/trained/logistic_regression.pkl"
    assert settings["output_report"] == "data/reports/evaluation_report.json"


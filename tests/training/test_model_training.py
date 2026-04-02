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

    output_model = tmp_path / "logistic_regression.pkl"
    output_report = tmp_path / "training_report.json"
    config = build_training_config(
        input_features=features_path,
        output_model=output_model,
        output_report=output_report,
        max_iter=300,
    )

    report = run_training(config)

    assert report["passed"] is True
    assert report["train_rows"] == 8
    assert report["validation_rows"] == 4
    assert report["num_features"] == 64
    assert output_model.exists()
    assert output_report.exists()

    with output_model.open("rb") as handle:
        model = pickle.load(handle)
    assert sorted(model.classes_.tolist()) == [0, 1, 2, 3]

    saved_report = json.loads(output_report.read_text(encoding="utf-8"))
    assert saved_report["passed"] is True


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

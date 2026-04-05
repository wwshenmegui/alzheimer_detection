from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image


PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from training.pipeline.run_pipeline import build_pipeline_config, run_training_pipeline


def create_test_image(file_path: Path, seed: int) -> None:
    rng = np.random.default_rng(seed)
    image = rng.integers(0, 255, size=(16, 16), dtype=np.uint8)
    file_path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(image).save(file_path, format="PNG")


def write_pipeline_config(config_path: Path, dataset_root: Path) -> None:
    config_path.write_text(
        f"""
ingestion:
  dataset_root: {dataset_root}
  output_manifest: {config_path.parent / 'data/processed/manifest.csv'}
  output_report: {config_path.parent / 'data/reports/ingestion_summary.json'}
  duplicate_report: {config_path.parent / 'data/reports/duplicate_report.csv'}
  download: false
  min_image_size:
    - 8
    - 8
  aspect_ratio_range:
    - 0.5
    - 2.0
  min_stddev: 0.0
  min_center_border_diff: 0.0
validation:
  output_report: {config_path.parent / 'data/reports/validation_report.json'}
  approved_manifest: {config_path.parent / 'data/processed/validated_manifest.csv'}
features:
  output_features: {config_path.parent / 'data/processed/features.npz'}
  output_report: {config_path.parent / 'data/reports/features_report.json'}
  image_size:
    - 8
    - 8
  split_ratios:
    - 0.6
    - 0.2
    - 0.2
  split_random_state: 42
model:
  output_model: {config_path.parent / 'models/trained/logistic_regression.pkl'}
  output_report: {config_path.parent / 'data/reports/training_report.json'}
  max_iter: 200
  model_name: test_detector
  model_version: v-test
evaluation:
  output_report: {config_path.parent / 'data/reports/evaluation_report.json'}
experiment_tracking:
  enabled: false
""".strip(),
        encoding="utf-8",
    )


def test_run_training_pipeline_executes_all_stages(tmp_path: Path) -> None:
    dataset_root = tmp_path / "dataset"
    labels = ["NonDemented", "VeryMildDemented", "MildDemented", "ModerateDemented"]
    for label_index, label_name in enumerate(labels):
        for sample_index in range(8):
            create_test_image(
                dataset_root / label_name / f"scan_{sample_index}.png",
                seed=(label_index * 10) + sample_index,
            )

    config_path = tmp_path / "training.yaml"
    write_pipeline_config(config_path, dataset_root)
    pipeline_config = build_pipeline_config(config_path=config_path)

    report = run_training_pipeline(pipeline_config)

    assert report["passed"] is True
    assert report["failed_stage"] is None
    assert list(report["stages"].keys()) == ["ingestion", "validation", "features", "training", "evaluation"]
    assert report["stages"]["ingestion"]["ingested_rows"] == 32
    assert report["stages"]["validation"]["passed"] is True
    assert report["stages"]["features"]["passed"] is True
    assert report["stages"]["training"]["passed"] is True
    assert report["stages"]["evaluation"]["passed"] is True

    training_metadata_path = Path(report["stages"]["training"]["output_metadata"])
    evaluation_report_path = tmp_path / "data/reports/evaluation_report.json"
    assert training_metadata_path.exists()
    assert evaluation_report_path.exists()

    metadata = json.loads(training_metadata_path.read_text(encoding="utf-8"))
    evaluation_report = json.loads(evaluation_report_path.read_text(encoding="utf-8"))
    assert metadata["model_name"] == "test_detector"
    assert metadata["model_version"] == "v-test"
    assert evaluation_report["passed"] is True
    assert evaluation_report["test_rows"] > 0


def test_run_training_pipeline_stops_on_ingestion_failure(tmp_path: Path) -> None:
    config_path = tmp_path / "training.yaml"
    write_pipeline_config(config_path, tmp_path / "missing-dataset")
    pipeline_config = build_pipeline_config(config_path=config_path)

    report = run_training_pipeline(pipeline_config)

    assert report["passed"] is False
    assert report["failed_stage"] == "ingestion"
    assert report["stages"]["ingestion"]["passed"] is False
    assert "Dataset root does not exist" in report["stages"]["ingestion"]["errors"][0]
    assert "validation" not in report["stages"]


def test_run_training_pipeline_reuses_local_dataset_after_download(tmp_path: Path, monkeypatch) -> None:
    cached_download_root = tmp_path / "kaggle-cache"
    source_root = cached_download_root / "combined_images"
    local_dataset_root = tmp_path / "data" / "raw" / "alzheimers_multiclass"
    labels = ["NonDemented", "VeryMildDemented", "MildDemented", "ModerateDemented"]
    for label_index, label_name in enumerate(labels):
        for sample_index in range(8):
            create_test_image(
                source_root / label_name / f"scan_{sample_index}.png",
                seed=(label_index * 10) + sample_index,
            )

    config_path = tmp_path / "training.yaml"
    write_pipeline_config(config_path, local_dataset_root)

    def fake_download_dataset(_: str) -> Path:
        return cached_download_root

    monkeypatch.setattr("training.pipeline.run_pipeline.download_dataset", fake_download_dataset)

    first_report = run_training_pipeline(build_pipeline_config(config_path=config_path, download=True))
    assert first_report["passed"] is True
    assert (local_dataset_root / "NonDemented" / "scan_0.png").exists()

    second_report = run_training_pipeline(build_pipeline_config(config_path=config_path))
    assert second_report["passed"] is True
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

from training.features.build_features import (
    assign_splits,
    build_features_config,
    load_feature_settings,
    preprocess_image,
    run_feature_build,
)


def create_test_image(file_path: Path, size: tuple[int, int], color: int) -> None:
    file_path.parent.mkdir(parents=True, exist_ok=True)
    image = Image.new("L", size=size, color=color)
    image.save(file_path)


def write_manifest(file_path: Path, content: str) -> None:
    file_path.parent.mkdir(parents=True, exist_ok=True)
    file_path.write_text(content, encoding="utf-8")


def test_preprocess_image_returns_grayscale_tensor(tmp_path: Path) -> None:
    image_path = tmp_path / "images" / "scan.png"
    create_test_image(image_path, size=(32, 32), color=128)

    tensor = preprocess_image(image_path, (16, 16))

    assert tensor.shape == (16, 16, 1)
    assert tensor.dtype == np.float32
    assert float(tensor.min()) >= 0.0
    assert float(tensor.max()) <= 1.0


def test_run_feature_build_creates_npz_and_report(tmp_path: Path) -> None:
    manifest_path = tmp_path / "validated_manifest.csv"
    lines = ["sample_id,image_path,label_name,label_id"]
    sample_number = 1
    for label_name, label_id, color in [
        ("NonDemented", 0, 32),
        ("VeryMildDemented", 1, 96),
        ("MildDemented", 2, 160),
        ("ModerateDemented", 3, 224),
    ]:
        for class_index in range(5):
            image_path = tmp_path / "images" / label_name / f"scan_{class_index}.png"
            create_test_image(image_path, size=(10, 10), color=color + class_index)
            lines.append(f"sample_{sample_number:05d},{image_path},{label_name},{label_id}")
            sample_number += 1
    write_manifest(manifest_path, "\n".join(lines) + "\n")

    output_features = tmp_path / "features.npz"
    output_report = tmp_path / "features_report.json"
    config = build_features_config(
        validated_manifest=manifest_path,
        output_features=output_features,
        output_report=output_report,
        image_size=(8, 8),
    )

    report = run_feature_build(config)

    assert report["passed"] is True
    assert report["total_rows"] == 20
    assert report["image_shape"] == [8, 8, 1]
    assert report["class_distribution"] == {
        "NonDemented": 5,
        "VeryMildDemented": 5,
        "MildDemented": 5,
        "ModerateDemented": 5,
    }
    assert report["split_distribution"] == {"train": 12, "validation": 4, "test": 4}
    assert output_features.exists()

    loaded = np.load(output_features)
    assert loaded["images"].shape == (20, 8, 8, 1)
    assert loaded["labels"].shape == (20,)
    assert loaded["sample_ids"].shape == (20,)
    assert loaded["splits"].shape == (20,)


def test_assign_splits_creates_6_2_2_partition() -> None:
    labels = np.asarray([0] * 5 + [1] * 5 + [2] * 5 + [3] * 5, dtype=np.int64)

    splits = assign_splits(labels, (0.6, 0.2, 0.2), 42)

    unique, counts = np.unique(splits, return_counts=True)
    assert dict(zip(unique.tolist(), counts.tolist())) == {"test": 4, "train": 12, "validation": 4}


def test_run_feature_build_fails_for_missing_manifest(tmp_path: Path) -> None:
    output_report = tmp_path / "features_report.json"
    config = build_features_config(
        validated_manifest=tmp_path / "missing.csv",
        output_features=tmp_path / "features.npz",
        output_report=output_report,
        image_size=(8, 8),
    )

    report = run_feature_build(config)

    assert report["passed"] is False
    assert "Validated manifest file does not exist" in report["errors"][0]


def test_load_feature_settings_reads_yaml_config(tmp_path: Path, monkeypatch) -> None:
    config_path = tmp_path / "training.yaml"
    config_path.write_text(
        """
features:
  validated_manifest: data/processed/validated_manifest.csv
  output_features: data/processed/features.npz
  output_report: data/reports/features_report.json
  image_size:
    - 128
    - 128
""".strip(),
        encoding="utf-8",
    )

    import yaml

    monkeypatch.setitem(sys.modules, "yaml", yaml)

    settings = load_feature_settings(config_path)

    assert settings["validated_manifest"] == "data/processed/validated_manifest.csv"
    assert settings["output_features"] == "data/processed/features.npz"
    assert settings["output_report"] == "data/reports/features_report.json"
    assert settings["image_size"] == [128, 128]

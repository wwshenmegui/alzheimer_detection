from __future__ import annotations

import csv
import sys
import types
from pathlib import Path

from PIL import Image


PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from training.ingestion.ingest import (
    DEFAULT_DATASET_HANDLE,
    IngestionConfig,
    build_ingestion_config,
    build_manifest,
    download_dataset,
    load_ingestion_settings,
    resolve_dataset_root,
    save_duplicate_report,
    save_manifest,
)


def write_file(file_path: Path, content: bytes) -> None:
    file_path.parent.mkdir(parents=True, exist_ok=True)
    file_path.write_bytes(content)


def create_test_image(file_path: Path, *, image_format: str = "PNG", size: tuple[int, int] = (128, 128), color: int = 128) -> None:
    file_path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("L", size=size, color=color).save(file_path, format=image_format)


def test_build_manifest_includes_only_known_readable_images(tmp_path: Path) -> None:
    dataset_root = tmp_path / "dataset"
    create_test_image(dataset_root / "NonDemented" / "scan_1.png", image_format="PNG")
    create_test_image(dataset_root / "MildDemented" / "scan_2.jpg", image_format="JPEG")
    create_test_image(dataset_root / "UnknownLabel" / "scan_3.png", image_format="PNG")
    write_file(dataset_root / "VeryMildDemented" / "broken.png", b"not-an-image")

    config = IngestionConfig(
        dataset_root=dataset_root,
        output_manifest=tmp_path / "manifest.csv",
    )

    manifest = build_manifest(config)

    assert len(manifest) == 2
    assert manifest[0]["sample_id"] == "sample_00001"
    assert manifest[0]["label_name"] == "MildDemented"
    assert manifest[0]["label_id"] == 2
    assert manifest[1]["sample_id"] == "sample_00002"
    assert manifest[1]["label_name"] == "NonDemented"
    assert manifest[1]["label_id"] == 0
    assert manifest[0]["group_id"]
    assert manifest[1]["group_id"]


def test_save_manifest_writes_csv(tmp_path: Path) -> None:
    output_path = tmp_path / "reports" / "manifest.csv"
    rows = [
        {
            "sample_id": "sample_00001",
            "image_path": "/tmp/example.png",
            "label_name": "NonDemented",
            "label_id": 0,
        }
    ]

    save_manifest(rows, output_path)

    with output_path.open("r", newline="", encoding="utf-8") as handle:
        saved_rows = list(csv.DictReader(handle))

    assert saved_rows == [
        {
            "sample_id": "sample_00001",
            "image_path": "/tmp/example.png",
            "label_name": "NonDemented",
            "label_id": "0",
            "patient_id": "",
            "group_id": "",
            "duplicate_group_id": "",
            "width": "",
            "height": "",
            "mode": "",
            "image_format": "",
            "file_size_bytes": "",
            "mean_intensity": "",
            "std_intensity": "",
            "center_mean_intensity": "",
            "border_mean_intensity": "",
            "sha256": "",
            "average_hash": "",
            "mri_is_valid": "",
            "mri_error_code": "",
            "mri_message": "",
        }
    ]


def test_save_duplicate_report_ignores_extra_manifest_fields(tmp_path: Path) -> None:
    output_path = tmp_path / "reports" / "duplicates.csv"
    rows = [
        {
            "sample_id": "sample_00001",
            "image_path": "/tmp/example.png",
            "label_name": "NonDemented",
            "label_id": 0,
            "patient_id": "patient_1",
            "group_id": "patient_1",
            "duplicate_group_id": "exact_duplicate_00001",
            "sha256": "abc123",
            "average_hash": "0101",
            "width": 128,
            "height": 128,
            "mri_is_valid": "True",
        }
    ]

    save_duplicate_report(rows, output_path)

    with output_path.open("r", newline="", encoding="utf-8") as handle:
        saved_rows = list(csv.DictReader(handle))

    assert saved_rows == [
        {
            "sample_id": "sample_00001",
            "image_path": "/tmp/example.png",
            "label_name": "NonDemented",
            "patient_id": "patient_1",
            "duplicate_group_id": "exact_duplicate_00001",
            "group_id": "patient_1",
            "sha256": "abc123",
            "average_hash": "0101",
        }
    ]


def test_resolve_dataset_root_finds_combined_images_directory(tmp_path: Path) -> None:
    dataset_root = tmp_path / "downloaded"
    combined_images = dataset_root / "combined_images"
    for label_name in ("NonDemented", "VeryMildDemented", "MildDemented", "ModerateDemented"):
        (combined_images / label_name).mkdir(parents=True)

    resolved_root = resolve_dataset_root(
        dataset_root,
        {
            "NonDemented": 0,
            "VeryMildDemented": 1,
            "MildDemented": 2,
            "ModerateDemented": 3,
        },
    )

    assert resolved_root == combined_images


def test_download_dataset_uses_kagglehub(monkeypatch, tmp_path: Path) -> None:
    fake_download_path = tmp_path / "downloaded-dataset"

    def fake_dataset_download(dataset_handle: str) -> str:
        assert dataset_handle == DEFAULT_DATASET_HANDLE
        return str(fake_download_path)

    fake_module = types.SimpleNamespace(dataset_download=fake_dataset_download)
    monkeypatch.setitem(sys.modules, "kagglehub", fake_module)

    resolved_path = download_dataset()

    assert resolved_path == fake_download_path


def test_load_ingestion_settings_reads_yaml_config(tmp_path: Path, monkeypatch) -> None:
    config_path = tmp_path / "training.yaml"
    config_path.write_text(
        """
ingestion:
  dataset_root: data/raw/alzheimers_multiclass
  output_manifest: data/processed/manifest.csv
  allowed_extensions:
    - .jpg
    - .png
  label_to_id:
    NonDemented: 0
    VeryMildDemented: 1
    MildDemented: 2
    ModerateDemented: 3
""".strip(),
        encoding="utf-8",
    )

    import yaml

    monkeypatch.setitem(sys.modules, "yaml", yaml)

    settings = load_ingestion_settings(config_path)

    assert settings["dataset_root"] == "data/raw/alzheimers_multiclass"
    assert settings["output_manifest"] == "data/processed/manifest.csv"
    assert settings["allowed_extensions"] == [".jpg", ".png"]


def test_build_ingestion_config_requires_paths(tmp_path: Path) -> None:
    config = build_ingestion_config(
        dataset_root=tmp_path / "dataset",
        output_manifest=tmp_path / "manifest.csv",
    )

    assert config.dataset_root == tmp_path / "dataset"
    assert config.output_manifest == tmp_path / "manifest.csv"
    assert config.label_to_id == {
        "NonDemented": 0,
        "VeryMildDemented": 1,
        "MildDemented": 2,
        "ModerateDemented": 3,
    }
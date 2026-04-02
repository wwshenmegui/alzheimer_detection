from __future__ import annotations

import csv
import sys
import types
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from training.ingestion.ingest import (
    DEFAULT_CONFIG_PATH,
    DEFAULT_DATASET_HANDLE,
    IngestionConfig,
    build_ingestion_config,
    build_manifest,
    download_dataset,
    load_ingestion_settings,
    resolve_dataset_root,
    save_manifest,
)


MINIMAL_PNG = (
    b"\x89PNG\r\n\x1a\n"
    b"\x00\x00\x00\rIHDR"
    b"\x00\x00\x00\x01\x00\x00\x00\x01\x08\x02\x00\x00\x00"
    b"\x90wS\xde"
    b"\x00\x00\x00\x0cIDATx\x9cc``\x00\x00\x00\x02\x00\x01"
    b"\xe2!\xbc3"
    b"\x00\x00\x00\x00IEND\xaeB`\x82"
)


def write_file(file_path: Path, content: bytes) -> None:
    file_path.parent.mkdir(parents=True, exist_ok=True)
    file_path.write_bytes(content)


def test_build_manifest_includes_only_known_readable_images(tmp_path: Path) -> None:
    dataset_root = tmp_path / "dataset"
    write_file(dataset_root / "NonDemented" / "scan_1.png", MINIMAL_PNG)
    write_file(dataset_root / "MildDemented" / "scan_2.jpg", b"\xff\xd8\xfftest-jpeg")
    write_file(dataset_root / "UnknownLabel" / "scan_3.png", MINIMAL_PNG)
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
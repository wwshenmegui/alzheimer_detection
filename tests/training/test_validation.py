from __future__ import annotations

import json
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from training.validation.validate import (
    build_validation_config,
    load_validation_settings,
    run_validation,
)


def write_manifest(file_path: Path, content: str) -> None:
    file_path.parent.mkdir(parents=True, exist_ok=True)
    file_path.write_text(content, encoding="utf-8")


def test_run_validation_passes_for_valid_manifest(tmp_path: Path) -> None:
    image_path = tmp_path / "images" / "scan_1.png"
    image_path.parent.mkdir(parents=True, exist_ok=True)
    image_path.write_bytes(b"file")

    manifest_path = tmp_path / "manifest.csv"
    write_manifest(
        manifest_path,
        "sample_id,image_path,label_name,label_id,patient_id,group_id,duplicate_group_id,width,height,mode,image_format,file_size_bytes,mean_intensity,std_intensity,center_mean_intensity,border_mean_intensity,sha256,average_hash,mri_is_valid,mri_error_code,mri_message\n"
        f"sample_00001,{image_path},NonDemented,0,,sample_00001,,128,128,L,PNG,10,0.5,0.1,0.6,0.4,abc,hash,True,,\n",
    )

    report_path = tmp_path / "validation_report.json"
    approved_manifest = tmp_path / "validated_manifest.csv"
    config = build_validation_config(
        manifest_path=manifest_path,
        output_report=report_path,
        approved_manifest=approved_manifest,
    )

    report = run_validation(config)

    assert report["passed"] is True
    assert report["errors"] == []
    assert report["warnings"] == []
    assert report["total_rows"] == 1
    assert report["class_distribution"] == {"NonDemented": 1}
    assert approved_manifest.exists()


def test_run_validation_fails_for_missing_manifest(tmp_path: Path) -> None:
    report_path = tmp_path / "validation_report.json"
    config = build_validation_config(
        manifest_path=tmp_path / "missing.csv",
        output_report=report_path,
        approved_manifest=None,
    )

    report = run_validation(config)

    assert report["passed"] is False
    assert "Manifest file does not exist" in report["errors"][0]


def test_run_validation_fails_for_invalid_rows(tmp_path: Path) -> None:
    missing_image = tmp_path / "images" / "missing.png"
    manifest_path = tmp_path / "manifest.csv"
    write_manifest(
        manifest_path,
        "sample_id,image_path,label_name,label_id,patient_id,group_id,duplicate_group_id,width,height,mode,image_format,file_size_bytes,mean_intensity,std_intensity,center_mean_intensity,border_mean_intensity,sha256,average_hash,mri_is_valid,mri_error_code,mri_message\n"
        f"sample_00001,{missing_image},WrongLabel,0,,sample_00001,,128,128,L,PNG,10,0.5,0.1,0.6,0.4,abc,hash,True,,\n"
        f"sample_00001,{missing_image},NonDemented,9,,sample_00001,,128,128,L,PNG,10,0.5,0.1,0.6,0.4,def,hash2,False,image_too_small,too small\n",
    )

    report_path = tmp_path / "validation_report.json"
    config = build_validation_config(
        manifest_path=manifest_path,
        output_report=report_path,
        approved_manifest=None,
    )

    report = run_validation(config)

    assert report["passed"] is False
    assert any("Image path does not exist" in error for error in report["errors"])
    assert any("Invalid label_name 'WrongLabel'" in error for error in report["errors"])
    assert any("Duplicate sample_id found: sample_00001" in error for error in report["errors"])
    assert any("Label mismatch" in error for error in report["errors"])
    assert any("failed MRI validation" in warning for warning in report["warnings"])

    saved_report = json.loads(report_path.read_text(encoding="utf-8"))
    assert saved_report["passed"] is False


def test_load_validation_settings_reads_yaml_config(tmp_path: Path, monkeypatch) -> None:
    config_path = tmp_path / "training.yaml"
    config_path.write_text(
        """
validation:
  manifest_path: data/processed/manifest.csv
  output_report: data/reports/validation_report.json
  approved_manifest: data/processed/validated_manifest.csv
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

    settings = load_validation_settings(config_path)

    assert settings["manifest_path"] == "data/processed/manifest.csv"
    assert settings["output_report"] == "data/reports/validation_report.json"
    assert settings["approved_manifest"] == "data/processed/validated_manifest.csv"

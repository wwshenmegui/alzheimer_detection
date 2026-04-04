from __future__ import annotations

import io
import sys
from pathlib import Path

from fastapi.testclient import TestClient
from PIL import Image


PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from serving.api.app import create_app
from serving.inference.predictor import ModelPredictor, build_serving_config
from training.models.train import build_training_config, run_training


def create_feature_artifact(file_path: Path) -> None:
    import numpy as np

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


def train_versioned_model(tmp_path: Path, model_version: str) -> None:
    features_path = tmp_path / "features.npz"
    if not features_path.exists():
        create_feature_artifact(features_path)
    model_path = tmp_path / "model.pkl"
    train_config = build_training_config(
        input_features=features_path,
        output_model=model_path,
        output_report=tmp_path / "training_report.json",
        max_iter=300,
        model_version=model_version,
    )
    report = run_training(train_config)
    assert report["passed"] is True


def create_predictor(tmp_path: Path) -> ModelPredictor:
    train_versioned_model(tmp_path, "v1")
    model_path = tmp_path / "model.pkl"
    return ModelPredictor(build_serving_config(model_path=model_path, image_size=(8, 8), min_image_size=(8, 8)))


def create_png_bytes() -> bytes:
    image = Image.new("L", (8, 8), color=20)
    for x in range(2, 6):
        for y in range(2, 6):
            image.putpixel((x, y), 200)
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return buffer.getvalue()


def test_health_endpoint_returns_ok(tmp_path: Path) -> None:
    client = TestClient(create_app(predictor=create_predictor(tmp_path)))

    response = client.get("/health")

    assert response.status_code == 200
    payload = response.json()
    assert payload["status"] == "ok"
    assert payload["model_loaded"] is True
    assert payload["model_version"] == "v1"


def test_model_metadata_endpoint_returns_active_model_metadata(tmp_path: Path) -> None:
    client = TestClient(create_app(predictor=create_predictor(tmp_path)))

    response = client.get("/model")

    assert response.status_code == 200
    payload = response.json()["metadata"]
    assert payload["model_version"] == "v1"
    assert payload["model_name"] == "model"


def test_models_endpoint_lists_registered_versions(tmp_path: Path) -> None:
    client = TestClient(create_app(predictor=create_predictor(tmp_path)))

    response = client.get("/models")

    assert response.status_code == 200
    payload = response.json()
    assert payload["active_model_version"] == "v1"
    assert len(payload["models"]) == 1
    assert payload["models"][0]["model_version"] == "v1"


def test_activate_model_endpoint_switches_active_version(tmp_path: Path) -> None:
    train_versioned_model(tmp_path, "v1")
    train_versioned_model(tmp_path, "v2")
    client = TestClient(create_app(predictor=create_predictor(tmp_path)))

    response = client.post("/model/activate", json={"model_version": "v2"})

    assert response.status_code == 200
    payload = response.json()
    assert payload["activated_model_version"] == "v2"
    assert payload["metadata"]["model_version"] == "v2"

    health_response = client.get("/health")
    assert health_response.status_code == 200
    assert health_response.json()["model_version"] == "v2"


def test_activate_model_endpoint_rejects_missing_version(tmp_path: Path) -> None:
    client = TestClient(create_app(predictor=create_predictor(tmp_path)))

    response = client.post("/model/activate", json={"model_version": "v9"})

    assert response.status_code == 404
    assert "does not exist" in response.json()["detail"]


def test_root_endpoint_returns_ui_page(tmp_path: Path) -> None:
    client = TestClient(create_app(predictor=create_predictor(tmp_path)))

    response = client.get("/")

    assert response.status_code == 200
    assert "Alzheimer Detection" in response.text
    assert "Please upload an MRI image" in response.text
    assert "Image preview" in response.text
    assert "Predict" in response.text


def test_predict_endpoint_returns_prediction(tmp_path: Path) -> None:
    client = TestClient(create_app(predictor=create_predictor(tmp_path)))

    response = client.post(
        "/predict",
        files={"file": ("scan.png", create_png_bytes(), "image/png")},
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["predicted_label"] in {
        "NonDemented",
        "VeryMildDemented",
        "MildDemented",
        "ModerateDemented",
    }
    assert set(payload["probabilities"].keys()) == {
        "NonDemented",
        "VeryMildDemented",
        "MildDemented",
        "ModerateDemented",
    }
    assert payload["model_version"] == "v1"


def test_predict_endpoint_rejects_non_image(tmp_path: Path) -> None:
    client = TestClient(create_app(predictor=create_predictor(tmp_path)))

    response = client.post(
        "/predict",
        files={"file": ("notes.txt", b"not an image", "text/plain")},
    )

    assert response.status_code == 400
    payload = response.json()
    assert payload["error_code"] == "unsupported_file_type"
    assert "Please upload" in payload["user_action"]


def test_predict_endpoint_reports_size_requirement(tmp_path: Path) -> None:
    client = TestClient(create_app(predictor=create_predictor(tmp_path)))
    tiny = Image.new("L", (4, 4), color=200)
    buffer = io.BytesIO()
    tiny.save(buffer, format="PNG")

    response = client.post(
        "/predict",
        files={"file": ("tiny.png", buffer.getvalue(), "image/png")},
    )

    assert response.status_code == 400
    payload = response.json()
    assert payload["error_code"] == "image_too_small"
    assert "at least" in payload["user_action"]
    assert payload["details"]["minimum_width"] == 8

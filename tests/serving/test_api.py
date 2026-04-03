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


def create_predictor(tmp_path: Path) -> ModelPredictor:
    features_path = tmp_path / "features.npz"
    create_feature_artifact(features_path)
    model_path = tmp_path / "model.pkl"
    train_config = build_training_config(
        input_features=features_path,
        output_model=model_path,
        output_report=tmp_path / "training_report.json",
        max_iter=300,
    )
    report = run_training(train_config)
    assert report["passed"] is True
    return ModelPredictor(build_serving_config(model_path=model_path, image_size=(8, 8)))


def create_png_bytes() -> bytes:
    image = Image.new("L", (8, 8), color=200)
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


def test_predict_endpoint_rejects_non_image(tmp_path: Path) -> None:
    client = TestClient(create_app(predictor=create_predictor(tmp_path)))

    response = client.post(
        "/predict",
        files={"file": ("notes.txt", b"not an image", "text/plain")},
    )

    assert response.status_code == 400

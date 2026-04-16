from __future__ import annotations

import io
import sys
from pathlib import Path

from PIL import Image
import pytest


PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from shared.data_quality import InputValidationError
from serving.inference.predictor import ModelPredictor, build_serving_config
from training.models.train import build_training_config, run_training


class DummyMlflowModel:
    def __init__(self) -> None:
        import numpy as np

        self.classes_ = np.asarray([0, 1, 2, 3], dtype=np.int64)

    def predict_proba(self, features):
        import numpy as np

        return np.asarray([[0.05, 0.1, 0.8, 0.05]], dtype=np.float32)


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


def train_model(tmp_path: Path) -> Path:
    features_path = tmp_path / "features.npz"
    create_feature_artifact(features_path)
    model_path = tmp_path / "model.pkl"
    config = build_training_config(
        input_features=features_path,
        output_model=model_path,
        output_report=tmp_path / "training_report.json",
        max_iter=300,
    )
    report = run_training(config)
    assert report["passed"] is True
    return model_path


def create_test_image_bytes(color: int = 180) -> bytes:
    image = Image.new("L", (8, 8), color=20)
    for x in range(2, 6):
        for y in range(2, 6):
            image.putpixel((x, y), color)
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return buffer.getvalue()


def test_predictor_returns_prediction_response(tmp_path: Path) -> None:
    model_path = train_model(tmp_path)
    predictor = ModelPredictor(
        build_serving_config(
            model_path=model_path,
            image_size=(8, 8),
            min_image_size=(8, 8),
        )
    )

    response = predictor.predict_bytes(create_test_image_bytes())

    assert response.predicted_label in {
        "NonDemented",
        "VeryMildDemented",
        "MildDemented",
        "ModerateDemented",
    }
    assert response.predicted_label_id in {0, 1, 2, 3}
    assert set(response.probabilities.keys()) == {
        "NonDemented",
        "VeryMildDemented",
        "MildDemented",
        "ModerateDemented",
    }
    assert response.input_shape == [8, 8, 1]
    assert response.model_version == "v1"


def test_predictor_rejects_non_mri_like_image(tmp_path: Path) -> None:
    model_path = train_model(tmp_path)
    predictor = ModelPredictor(build_serving_config(model_path=model_path, image_size=(8, 8), min_image_size=(8, 8)))

    colorful = Image.new("RGB", (8, 8), color=(200, 200, 200))
    buffer = io.BytesIO()
    colorful.save(buffer, format="PNG")

    with pytest.raises(InputValidationError) as exc_info:
        predictor.predict_bytes(buffer.getvalue())

    assert exc_info.value.feedback.error_code in {"low_signal_image", "non_mri_like_image"}


def test_predictor_uses_mlflow_champion_alias(monkeypatch, tmp_path: Path) -> None:
    captured = {}

    def fake_load_mlflow_model_metadata(*, model_name: str, tracking_uri: str | None, model_alias: str):
        captured["metadata"] = {
            "model_name": model_name,
            "tracking_uri": tracking_uri,
            "model_alias": model_alias,
        }
        return {
            "model_name": model_name,
            "model_version": "7",
            "model_alias": model_alias,
            "model_source": "mlflow",
        }

    def fake_load_mlflow_model(*, model_name: str, tracking_uri: str | None, model_alias: str):
        captured["model"] = {
            "model_name": model_name,
            "tracking_uri": tracking_uri,
            "model_alias": model_alias,
        }
        return DummyMlflowModel()

    def fake_list_mlflow_registered_models(*, model_name: str, tracking_uri: str | None, model_alias: str):
        return [
            {"model_name": model_name, "model_version": "7", "model_aliases": [model_alias]},
            {"model_name": model_name, "model_version": "6", "model_aliases": []},
        ]

    def fake_activate_mlflow_model_alias(*, model_name: str, model_version: str, tracking_uri: str | None, model_alias: str):
        captured["activation"] = {
            "model_name": model_name,
            "model_version": model_version,
            "tracking_uri": tracking_uri,
            "model_alias": model_alias,
        }
        return {
            "model_name": model_name,
            "model_version": model_version,
            "model_alias": model_alias,
            "model_source": "mlflow",
        }

    monkeypatch.setattr("serving.inference.predictor.load_mlflow_model_metadata", fake_load_mlflow_model_metadata)
    monkeypatch.setattr("serving.inference.predictor.load_mlflow_model", fake_load_mlflow_model)
    monkeypatch.setattr("serving.inference.predictor.list_mlflow_registered_models", fake_list_mlflow_registered_models)
    monkeypatch.setattr("serving.inference.predictor.activate_mlflow_model_alias", fake_activate_mlflow_model_alias)

    predictor = ModelPredictor(
        build_serving_config(
            model_path=tmp_path / "placeholder.pkl",
            model_name="alzheimer_detector",
            model_source="mlflow",
            mlflow_tracking_uri="http://127.0.0.1:5000",
            mlflow_model_alias="champion",
            image_size=(8, 8),
            min_image_size=(8, 8),
        )
    )

    response = predictor.predict_bytes(create_test_image_bytes())

    assert predictor.model_path == Path("models:/alzheimer_detector@champion")
    assert predictor.model_version == "7"
    assert response.predicted_label == "MildDemented"
    assert response.model_version == "7"
    assert captured["metadata"] == {
        "model_name": "alzheimer_detector",
        "tracking_uri": "http://127.0.0.1:5000",
        "model_alias": "champion",
    }
    assert captured["model"] == captured["metadata"]
    assert predictor.list_registered_models()[0]["model_aliases"] == ["champion"]

    activation_metadata = predictor.activate_model_version("6")

    assert activation_metadata["model_version"] == "6"
    assert captured["activation"] == {
        "model_name": "alzheimer_detector",
        "model_version": "6",
        "tracking_uri": "http://127.0.0.1:5000",
        "model_alias": "champion",
    }

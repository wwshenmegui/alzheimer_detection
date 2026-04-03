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


def test_predictor_rejects_non_mri_like_image(tmp_path: Path) -> None:
    model_path = train_model(tmp_path)
    predictor = ModelPredictor(build_serving_config(model_path=model_path, image_size=(8, 8), min_image_size=(8, 8)))

    colorful = Image.new("RGB", (8, 8), color=(200, 200, 200))
    buffer = io.BytesIO()
    colorful.save(buffer, format="PNG")

    with pytest.raises(InputValidationError) as exc_info:
        predictor.predict_bytes(buffer.getvalue())

    assert exc_info.value.feedback.error_code in {"low_signal_image", "non_mri_like_image"}

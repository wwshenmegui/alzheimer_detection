from __future__ import annotations

import importlib
import logging
import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from training.features.build_features import DEFAULT_IMAGE_SIZE
from training.ingestion.ingest import DEFAULT_CONFIG_PATH, DEFAULT_LABEL_TO_ID

from .preprocess import preprocess_uploaded_image
from .schemas import PredictionResponse


LOGGER = logging.getLogger(__name__)


@dataclass
class ServingConfig:
    model_path: Path
    image_size: tuple[int, int] = DEFAULT_IMAGE_SIZE
    host: str = "127.0.0.1"
    port: int = 8000
    log_level: str = "INFO"
    label_to_id: dict[str, int] | None = None

    @property
    def id_to_label(self) -> dict[int, str]:
        label_to_id = self.label_to_id or dict(DEFAULT_LABEL_TO_ID)
        return {label_id: label_name for label_name, label_id in label_to_id.items()}


def configure_logging(log_level: str) -> None:
    logging.basicConfig(
        level=getattr(logging, log_level.upper(), logging.INFO),
        format="%(asctime)s | %(levelname)s | %(message)s",
    )


def load_serving_settings(config_path: Path) -> dict[str, Any]:
    if not config_path.exists():
        return {}

    try:
        yaml = importlib.import_module("yaml")
    except ImportError as exc:
        raise RuntimeError(
            "PyYAML is required for config loading. Install it with 'pip install pyyaml'."
        ) from exc

    with config_path.open("r", encoding="utf-8") as handle:
        config_data = yaml.safe_load(handle) or {}

    serving_settings = config_data.get("serving", {})
    if not isinstance(serving_settings, dict):
        raise ValueError("The 'serving' section in the config file must be a mapping.")

    return serving_settings


def build_serving_config(
    *,
    model_path: Path | None,
    image_size: tuple[int, int] = DEFAULT_IMAGE_SIZE,
    host: str = "127.0.0.1",
    port: int = 8000,
    log_level: str = "INFO",
    label_to_id: dict[str, int] | None = None,
) -> ServingConfig:
    if model_path is None:
        raise ValueError("A model path is required to build serving config.")

    return ServingConfig(
        model_path=model_path,
        image_size=image_size,
        host=host,
        port=port,
        log_level=log_level,
        label_to_id=label_to_id or dict(DEFAULT_LABEL_TO_ID),
    )


class ModelPredictor:
    def __init__(self, config: ServingConfig):
        self.config = config
        self.model = self._load_model(config.model_path)
        self.id_to_label = config.id_to_label

    def _load_model(self, model_path: Path):
        LOGGER.info("Loading serving model from %s", model_path)
        with model_path.open("rb") as handle:
            return pickle.load(handle)

    def predict_bytes(self, image_bytes: bytes) -> PredictionResponse:
        features = preprocess_uploaded_image(image_bytes, self.config.image_size)
        probabilities = self.model.predict_proba(features)[0]
        predicted_index = int(np.argmax(probabilities))
        predicted_label_id = int(self.model.classes_[predicted_index])
        predicted_label = self.id_to_label[predicted_label_id]

        probability_map = {
            self.id_to_label[int(class_id)]: float(probability)
            for class_id, probability in zip(self.model.classes_.tolist(), probabilities.tolist())
        }

        return PredictionResponse(
            predicted_label=predicted_label,
            predicted_label_id=predicted_label_id,
            probabilities=probability_map,
            input_shape=[self.config.image_size[0], self.config.image_size[1], 1],
            model_name=type(self.model).__name__,
        )


def create_predictor(config_path: Path = DEFAULT_CONFIG_PATH, config: ServingConfig | None = None) -> ModelPredictor:
    if config is None:
        settings = load_serving_settings(config_path)
        image_size_value = settings.get("image_size", list(DEFAULT_IMAGE_SIZE))
        config = build_serving_config(
            model_path=Path(settings.get("model_path")) if settings.get("model_path") else None,
            image_size=(int(image_size_value[0]), int(image_size_value[1])),
            host=str(settings.get("host", "127.0.0.1")),
            port=int(settings.get("port", 8000)),
            log_level=str(settings.get("log_level", "INFO")),
            label_to_id={str(key): int(value) for key, value in settings.get("label_to_id", dict(DEFAULT_LABEL_TO_ID)).items()},
        )
    configure_logging(config.log_level)
    return ModelPredictor(config)

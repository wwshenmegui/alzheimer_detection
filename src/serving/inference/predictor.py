from __future__ import annotations

import importlib
import logging
import pickle
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

import numpy as np

from shared.data_quality import (
    DEFAULT_ASPECT_RATIO_RANGE,
    DEFAULT_MIN_CENTER_BORDER_DIFF,
    DEFAULT_MIN_IMAGE_SIZE,
    DEFAULT_MIN_STDDEV,
    InputValidationError,
    validate_mri_image_bytes,
)
from shared.model_registry import activate_model_version, list_registered_models, load_model_metadata, resolve_model_artifacts
from training.features.build_features import DEFAULT_IMAGE_SIZE
from training.ingestion.ingest import DEFAULT_CONFIG_PATH, DEFAULT_LABEL_TO_ID

from .preprocess import preprocess_uploaded_image
from .schemas import PredictionResponse


LOGGER = logging.getLogger(__name__)


@dataclass
class ServingConfig:
    model_path: Path
    model_name: str | None = None
    model_version: str | None = None
    image_size: tuple[int, int] = DEFAULT_IMAGE_SIZE
    host: str = "127.0.0.1"
    port: int = 8000
    log_level: str = "INFO"
    label_to_id: dict[str, int] | None = None
    min_image_size: tuple[int, int] = DEFAULT_MIN_IMAGE_SIZE
    aspect_ratio_range: tuple[float, float] = DEFAULT_ASPECT_RATIO_RANGE
    min_stddev: float = DEFAULT_MIN_STDDEV
    min_center_border_diff: float = DEFAULT_MIN_CENTER_BORDER_DIFF

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
    model_name: str | None = None,
    model_version: str | None = None,
    image_size: tuple[int, int] = DEFAULT_IMAGE_SIZE,
    host: str = "127.0.0.1",
    port: int = 8000,
    log_level: str = "INFO",
    label_to_id: dict[str, int] | None = None,
    min_image_size: tuple[int, int] = DEFAULT_MIN_IMAGE_SIZE,
    aspect_ratio_range: tuple[float, float] = DEFAULT_ASPECT_RATIO_RANGE,
    min_stddev: float = DEFAULT_MIN_STDDEV,
    min_center_border_diff: float = DEFAULT_MIN_CENTER_BORDER_DIFF,
) -> ServingConfig:
    if model_path is None:
        raise ValueError("A model path is required to build serving config.")

    return ServingConfig(
        model_path=model_path,
        model_name=model_name,
        model_version=model_version,
        image_size=image_size,
        host=host,
        port=port,
        log_level=log_level,
        label_to_id=label_to_id or dict(DEFAULT_LABEL_TO_ID),
        min_image_size=min_image_size,
        aspect_ratio_range=aspect_ratio_range,
        min_stddev=min_stddev,
        min_center_border_diff=min_center_border_diff,
    )


class ModelPredictor:
    def __init__(self, config: ServingConfig):
        self.config = config
        self.model_path, self.metadata_path = resolve_model_artifacts(
            config.model_path,
            model_version=config.model_version,
            model_name=config.model_name,
        )
        self.model_metadata = load_model_metadata(self.metadata_path) or self._build_fallback_metadata()
        self.model = self._load_model(self.model_path)
        self.id_to_label = config.id_to_label

    def _load_model(self, model_path: Path):
        LOGGER.info("Loading serving model from %s", model_path)
        with model_path.open("rb") as handle:
            return pickle.load(handle)

    def _build_fallback_metadata(self) -> dict[str, Any]:
        return {
            "model_name": self.config.model_name or self.config.model_path.stem,
            "model_version": self.config.model_version or "unversioned",
            "model_type": "unknown",
            "model_path": str(self.model_path),
            "metadata_path": str(self.metadata_path) if self.metadata_path else "",
        }

    @property
    def model_version(self) -> str:
        return str(self.model_metadata.get("model_version", self.config.model_version or "unversioned"))

    def get_model_metadata(self) -> dict[str, Any]:
        return dict(self.model_metadata)

    def list_registered_models(self) -> list[dict[str, Any]]:
        return list_registered_models(self.config.model_path, model_name=self.config.model_name)

    def activate_model_version(self, model_version: str) -> dict[str, Any]:
        metadata = activate_model_version(
            self.config.model_path,
            model_version=model_version,
            model_name=self.config.model_name,
        )
        refreshed = ModelPredictor(replace(self.config, model_version=None))
        self.config = refreshed.config
        self.model_path = refreshed.model_path
        self.metadata_path = refreshed.metadata_path
        self.model_metadata = refreshed.model_metadata
        self.model = refreshed.model
        self.id_to_label = refreshed.id_to_label
        return metadata

    def predict_bytes(self, image_bytes: bytes) -> PredictionResponse:
        feedback, _ = validate_mri_image_bytes(
            image_bytes,
            min_image_size=self.config.min_image_size,
            aspect_ratio_range=self.config.aspect_ratio_range,
            min_stddev=self.config.min_stddev,
            min_center_border_diff=self.config.min_center_border_diff,
        )
        if not feedback.passed:
            raise InputValidationError(feedback)

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
            model_version=self.model_version,
        )


def create_predictor(config_path: Path = DEFAULT_CONFIG_PATH, config: ServingConfig | None = None) -> ModelPredictor:
    if config is None:
        settings = load_serving_settings(config_path)
        image_size_value = settings.get("image_size", list(DEFAULT_IMAGE_SIZE))
        min_image_size_value = settings.get("min_image_size", list(DEFAULT_MIN_IMAGE_SIZE))
        aspect_ratio_range_value = settings.get("aspect_ratio_range", list(DEFAULT_ASPECT_RATIO_RANGE))
        config = build_serving_config(
            model_path=Path(settings.get("model_path")) if settings.get("model_path") else None,
            model_name=str(settings.get("model_name")) if settings.get("model_name") else None,
            model_version=str(settings.get("model_version")) if settings.get("model_version") else None,
            image_size=(int(image_size_value[0]), int(image_size_value[1])),
            host=str(settings.get("host", "127.0.0.1")),
            port=int(settings.get("port", 8000)),
            log_level=str(settings.get("log_level", "INFO")),
            label_to_id={str(key): int(value) for key, value in settings.get("label_to_id", dict(DEFAULT_LABEL_TO_ID)).items()},
            min_image_size=(int(min_image_size_value[0]), int(min_image_size_value[1])),
            aspect_ratio_range=(float(aspect_ratio_range_value[0]), float(aspect_ratio_range_value[1])),
            min_stddev=float(settings.get("min_stddev", DEFAULT_MIN_STDDEV)),
            min_center_border_diff=float(settings.get("min_center_border_diff", DEFAULT_MIN_CENTER_BORDER_DIFF)),
        )
    configure_logging(config.log_level)
    return ModelPredictor(config)

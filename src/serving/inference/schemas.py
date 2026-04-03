from __future__ import annotations

from typing import Any

from pydantic import BaseModel


class PredictionResponse(BaseModel):
    predicted_label: str
    predicted_label_id: int
    probabilities: dict[str, float]
    input_shape: list[int]
    model_name: str


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    model_name: str
    model_path: str


class ErrorResponse(BaseModel):
    error_code: str
    message: str
    user_action: str
    details: dict[str, Any]

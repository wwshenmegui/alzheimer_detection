from __future__ import annotations

import os
from pathlib import Path

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import FileResponse, JSONResponse

from shared.data_quality import InputValidationError, ValidationFeedback
from training.ingestion.ingest import DEFAULT_CONFIG_PATH

from serving.inference.predictor import ModelPredictor, create_predictor
from serving.inference.schemas import (
    ActivateModelRequest,
    ActivateModelResponse,
    ErrorResponse,
    HealthResponse,
    ModelCatalogResponse,
    ModelMetadataResponse,
    PredictionResponse,
)


STATIC_DIR = Path(__file__).resolve().parent / "static"
CONFIG_ENV_VAR = "ALZHEIMER_CONFIG_PATH"


def create_app(
    config_path: Path | None = None,
    predictor: ModelPredictor | None = None,
) -> FastAPI:
    resolved_config_path = config_path or Path(os.environ.get(CONFIG_ENV_VAR, str(DEFAULT_CONFIG_PATH)))
    app = FastAPI(title="Alzheimer Detection Serving API", version="0.1.0")
    app.state.predictor = predictor or create_predictor(config_path=resolved_config_path)

    def serialize_error_response(feedback: ValidationFeedback) -> dict:
        payload = ErrorResponse(**feedback.to_dict())
        if hasattr(payload, "model_dump"):
            return payload.model_dump()
        return payload.dict()

    @app.get("/")
    def index() -> FileResponse:
        return FileResponse(STATIC_DIR / "index.html")

    @app.get("/health", response_model=HealthResponse)
    def health() -> HealthResponse:
        active_predictor: ModelPredictor = app.state.predictor
        return HealthResponse(
            status="ok",
            model_loaded=active_predictor.model is not None,
            model_name=type(active_predictor.model).__name__,
            model_path=str(active_predictor.model_path),
            model_version=active_predictor.model_version,
        )

    @app.get("/model", response_model=ModelMetadataResponse)
    def model_metadata() -> ModelMetadataResponse:
        active_predictor: ModelPredictor = app.state.predictor
        return ModelMetadataResponse(metadata=active_predictor.get_model_metadata())

    @app.get("/models", response_model=ModelCatalogResponse)
    def models() -> ModelCatalogResponse:
        active_predictor: ModelPredictor = app.state.predictor
        return ModelCatalogResponse(
            active_model_version=active_predictor.model_version,
            models=active_predictor.list_registered_models(),
        )

    @app.post("/model/activate", response_model=ActivateModelResponse)
    def activate_model(request: ActivateModelRequest) -> ActivateModelResponse:
        active_predictor: ModelPredictor = app.state.predictor
        try:
            metadata = active_predictor.activate_model_version(request.model_version)
        except FileNotFoundError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        return ActivateModelResponse(
            activated_model_version=str(metadata.get("model_version", request.model_version)),
            metadata=metadata,
        )

    @app.post("/predict", response_model=PredictionResponse)
    async def predict(file: UploadFile = File(...)) -> PredictionResponse:
        if not file.content_type or not file.content_type.startswith("image/"):
            feedback = ValidationFeedback(
                passed=False,
                error_code="unsupported_file_type",
                message="Unsupported file type.",
                user_action="Please upload a PNG, JPG, or JPEG brain MRI image.",
                details={"content_type": file.content_type or "unknown"},
            )
            return JSONResponse(status_code=400, content=serialize_error_response(feedback))

        file_bytes = await file.read()
        if not file_bytes:
            feedback = ValidationFeedback(
                passed=False,
                error_code="empty_upload",
                message="Uploaded file is empty.",
                user_action="Please choose an MRI image file and try again.",
                details={},
            )
            return JSONResponse(status_code=400, content=serialize_error_response(feedback))

        try:
            active_predictor: ModelPredictor = app.state.predictor
            return active_predictor.predict_bytes(file_bytes)
        except InputValidationError as exc:
            return JSONResponse(status_code=400, content=serialize_error_response(exc.feedback))
        except Exception as exc:  # pragma: no cover
            raise HTTPException(status_code=400, detail=f"Prediction failed: {exc}") from exc

    return app

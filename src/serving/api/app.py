from __future__ import annotations

import os
from pathlib import Path

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import FileResponse

from training.ingestion.ingest import DEFAULT_CONFIG_PATH

from serving.inference.predictor import ModelPredictor, create_predictor
from serving.inference.schemas import HealthResponse, PredictionResponse


STATIC_DIR = Path(__file__).resolve().parent / "static"
CONFIG_ENV_VAR = "ALZHEIMER_CONFIG_PATH"


def create_app(
    config_path: Path | None = None,
    predictor: ModelPredictor | None = None,
) -> FastAPI:
    resolved_config_path = config_path or Path(os.environ.get(CONFIG_ENV_VAR, str(DEFAULT_CONFIG_PATH)))
    app = FastAPI(title="Alzheimer Detection Serving API", version="0.1.0")
    app.state.predictor = predictor or create_predictor(config_path=resolved_config_path)

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
            model_path=str(active_predictor.config.model_path),
        )

    @app.post("/predict", response_model=PredictionResponse)
    async def predict(file: UploadFile = File(...)) -> PredictionResponse:
        if not file.content_type or not file.content_type.startswith("image/"):
            raise HTTPException(status_code=400, detail="Uploaded file must be an image.")

        file_bytes = await file.read()
        if not file_bytes:
            raise HTTPException(status_code=400, detail="Uploaded file is empty.")

        try:
            active_predictor: ModelPredictor = app.state.predictor
            return active_predictor.predict_bytes(file_bytes)
        except Exception as exc:  # pragma: no cover
            raise HTTPException(status_code=400, detail=f"Prediction failed: {exc}") from exc

    return app

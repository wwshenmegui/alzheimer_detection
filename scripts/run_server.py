from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import uvicorn


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"

if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

os.chdir(PROJECT_ROOT)

from serving.api.app import CONFIG_ENV_VAR, create_app
from serving.inference.predictor import build_serving_config, create_predictor, load_serving_settings
from training.features.build_features import DEFAULT_IMAGE_SIZE
from training.ingestion.ingest import DEFAULT_CONFIG_PATH


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the Alzheimer detection serving API.")
    parser.add_argument("--config", default=str(DEFAULT_CONFIG_PATH), help="Path to the YAML config file.")
    parser.add_argument("--host", help="Host to bind the server to.")
    parser.add_argument("--port", type=int, help="Port to bind the server to.")
    parser.add_argument("--log-level", help="Logging level, for example DEBUG, INFO, WARNING.")
    parser.add_argument("--reload", action="store_true", help="Enable auto-reload for development.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    settings = load_serving_settings(Path(args.config))
    image_size_value = settings.get("image_size", list(DEFAULT_IMAGE_SIZE))
    host = str(args.host or settings.get("host", "127.0.0.1"))
    port = int(args.port if args.port is not None else settings.get("port", 8000))
    log_level = str(args.log_level or settings.get("log_level", "INFO"))
    os.environ[CONFIG_ENV_VAR] = str(Path(args.config))

    if args.reload:
        uvicorn.run(
            "serving.api.app:create_app",
            factory=True,
            host=host,
            port=port,
            log_level=log_level.lower(),
            reload=True,
        )
        return

    predictor = create_predictor(
        config=build_serving_config(
            model_path=Path(settings.get("model_path")) if settings.get("model_path") else None,
            image_size=(int(image_size_value[0]), int(image_size_value[1])),
            host=host,
            port=port,
            log_level=log_level,
            label_to_id={str(key): int(value) for key, value in settings.get("label_to_id", {}).items()} or None,
        )
    )
    app = create_app(predictor=predictor)
    uvicorn.run(app, host=predictor.config.host, port=predictor.config.port, log_level=predictor.config.log_level.lower())


if __name__ == "__main__":
    main()
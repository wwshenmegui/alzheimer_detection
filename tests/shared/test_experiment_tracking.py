from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from shared.experiment_tracking import log_remote_evaluation_run, log_remote_training_run


class FakeRunContext:
    def __init__(self, run_id: str) -> None:
        self.info = SimpleNamespace(run_id=run_id)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        return False


class FakeMlflow:
    def __init__(self) -> None:
        self.tracking_uri = None
        self.experiment_name = None
        self.params = []
        self.metrics = []
        self.tags = []
        self.artifacts = []
        self.started_runs = []
        self.logged_models = []
        self.evaluations = []
        self.models = SimpleNamespace(infer_signature=self.infer_signature)
        self.sklearn = SimpleNamespace(log_model=self.log_model)

    def set_tracking_uri(self, tracking_uri: str) -> None:
        self.tracking_uri = tracking_uri

    def set_experiment(self, experiment_name: str) -> None:
        self.experiment_name = experiment_name

    def start_run(self, run_name: str | None = None, run_id: str | None = None):
        resolved_run_id = run_id or "remote-run-123"
        self.started_runs.append({"run_name": run_name, "run_id": resolved_run_id})
        return FakeRunContext(resolved_run_id)

    def log_params(self, params: dict[str, object]) -> None:
        self.params.append(params)

    def log_metrics(self, metrics: dict[str, float]) -> None:
        self.metrics.append(metrics)

    def set_tags(self, tags: dict[str, str]) -> None:
        self.tags.append(tags)

    def log_artifact(self, artifact_path: str) -> None:
        self.artifacts.append(artifact_path)

    def infer_signature(self, inputs, outputs):
        return {"inputs": getattr(inputs, "shape", None), "outputs": getattr(outputs, "shape", None)}

    def log_model(self, *, sk_model, artifact_path: str, input_example, signature):
        self.logged_models.append(
            {
                "model": sk_model,
                "artifact_path": artifact_path,
                "input_example_shape": getattr(input_example, "shape", None),
                "signature": signature,
            }
        )
        return SimpleNamespace(model_uri=f"runs:/remote-run-123/{artifact_path}")

    def evaluate(self, *, model: str, data, targets: str, model_type: str, evaluator_config: dict[str, object]) -> None:
        self.evaluations.append(
            {
                "model": model,
                "data": data,
                "targets": targets,
                "model_type": model_type,
                "evaluator_config": evaluator_config,
            }
        )


class FakeDataFrame(dict):
    def __init__(self, data, columns):
        super().__init__()
        self["_rows"] = data
        self["_columns"] = columns


class FakePandas:
    @staticmethod
    def DataFrame(data, columns):
        return FakeDataFrame(data, columns)


def test_log_remote_training_run_logs_mlflow_model(tmp_path: Path, monkeypatch) -> None:
    fake_mlflow = FakeMlflow()
    monkeypatch.setitem(sys.modules, "mlflow", fake_mlflow)
    artifact_path = tmp_path / "training_report.json"
    artifact_path.write_text("{}", encoding="utf-8")

    result = log_remote_training_run(
        SimpleNamespace(tracking_uri="http://127.0.0.1:5000", experiment_name="alzheimer_detection"),
        run_id="local-run-001",
        params={"model_name": "alzheimer_detector"},
        metrics={"validation_accuracy": 0.91},
        artifact_paths=[artifact_path],
        model=object(),
        input_example=np.ones((2, 4), dtype=np.float32),
        signature_inputs=np.ones((3, 4), dtype=np.float32),
        signature_outputs=np.array([1, 0, 1]),
        tags={"stage": "training"},
    )

    assert result["run_id"] == "remote-run-123"
    assert result["logged_model_uri"] == "runs:/remote-run-123/model"
    assert fake_mlflow.experiment_name == "alzheimer_detection"
    assert fake_mlflow.logged_models[0]["artifact_path"] == "model"
    assert fake_mlflow.logged_models[0]["input_example_shape"] == (2, 4)
    assert fake_mlflow.metrics == [{"validation_accuracy": 0.91}]


def test_log_remote_evaluation_run_logs_mlflow_evaluation(monkeypatch) -> None:
    fake_mlflow = FakeMlflow()
    monkeypatch.setitem(sys.modules, "mlflow", fake_mlflow)
    monkeypatch.setitem(sys.modules, "pandas", FakePandas())

    log_remote_evaluation_run(
        SimpleNamespace(tracking_uri="http://127.0.0.1:5000"),
        remote_run_id="remote-run-xyz",
        metrics={"evaluation_accuracy": 0.87},
        artifact_paths=[],
        evaluation_features=np.array([[0.1, 0.2], [0.3, 0.4]], dtype=np.float32),
        evaluation_labels=np.array([0, 1], dtype=np.int64),
    )

    assert fake_mlflow.metrics == [{"evaluation_accuracy": 0.87}]
    assert fake_mlflow.evaluations[0]["model"] == "runs:/remote-run-xyz/model"
    assert fake_mlflow.evaluations[0]["targets"] == "target"
    assert fake_mlflow.evaluations[0]["model_type"] == "classifier"
    assert fake_mlflow.evaluations[0]["data"]["_columns"] == ["feature_00000", "feature_00001"]
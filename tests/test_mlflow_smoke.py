import os
import mlflow


def test_mlflow_smoke(tmp_path, monkeypatch):
    monkeypatch.setenv("MLFLOW_TRACKING_URI", str(tmp_path))
    mlflow.set_experiment("fiam_llm_alpha_test")
    with mlflow.start_run(run_name="smoke"):
        mlflow.log_params({"a": 1})
        mlflow.log_metric("m", 0.1)
    assert any(p.name.startswith("mlruns") or p.name.isdigit() for p in tmp_path.iterdir())

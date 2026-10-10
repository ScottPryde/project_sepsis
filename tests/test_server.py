import json
import sys
import threading
import time
from pathlib import Path
from urllib.request import Request, urlopen

import pandas as pd
import pytest

from sepsis_prediction.application import MODEL_IDS, create_application_server


def test_local_report_server_serves_report_and_runs_pipeline_command(tmp_path) -> None:
    report_path = tmp_path / "report" / "report.html"
    report_path.parent.mkdir()
    report_path.write_text("<h1>Full report</h1>", encoding="utf-8")
    model_dir = tmp_path / "models" / "logistic_regression"
    model_dir.mkdir(parents=True)
    (model_dir / "metrics.json").write_text(json.dumps({
        "logistic_regression": {"auroc": 0.9, "auprc_average_precision": 0.8},
    }), encoding="utf-8")
    pd.DataFrame([{
        "model": "logistic_regression", "feature": "HR_mean", "value": 0.5,
        "importance_type": "coefficient", "absolute_value": 0.5,
    }]).to_csv(model_dir / "feature_importance.csv", index=False)
    commands = {
        model: [sys.executable, "-c", "print('pipeline finished')"]
        for model in MODEL_IDS
    }
    server = create_application_server(
        tmp_path,
        report_path,
        {model: tmp_path / "models" / model for model in MODEL_IDS},
        commands,
        tmp_path / "fallback",
        Path(__file__).parents[1] / "src" / "sepsis_prediction" / "web",
        False,
        port=0,
    )
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    base_url = f"http://127.0.0.1:{server.server_port}"
    try:
        with urlopen(base_url + "/") as response:
            assert b"Model runs" in response.read()
        with urlopen(base_url + "/app.js") as response:
            assert b"formatElapsed" in response.read()
        with urlopen(base_url + "/report.html") as response:
            assert response.read() == b"<h1>Full report</h1>"
        with urlopen(base_url + "/api/results") as response:
            results = json.loads(response.read())
            assert set(results["models"]) == set(MODEL_IDS)
            assert results["datasource_description"]
            assert results["models"]["logistic_regression"]["metrics"]["auroc"] == 0.9
            assert results["models"]["logistic_regression"]["leading_features"][0]["feature"] == "HR_mean"

        request = Request(base_url + "/api/run/logistic_regression", method="POST", data=b"")
        with urlopen(request) as response:
            assert response.status == 202
            assert json.loads(response.read()) == {"state": "running", "model": "logistic_regression"}

        for _ in range(100):
            with urlopen(base_url + "/api/status") as response:
                status = json.loads(response.read())
            if status["state"] != "running":
                break
            time.sleep(0.02)
        assert status["state"] == "complete"
        assert status["active_model"] == "logistic_regression"
        assert status["returncode"] == 0
        assert status["elapsed_seconds"] >= 0
        assert "pipeline finished" in status["logs"]
        with urlopen(base_url + "/api/results") as response:
            results = json.loads(response.read())
        assert results["models"]["logistic_regression"]["status"]["elapsed_seconds"] >= 0
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)


def test_application_server_rejects_non_loopback_bind(tmp_path) -> None:
    with pytest.raises(ValueError, match="loopback"):
        create_application_server(
            tmp_path, tmp_path / "report.html", {model: tmp_path / model for model in MODEL_IDS},
            {model: [sys.executable, "-c", "pass"] for model in MODEL_IDS},
            tmp_path, tmp_path, False, host="0.0.0.0",
        )
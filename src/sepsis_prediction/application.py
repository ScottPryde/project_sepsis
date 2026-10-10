"""Local JavaScript application for independent per-model pipeline runs."""

from __future__ import annotations

import ipaddress
import json
import os
import subprocess
import sys
import threading
import time
from datetime import datetime
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import unquote, urlsplit

import pandas as pd

from sepsis_prediction.reporting import write_html_report


MODEL_IDS = ("logistic_regression", "random_forest", "xgboost", "cnn_1d")
CHART_FILES = {"model_curves.png", "model_forest.png", "calibration.png", "decision_curve.png"}


class _ApplicationHandler(BaseHTTPRequestHandler):
    output_dir: Path
    report_path: Path
    app_dir: Path
    model_dirs: dict[str, Path]
    commands: dict[str, list[str]]
    fallback_dir: Path
    synthetic_demo: bool
    job_lock = threading.Lock()
    job_status: dict[str, Any] = {"state": "idle", "active_model": None, "returncode": None, "logs": ""}

    def do_GET(self) -> None:
        path = unquote(urlsplit(self.path).path)
        if path in ("/", "/index.html"):
            self._send_file(self.app_dir / "index.html", "text/html; charset=utf-8")
        elif path == "/app.js":
            self._send_file(self.app_dir / "app.js", "text/javascript; charset=utf-8")
        elif path == "/report.html":
            report_path = self._latest_report_path()
            if not report_path.exists():
                write_html_report(report_path.parent)
            self._send_file(report_path, "text/html; charset=utf-8", no_store=True)
        elif path == "/api/status":
            handler_type = type(self)
            with handler_type.job_lock:
                status = dict(handler_type.job_status)
            started_at = status.get("started_at")
            if status.get("state") == "running" and started_at:
                status["elapsed_seconds"] = round(time.time() - started_at, 1)
            else:
                status.setdefault("elapsed_seconds", 0)
            self._send_json(200, status)
        elif path == "/api/results":
            self._send_json(200, self._results_payload())
        elif path.startswith("/assets/"):
            self._serve_chart(path)
        else:
            self.send_error(404)

    def do_POST(self) -> None:
        path = urlsplit(self.path).path
        model = path.removeprefix("/api/run/")
        if not path.startswith("/api/run/") or model not in self.commands:
            self.send_error(404)
            return
        if not self._request_is_local():
            self.send_error(403)
            return
        handler_type = type(self)
        with handler_type.job_lock:
            if handler_type.job_status["state"] == "running":
                self._send_json(409, {"error": f"{handler_type.job_status['active_model']} is already running"})
                return
            handler_type.job_status = {
                "state": "running",
                "active_model": model,
                "started_at": time.time(),
                "returncode": None,
                "logs": "",
            }
        threading.Thread(target=self._run_model, args=(model,), daemon=True).start()
        self._send_json(202, {"state": "running", "model": model})

    def log_message(self, format_string: str, *args: Any) -> None:
        return

    def _run_model(self, model: str) -> None:
        handler_type = type(self)
        with handler_type.job_lock:
            started_at = handler_type.job_status.get("started_at", time.time())
        environment = os.environ.copy()
        environment.setdefault("OMP_NUM_THREADS", "1")
        try:
            result = subprocess.run(
                self.commands[model],
                cwd=self.output_dir.parent if self.synthetic_demo else self.output_dir,
                capture_output=True,
                text=True,
                env=environment,
                check=False,
            )
            elapsed_seconds = round(time.time() - started_at, 1)
            status = {
                "state": "complete" if result.returncode == 0 else "failed",
                "active_model": model,
                "returncode": result.returncode,
                "logs": (result.stdout + "\n" + result.stderr)[-12000:],
                "elapsed_seconds": elapsed_seconds,
            }
            duration_path = self.model_dirs[model] / "run_duration.json"
            duration_path.parent.mkdir(parents=True, exist_ok=True)
            duration_path.write_text(json.dumps({"elapsed_seconds": elapsed_seconds}), encoding="utf-8")
        except OSError as error:
            status = {
                "state": "failed", "active_model": model, "returncode": None,
                "logs": str(error), "elapsed_seconds": round(time.time() - started_at, 1),
            }
        with handler_type.job_lock:
            handler_type.job_status = status

    def _results_payload(self) -> dict[str, Any]:
        handler_type = type(self)
        with handler_type.job_lock:
            job_status = dict(handler_type.job_status)
        snapshot_dir = self.fallback_dir
        if not (snapshot_dir / "run_config.json").exists():
            available_dirs = [directory for directory in self.model_dirs.values()
                              if (directory / "run_config.json").exists()]
            if available_dirs:
                snapshot_dir = max(available_dirs, key=lambda directory: (directory / "run_config.json").stat().st_mtime)
        snapshot = {
            "config": self._read_json(snapshot_dir / "run_config.json"),
            "cohort": self._read_json(snapshot_dir / "cohort_summary.json"),
        }
        models: dict[str, Any] = {}
        for model in MODEL_IDS:
            model_dir = self.model_dirs[model]
            run_dir = self._model_run_dir(model)
            metrics = self._read_json(run_dir / "metrics.json")
            evaluation = self._read_json(run_dir / "model_evaluation.json")
            config = self._read_json(run_dir / "run_config.json")
            importance = self._read_importance(run_dir / "feature_importance.csv", model)
            models[model] = {
                "available": model in metrics,
                "isolated_run": run_dir == model_dir and (model_dir / "metrics.json").exists(),
                "metrics": metrics.get(model, {}),
                "evaluation": next((item for item in evaluation.get("models", [])
                                     if item.get("model") == model), {}),
                "config": config,
                "leading_features": importance,
                "target_sensitivity": evaluation.get("target_sensitivity", config.get("target_sensitivity", 0.8)),
                "synthetic_demo": bool(config.get("synthetic_demo", self.synthetic_demo)),
                "updated_at": datetime.fromtimestamp((run_dir / "metrics.json").stat().st_mtime).isoformat(timespec="minutes")
                if (run_dir / "metrics.json").exists() else None,
                "status": self._status_for(model),
            }
        source = (
            "Fabricated synthetic records with deliberately label-linked trends; smoke-test data, not clinical evidence."
            if self.synthetic_demo else
            "PhysioNet/Computing in Cardiology Challenge 2019 PSV data; one pipe-separated, hourly patient record per file."
        )
        return {
            "snapshot": snapshot,
            "models": models,
            "datasource_description": source,
            "job_status": job_status,
        }

    def _status_for(self, model: str) -> dict[str, Any]:
        handler_type = type(self)
        with handler_type.job_lock:
            status = dict(handler_type.job_status)
        if status.get("active_model") != model:
                run_dir = self._model_run_dir(model)
                duration = self._read_json(self.model_dirs[model] / "run_duration.json")
                return {
                    "state": "complete" if (run_dir / "metrics.json").exists() else "idle",
                    "elapsed_seconds": duration.get("elapsed_seconds"),
                }
        started_at = status.get("started_at")
        if status.get("state") == "running" and started_at:
            status["elapsed_seconds"] = round(time.time() - started_at, 1)
        else:
            status.setdefault("elapsed_seconds", 0)
        return status

    def _model_run_dir(self, model: str) -> Path:
        candidate = self.model_dirs[model]
        if (candidate / "metrics.json").exists():
            return candidate
        return self.fallback_dir

    def _latest_report_path(self) -> Path:
        candidates = [self.report_path]
        candidates.extend(directory / "report.html" for directory in self.model_dirs.values())
        available = [path for path in candidates if path.is_file()]
        return max(available, key=lambda path: path.stat().st_mtime) if available else self.report_path

    @staticmethod
    def _read_json(path: Path) -> dict[str, Any]:
        if not path.exists():
            return {}
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return {}

    @staticmethod
    def _read_importance(path: Path, model: str) -> list[dict[str, Any]]:
        if not path.exists():
            return []
        try:
            table = pd.read_csv(path)
            if "model" not in table or "absolute_value" not in table:
                return []
            selected = table.loc[table["model"] == model].nlargest(3, "absolute_value")
            rows = selected[["feature", "value", "importance_type"]].to_dict(orient="records")
            return [{
                "feature": str(row["feature"]),
                "value": float(row["value"]),
                "importance_type": str(row["importance_type"]),
            } for row in rows]
        except (OSError, ValueError, KeyError):
            return []

    def _serve_chart(self, path: str) -> None:
        parts = path.strip("/").split("/")
        if len(parts) != 3 or parts[1] not in MODEL_IDS or parts[2] not in CHART_FILES:
            self.send_error(404)
            return
        chart = self._model_run_dir(parts[1]) / parts[2]
        self._send_file(chart, "image/png", no_store=True)

    def _request_is_local(self) -> bool:
        try:
            if not ipaddress.ip_address(self.client_address[0]).is_loopback:
                return False
        except ValueError:
            return False
        origin = self.headers.get("Origin")
        if not origin:
            return True
        origin_host = urlsplit(origin).hostname
        request_host = urlsplit(f"//{self.headers.get('Host', '')}").hostname
        return origin_host is not None and origin_host == request_host

    def _send_file(self, path: Path, content_type: str, no_store: bool = False) -> None:
        if not path.is_file():
            self.send_error(404)
            return
        content = path.read_bytes()
        self.send_response(200)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(content)))
        if no_store:
            self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(content)

    def _send_json(self, status_code: int, value: dict[str, Any]) -> None:
        content = json.dumps(value, allow_nan=False).encode("utf-8")
        self.send_response(status_code)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(content)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(content)


def create_application_server(
    output_dir: str | Path,
    report_path: str | Path,
    model_dirs: dict[str, Path],
    commands: dict[str, list[str]],
    fallback_dir: str | Path,
    app_dir: str | Path,
    synthetic_demo: bool,
    host: str = "127.0.0.1",
    port: int = 8765,
) -> ThreadingHTTPServer:
    try:
        if not ipaddress.ip_address(host).is_loopback:
            raise ValueError("host must be loopback")
    except ValueError as error:
        raise ValueError("host must be a loopback IP address such as 127.0.0.1") from error
    handler = type("ConfiguredApplicationHandler", (_ApplicationHandler,), {
        "output_dir": Path(output_dir).resolve(),
        "report_path": Path(report_path).resolve(),
        "app_dir": Path(app_dir).resolve(),
        "model_dirs": {name: Path(path).resolve() for name, path in model_dirs.items()},
        "commands": {name: list(command) for name, command in commands.items()},
        "fallback_dir": Path(fallback_dir).resolve(),
        "synthetic_demo": synthetic_demo,
        "job_lock": threading.Lock(),
        "job_status": {"state": "idle", "active_model": None, "returncode": None, "logs": ""},
    })
    return ThreadingHTTPServer((host, port), handler)


def serve_application(
    *,
    output_dir: str | Path,
    data_dir: str | Path | None = None,
    synthetic_demo: bool = False,
    host: str = "127.0.0.1",
    port: int = 8765,
    run_options: dict[str, Any] | None = None,
) -> None:
    project_root = Path(__file__).resolve().parents[2]
    output = Path(output_dir).resolve()
    if synthetic_demo:
        fallback_dir = output / "run"
        report_path = fallback_dir / "report.html"
    else:
        fallback_dir = output
        report_path = output / "report.html"
    fallback_dir.mkdir(parents=True, exist_ok=True)
    if not report_path.exists():
        write_html_report(fallback_dir)

    options = run_options or {}
    model_dirs: dict[str, Path] = {}
    commands: dict[str, list[str]] = {}
    for model in MODEL_IDS:
        model_root = output / "models" / model
        model_dirs[model] = model_root / "run" if synthetic_demo else model_root
        if synthetic_demo:
            commands[model] = [
                sys.executable,
                str(project_root / "scripts" / "run_synthetic_demo.py"),
                "--output-dir", str(model_root),
                "--model", model,
            ]
            continue
        if data_dir is None:
            raise ValueError("--data-dir is required unless --synthetic-demo is selected")
        selected_models = ["logistic_regression"] if model == "cnn_1d" else [model]
        command = [
            sys.executable, "-m", "sepsis_prediction.cli", "run",
            "--data-dir", str(Path(data_dir).resolve()),
            "--output-dir", str(model_root),
            "--test-size", str(options.get("test_size", 0.2)),
            "--validation-size", str(options.get("validation_size", 0.2)),
            "--random-state", str(options.get("random_state", 42)),
            "--target-sensitivity", str(options.get("target_sensitivity", 0.8)),
            "--horizon-hours", str(options.get("horizon_hours", 24)),
            "--models", *selected_models,
        ]
        if options.get("threshold") is not None:
            command.extend(["--threshold", str(options["threshold"])])
        if model == "cnn_1d":
            command.extend(["--cnn", "--cnn-epochs", str(options.get("cnn_epochs", 20))])
        commands[model] = command

    server = create_application_server(
        output, report_path, model_dirs, commands, fallback_dir,
        Path(__file__).resolve().parent / "web", synthetic_demo, host, port,
    )
    print(f"Sepsis pipeline application: http://{host}:{server.server_port}/")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
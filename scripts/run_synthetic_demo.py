"""Run the complete workflow on generated, non-clinical example records."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import pandas as pd


def main() -> None:
    project_root = Path(__file__).resolve().parents[1]
    output_dir = project_root / "outputs" / "synthetic_e2e"
    output_dir.mkdir(parents=True, exist_ok=True)
    random_generator = np.random.default_rng(20261009)

    with TemporaryDirectory(prefix="sepsis_synthetic_") as temporary_dir:
        data_dir = Path(temporary_dir) / "psv"
        data_dir.mkdir()
        for patient in range(72):
            positive = patient % 4 == 0
            rows = []
            for hour in range(8):
                rows.append({
                    "HR": 68 + patient % 13 + hour * (0.8 if positive else 0.15)
                    + random_generator.normal(0, 1),
                    "O2Sat": 97 - (hour * 0.25 if positive else 0)
                    + random_generator.normal(0, 0.4),
                    "Temp": 36.5 + (hour * 0.05 if positive else 0)
                    + random_generator.normal(0, 0.08),
                    "SBP": 118 - (hour * 0.7 if positive else 0)
                    + random_generator.normal(0, 2),
                    "MAP": 82 - (hour * 0.45 if positive else 0)
                    + random_generator.normal(0, 1.5),
                    "Resp": 16 + (hour * 0.35 if positive else 0)
                    + random_generator.normal(0, 0.8),
                    "Age": 25 + patient % 60,
                    "Gender": patient % 2,
                    "HospAdmTime": -4 - patient % 20,
                    "SepsisLabel": int(positive and hour >= 6),
                })
            frame = pd.DataFrame(rows)
            if patient % 5 == 0:
                frame.loc[2, "Temp"] = np.nan
            if patient % 7 == 0:
                frame.loc[1:3, "O2Sat"] = np.nan
            frame.to_csv(data_dir / f"patient_{patient:03d}.psv", sep="|", index=False)

        commands = [
            [sys.executable, "-m", "sepsis_prediction.cli", "validate", "--data-dir", str(data_dir)],
            [
                sys.executable, "-m", "sepsis_prediction.cli", "eda",
                "--data-dir", str(data_dir), "--output-dir", str(output_dir / "eda"),
            ],
            [
                sys.executable, "-m", "sepsis_prediction.cli", "run",
                "--data-dir", str(data_dir), "--output-dir", str(output_dir / "run"),
                "--models", "logistic_regression", "random_forest", "xgboost",
                "--cnn", "--cnn-epochs", "1", "--random-state", "2026",
            ],
        ]
        for command in commands:
            result = subprocess.run(command, cwd=project_root, text=True, capture_output=True)
            print(result.stdout.strip())
            if result.returncode:
                print(result.stderr.strip())
                raise subprocess.CalledProcessError(result.returncode, command)

    report_path = output_dir / "run" / "report.html"
    report = report_path.read_text(encoding="utf-8")
    required_sections = (
        "Model results",
        "Sample of source observations",
        "Pipeline and data lineage",
        "Discrimination curves",
        "cnn_1d",
        "xgboost",
    )
    missing_sections = [section for section in required_sections if section.lower() not in report.lower()]
    if missing_sections:
        raise RuntimeError(f"Generated HTML report is missing sections: {missing_sections}")

    metrics = json.loads((output_dir / "run" / "metrics.json").read_text(encoding="utf-8"))
    print(f"Synthetic end-to-end run complete. Models: {', '.join(metrics)}")
    print(f"HTML report: {report_path}")
    print("Synthetic outcomes are fabricated for workflow verification, not clinical evidence.")


if __name__ == "__main__":
    main()
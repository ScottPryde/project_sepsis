"""Run the complete workflow on generated, non-clinical example records."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import pandas as pd


def main() -> None:
    os.environ["OMP_NUM_THREADS"] = "1"
    parser = argparse.ArgumentParser(description="Run the workflow on fabricated, non-clinical records.")
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument(
        "--model",
        choices=("logistic_regression", "random_forest", "xgboost", "cnn_1d"),
        default=None,
        help="run one model only; the CNN uses Logistic Regression for its shared split",
    )
    args = parser.parse_args()
    project_root = Path(__file__).resolve().parents[1]
    output_dir = args.output_dir or project_root / "outputs" / "synthetic_e2e"
    output_dir = output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    random_generator = np.random.default_rng(20261009)

    with TemporaryDirectory(prefix="sepsis_synthetic_") as temporary_dir:
        # Two source folders mimic PhysioNet's training_setA/B so source tagging,
        # every cohort exclusion, sparse labs, and late record starts are exercised.
        source_dirs = [Path(temporary_dir) / "synthetic_hospital_A", Path(temporary_dir) / "synthetic_hospital_B"]
        for source_dir in source_dirs:
            source_dir.mkdir()
        for patient in range(160):
            source = patient % 2
            positive = patient % 6 == 0
            onset = 3 if patient % 24 == 0 else 6 + patient % 10
            if patient % 37 == 0:
                length = 4
            elif patient % 41 == 0 and not positive:
                length = 6
            elif patient % 9 == 0 and not positive:
                length = 18
            else:
                length = 30 + patient % 11
            first_iculos = 1 + (patient % 7 if patient % 5 == 0 else 0)
            rows = []
            for hour in range(length):
                trend = hour * 0.4 if positive else 0
                row = {
                    "HR": 68 + patient % 13 + trend * 0.8 + hour * 0.15
                    + random_generator.normal(0, 3),
                    "O2Sat": 97 - trend * 0.25 + random_generator.normal(0, 0.8),
                    "Temp": 36.5 + trend * 0.05 + random_generator.normal(0, 0.2),
                    "SBP": 118 - trend * 0.7 + random_generator.normal(0, 5),
                    "MAP": 82 - trend * 0.45 + random_generator.normal(0, 3),
                    "Resp": 16 + trend * 0.35 + random_generator.normal(0, 1.5),
                    "Lactate": np.nan,
                    "WBC": np.nan,
                    "Creatinine": np.nan,
                    "Age": 25 + patient % 60,
                    "Gender": patient % 2,
                    "Unit1": float(source == 0),
                    "Unit2": float(source == 1),
                    "HospAdmTime": -4 - patient % 20,
                    "ICULOS": first_iculos + hour,
                    "SepsisLabel": int(positive and hour >= onset),
                }
                if hour % 8 == patient % 8:
                    row["Lactate"] = 1.2 + (0.6 if positive else 0) + random_generator.normal(0, 0.6)
                    row["WBC"] = 9 + (2 if positive else 0) + random_generator.normal(0, 3)
                    row["Creatinine"] = 1.0 + random_generator.normal(0, 0.2)
                rows.append(row)
            frame = pd.DataFrame(rows)
            if patient % 5 == 0:
                frame.loc[2, "Temp"] = np.nan
            if patient % 7 == 0:
                frame.loc[1:3, "O2Sat"] = np.nan
            if source == 1:
                frame["Temp"] = frame["Temp"].where(frame.index % 4 == 0)
            prefix = "p0" if source == 0 else "p1"
            frame.to_csv(source_dirs[source] / f"{prefix}{patient:05d}.psv", sep="|", index=False)
        data_arguments = ["--data-dir", *(str(source_dir) for source_dir in source_dirs)]

        commands = []
        if args.model is None:
            commands.extend([
                [sys.executable, "-m", "sepsis_prediction.cli", "validate", *data_arguments],
                [
                sys.executable, "-m", "sepsis_prediction.cli", "eda",
                *data_arguments, "--output-dir", str(output_dir / "eda"),
                ],
            ])
        if args.model == "cnn_1d":
            selected_models = ("logistic_regression",)
            run_cnn = True
        elif args.model:
            selected_models = (args.model,)
            run_cnn = False
        else:
            selected_models = ("logistic_regression", "random_forest", "xgboost")
            run_cnn = True
        run_command = [
            sys.executable, "-m", "sepsis_prediction.cli", "run",
            *data_arguments, "--output-dir", str(output_dir / "run"),
            "--models", *selected_models,
            "--random-state", "2026", "--synthetic-demo",
        ]
        if run_cnn:
            run_command.extend(["--cnn", "--cnn-epochs", "1"])
        commands.append(run_command)
        for command in commands:
            result = subprocess.run(command, cwd=project_root, text=True, capture_output=True)
            print(result.stdout.strip())
            if result.returncode:
                print(result.stderr.strip())
                raise subprocess.CalledProcessError(result.returncode, command)

    report_path = output_dir / "run" / "report.html"
    report = report_path.read_text(encoding="utf-8")
    required_sections = [
        "Model results",
        "Sample of source observations",
        "Pipeline and data lineage",
        "Discrimination curves",
    ]
    if args.model in (None, "cnn_1d"):
        required_sections.append("cnn_1d")
    if args.model in (None, "xgboost"):
        required_sections.append("xgboost")
    missing_sections = [section for section in required_sections if section.lower() not in report.lower()]
    if missing_sections:
        raise RuntimeError(f"Generated HTML report is missing sections: {missing_sections}")

    cohort = json.loads((output_dir / "run" / "cohort_summary.json").read_text(encoding="utf-8"))
    untriggered = [reason for reason, count in cohort["exclusions"].items() if count == 0]
    if untriggered or len(cohort["by_source"]) != 2:
        raise RuntimeError(f"Synthetic cohort no longer exercises exclusions {untriggered} or both sources")

    metrics = json.loads((output_dir / "run" / "metrics.json").read_text(encoding="utf-8"))
    print(f"Synthetic end-to-end run complete. Models: {', '.join(metrics)}")
    print(f"HTML report: {report_path}")
    print("Synthetic outcomes are fabricated for workflow verification, not clinical evidence.")


if __name__ == "__main__":
    main()
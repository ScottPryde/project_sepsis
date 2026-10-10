import json
import sys

import pandas as pd

from sepsis_prediction.cli import main


def test_cli_validate_prints_json_serializable_counts(tmp_path, monkeypatch, capsys) -> None:
    data_dir = tmp_path / "psv"
    data_dir.mkdir()
    pd.DataFrame({"SepsisLabel": [0, 1]}).to_csv(
        data_dir / "patient.psv", sep="|", index=False,
    )
    monkeypatch.setattr(sys, "argv", [
        "sepsis-pipeline", "validate", "--data-dir", str(data_dir),
    ])

    main()

    report = json.loads(capsys.readouterr().out)
    assert report["patients"] == 1
    assert report["rows"] == 2
    assert report["positive_patient_files"] == 1
    assert "files" not in report


def test_cli_validate_lists_files_only_when_verbose(tmp_path, monkeypatch, capsys) -> None:
    data_dir = tmp_path / "psv"
    data_dir.mkdir()
    pd.DataFrame({"SepsisLabel": [0]}).to_csv(data_dir / "patient.psv", sep="|", index=False)
    monkeypatch.setattr(sys, "argv", [
        "sepsis-pipeline", "validate", "--data-dir", str(data_dir), "--verbose",
    ])

    main()

    assert json.loads(capsys.readouterr().out)["files"] == ["patient"]


def test_cli_run_ingests_psv_and_writes_cohort_and_model_artifacts(tmp_path, monkeypatch) -> None:
    data_dir = tmp_path / "psv"
    output_dir = tmp_path / "run"
    data_dir.mkdir()

    for patient_number in range(30):
        labels = [0] * 8
        if patient_number % 3 == 0:
            labels[6] = 1
        pd.DataFrame({
            "HR": [60 + patient_number + hour for hour in range(8)],
            "SepsisLabel": labels,
        }).to_csv(data_dir / f"patient_{patient_number:03d}.psv", sep="|", index=False)

    monkeypatch.setattr(sys, "argv", [
        "sepsis-pipeline", "run",
        "--data-dir", str(data_dir),
        "--output-dir", str(output_dir),
        "--models", "logistic_regression",
        "--random-state", "17",
        "--horizon-hours", "2",
        "--synthetic-demo",
    ])
    main()

    cohort = json.loads((output_dir / "cohort_summary.json").read_text(encoding="utf-8"))
    metrics = json.loads((output_dir / "metrics.json").read_text(encoding="utf-8"))
    assert cohort == {
        "eligible_patients": 30,
        "positive_outcomes": 10,
        "horizon_hours": 2,
        "exclusions": {
            "too_short": 0,
            "no_outcome_followup": 0,
            "incomplete_horizon_followup": 0,
            "positive_in_lookback": 0,
        },
    }
    assert set(metrics) == {"logistic_regression"}
    assert (output_dir / "logistic_regression.joblib").is_file()
    assert (output_dir / "test_predictions.csv").is_file()
    assert (output_dir / "model_comparison.csv").is_file()
    assert (output_dir / "model_curves.png").is_file()
    report = (output_dir / "report.html").read_text(encoding="utf-8")
    assert "Synthetic smoke test" in report
    assert "Methodology" in report
    assert "Written performance review" in report
    assert "Logistic Regression" in report
    assert "Leading fit-set features" in report
    assert "constructed separability" in report
    assert "Datasource, ingestion, and field validation" in report
    assert "SepsisLabel" in report
    assert "No physiologic range checks" in report
    assert "validation patients" in report
    assert (output_dir / "model_evaluation.json").is_file()
    assert (output_dir / "model_ranking.csv").is_file()
    assert "Model results" in report
    assert "Pipeline and data lineage" in report
    assert "Sample of source observations" in report
    assert "Sample 1" in report
    assert "data:image/png;base64," in report
    assert "<svg" in report
    assert "https://" not in report


def test_cli_eda_writes_dataset_summaries_and_plots(tmp_path, monkeypatch) -> None:
    data_dir = tmp_path / "psv"
    output_dir = tmp_path / "eda"
    data_dir.mkdir()
    pd.DataFrame({
        "HR": [70.0, None, 72.0],
        "SepsisLabel": [0, 0, 1],
    }).to_csv(data_dir / "patient.psv", sep="|", index=False)
    monkeypatch.setattr(sys, "argv", [
        "sepsis-pipeline", "eda",
        "--data-dir", str(data_dir),
        "--output-dir", str(output_dir),
    ])

    main()

    summary = json.loads((output_dir / "dataset_summary.json").read_text(encoding="utf-8"))
    assert summary["patient_count"] == 1
    assert summary["hourly_observation_count"] == 3
    assert summary["label_counts"] == {"0": 2, "1": 1}
    assert (output_dir / "missingness.csv").is_file()
    assert (output_dir / "record_lengths.csv").is_file()
    assert (output_dir / "labels_by_hour.csv").is_file()
    assert (output_dir / "missingness.png").is_file()
    assert (output_dir / "record_lengths.png").is_file()
    assert (output_dir / "labels_by_hour.png").is_file()
    report = (output_dir / "report.html").read_text(encoding="utf-8")
    assert "Dataset summary" in report or "Study snapshot" in report
    assert "Sample of source observations" in report
    assert "Clinical variable missingness chart" in report
    assert "<svg" in report
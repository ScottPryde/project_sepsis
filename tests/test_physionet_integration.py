import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from sklearn.model_selection import StratifiedShuffleSplit

from sepsis_prediction.cli import main
from sepsis_prediction.data import (
    KNOWN_CLINICAL_COLUMNS,
    PatientRecord,
    load_patient_files,
    read_record_cache,
    write_record_cache,
)
from sepsis_prediction.features import (
    LOOKBACK_HOURS,
    STATIC_COLUMNS,
    cohort_audit,
    create_examples,
    summarise_cohort,
)
from sepsis_prediction.modeling import split_patients


def write_patient(directory: Path, patient_id: str, labels: list[int], start_hour: int = 1, hr_offset: float = 0) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({
        "HR": [70 + hr_offset + hour for hour in range(len(labels))],
        "ICULOS": list(range(start_hour, start_hour + len(labels))),
        "SepsisLabel": labels,
    }).to_csv(directory / f"{patient_id}.psv", sep="|", index=False)


def two_hospitals(root: Path, patients_per_source: int = 30) -> list[Path]:
    folders = [root / "training_setA", root / "training_setB"]
    for prefix, folder in zip(("p0", "p1"), folders):
        for number in range(patients_per_source):
            labels = [0] * 8
            if number % 3 == 0:
                labels[6] = 1
            write_patient(folder, f"{prefix}{number:05d}", labels, start_hour=1 + number % 4, hr_offset=5 * (labels[6]))
    return folders


def reference_features(frame: pd.DataFrame) -> dict[str, float]:
    """The original per-patient pandas implementation, kept as an oracle."""
    window = frame.iloc[:LOOKBACK_HOURS]
    features: dict[str, float] = {}
    for column in KNOWN_CLINICAL_COLUMNS:
        if column in window.columns:
            values = pd.to_numeric(window[column], errors="coerce").astype(float)
        else:
            values = pd.Series(np.nan, index=window.index, dtype=float)
        observed = values.dropna()
        features[f"{column}_latest"] = float(observed.iloc[-1]) if not observed.empty else np.nan
        features[f"{column}_missing_fraction"] = float(values.isna().mean())
        features[f"{column}_any_missing"] = float(values.isna().any())
        if column in STATIC_COLUMNS:
            continue
        features[f"{column}_mean"] = float(observed.mean()) if not observed.empty else np.nan
        features[f"{column}_min"] = float(observed.min()) if not observed.empty else np.nan
        features[f"{column}_max"] = float(observed.max()) if not observed.empty else np.nan
        features[f"{column}_std"] = float(observed.std(ddof=0)) if not observed.empty else np.nan
        features[f"{column}_delta"] = float(observed.iloc[-1] - observed.iloc[0]) if len(observed) >= 2 else np.nan
    return features


def test_multiple_folders_are_merged_in_filename_order_with_sources(tmp_path: Path) -> None:
    folders = two_hospitals(tmp_path, patients_per_source=3)

    records = load_patient_files(folders)

    assert [record.patient_id for record in records] == [
        "p000000", "p000001", "p000002", "p100000", "p100001", "p100002",
    ]
    assert [record.source for record in records] == ["training_setA"] * 3 + ["training_setB"] * 3
    assert [record.patient_id for record in load_patient_files(folders, workers=2)] == [
        record.patient_id for record in records
    ]


def test_duplicate_patient_ids_across_folders_are_rejected(tmp_path: Path) -> None:
    write_patient(tmp_path / "a", "p000001", [0] * 7)
    write_patient(tmp_path / "b", "p000001", [0] * 7)

    with pytest.raises(ValueError, match="Duplicate patient identifiers"):
        load_patient_files([tmp_path / "a", tmp_path / "b"])


def test_cache_round_trip_preserves_records_and_features(tmp_path: Path) -> None:
    pytest.importorskip("pyarrow")
    folders = two_hospitals(tmp_path / "raw", patients_per_source=6)
    pd.DataFrame({"SepsisLabel": [0] * 31, "Temp": [37.0] * 31}).to_csv(
        folders[0] / "p0no_iculos.psv", sep="|", index=False,
    )
    records = load_patient_files(folders)

    cached = read_record_cache(write_record_cache(records, tmp_path / "cache" / "records.parquet"))

    assert [(record.patient_id, record.source) for record in cached] == [
        (record.patient_id, record.source) for record in records
    ]
    assert "ICULOS" not in next(record for record in cached if record.patient_id == "p0no_iculos").frame
    original = create_examples(records)
    restored = create_examples(cached)
    pd.testing.assert_frame_equal(original[0], restored[0])
    assert original[1].equals(restored[1])
    assert original[3] == restored[3]


def test_vectorised_features_match_reference_implementation() -> None:
    rng = np.random.default_rng(7)
    records = []
    for number in range(25):
        frame = pd.DataFrame(
            rng.normal(50, 10, size=(10, len(KNOWN_CLINICAL_COLUMNS))), columns=list(KNOWN_CLINICAL_COLUMNS),
        )
        frame = frame.mask(rng.random(frame.shape) < 0.5 + 0.02 * number)
        frame = frame.drop(columns=rng.choice(frame.columns, size=3, replace=False))
        frame["SepsisLabel"] = [0] * 10
        records.append(PatientRecord(f"p{number:03d}", frame))

    features, _, _, _ = create_examples(records, horizon_hours=4)

    expected = pd.DataFrame([reference_features(record.frame) for record in records], index=features.index)
    pd.testing.assert_frame_equal(features, expected, check_exact=False, rtol=1e-12, atol=1e-12)


def test_pooled_split_matches_previous_stratified_split() -> None:
    labels = pd.Series([index % 4 == 0 for index in range(80)], dtype="int64", index=[f"p{i}" for i in range(80)])
    groups = pd.Series(labels.index, index=labels.index, dtype="string")

    split = split_patients(labels, groups, random_state=9)

    _, expected_test = next(StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=9).split(labels, labels))
    assert np.array_equal(split["test"], expected_test)


def test_hospital_split_tests_on_unseen_hospital_only() -> None:
    index = [f"p{i}" for i in range(80)]
    labels = pd.Series([i % 4 == 0 for i in range(80)], dtype="int64", index=index)
    groups = pd.Series(index, index=index, dtype="string")
    sources = pd.Series(["A"] * 50 + ["B"] * 30, index=index)

    split = split_patients(
        labels, groups, split_mode="hospital", sources=sources, train_sources=("A",), test_sources=("B",),
    )

    assert set(sources.iloc[split["test"]]) == {"B"}
    assert len(split["test"]) == 30
    assert set(sources.iloc[np.concatenate([split["fit"], split["validation"]])]) == {"A"}
    with pytest.raises(ValueError, match="Unknown sources"):
        split_patients(labels, groups, split_mode="hospital", sources=sources, train_sources=("A",), test_sources=("C",))


def test_cohort_audit_records_start_hour_and_source(tmp_path: Path) -> None:
    folders = two_hospitals(tmp_path, patients_per_source=4)
    write_patient(folders[1], "p1short", [0] * 7, start_hour=12)

    audit = cohort_audit(load_patient_files(folders), horizon_hours=2)
    summary = summarise_cohort(audit, horizon_hours=2)

    assert audit.set_index("patient_id").loc["p1short", "iculos_at_row_0"] == 12
    assert audit.set_index("patient_id").loc["p1short", "status"] == "incomplete_horizon_followup"
    assert summary["by_source"]["training_setB"]["exclusions"]["incomplete_horizon_followup"] == 1
    assert summary["eligible_iculos_at_row_0"]["starts_after_hour_1"] == 6


def test_cli_hospital_split_run_from_cache(tmp_path: Path, monkeypatch, capsys) -> None:
    pytest.importorskip("pyarrow")
    folders = two_hospitals(tmp_path / "raw")
    cache = tmp_path / "records.parquet"
    monkeypatch.setattr(sys, "argv", [
        "sepsis-pipeline", "cache", "--data-dir", *map(str, folders), "--output", str(cache),
    ])
    main()
    assert json.loads(capsys.readouterr().out)["sources"] == {"training_setA": 30, "training_setB": 30}

    output_dir = tmp_path / "a_to_b"
    monkeypatch.setattr(sys, "argv", [
        "sepsis-pipeline", "run", "--cache", str(cache), "--output-dir", str(output_dir),
        "--models", "logistic_regression", "--horizon-hours", "2",
        "--split-mode", "hospital", "--train-sources", "training_setA", "--test-sources", "training_setB",
    ])
    main()

    config = json.loads((output_dir / "run_config.json").read_text(encoding="utf-8"))
    split = json.loads((output_dir / "split_patients.json").read_text(encoding="utf-8"))
    cohort = json.loads((output_dir / "cohort_summary.json").read_text(encoding="utf-8"))
    assert config["split_mode"] == "hospital"
    assert config["split_composition"]["test"]["by_source"] == {"training_setB": 30}
    assert all(patient.startswith("p1") for patient in split["test"])
    assert all(patient.startswith("p0") for patient in split["fit"] + split["validation"])
    assert set(cohort["by_source"]) == {"training_setA", "training_setB"}
    comparison = pd.read_csv(output_dir / "model_comparison.csv")
    expected_lift = comparison["auprc_average_precision"] / (10 / 30)
    assert np.allclose(comparison["auprc_over_prevalence"], expected_lift)
    report = (output_dir / "report.html").read_text(encoding="utf-8")
    assert "Cohort by source" in report
    assert "Eligible records starting after ICU hour 1" in report

    comparison_dir = tmp_path / "comparison"
    monkeypatch.setattr(sys, "argv", [
        "sepsis-pipeline", "compare", "--run", f"a_to_b={output_dir}", "--output-dir", str(comparison_dir),
    ])
    main()
    cohorts = pd.read_csv(comparison_dir / "comparison_cohorts.csv")
    models = pd.read_csv(comparison_dir / "comparison_models.csv")
    assert cohorts.loc[0, "test_sources"] == "training_setB"
    assert cohorts.loc[0, "test_patients"] == 30
    assert models["model"].tolist() == ["logistic_regression"]
    page = (comparison_dir / "comparison.html").read_text(encoding="utf-8")
    assert "AUPRC ÷ prevalence" in page
    assert "https://" not in page


def test_eda_reports_each_source(tmp_path: Path, monkeypatch) -> None:
    folders = two_hospitals(tmp_path / "raw", patients_per_source=3)
    output_dir = tmp_path / "eda"
    monkeypatch.setattr(sys, "argv", [
        "sepsis-pipeline", "eda", "--data-dir", *map(str, folders), "--output-dir", str(output_dir),
    ])

    main()

    summary = json.loads((output_dir / "dataset_summary.json").read_text(encoding="utf-8"))
    assert summary["patient_count"] == 6
    assert summary["by_source"]["training_setB"]["patient_count"] == 3
    hourly = pd.read_csv(output_dir / "labels_by_hour.csv")
    assert hourly.loc[hourly["hour_index"] == 6, "positive_labels"].item() == 2

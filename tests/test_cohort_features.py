from pathlib import Path

import pandas as pd

from sepsis_prediction.data import load_patient_files, validate_patient_frame
from sepsis_prediction.features import (
    LOOKBACK_HOURS,
    create_examples,
    create_patient_example,
    create_sequence_examples,
)


def write_patient(directory: Path, patient_id: str, labels: list[int], hr: list[float] | None = None) -> None:
    frame = pd.DataFrame({"SepsisLabel": labels})
    if hr is not None:
        frame["HR"] = hr
    frame.to_csv(directory / f"{patient_id}.psv", sep="|", index=False)


def test_outcome_uses_bounded_horizon_and_excludes_lookback_positive(tmp_path: Path) -> None:
    write_patient(tmp_path, "early", [0, 0, 0, 0, 0, 1, 1])
    write_patient(tmp_path, "hour_six", [0, 0, 0, 0, 0, 0, 1])
    write_patient(tmp_path, "negative", [0] * 8)
    write_patient(tmp_path, "incomplete_negative", [0] * 7)
    write_patient(tmp_path, "outside_horizon", [0] * 8 + [1])
    write_patient(tmp_path, "no_followup", [0] * LOOKBACK_HOURS)
    write_patient(tmp_path, "short", [0] * (LOOKBACK_HOURS - 1))

    records = load_patient_files(tmp_path)
    features, labels, groups, exclusions = create_examples(records, horizon_hours=2)

    assert labels.to_dict() == {"hour_six": 1, "negative": 0, "outside_horizon": 0}
    assert groups.tolist() == ["hour_six", "negative", "outside_horizon"]
    assert features.index.tolist() == ["hour_six", "negative", "outside_horizon"]
    assert exclusions == {
        "too_short": 1,
        "no_outcome_followup": 1,
        "incomplete_horizon_followup": 1,
        "positive_in_lookback": 1,
    }


def test_positive_inside_horizon_is_kept_with_short_followup(tmp_path: Path) -> None:
    write_patient(tmp_path, "inside_horizon", [0] * 6 + [0, 1])
    records = load_patient_files(tmp_path)

    _, labels, _, exclusions = create_examples(records, horizon_hours=3)

    assert labels.to_dict() == {"inside_horizon": 1}
    assert exclusions["incomplete_horizon_followup"] == 0


def test_predictors_exclude_target_patient_id_and_iculos() -> None:
    frame = pd.DataFrame({
        "SepsisLabel": [0, 0, 0, 0, 0, 0, 1],
        "HR": [70, 71, 72, 73, 74, 75, 200],
        "ICULOS": list(range(7)),
    })

    example = create_patient_example(frame)

    assert example is not None
    features, outcome = example
    assert outcome == 1
    assert "SepsisLabel" not in features
    assert "patient_id" not in features
    assert not any(name.startswith("ICULOS_") for name in features)
    assert features["HR_latest"] == 75
    assert features["HR_mean"] == 72.5


def test_missing_and_all_nan_clinical_columns_have_stable_features() -> None:
    frame = pd.DataFrame({"SepsisLabel": [0] * 6 + [1], "HR": [float("nan")] * 7})

    example = create_patient_example(frame)

    assert example is not None
    features, outcome = example
    assert outcome == 1
    assert pd.isna(features["HR_latest"])
    assert pd.isna(features["HR_delta"])
    assert features["HR_missing_fraction"] == 1
    assert features["HR_any_missing"] == 1
    assert pd.isna(features["O2Sat_latest"])


def test_static_variables_keep_only_value_and_missingness_features() -> None:
    frame = pd.DataFrame({"SepsisLabel": [0] * 6 + [1], "Age": [65] * 7})

    example = create_patient_example(frame)

    assert example is not None
    features, _ = example
    assert features["Age_latest"] == 65
    assert features["Age_any_missing"] == 0
    assert "Age_mean" not in features


def test_iculos_must_be_contiguous_when_present() -> None:
    valid = pd.DataFrame({"SepsisLabel": [0, 0, 0], "ICULOS": [4, 5, 6]})
    invalid = pd.DataFrame({"SepsisLabel": [0, 0, 0], "ICULOS": [4, 6, 7]})

    assert validate_patient_frame(valid) == []
    assert any("increase by exactly one" in error for error in validate_patient_frame(invalid))


def test_sequence_examples_share_cohort_and_only_use_lookback(tmp_path: Path) -> None:
    write_patient(tmp_path, "eligible", [0, 0, 0, 0, 0, 0, 1], [60, 61, 62, 63, 64, 65, 250])
    write_patient(tmp_path, "early_positive", [0, 0, 0, 0, 0, 1, 1], [60] * 7)
    records = load_patient_files(tmp_path)

    tabular_features, tabular_labels, tabular_groups, _ = create_examples(records)
    sequences, sequence_labels, sequence_groups, _ = create_sequence_examples(records)

    assert sequence_labels.equals(tabular_labels)
    assert sequence_groups.equals(tabular_groups)
    assert sequences.shape == (1, LOOKBACK_HOURS, 39)
    assert sequences[0, :, 0].tolist() == [60, 61, 62, 63, 64, 65]
    assert tabular_features.loc["eligible", "HR_latest"] == 65
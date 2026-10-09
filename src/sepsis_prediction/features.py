"""Leakage-conscious features from each patient's fixed initial window."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from sepsis_prediction.data import KNOWN_CLINICAL_COLUMNS, TARGET_COLUMN


LOOKBACK_HOURS = 6
DEFAULT_HORIZON_HOURS = 24
STATIC_COLUMNS = {"Age", "Gender", "Unit1", "Unit2", "HospAdmTime"}


def _create_patient_example(
    frame: pd.DataFrame,
    horizon_hours: int,
) -> tuple[tuple[dict[str, float], int] | None, str | None]:
    """Build one bounded-horizon example and explain exclusions."""
    if horizon_hours < 1:
        raise ValueError("horizon_hours must be a positive integer")
    if len(frame) < LOOKBACK_HOURS:
        return None, "too_short"
    labels = pd.to_numeric(frame[TARGET_COLUMN], errors="coerce")
    if labels.iloc[:LOOKBACK_HOURS].eq(1).any():
        return None, "positive_in_lookback"
    outcome_labels = labels.iloc[LOOKBACK_HOURS:LOOKBACK_HOURS + horizon_hours]
    if outcome_labels.eq(1).any():
        outcome = 1
    elif outcome_labels.empty:
        return None, "no_outcome_followup"
    elif len(outcome_labels) < horizon_hours:
        return None, "incomplete_horizon_followup"
    else:
        outcome = 0

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
        features[f"{column}_delta"] = (
            float(observed.iloc[-1] - observed.iloc[0]) if len(observed) >= 2 else np.nan
        )

    return (features, outcome), None


def create_patient_example(
    frame: pd.DataFrame,
    horizon_hours: int = DEFAULT_HORIZON_HOURS,
) -> tuple[dict[str, float], int] | None:
    """Build a bounded-horizon example, or return None when it is ineligible.

    Rows are treated as hourly observations in file order. Rows 0-5 provide all
    features; labels from row 6 through ``6 + horizon_hours - 1`` define whether
    sepsis occurs. Negative patients must have complete follow-up through that
    horizon. A positive observed within it remains eligible with shorter follow-up.
    A patient labeled positive anywhere in rows 0-5 is excluded.
    ICULOS is intentionally omitted because elapsed ICU time is not part of the
    stated six-row prediction design and may encode care-process timing.
    """
    example, _ = _create_patient_example(frame, horizon_hours)
    return example


def create_sequence_examples(
    records: list[Any],
    horizon_hours: int = DEFAULT_HORIZON_HOURS,
) -> tuple[np.ndarray, pd.Series, pd.Series, dict[str, int]]:
    """Build raw ordered look-back tensors with the tabular cohort definition.

    The returned array has shape ``(patients, hours, variables)``. Missing
    values remain NaN for train-only imputation in the sequence model.
    """
    sequences: list[np.ndarray] = []
    labels: list[int] = []
    patient_ids: list[str] = []
    exclusions = {
        "too_short": 0,
        "no_outcome_followup": 0,
        "incomplete_horizon_followup": 0,
        "positive_in_lookback": 0,
    }

    for record in records:
        example, exclusion = _create_patient_example(record.frame, horizon_hours)
        if example is None:
            exclusions[exclusion] += 1
            continue

        window = record.frame.iloc[:LOOKBACK_HOURS]
        values = window.reindex(columns=KNOWN_CLINICAL_COLUMNS)
        sequences.append(values.apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float))
        labels.append(example[1])
        patient_ids.append(record.patient_id)

    if not sequences:
        raise ValueError("No eligible patients remain after applying the prediction design")

    patient_index = pd.Index(patient_ids)
    return (
        np.stack(sequences),
        pd.Series(labels, index=patient_index, name=TARGET_COLUMN, dtype="int64"),
        pd.Series(patient_ids, index=patient_index, name="patient_id", dtype="string"),
        exclusions,
    )


def create_examples(
    records: list[Any],
    horizon_hours: int = DEFAULT_HORIZON_HOURS,
) -> tuple[pd.DataFrame, pd.Series, pd.Series, dict[str, int]]:
    """Create feature, outcome, and group arrays plus cohort exclusion counts."""
    feature_rows: list[dict[str, float]] = []
    labels: list[int] = []
    patient_ids: list[str] = []
    exclusions = {
        "too_short": 0,
        "no_outcome_followup": 0,
        "incomplete_horizon_followup": 0,
        "positive_in_lookback": 0,
    }

    for record in records:
        example, exclusion = _create_patient_example(record.frame, horizon_hours)
        if example is None:
            exclusions[exclusion] += 1
            continue
        features, label = example
        feature_rows.append(features)
        labels.append(label)
        patient_ids.append(record.patient_id)

    if not feature_rows:
        raise ValueError("No eligible patients remain after applying the prediction design")
    return (
        pd.DataFrame(feature_rows, index=patient_ids),
        pd.Series(labels, index=patient_ids, name=TARGET_COLUMN, dtype="int64"),
        pd.Series(patient_ids, index=patient_ids, name="patient_id", dtype="string"),
        exclusions,
    )
"""Leakage-conscious features from each patient's fixed initial window."""

from __future__ import annotations

import warnings
from typing import Any

import numpy as np
import pandas as pd

from sepsis_prediction.data import KNOWN_CLINICAL_COLUMNS, TARGET_COLUMN


LOOKBACK_HOURS = 6
DEFAULT_HORIZON_HOURS = 24
STATIC_COLUMNS = {"Age", "Gender", "Unit1", "Unit2", "HospAdmTime"}
EXCLUSION_REASONS = (
    "too_short",
    "no_outcome_followup",
    "incomplete_horizon_followup",
    "positive_in_lookback",
)


def _outcome_status(frame: pd.DataFrame, horizon_hours: int) -> tuple[int | None, str | None]:
    """Return the bounded-horizon outcome, or the reason the patient is excluded."""
    if horizon_hours < 1:
        raise ValueError("horizon_hours must be a positive integer")
    if len(frame) < LOOKBACK_HOURS:
        return None, "too_short"
    labels = pd.to_numeric(frame[TARGET_COLUMN], errors="coerce").to_numpy()
    if (labels[:LOOKBACK_HOURS] == 1).any():
        return None, "positive_in_lookback"
    outcome_labels = labels[LOOKBACK_HOURS:LOOKBACK_HOURS + horizon_hours]
    if (outcome_labels == 1).any():
        return 1, None
    if outcome_labels.size == 0:
        return None, "no_outcome_followup"
    if outcome_labels.size < horizon_hours:
        return None, "incomplete_horizon_followup"
    return 0, None


def _lookback_window(frame: pd.DataFrame) -> np.ndarray:
    """Rows 0-5 as an hours x variables array; absent columns become NaN."""
    window = frame.iloc[:LOOKBACK_HOURS].reindex(columns=KNOWN_CLINICAL_COLUMNS)
    return window.apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)


def _window_features(windows: np.ndarray) -> pd.DataFrame:
    """Summarise ``(patients, hours, variables)`` windows into the tabular feature schema.

    Per variable: latest observed value, missing fraction, and any-missing flag;
    dynamic variables also get mean, minimum, maximum, population standard
    deviation, and first-to-last observed delta.
    """
    observed = ~np.isnan(windows)
    counts = observed.sum(axis=1)
    hours = windows.shape[1]
    last_index = hours - 1 - np.argmax(observed[:, ::-1, :], axis=1)
    first_index = np.argmax(observed, axis=1)
    latest = np.take_along_axis(windows, last_index[:, None, :], axis=1)[:, 0, :]
    first = np.take_along_axis(windows, first_index[:, None, :], axis=1)[:, 0, :]
    latest = np.where(counts > 0, latest, np.nan)
    delta = np.where(counts >= 2, latest - first, np.nan)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        summaries = {
            "mean": np.nanmean(windows, axis=1),
            "min": np.nanmin(windows, axis=1),
            "max": np.nanmax(windows, axis=1),
            "std": np.nanstd(windows, axis=1),
        }
    missing_fraction = 1.0 - counts / hours
    any_missing = (counts < hours).astype(float)

    columns: dict[str, np.ndarray] = {}
    for position, column in enumerate(KNOWN_CLINICAL_COLUMNS):
        columns[f"{column}_latest"] = latest[:, position]
        columns[f"{column}_missing_fraction"] = missing_fraction[:, position]
        columns[f"{column}_any_missing"] = any_missing[:, position]
        if column in STATIC_COLUMNS:
            continue
        for name, values in summaries.items():
            columns[f"{column}_{name}"] = values[:, position]
        columns[f"{column}_delta"] = delta[:, position]
    return pd.DataFrame(columns)


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
    outcome, _ = _outcome_status(frame, horizon_hours)
    if outcome is None:
        return None
    features = _window_features(_lookback_window(frame)[None, :, :]).iloc[0]
    return {name: float(value) for name, value in features.items()}, outcome


def _eligible_records(
    records: list[Any],
    horizon_hours: int,
) -> tuple[list[Any], list[int], dict[str, int]]:
    eligible: list[Any] = []
    outcomes: list[int] = []
    exclusions = {reason: 0 for reason in EXCLUSION_REASONS}
    for record in records:
        outcome, exclusion = _outcome_status(record.frame, horizon_hours)
        if outcome is None:
            exclusions[exclusion] += 1
            continue
        eligible.append(record)
        outcomes.append(outcome)
    if not eligible:
        raise ValueError("No eligible patients remain after applying the prediction design")
    return eligible, outcomes, exclusions


def create_sequence_examples(
    records: list[Any],
    horizon_hours: int = DEFAULT_HORIZON_HOURS,
) -> tuple[np.ndarray, pd.Series, pd.Series, dict[str, int]]:
    """Build raw ordered look-back tensors with the tabular cohort definition.

    The returned array has shape ``(patients, hours, variables)``. Missing
    values remain NaN for train-only imputation in the sequence model.
    """
    eligible, outcomes, exclusions = _eligible_records(records, horizon_hours)
    patient_ids = [record.patient_id for record in eligible]
    patient_index = pd.Index(patient_ids)
    return (
        np.stack([_lookback_window(record.frame) for record in eligible]),
        pd.Series(outcomes, index=patient_index, name=TARGET_COLUMN, dtype="int64"),
        pd.Series(patient_ids, index=patient_index, name="patient_id", dtype="string"),
        exclusions,
    )


def create_examples(
    records: list[Any],
    horizon_hours: int = DEFAULT_HORIZON_HOURS,
) -> tuple[pd.DataFrame, pd.Series, pd.Series, dict[str, int]]:
    """Create feature, outcome, and group arrays plus cohort exclusion counts."""
    eligible, outcomes, exclusions = _eligible_records(records, horizon_hours)
    patient_ids = [record.patient_id for record in eligible]
    features = _window_features(np.stack([_lookback_window(record.frame) for record in eligible]))
    features.index = patient_ids
    return (
        features,
        pd.Series(outcomes, index=patient_ids, name=TARGET_COLUMN, dtype="int64"),
        pd.Series(patient_ids, index=patient_ids, name="patient_id", dtype="string"),
        exclusions,
    )


def cohort_audit(records: list[Any], horizon_hours: int = DEFAULT_HORIZON_HOURS) -> pd.DataFrame:
    """One row per loaded patient: source, length, record start, and cohort status.

    ``iculos_at_row_0`` records when each file starts in the ICU stay. The
    look-back window is always the first six rows; this column documents how
    far those rows are from ICU admission and is never a predictor.
    """
    rows = []
    for record in records:
        outcome, exclusion = _outcome_status(record.frame, horizon_hours)
        iculos = (
            pd.to_numeric(record.frame["ICULOS"], errors="coerce").iloc[0]
            if "ICULOS" in record.frame.columns and len(record.frame) else np.nan
        )
        rows.append({
            "patient_id": record.patient_id,
            "source": getattr(record, "source", ""),
            "rows": len(record.frame),
            "iculos_at_row_0": float(iculos),
            "status": "eligible" if outcome is not None else exclusion,
            "outcome": outcome,
        })
    return pd.DataFrame(rows).astype({"outcome": "Int64"})


def summarise_cohort(audit: pd.DataFrame, horizon_hours: int) -> dict[str, Any]:
    """Cohort counts overall and per source, plus the record-start distribution."""
    def counts(table: pd.DataFrame) -> dict[str, Any]:
        eligible = table[table["status"] == "eligible"]
        positives = int((eligible["outcome"] == 1).sum())
        iculos = eligible["iculos_at_row_0"].dropna()
        return {
            "loaded_patients": int(len(table)),
            "eligible_patients": int(len(eligible)),
            "positive_outcomes": positives,
            "prevalence": positives / len(eligible) if len(eligible) else None,
            "exclusions": {reason: int((table["status"] == reason).sum()) for reason in EXCLUSION_REASONS},
            "eligible_iculos_at_row_0": {
                "observed": int(len(iculos)),
                "starts_after_hour_1": int((iculos > 1).sum()),
                "median": float(iculos.median()) if len(iculos) else None,
                "p90": float(iculos.quantile(0.9)) if len(iculos) else None,
                "max": float(iculos.max()) if len(iculos) else None,
            },
        }

    return {
        "horizon_hours": horizon_hours,
        **counts(audit),
        "by_source": {str(source): counts(table) for source, table in audit.groupby("source", sort=True)},
    }

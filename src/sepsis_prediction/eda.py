"""Dataset-level summaries and plots that do not fit predictive models."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from sepsis_prediction.data import KNOWN_CLINICAL_COLUMNS, TARGET_COLUMN, PatientRecord


def run_eda(records: list[PatientRecord], output_dir: str | Path) -> dict[str, Any]:
    """Write cohort-level CSV, JSON, and PNG summaries for loaded records."""
    if not records:
        raise ValueError("EDA requires at least one patient record")

    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    lengths = pd.DataFrame({
        "patient_id": [record.patient_id for record in records],
        "rows": [len(record.frame) for record in records],
    })

    long = pd.concat(
        [
            record.frame.reindex(columns=[*KNOWN_CLINICAL_COLUMNS, TARGET_COLUMN]).assign(
                hour_index=range(len(record.frame)), source=record.source,
            )
            for record in records
        ],
        ignore_index=True,
    )
    for column in (*KNOWN_CLINICAL_COLUMNS, TARGET_COLUMN):
        long[column] = pd.to_numeric(long[column], errors="coerce")
    total_rows = len(long)
    observed_counts = long[list(KNOWN_CLINICAL_COLUMNS)].notna().sum()
    missingness = pd.DataFrame({
        "variable": list(KNOWN_CLINICAL_COLUMNS),
        "observed_count": [int(observed_counts[column]) for column in KNOWN_CLINICAL_COLUMNS],
    })
    missingness["missing_count"] = total_rows - missingness["observed_count"]
    missingness["missing_fraction"] = missingness["missing_count"] / total_rows if total_rows else None
    missingness = missingness.sort_values(
        ["missing_fraction", "variable"], ascending=[False, True], ignore_index=True,
    )

    positive = long[TARGET_COLUMN].eq(1)
    label_counts = {"0": int(long[TARGET_COLUMN].eq(0).sum()), "1": int(positive.sum())}
    hourly = (
        long.assign(positive=positive.astype(int))
        .groupby("hour_index", sort=True)
        .agg(patients_observed=("positive", "size"), positive_labels=("positive", "sum"))
        .reset_index()
    )
    hourly["positive_fraction"] = hourly["positive_labels"] / hourly["patients_observed"]

    lengths["source"] = [record.source for record in records]
    patient_positive = pd.Series(
        [bool(pd.to_numeric(record.frame[TARGET_COLUMN], errors="coerce").eq(1).any()) for record in records],
    )
    by_source = {}
    for source, rows in long.groupby("source", sort=True):
        in_source = lengths["source"] == source
        by_source[str(source)] = {
            "patient_count": int(in_source.sum()),
            "hourly_observation_count": int(len(rows)),
            "record_length_median": float(lengths.loc[in_source, "rows"].median()),
            "positive_patient_fraction": float(patient_positive[in_source.to_numpy()].mean()),
            "positive_label_fraction": float(rows[TARGET_COLUMN].eq(1).mean()),
        }
    lengths.to_csv(output / "record_lengths.csv", index=False)
    missingness.to_csv(output / "missingness.csv", index=False, float_format="%.8g")
    hourly.to_csv(output / "labels_by_hour.csv", index=False, float_format="%.8g")

    summary = {
        "patient_count": len(records),
        "hourly_observation_count": total_rows,
        "record_length_min": int(lengths["rows"].min()),
        "record_length_median": float(lengths["rows"].median()),
        "record_length_max": int(lengths["rows"].max()),
        "label_counts": label_counts,
        "positive_label_fraction": label_counts["1"] / total_rows if total_rows else None,
        "variables": list(KNOWN_CLINICAL_COLUMNS),
        "by_source": by_source,
    }
    (output / "dataset_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )

    figure, axis = plt.subplots(figsize=(9, 7))
    plotted = missingness.sort_values("missing_fraction", ascending=True)
    axis.barh(plotted["variable"], plotted["missing_fraction"], color="#247a78")
    axis.set(xlim=(0, 1), xlabel="Missing fraction", title="Clinical variable missingness")
    figure.tight_layout()
    figure.savefig(output / "missingness.png", dpi=160)
    plt.close(figure)

    figure, axis = plt.subplots(figsize=(8, 4.5))
    bins = min(40, max(8, int(lengths["rows"].nunique())))
    axis.hist(lengths["rows"], bins=bins, color="#d17a35", edgecolor="white")
    axis.set(xlabel="Hourly rows per patient", ylabel="Patients", title="Patient record lengths")
    figure.tight_layout()
    figure.savefig(output / "record_lengths.png", dpi=160)
    plt.close(figure)

    figure, axis = plt.subplots(figsize=(9, 4.5))
    axis.plot(hourly["hour_index"], hourly["positive_fraction"], color="#9d3e4b", linewidth=1.8)
    axis.set(
        xlabel="Hour index in patient record",
        ylabel="Fraction labeled positive",
        ylim=(0, 1),
        title="Sepsis labels by hour index",
    )
    figure.tight_layout()
    figure.savefig(output / "labels_by_hour.png", dpi=160)
    plt.close(figure)
    return summary
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

    total_rows = sum(len(record.frame) for record in records)
    missing_rows: list[dict[str, Any]] = []
    for column in KNOWN_CLINICAL_COLUMNS:
        observed_count = 0
        for record in records:
            if column in record.frame.columns:
                observed_count += int(pd.to_numeric(record.frame[column], errors="coerce").notna().sum())
        missing_count = total_rows - observed_count
        missing_rows.append({
            "variable": column,
            "observed_count": observed_count,
            "missing_count": missing_count,
            "missing_fraction": missing_count / total_rows if total_rows else None,
        })
    missingness = pd.DataFrame(missing_rows).sort_values(
        ["missing_fraction", "variable"], ascending=[False, True], ignore_index=True,
    )

    label_counts: dict[str, int] = {"0": 0, "1": 0}
    hourly_positive: dict[int, int] = {}
    hourly_rows: dict[int, int] = {}
    for record in records:
        labels = pd.to_numeric(record.frame[TARGET_COLUMN], errors="coerce").astype(int)
        counts = labels.value_counts()
        label_counts["0"] += int(counts.get(0, 0))
        label_counts["1"] += int(counts.get(1, 0))
        for hour, label in enumerate(labels):
            hourly_rows[hour] = hourly_rows.get(hour, 0) + 1
            hourly_positive[hour] = hourly_positive.get(hour, 0) + int(label == 1)

    hourly = pd.DataFrame([
        {
            "hour_index": hour,
            "patients_observed": hourly_rows[hour],
            "positive_labels": hourly_positive[hour],
            "positive_fraction": hourly_positive[hour] / hourly_rows[hour],
        }
        for hour in sorted(hourly_rows)
    ])
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
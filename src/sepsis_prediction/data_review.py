"""Field-by-field review of raw patient records, written as JSON, CSV, and one HTML page.

The review runs on records before any quality rule, so it shows the data as
delivered. It measures completeness, measurement cadence, distributions,
plausibility, outliers, carried-forward values, hospital differences, and
association with the cohort outcome, plus cross-field and record-level
integrity checks. It then counts what each quality rule in ``quality.py``
would change, which is the evidence for turning a rule on.
"""

from __future__ import annotations

import base64
import html
import json
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import ks_2samp, skew

from sepsis_prediction.data import PatientRecord
from sepsis_prediction.features import DEFAULT_HORIZON_HOURS, LOOKBACK_HOURS, cohort_audit, create_examples
from sepsis_prediction.quality import (
    AGE_TOP_CODE,
    CATEGORY_ORDER,
    FIELD_SPECS,
    QUALITY_RULES,
    evaluate_rules,
    records_to_table,
)


SOURCE_COLOURS = ("#2a6fb0", "#c9821f")
NEUTRAL = "#a9b3b0"
VIOLATION = "#c45042"
INK = "#1c2727"
INK_MUTED = "#52605f"
SURFACE = "#fcfcfb"
HOURS_PLOTTED = 48
KS_SAMPLE = 50000
SCATTER_SAMPLE = 15000
HIGH_MISSINGNESS = 0.9
SHIFT_KS = 0.2
AVAILABILITY_GAP = 0.3
CARRY_FORWARD_SHARE = 0.5
SMD_NOTABLE = 0.2
IQR_FENCE = 3.0

TESTS = (
    ("Completeness", "Share of hourly rows and of patients with any value, overall and per hospital.",
     "Missingness drives imputation and can encode care processes (which tests were ordered) rather than physiology."),
    ("Measurement cadence", "Observations per patient, median hours between observations, and observation rate by hour.",
     "The model only sees rows 0-5; a field measured once a day rarely appears in that window."),
    ("Distribution", "Mean, spread, quantiles, skew, distinct values, zeros and negatives.",
     "Confirms units and scale, and shows heavy tails that dominate scaling and distance-based methods."),
    ("Plausibility limits", "Values outside wide hard limits (impossible or instrument/entry errors).",
     "Single impossible values (e.g. FiO2 of 4000) distort means, scaling, and clustering."),
    ("Outliers", f"Values beyond {IQR_FENCE:g}× the interquartile range from the quartiles.",
     "Separates rare-but-real extremes from errors; reported, not removed."),
    ("Carried-forward values", "Share of consecutive hourly observations with an identical value.",
     "High repetition suggests values were copied forward at source, which overstates measurement frequency."),
    ("Hospital comparison", "Missingness, medians, and a Kolmogorov-Smirnov distance between hospitals A and B.",
     "Differences in practice or devices between sites limit how well models trained at one site transfer."),
    ("Outcome association", "Standardised mean difference of the six-hour summary between cohort outcome groups.",
     "Descriptive only: shows which fields carry early signal, and whether missingness itself does."),
    ("Cross-field consistency", "Pairs that must agree (DBP ≤ MAP ≤ SBP, Hct ≈ 3 × Hgb, direct ≤ total bilirubin, pH vs bicarbonate and PaCO2).",
     "Impossible combinations reveal errors that single-field limits miss."),
    ("Record integrity", "Label behaviour, ICU time continuity and start, admission fields, unit flags, duplicate rows.",
     "Confirms the assumptions the cohort definition relies on."),
    ("Rule impact", "Values, rows, and patients each proposed quality rule would change.",
     "Turns findings into rules whose cost is visible before they are applied."),
)


def _png(figure: plt.Figure, path: Path) -> str:
    figure.savefig(path, dpi=130, facecolor=SURFACE)
    plt.close(figure)
    return base64.b64encode(path.read_bytes()).decode("ascii")


def _style(axis: plt.Axes) -> None:
    axis.set_facecolor(SURFACE)
    for spine in ("top", "right"):
        axis.spines[spine].set_visible(False)
    for spine in ("left", "bottom"):
        axis.spines[spine].set_color("#c9d1ce")
    axis.tick_params(colors=INK_MUTED, labelsize=8)
    axis.xaxis.label.set_color(INK_MUTED)
    axis.yaxis.label.set_color(INK_MUTED)
    axis.title.set_color(INK)
    axis.title.set_fontsize(10)


def _number(value: Any, digits: int = 3) -> Any:
    if value is None or (isinstance(value, float) and not np.isfinite(value)):
        return None
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (float, np.floating)):
        return float(round(float(value), digits))
    return value


def _hour_rates(table: pd.DataFrame, fields: list[str]) -> pd.DataFrame:
    early = table[table["hour_index"] < HOURS_PLOTTED]
    observed = early.groupby(["source", "hour_index"])[fields].count()
    present = early.groupby(["source", "hour_index"]).size()
    return observed.div(present, axis=0)


def _cadence(table: pd.DataFrame, field: str) -> tuple[float | None, float | None]:
    observed = table.loc[table[field].notna(), ["patient_id", "hour_index", field]]
    if len(observed) < 2:
        return None, None
    same_patient = observed["patient_id"].eq(observed["patient_id"].shift())
    gaps = observed["hour_index"].diff()[same_patient]
    consecutive = same_patient & observed["hour_index"].diff().eq(1)
    repeated = observed[field].eq(observed[field].shift())[consecutive]
    return (
        float(gaps.median()) if len(gaps) else None,
        float(repeated.mean()) if len(repeated) else None,
    )


def _field_review(
    table: pd.DataFrame,
    field: str,
    sources: list[str],
    patient_counts: pd.DataFrame,
    patient_sources: pd.Series,
    cohort_features: pd.DataFrame,
    cohort_labels: pd.Series,
    rng: np.random.Generator,
) -> dict[str, Any]:
    spec = FIELD_SPECS[field]
    values = table[field]
    observed = values.dropna()
    review: dict[str, Any] = {
        "field": field, "category": spec.category, "unit": spec.unit, "description": spec.description,
        "limits": {"low": spec.low, "high": spec.high, "allowed": list(spec.allowed) if spec.allowed else None},
        "observed": int(len(observed)),
        "missing_fraction": _number(1 - len(observed) / len(values)),
        "patients_with_any": _number(float((patient_counts[field] > 0).mean())),
    }
    per_patient = patient_counts.loc[patient_counts[field] > 0, field]
    review["median_observations_per_patient"] = _number(per_patient.median()) if len(per_patient) else None
    review["median_hours_between_observations"], review["carry_forward_share"] = (
        _number(value) for value in _cadence(table, field)
    )

    if len(observed):
        quantiles = observed.quantile([0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99])
        iqr = quantiles[0.75] - quantiles[0.25]
        review["distribution"] = {
            "mean": _number(observed.mean()), "std": _number(observed.std()),
            "min": _number(observed.min()), "p1": _number(quantiles[0.01]), "p5": _number(quantiles[0.05]),
            "p25": _number(quantiles[0.25]), "median": _number(quantiles[0.5]), "p75": _number(quantiles[0.75]),
            "p95": _number(quantiles[0.95]), "p99": _number(quantiles[0.99]), "max": _number(observed.max()),
            "skew": _number(skew(observed.to_numpy())) if observed.nunique() > 1 else None,
            "distinct_values": int(observed.nunique()),
            "zeros": int((observed == 0).sum()), "negatives": int((observed < 0).sum()),
        }
        review["outliers_iqr"] = int(
            ((observed < quantiles[0.25] - IQR_FENCE * iqr) | (observed > quantiles[0.75] + IQR_FENCE * iqr)).sum()
        ) if iqr > 0 else 0
    else:
        review["distribution"] = {}
        review["outliers_iqr"] = 0

    if spec.allowed is not None:
        invalid = values.notna() & ~values.isin(spec.allowed)
        review["implausible"] = {"below": 0, "above": 0, "not_allowed": int(invalid.sum())}
        extremes = table.loc[invalid, ["patient_id", "source", "hour_index", field]].head(5)
    else:
        below, above = values < spec.low, values > spec.high
        review["implausible"] = {"below": int(below.sum()), "above": int(above.sum()), "not_allowed": 0}
        extremes = pd.concat([
            table.loc[below, ["patient_id", "source", "hour_index", field]].nsmallest(5, field),
            table.loc[above, ["patient_id", "source", "hour_index", field]].nlargest(5, field),
        ])
    review["implausible"]["share_of_observed"] = _number(
        sum(review["implausible"][key] for key in ("below", "above", "not_allowed")) / len(observed), 5,
    ) if len(observed) else None
    review["implausible_examples"] = [
        {"patient_id": row.patient_id, "source": row.source, "row": int(row.hour_index), "value": _number(getattr(row, field))}
        for row in extremes.itertuples()
    ]

    by_source = {}
    samples = {}
    for source in sources:
        in_source = table["source"] == source
        source_values = values[in_source]
        source_observed = source_values.dropna()
        source_patients = patient_counts.loc[patient_sources == source, field]
        by_source[source] = {
            "missing_fraction": _number(1 - len(source_observed) / len(source_values)) if len(source_values) else None,
            "patients_with_any": _number(float((source_patients > 0).mean())) if len(source_patients) else None,
            "median": _number(source_observed.median()) if len(source_observed) else None,
            "p5": _number(source_observed.quantile(0.05)) if len(source_observed) else None,
            "p95": _number(source_observed.quantile(0.95)) if len(source_observed) else None,
            "carry_forward_share": _number(_cadence(table[in_source], field)[1]),
        }
        if len(source_observed):
            take = min(KS_SAMPLE, len(source_observed))
            samples[source] = rng.choice(source_observed.to_numpy(), size=take, replace=False)
    review["by_source"] = by_source
    review["hospital_ks"] = (
        _number(ks_2samp(samples[sources[0]], samples[sources[1]]).statistic)
        if len(sources) == 2 and len(samples) == 2 else None
    )
    missing_values = [entry["missing_fraction"] for entry in by_source.values() if entry["missing_fraction"] is not None]
    review["hospital_missing_gap"] = _number(max(missing_values) - min(missing_values)) if len(missing_values) == 2 else None

    summary_column = f"{field}_mean" if f"{field}_mean" in cohort_features else f"{field}_latest"
    review["outcome"] = None
    if summary_column in cohort_features and cohort_labels.nunique() == 2:
        summary = cohort_features[summary_column]
        positive, negative = summary[cohort_labels == 1].dropna(), summary[cohort_labels == 0].dropna()
        pooled = np.sqrt((positive.var() + negative.var()) / 2) if len(positive) > 1 and len(negative) > 1 else np.nan
        missing_column = f"{field}_missing_fraction"
        review["outcome"] = {
            "summary_feature": summary_column,
            "smd": _number((positive.mean() - negative.mean()) / pooled) if pooled and np.isfinite(pooled) and pooled > 0 else None,
            "median_sepsis": _number(positive.median()) if len(positive) else None,
            "median_no_sepsis": _number(negative.median()) if len(negative) else None,
            "lookback_missing_sepsis": _number(cohort_features.loc[cohort_labels == 1, missing_column].mean())
            if missing_column in cohort_features else None,
            "lookback_missing_no_sepsis": _number(cohort_features.loc[cohort_labels == 0, missing_column].mean())
            if missing_column in cohort_features else None,
        }

    flags = []
    if review["implausible"]["share_of_observed"]:
        flags.append("implausible values")
    if review["missing_fraction"] is not None and review["missing_fraction"] > HIGH_MISSINGNESS:
        flags.append("mostly missing")
    if review["hospital_ks"] is not None and review["hospital_ks"] > SHIFT_KS:
        flags.append("hospital distribution shift")
    if review["hospital_missing_gap"] is not None and review["hospital_missing_gap"] > AVAILABILITY_GAP:
        flags.append("hospital-specific availability")
    if review["carry_forward_share"] is not None and review["carry_forward_share"] > CARRY_FORWARD_SHARE and spec.allowed is None \
            and spec.category not in ("Demographics and admission", "Time and target"):
        flags.append("repeated values")
    if review["outcome"] and review["outcome"]["smd"] is not None and abs(review["outcome"]["smd"]) >= SMD_NOTABLE:
        flags.append("outcome signal")
    review["flags"] = flags
    return review


def _field_chart(
    table: pd.DataFrame, field: str, sources: list[str], hour_rates: pd.DataFrame, path: Path,
) -> str:
    spec = FIELD_SPECS[field]
    figure, axes = plt.subplots(1, 2, figsize=(10, 3.2), gridspec_kw={"width_ratios": [1.35, 1]})
    figure.patch.set_facecolor(SURFACE)
    distribution, cadence = axes
    observed = table[field].dropna()
    if spec.allowed is not None:
        positions = np.arange(len(spec.allowed))
        width = 0.8 / max(len(sources), 1)
        for index, source in enumerate(sources):
            source_values = table.loc[table["source"] == source, field].dropna()
            shares = [float((source_values == value).mean()) if len(source_values) else 0 for value in spec.allowed]
            distribution.bar(positions + index * width, shares, width=width - 0.02, color=SOURCE_COLOURS[index % 2], label=source)
        distribution.set_xticks(positions + width * (len(sources) - 1) / 2, [f"{value:g}" for value in spec.allowed])
        distribution.set(ylabel="Share of observed rows", title=f"{field}: values by hospital")
    elif len(observed):
        low, high = observed.quantile([0.005, 0.995])
        if low == high:
            low, high = low - 0.5, high + 0.5
        bins = np.linspace(low, high, 50)
        for index, source in enumerate(sources):
            source_values = table.loc[table["source"] == source, field].dropna()
            if len(source_values):
                distribution.hist(
                    source_values.clip(low, high), bins=bins, density=True, histtype="step", linewidth=2,
                    color=SOURCE_COLOURS[index % 2], label=source,
                )
        for limit in (spec.low, spec.high):
            if limit is not None and low <= limit <= high:
                distribution.axvline(limit, color=INK_MUTED, linestyle="--", linewidth=1)
        outside = int(((observed < low) | (observed > high)).sum())
        distribution.set(xlabel=spec.unit, ylabel="Density", title=f"{field}: distribution by hospital (central 99%)")
        if outside:
            distribution.text(0.99, 0.97, f"{outside:,} values outside view", transform=distribution.transAxes,
                              ha="right", va="top", fontsize=8, color=INK_MUTED)
    else:
        distribution.text(0.5, 0.5, "No observed values", ha="center", va="center", color=INK_MUTED)
    for index, source in enumerate(sources):
        if (source, 0) in hour_rates.index:
            rates = hour_rates.loc[source, field]
            cadence.plot(rates.index, rates.to_numpy(), color=SOURCE_COLOURS[index % 2], linewidth=2, label=source)
    cadence.axvspan(-0.5, LOOKBACK_HOURS - 0.5, color="#e6eae7", zorder=0)
    cadence.text(LOOKBACK_HOURS / 2 - 0.5, 1.02, "look-back", ha="center", va="bottom", fontsize=8, color=INK_MUTED,
                 transform=cadence.get_xaxis_transform())
    cadence.set(xlabel="Row (hour index)", ylabel="Share of patients observed", ylim=(0, 1.05),
                title=f"{field}: observation rate by hour")
    for axis in axes:
        _style(axis)
    if sources:
        distribution.legend(frameon=False, fontsize=8)
    figure.tight_layout()
    return _png(figure, path)


def _cross_field(table: pd.DataFrame, rng: np.random.Generator, path: Path) -> tuple[list[dict[str, Any]], str]:
    def both(*columns: str) -> pd.DataFrame:
        present = [column for column in columns if column in table]
        if len(present) != len(columns):
            return pd.DataFrame(columns=list(columns))
        return table.loc[table[list(columns)].notna().all(axis=1), ["patient_id", *columns]]

    checks = []
    panels = []
    pressure = both("SBP", "DBP")
    inverted = pressure["DBP"] > pressure["SBP"]
    checks.append({"check": "DBP above SBP", "rule": "Q3_inverted_pressure", "pairs": int(len(pressure)),
                   "violations": int(inverted.sum())})
    panels.append(("SBP", "DBP", pressure, inverted, "identity", "DBP vs SBP (above the line: DBP > SBP)"))
    triple = both("SBP", "DBP", "MAP")
    outside = (triple["MAP"] < triple["DBP"]) | (triple["MAP"] > triple["SBP"])
    checks.append({"check": "MAP outside DBP-SBP", "rule": "flag only", "pairs": int(len(triple)),
                   "violations": int(outside.sum())})
    if len(triple):
        estimate = triple["DBP"] + (triple["SBP"] - triple["DBP"]) / 3
        panels.append(("MAP_estimate", "MAP", triple.assign(MAP_estimate=estimate), outside, "identity",
                       "MAP vs DBP + (SBP − DBP)/3"))
    blood = both("Hgb", "Hct")
    ratio = blood["Hct"] / blood["Hgb"]
    odd_ratio = (ratio < 2) | (ratio > 4.5)
    checks.append({"check": "Hct/Hgb outside 2-4.5", "rule": "flag only", "pairs": int(len(blood)),
                   "violations": int(odd_ratio.sum())})
    panels.append(("Hgb", "Hct", blood, odd_ratio, "triple", "Hct vs Hgb (line: Hct = 3 × Hgb)"))
    bilirubin = both("Bilirubin_total", "Bilirubin_direct")
    direct_high = bilirubin["Bilirubin_direct"] > bilirubin["Bilirubin_total"]
    checks.append({"check": "Direct bilirubin above total", "rule": "Q4_bilirubin_direct_above_total",
                   "pairs": int(len(bilirubin)), "violations": int(direct_high.sum())})
    panels.append(("Bilirubin_total", "Bilirubin_direct", bilirubin, direct_high, "identity", "Direct vs total bilirubin"))
    gas = both("pH", "HCO3", "PaCO2")
    if len(gas):
        positive = (gas["HCO3"] > 0) & (gas["PaCO2"] > 0)
        predicted = pd.Series(np.nan, index=gas.index)
        predicted[positive] = 6.1 + np.log10(gas.loc[positive, "HCO3"] / (0.03 * gas.loc[positive, "PaCO2"]))
        gas = gas.assign(pH_predicted=predicted)
        mismatch = (gas["pH"] - gas["pH_predicted"]).abs() > 0.15
    else:
        mismatch = pd.Series(dtype=bool)
    checks.append({"check": "pH vs Henderson-Hasselbalch > 0.15", "rule": "flag only", "pairs": int(len(gas)),
                   "violations": int(mismatch.sum())})
    if len(gas):
        panels.append(("pH_predicted", "pH", gas, mismatch, "identity", "Measured vs predicted pH"))
    saturation = both("O2Sat", "SaO2")
    gap = (saturation["O2Sat"] - saturation["SaO2"]).abs() > 10
    checks.append({"check": "Pulse vs arterial saturation differ by > 10", "rule": "flag only",
                   "pairs": int(len(saturation)), "violations": int(gap.sum())})
    panels.append(("SaO2", "O2Sat", saturation, gap, "identity", "Pulse oximetry vs arterial saturation"))

    figure, axes = plt.subplots(2, 3, figsize=(12, 7.4))
    figure.patch.set_facecolor(SURFACE)
    for axis, (x, y, frame, flagged, line, title) in zip(axes.flat, panels):
        _style(axis)
        if not len(frame):
            axis.text(0.5, 0.5, "No paired values", ha="center", va="center", color=INK_MUTED, transform=axis.transAxes)
            axis.set_title(title)
            continue
        take = rng.choice(len(frame), size=min(SCATTER_SAMPLE, len(frame)), replace=False)
        sample = frame.iloc[take]
        sample_flagged = flagged.iloc[take].to_numpy()
        axis.scatter(sample.loc[~sample_flagged, x], sample.loc[~sample_flagged, y], s=4, color=NEUTRAL, alpha=0.35,
                     linewidths=0, rasterized=True, label="consistent")
        violations = frame[flagged.to_numpy()]
        axis.scatter(violations[x], violations[y], s=14, color=VIOLATION, edgecolors=SURFACE, linewidths=0.4,
                     marker="^", label=f"flagged (all {int(flagged.sum()):,})")
        low, high = np.nanpercentile(frame[[x, y]].to_numpy(dtype=float), [0.5, 99.5])
        guide = np.linspace(low, high, 10)
        axis.plot(guide, guide * (3 if line == "triple" else 1), color=INK_MUTED, linestyle="--", linewidth=1)
        axis.set(xlabel=x, ylabel=y, title=title)
        axis.legend(frameon=False, fontsize=7, loc="upper left")
    for axis in list(axes.flat)[len(panels):]:
        axis.set_visible(False)
    figure.tight_layout()
    return checks, _png(figure, path)


def _record_checks(table: pd.DataFrame, records: list[PatientRecord]) -> list[dict[str, Any]]:
    by_patient = table.groupby("patient_id", sort=False)
    first = by_patient.first()
    label_drop = by_patient["SepsisLabel"].diff().lt(0).groupby(table["patient_id"]).any()
    ever_positive = by_patient["SepsisLabel"].max() == 1
    onset = table[table["SepsisLabel"] == 1].groupby("patient_id")["hour_index"].min()
    checks = [
        {"check": "SepsisLabel switches back from 1 to 0", "patients": int(label_drop.sum()),
         "note": "The label should stay on once sepsis is labelled; reversions would break the outcome definition."},
        {"check": "Patients labelled positive at row 0", "patients": int((onset == 0).sum()),
         "note": "Excluded by the cohort rules (positive in look-back)."},
        {"check": "Patients ever labelled positive", "patients": int(ever_positive.sum()),
         "note": f"Median first positive row {float(onset.median()):.0f}." if len(onset) else ""},
    ]
    if "ICULOS" in table:
        diffs = by_patient["ICULOS"].diff()
        gaps = diffs.notna() & diffs.ne(1)
        checks.append({"check": "ICULOS not increasing by 1", "patients": int(gaps.groupby(table["patient_id"]).any().sum()),
                       "note": "Hourly continuity the look-back window relies on."})
        checks.append({"check": "Record starts after ICU hour 1", "patients": int((first["ICULOS"] > 1).sum()),
                       "note": f"Maximum ICULOS at row 0: {float(first['ICULOS'].max()):.0f}. Rows 0-5 are then not the first ICU hours."})
    if "HospAdmTime" in table:
        checks.append({"check": "HospAdmTime above 0 (ICU before hospital admission)",
                       "patients": int((first["HospAdmTime"] > 0).sum()),
                       "note": "Kept; small positive values may reflect transfer timing."})
    if "Age" in table:
        ages = first["Age"]
        checks.append({"check": f"Age recorded as {AGE_TOP_CODE}", "patients": int((ages == AGE_TOP_CODE).sum()),
                       "note": f"Patients aged 90-99: {int(ages.between(90, 99.99).sum())}. A spike at 100 with none in 90-99 indicates top-coding."})
        checks.append({"check": "Age below 18", "patients": int((ages < 18).sum()),
                       "note": "Kept; the source cohort is described as adult."})
    if {"Unit1", "Unit2"} <= set(table.columns):
        units = first[["Unit1", "Unit2"]]
        checks.append({"check": "ICU unit type missing", "patients": int(units.isna().all(axis=1).sum()),
                       "note": "Unit1/Unit2 absent for the whole stay."})
        checks.append({"check": "Both or neither unit flag set", "patients": int(((units["Unit1"] + units["Unit2"]) != 1).where(units.notna().all(axis=1), False).sum()),
                       "note": "Exactly one of Unit1 (medical) and Unit2 (surgical) should be 1."})
    clinical = [name for name, spec in FIELD_SPECS.items() if spec.category not in ("Time and target", "Demographics and admission") and name in table]
    same = table[clinical].eq(table[clinical].shift()) | (table[clinical].isna() & table[clinical].shift().isna())
    duplicate = same.all(axis=1) & table["patient_id"].eq(table["patient_id"].shift()) & table[clinical].notna().any(axis=1)
    checks.append({"check": "Rows identical to the previous row (clinical fields)", "patients": int(duplicate.groupby(table["patient_id"]).any().sum()),
                   "note": f"{int(duplicate.sum()):,} rows; suggests copied-forward records."})
    checks.append({"check": "Records shorter than the six-row look-back",
                   "patients": int(sum(len(record.frame) < LOOKBACK_HOURS for record in records)),
                   "note": "Excluded by the cohort rules (too short)."})
    return checks


def _recommendations(fields: dict[str, dict[str, Any]], cross: list[dict[str, Any]], rules: list[dict[str, Any]],
                     record_checks: list[dict[str, Any]]) -> list[dict[str, Any]]:
    impact = {rule["rule_id"]: rule for rule in rules}
    recommendations = []
    for rule in QUALITY_RULES:
        effect = impact.get(rule.rule_id, {})
        if not effect.get("values_changed"):
            continue
        top = sorted(effect.get("values_changed_by_field", {}).items(), key=lambda item: -item[1])[:6]
        recommendations.append({
            "priority": "Implement",
            "area": ", ".join(name for name, _ in top) or ", ".join(rule.fields),
            "finding": f"{rule.title}: {effect['values_changed']:,} values in {effect['patients_affected']:,} patients.",
            "action": f"{rule.rule_id}: {rule.action}.",
            "rationale": rule.rationale,
        })
    shifted = [name for name, review in fields.items() if "hospital distribution shift" in review["flags"]]
    availability = [name for name, review in fields.items() if "hospital-specific availability" in review["flags"]]
    if shifted or availability:
        recommendations.append({
            "priority": "Pipeline",
            "area": ", ".join(sorted(set(shifted + availability))[:12]) + (" …" if len(set(shifted + availability)) > 12 else ""),
            "finding": f"{len(shifted)} fields differ in distribution and {len(availability)} in availability between hospitals.",
            "action": "Keep the cross-hospital split as a standing evaluation, report per-hospital metrics, and run "
                      "explore with measured values only alongside all features.",
            "rationale": "Site differences explain the drop from pooled to cross-hospital performance; they cannot be fixed by cleaning.",
        })
    sparse = [name for name, review in fields.items() if "mostly missing" in review["flags"]]
    if sparse:
        recommendations.append({
            "priority": "Pipeline",
            "area": f"{len(sparse)} fields",
            "finding": f"{len(sparse)} fields are missing in over {HIGH_MISSINGNESS:.0%} of rows; their six-hour summaries are mostly imputed medians.",
            "action": "Keep the missingness indicators as features, keep the ±10 SD cap in explore, and treat clusters driven by these fields with caution.",
            "rationale": "Median imputation of sparse labs creates point masses that dominate scaling and clustering.",
        })
    lab_rule_fields = set(next(rule.fields for rule in QUALITY_RULES if rule.rule_id == "Q6_lab_carry_forward"))
    repeated = [
        name for name, review in fields.items()
        if "repeated values" in review["flags"] and name not in lab_rule_fields
    ]
    if repeated:
        recommendations.append({
            "priority": "Monitor",
            "area": ", ".join(repeated),
            "finding": "Over half of consecutive hourly observations repeat the previous value.",
            "action": "Report repetition per field; do not treat repeated values as new measurements in cadence analyses.",
            "rationale": "Copied-forward values overstate measurement frequency but are not wrong values.",
        })
    for check in cross:
        if check["rule"] == "flag only" and check["violations"]:
            recommendations.append({
                "priority": "Monitor",
                "area": check["check"],
                "finding": f"{check['violations']:,} of {check['pairs']:,} paired rows ({check['violations'] / max(check['pairs'], 1):.2%}).",
                "action": "Report only; do not clean.",
                "rationale": "Values within an hour may come from different measurement times or methods, so the pair is not strictly comparable.",
            })
    for check in record_checks:
        if check["check"].startswith("Record starts after ICU hour 1") and check["patients"]:
            recommendations.append({
                "priority": "Monitor",
                "area": "ICULOS",
                "finding": f"{check['patients']:,} records start after ICU hour 1.",
                "action": "Keep the audit column and per-hospital count in the cohort summary (already implemented).",
                "rationale": "The look-back is defined as the first six rows by design decision.",
            })
    return recommendations


def run_data_review(
    records: list[PatientRecord],
    output_dir: str | Path,
    *,
    horizon_hours: int = DEFAULT_HORIZON_HOURS,
    random_state: int = 42,
) -> dict[str, Any]:
    """Review raw ``records`` and write ``data_review.json``, ``field_summary.csv``, charts, and ``data_review.html``."""
    if not records:
        raise ValueError("Data review requires at least one record")
    output = Path(output_dir)
    charts = output / "charts"
    charts.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(random_state)
    table = records_to_table(records)
    fields = [name for name in FIELD_SPECS if name in table.columns]
    sources = sorted(table["source"].astype(str).unique())
    patient_counts = table.groupby("patient_id", sort=False)[fields].count()
    patient_sources = table.groupby("patient_id", sort=False)["source"].first()
    try:
        cohort_features, cohort_labels, _, _ = create_examples(records, horizon_hours=horizon_hours)
    except ValueError:
        cohort_features, cohort_labels = pd.DataFrame(), pd.Series(dtype="int64")
    audit = cohort_audit(records, horizon_hours=horizon_hours)
    hour_rates = _hour_rates(table, fields)

    field_reviews: dict[str, dict[str, Any]] = {}
    chart_data: dict[str, str] = {}
    for field in fields:
        field_reviews[field] = _field_review(
            table, field, sources, patient_counts, patient_sources, cohort_features, cohort_labels, rng,
        )
        chart_data[field] = _field_chart(table, field, sources, hour_rates, charts / f"field_{field}.png")
    cross, cross_chart = _cross_field(table, rng, charts / "cross_field.png")
    record_checks = _record_checks(table, records)
    rule_impact = evaluate_rules(table, "standard")
    overview_chart = _overview_chart(field_reviews, sources, charts / "missingness_overview.png")
    recommendations = _recommendations(field_reviews, cross, rule_impact, record_checks)

    review = {
        "dataset": {
            "patients": int(table["patient_id"].nunique()),
            "rows": int(len(table)),
            "sources": {source: int((patient_sources == source).sum()) for source in sources},
            "fields": len(fields),
            "eligible_patients": int((audit["status"] == "eligible").sum()),
            "eligible_positive": int((audit["outcome"] == 1).sum()),
            "horizon_hours": horizon_hours,
        },
        "fields": field_reviews,
        "cross_field": cross,
        "record_checks": record_checks,
        "rule_impact": rule_impact,
        "recommendations": recommendations,
        "thresholds": {
            "high_missingness": HIGH_MISSINGNESS, "hospital_ks": SHIFT_KS, "availability_gap": AVAILABILITY_GAP,
            "carry_forward_share": CARRY_FORWARD_SHARE, "outcome_smd": SMD_NOTABLE, "iqr_fence": IQR_FENCE,
        },
    }
    (output / "data_review.json").write_text(json.dumps(review, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    pd.DataFrame([
        {
            "field": name, "category": item["category"], "unit": item["unit"],
            "missing_fraction": item["missing_fraction"], "patients_with_any": item["patients_with_any"],
            "median": item["distribution"].get("median"), "p1": item["distribution"].get("p1"),
            "p99": item["distribution"].get("p99"), "min": item["distribution"].get("min"),
            "max": item["distribution"].get("max"),
            "implausible": item["implausible"]["below"] + item["implausible"]["above"] + item["implausible"]["not_allowed"],
            "outliers_iqr": item["outliers_iqr"], "carry_forward_share": item["carry_forward_share"],
            "hospital_ks": item["hospital_ks"], "hospital_missing_gap": item["hospital_missing_gap"],
            "outcome_smd": (item["outcome"] or {}).get("smd"), "flags": "; ".join(item["flags"]),
        }
        for name, item in field_reviews.items()
    ]).to_csv(output / "field_summary.csv", index=False, float_format="%.6g")
    page = render_review_html(review, chart_data, cross_chart, overview_chart, sources)
    (output / "data_review.html").write_text(page, encoding="utf-8")
    return review


def _overview_chart(fields: dict[str, dict[str, Any]], sources: list[str], path: Path) -> str:
    names = [name for name in fields if name != "SepsisLabel"]
    names.sort(key=lambda name: fields[name]["missing_fraction"])
    figure, axis = plt.subplots(figsize=(10, 9))
    figure.patch.set_facecolor(SURFACE)
    positions = np.arange(len(names))
    height = 0.8 / max(len(sources), 1)
    for index, source in enumerate(sources):
        shares = [fields[name]["by_source"][source]["missing_fraction"] or 0 for name in names]
        axis.barh(positions + index * height, shares, height=height - 0.03, color=SOURCE_COLOURS[index % 2], label=source)
    axis.set_yticks(positions + height * (len(sources) - 1) / 2, names)
    axis.set(xlim=(0, 1), xlabel="Share of hourly rows missing", title="Missingness by field and hospital")
    _style(axis)
    axis.tick_params(axis="y", labelsize=8)
    axis.legend(frameon=False, fontsize=8, loc="lower right")
    figure.tight_layout()
    return _png(figure, path)


def _pipeline_svg() -> str:
    boxes = [
        (20, "PSV files", "hospitals A and B"),
        (170, "Parquet cache", "validated schema"),
        (320, "Data review", "this page (raw values)"),
        (470, "Quality rules", "Q1-Q5, audited"),
        (620, "Cohort + features", "rows 0-5, outcome"),
        (770, "Models + explore", "fit-only transforms"),
        (920, "Reports", "metrics, compare"),
    ]
    parts = [
        '<svg viewBox="0 0 1080 150" role="img" aria-label="Pipeline: PSV files, cache, data review, quality rules, '
        'cohort and features, models and exploration, reports">',
        '<defs><marker id="dr-arrow" markerWidth="8" markerHeight="8" refX="7" refY="4" orient="auto">'
        '<path d="M0,0 L8,4 L0,8 z" fill="#536966"/></marker></defs><g font-family="Segoe UI, sans-serif" text-anchor="middle">',
    ]
    for x, title, note in boxes:
        highlight = title == "Data review"
        rules = title == "Quality rules"
        stroke = "#2a6fb0" if highlight else "#c45042" if rules else "#8b9a91"
        fill = "#e8f0f8" if highlight else "#fbebe7" if rules else "#f4f6f3"
        parts.append(f'<rect x="{x}" y="30" width="135" height="56" rx="5" fill="{fill}" stroke="{stroke}" stroke-width="{2 if highlight or rules else 1}"/>')
        parts.append(f'<text x="{x + 67.5}" y="54" font-size="13" font-weight="700" fill="{INK}">{title}</text>')
        parts.append(f'<text x="{x + 67.5}" y="73" font-size="10" fill="{INK_MUTED}">{note}</text>')
        if x < 920:
            parts.append(f'<path d="M{x + 137} 58 H{x + 148}" stroke="#536966" stroke-width="1.6" marker-end="url(#dr-arrow)"/>')
    parts.append('<path d="M387 88 C387 128 537 128 537 92" fill="none" stroke="#2a6fb0" stroke-width="1.6" stroke-dasharray="4 3" marker-end="url(#dr-arrow)"/>')
    parts.append(f'<text x="462" y="140" font-size="10" fill="{INK_MUTED}">findings define and size the rules</text>')
    parts.append('</g></svg>')
    return "".join(parts)


def _fmt(value: Any, digits: int = 3) -> str:
    if value is None:
        return "—"
    if isinstance(value, (int, np.integer)):
        return f"{int(value):,}"
    if isinstance(value, float):
        if value == 0:
            return "0"
        if abs(value) >= 1000:
            return f"{value:,.0f}"
        if abs(value) < 0.001:
            return f"{value:.2g}"
        return f"{value:.{max(digits, 4)}g}"
    return html.escape(str(value))


def _pct(value: Any) -> str:
    return "—" if value is None else f"{float(value):.1%}"


def _table(headers: list[str], rows: list[list[Any]]) -> str:
    head = "".join(f"<th>{html.escape(header)}</th>" for header in headers)
    body = "".join("<tr>" + "".join(f"<td>{cell}</td>" for cell in row) + "</tr>" for row in rows)
    return f'<div class="table-wrap"><table><thead><tr>{head}</tr></thead><tbody>{body}</tbody></table></div>'


def _field_section(review: dict[str, Any], chart: str, sources: list[str]) -> str:
    distribution = review["distribution"]
    flags = "".join(f'<span class="badge">{html.escape(flag)}</span>' for flag in review["flags"])
    limits = review["limits"]
    limit_text = (
        f"allowed values {', '.join(f'{value:g}' for value in limits['allowed'])}" if limits["allowed"]
        else f"{_fmt(limits['low'])} to {_fmt(limits['high'])} {html.escape(review['unit'])}"
    )
    implausible = review["implausible"]
    stats = _table(
        ["Observed rows", "Missing", "Patients with any", "Obs / patient (median)", "Hours between obs", "Repeated consecutive"],
        [[_fmt(review["observed"]), _pct(review["missing_fraction"]), _pct(review["patients_with_any"]),
          _fmt(review["median_observations_per_patient"]), _fmt(review["median_hours_between_observations"]),
          _pct(review["carry_forward_share"])]],
    )
    spread = _table(
        ["Mean", "SD", "Min", "P1", "P25", "Median", "P75", "P99", "Max", "Skew", "Distinct", "Zeros", "Negatives"],
        [[_fmt(distribution.get(key)) for key in
          ("mean", "std", "min", "p1", "p25", "median", "p75", "p99", "max", "skew", "distinct_values", "zeros", "negatives")]],
    ) if distribution else '<p class="muted">No observed values.</p>'
    plausibility = (
        f"<p>Hard limits: {limit_text}. Below: <strong>{implausible['below']:,}</strong>, above: "
        f"<strong>{implausible['above']:,}</strong>, not allowed: <strong>{implausible['not_allowed']:,}</strong> "
        f"({_pct(implausible['share_of_observed'])} of observed). Beyond {IQR_FENCE:g}× IQR: {review['outliers_iqr']:,}.</p>"
    )
    if review["implausible_examples"]:
        plausibility += _table(
            ["Patient", "Hospital", "Row", "Value"],
            [[html.escape(item["patient_id"]), html.escape(item["source"]), item["row"], _fmt(item["value"])]
             for item in review["implausible_examples"]],
        )
    hospital = _table(
        ["Hospital", "Missing", "Patients with any", "Median", "P5", "P95", "Repeated consecutive"],
        [[html.escape(source), _pct(review["by_source"][source]["missing_fraction"]),
          _pct(review["by_source"][source]["patients_with_any"]), _fmt(review["by_source"][source]["median"]),
          _fmt(review["by_source"][source]["p5"]), _fmt(review["by_source"][source]["p95"]),
          _pct(review["by_source"][source]["carry_forward_share"])] for source in sources],
    ) + f'<p class="muted">Kolmogorov-Smirnov distance between hospitals: {_fmt(review["hospital_ks"])} (flagged above {SHIFT_KS}).</p>'
    outcome = review["outcome"]
    outcome_html = (
        _table(
            ["Six-hour summary", "Median, sepsis", "Median, no sepsis", "SMD", "Look-back missing, sepsis", "Look-back missing, no sepsis"],
            [[html.escape(outcome["summary_feature"]), _fmt(outcome["median_sepsis"]), _fmt(outcome["median_no_sepsis"]),
              _fmt(outcome["smd"]), _pct(outcome["lookback_missing_sepsis"]), _pct(outcome["lookback_missing_no_sepsis"])]],
        ) if outcome else '<p class="muted">Not applicable for this field.</p>'
    )
    return (
        f'<details class="field" id="field-{html.escape(review["field"])}"><summary><span class="field-name">{html.escape(review["field"])}</span>'
        f'<span class="field-desc">{html.escape(review["description"])} · {html.escape(review["unit"])}</span>'
        f'<span class="field-missing">{_pct(review["missing_fraction"])} missing</span>{flags}</summary>'
        f'<div class="field-body"><img alt="{html.escape(review["field"])} distribution by hospital and observation rate by hour" '
        f'src="data:image/png;base64,{chart}">'
        f"<h4>Completeness and cadence</h4>{stats}<h4>Distribution</h4>{spread}"
        f"<h4>Plausibility and outliers</h4>{plausibility}<h4>Hospital comparison</h4>{hospital}"
        f"<h4>Association with the cohort outcome (descriptive)</h4>{outcome_html}</div></details>"
    )


def render_review_html(
    review: dict[str, Any], charts: dict[str, str], cross_chart: str, overview_chart: str, sources: list[str],
) -> str:
    dataset = review["dataset"]
    flagged = sum(1 for item in review["fields"].values() if item["flags"])
    implausible_total = sum(
        item["implausible"]["below"] + item["implausible"]["above"] + item["implausible"]["not_allowed"]
        for item in review["fields"].values()
    )
    tiles = [
        (f"{dataset['patients']:,}", "patients"), (f"{dataset['rows']:,}", "hourly rows"),
        (" / ".join(f"{count:,}" for count in dataset["sources"].values()), " / ".join(dataset["sources"])),
        (str(dataset["fields"]), "fields reviewed"), (str(flagged), "fields with flags"),
        (f"{implausible_total:,}", "implausible values"),
        (f"{dataset['eligible_patients']:,}", f"eligible ({dataset['horizon_hours']} h horizon)"),
    ]
    tiles_html = "".join(f'<div class="stat"><strong>{value}</strong><span>{html.escape(label)}</span></div>' for value, label in tiles)
    tests_html = _table(["Test", "What is measured", "Why it matters"],
                        [[f"<strong>{html.escape(name)}</strong>", html.escape(what), html.escape(why)] for name, what, why in TESTS])
    recommendations_html = _table(
        ["Priority", "Area", "Finding", "Recommendation", "Rationale"],
        [[f'<span class="badge {item["priority"].lower()}">{html.escape(item["priority"])}</span>', html.escape(item["area"]),
          html.escape(item["finding"]), html.escape(item["action"]), html.escape(item["rationale"])]
         for item in review["recommendations"]],
    )
    rules_html = _table(
        ["Rule", "What it does", "Values changed", "Rows", "Patients", "Patients by hospital", "Top fields"],
        [[f"<strong>{html.escape(rule['rule_id'])}</strong><br>{html.escape(rule['title'])}", html.escape(rule["action"]),
          _fmt(rule["values_changed"]), _fmt(rule["rows_affected"]), _fmt(rule["patients_affected"]),
          html.escape(", ".join(f"{source}: {count:,}" for source, count in rule["patients_affected_by_source"].items())),
          html.escape(", ".join(f"{name} ({count:,})" for name, count in sorted(
              rule["values_changed_by_field"].items(), key=lambda item: -item[1])[:5]))]
         for rule in review["rule_impact"]],
    )
    groups = []
    for category in CATEGORY_ORDER:
        members = [item for item in review["fields"].values() if item["category"] == category]
        if not members:
            continue
        sections = "".join(_field_section(item, charts[item["field"]], sources) for item in members)
        flagged_count = sum(1 for item in members if item["flags"])
        groups.append(f'<details class="group" open><summary>{html.escape(category)} '
                      f'<span class="muted">· {len(members)} fields, {flagged_count} flagged</span></summary>{sections}</details>')
    summary_rows = [
        [f'<a href="#field-{html.escape(item["field"])}">{html.escape(item["field"])}</a>', html.escape(item["category"]),
         _pct(item["missing_fraction"]), _fmt(item["distribution"].get("median")),
         _fmt(item["implausible"]["below"] + item["implausible"]["above"] + item["implausible"]["not_allowed"]),
         _fmt(item["hospital_ks"]), _fmt((item["outcome"] or {}).get("smd")),
         "".join(f'<span class="badge">{html.escape(flag)}</span>' for flag in item["flags"])]
        for item in review["fields"].values()
    ]
    cross_html = _table(
        ["Check", "Paired rows", "Flagged", "Share", "Handling"],
        [[html.escape(item["check"]), _fmt(item["pairs"]), _fmt(item["violations"]),
          _pct(item["violations"] / item["pairs"]) if item["pairs"] else "—", html.escape(item["rule"])]
         for item in review["cross_field"]],
    )
    record_html = _table(
        ["Check", "Patients", "Note"],
        [[html.escape(item["check"]), _fmt(item["patients"]), html.escape(item["note"])] for item in review["record_checks"]],
    )
    return f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>Data review | Early Sepsis Prediction</title>
<style>
:root {{ --ink:#1c2727; --muted:#52605f; --line:#d8dfdc; --paper:#f4f5f0; --white:#fff; --teal:#167b73; --blue:#2a6fb0; --coral:#c45042; --font:"Aptos","Segoe UI",sans-serif; }}
* {{ box-sizing:border-box; }} body {{ margin:0; color:var(--ink); background:var(--paper); font-family:var(--font); line-height:1.5; }}
header {{ padding:24px max(24px, calc((100vw - 1180px)/2)); color:#f8f7ef; background:linear-gradient(110deg,#123d3a,#1a6258 62%,#247c6c); border-bottom:5px solid #d89a38; }}
header h1 {{ margin:6px 0; font-size:36px; }} header p {{ margin:0; color:#e1eee9; }} header a {{ color:#fff; }}
main {{ max-width:1180px; margin:0 auto; padding:24px; }} section {{ padding:18px 0 22px; border-top:1px solid var(--line); }}
h2 {{ font-size:22px; margin:0 0 12px; }} h4 {{ margin:16px 0 6px; font-size:14px; color:var(--muted); }}
.notice {{ padding:12px 16px; margin-bottom:20px; border-left:4px solid var(--coral); background:#fff4ef; color:#542d28; }}
.prose {{ max-width:90ch; }} .muted {{ color:var(--muted); font-size:13px; }}
.summary-grid {{ display:grid; grid-template-columns:repeat(auto-fit,minmax(140px,1fr)); gap:10px; margin:12px 0; }}
.stat {{ padding:12px 14px; background:var(--white); border-top:3px solid var(--teal); }} .stat strong {{ display:block; font-size:22px; }} .stat span {{ color:var(--muted); font-size:12px; }}
.table-wrap {{ overflow-x:auto; background:var(--white); border:1px solid var(--line); margin:6px 0 10px; }}
table {{ width:100%; border-collapse:collapse; font-size:13px; }} th,td {{ padding:7px 10px; text-align:left; border-bottom:1px solid #e6eae7; vertical-align:top; }}
th {{ background:#edf2ee; color:#35504d; }} tbody tr:nth-child(even) {{ background:#f8faf8; }}
details {{ background:var(--white); border:1px solid var(--line); margin:8px 0; }} details > summary {{ cursor:pointer; padding:10px 14px; font-weight:700; }}
details.group > summary {{ font-size:17px; background:#edf2ee; }} details.group {{ padding-bottom:6px; }}
details.field {{ margin:6px 10px; }} details.field > summary {{ display:flex; flex-wrap:wrap; gap:10px; align-items:baseline; font-weight:400; }}
.field-name {{ font-weight:700; min-width:130px; }} .field-desc {{ color:var(--muted); flex:1; }} .field-missing {{ font-size:12px; color:var(--muted); }}
.field-body {{ padding:4px 16px 14px; }} .field-body img, .panel img {{ display:block; width:100%; height:auto; }}
.badge {{ display:inline-block; padding:1px 8px; margin:1px 3px 1px 0; border-radius:10px; font-size:11px; background:#fff0d9; color:#6b4a12; border:1px solid #ecd2a2; font-weight:400; }}
.badge.implement {{ background:#fbebe7; color:#7a2c22; border-color:#efc2b8; }} .badge.pipeline {{ background:#e8f0f8; color:#1d4a78; border-color:#bfd3ea; }}
.badge.monitor {{ background:#eef2ed; color:#35504d; border-color:#cfd9d3; }}
.diagram {{ background:#fff; border:1px solid var(--line); padding:10px; }} svg {{ display:block; max-width:100%; height:auto; }}
.panel {{ background:#fff; border:1px solid var(--line); padding:10px; }}
</style></head><body>
<header><p>PhysioNet 2019 · Data review</p><h1>Field-by-field data review</h1><p>Raw records before quality rules · <a href="/">back to the pipeline application</a></p></header>
<main>
<div class="notice"><strong>Contains patient-derived values.</strong> Example rows list patient identifiers from the source files. Handle this page under the dataset's terms.</div>
<section><h2>Introduction</h2><div class="prose">
<p>This review describes the data as delivered, before any cleaning, so that every quality rule applied later is justified by a measured finding and its cost is visible. Each field is checked for completeness, measurement cadence, distribution, plausibility, outliers, repeated values, differences between the two hospitals, and descriptive association with the cohort outcome. Cross-field and record-level checks then test relationships single-field checks cannot.</p>
<p>Plausibility limits are deliberately wide: they reject impossible or instrument and entry errors, not abnormal values, because abnormal values are exactly what an early-warning model needs. Outcome associations use the six-hour look-back summaries of the eligible cohort and are descriptive only.</p></div>
<h3>Where the review sits in the pipeline</h3><div class="diagram">{_pipeline_svg()}</div>
<h3>Tests and their rationale</h3>{tests_html}</section>
<section><h2>Dataset overview</h2><div class="summary-grid">{tiles_html}</div>
<div class="panel"><img alt="Missingness by field and hospital" src="data:image/png;base64,{overview_chart}"></div></section>
<section><h2>Recommendations</h2><p class="prose">Generated from the findings below. <em>Implement</em> items are rules in <code>quality.py</code>; <em>Pipeline</em> items change how the pipeline is run or evaluated; <em>Monitor</em> items are reported but not cleaned.</p>{recommendations_html}
<h3>Quality rule impact on the raw data</h3><p class="muted">Rules run in order; counts are what each rule changes after the earlier ones. SepsisLabel and ICULOS are never changed, so cohort membership and outcomes are unaffected.</p>{rules_html}</section>
<section><h2>Field summary</h2><p class="muted">Select a field to jump to its detail. SMD: standardised mean difference of the six-hour summary, sepsis minus no sepsis.</p>
{_table(["Field", "Category", "Missing", "Median", "Implausible", "Hospital KS", "Outcome SMD", "Flags"], summary_rows)}</section>
<section><h2>Field-by-field detail</h2><p class="muted">Each chart shows the distribution by hospital (dashed lines are plausibility limits) and the share of patients with a value at each hour, with the six-hour look-back shaded.</p>{"".join(groups)}</section>
<section><h2>Cross-field consistency</h2>{cross_html}<details><summary>Scatter plots (sample of {SCATTER_SAMPLE:,} pairs per panel; all flagged pairs shown)</summary><div class="panel"><img alt="Cross-field consistency scatter plots" src="data:image/png;base64,{cross_chart}"></div></details></section>
<section><h2>Record-level integrity</h2>{record_html}</section>
<p class="muted">Generated by <code>sepsis-pipeline review</code>. Educational, retrospective analysis; not for clinical use.</p>
</main></body></html>
"""

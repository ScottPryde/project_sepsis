"""Field specifications and data quality rules for PhysioNet 2019 records.

``FIELD_SPECS`` is the single definition of each field's unit and hard
plausibility limits. Limits are deliberately wide: they reject values that are
physiologically impossible or instrument/entry errors, not values that are
merely abnormal. ``QUALITY_RULES`` turn the data review's findings into
reproducible cleaning steps. Every rule either sets values to missing (so the
fit-only imputer handles them like any other gap) or recodes a documented
convention; none invents a measurement. Flag-only checks are reported by the
data review but change nothing.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

import numpy as np
import pandas as pd

from sepsis_prediction.data import PatientRecord


@dataclass(frozen=True)
class FieldSpec:
    name: str
    category: str
    unit: str
    description: str
    low: float | None = None
    high: float | None = None
    allowed: tuple[float, ...] | None = None


FIELD_SPECS: dict[str, FieldSpec] = {spec.name: spec for spec in (
    FieldSpec("HR", "Vital signs", "beats/min", "Heart rate", 20, 300),
    FieldSpec("O2Sat", "Vital signs", "%", "Pulse oximetry", 50, 100),
    FieldSpec("Temp", "Vital signs", "°C", "Temperature", 25, 45),
    FieldSpec("SBP", "Vital signs", "mmHg", "Systolic blood pressure", 30, 300),
    FieldSpec("MAP", "Vital signs", "mmHg", "Mean arterial pressure", 20, 250),
    FieldSpec("DBP", "Vital signs", "mmHg", "Diastolic blood pressure", 10, 200),
    FieldSpec("Resp", "Vital signs", "breaths/min", "Respiration rate", 3, 80),
    FieldSpec("EtCO2", "Vital signs", "mmHg", "End-tidal carbon dioxide", 5, 100),
    FieldSpec("BaseExcess", "Blood gas", "mmol/L", "Excess bicarbonate", -40, 40),
    FieldSpec("HCO3", "Blood gas", "mmol/L", "Bicarbonate", 2, 60),
    FieldSpec("FiO2", "Blood gas", "fraction", "Fraction of inspired oxygen", 0.21, 1.0),
    FieldSpec("pH", "Blood gas", "pH", "Arterial pH", 6.5, 8.0),
    FieldSpec("PaCO2", "Blood gas", "mmHg", "Arterial carbon dioxide", 5, 150),
    FieldSpec("SaO2", "Blood gas", "%", "Arterial oxygen saturation", 30, 100),
    FieldSpec("AST", "Chemistry", "IU/L", "Aspartate transaminase", 1, 20000),
    FieldSpec("BUN", "Chemistry", "mg/dL", "Blood urea nitrogen", 1, 300),
    FieldSpec("Alkalinephos", "Chemistry", "IU/L", "Alkaline phosphatase", 5, 5000),
    FieldSpec("Calcium", "Chemistry", "mg/dL", "Calcium (total)", 3, 16),
    FieldSpec("Chloride", "Chemistry", "mmol/L", "Chloride", 60, 150),
    FieldSpec("Creatinine", "Chemistry", "mg/dL", "Creatinine", 0.1, 30),
    FieldSpec("Bilirubin_direct", "Chemistry", "mg/dL", "Direct bilirubin", 0, 50),
    FieldSpec("Glucose", "Chemistry", "mg/dL", "Serum glucose", 10, 1500),
    FieldSpec("Lactate", "Chemistry", "mmol/L", "Lactic acid", 0.1, 35),
    FieldSpec("Magnesium", "Chemistry", "mmol/dL", "Magnesium", 0.3, 10),
    FieldSpec("Phosphate", "Chemistry", "mg/dL", "Phosphate", 0.3, 20),
    FieldSpec("Potassium", "Chemistry", "mmol/L", "Potassium", 1.5, 10),
    FieldSpec("Bilirubin_total", "Chemistry", "mg/dL", "Total bilirubin", 0.05, 80),
    FieldSpec("TroponinI", "Chemistry", "ng/mL", "Troponin I", 0, 500),
    FieldSpec("Hct", "Haematology", "%", "Haematocrit", 5, 75),
    FieldSpec("Hgb", "Haematology", "g/dL", "Haemoglobin", 1, 25),
    FieldSpec("PTT", "Haematology", "seconds", "Partial thromboplastin time", 10, 250),
    FieldSpec("WBC", "Haematology", "count×10³/µL", "Leukocyte count", 0.05, 500),
    FieldSpec("Fibrinogen", "Haematology", "mg/dL", "Fibrinogen", 20, 2000),
    FieldSpec("Platelets", "Haematology", "count×10³/µL", "Platelets", 1, 2500),
    FieldSpec("Age", "Demographics and admission", "years", "Age (100 = top-coded)", 14, 100),
    FieldSpec("Gender", "Demographics and admission", "code", "Female (0) or male (1)", allowed=(0, 1)),
    FieldSpec("Unit1", "Demographics and admission", "flag", "Medical ICU", allowed=(0, 1)),
    FieldSpec("Unit2", "Demographics and admission", "flag", "Surgical ICU", allowed=(0, 1)),
    FieldSpec("HospAdmTime", "Demographics and admission", "hours", "Hours between hospital and ICU admission", -10000, 100),
    FieldSpec("ICULOS", "Time and target", "hours", "ICU length of stay at this row", 1, 1000),
    FieldSpec("SepsisLabel", "Time and target", "flag", "Sepsis label (on 6 h before onset)", allowed=(0, 1)),
)}
CATEGORY_ORDER = (
    "Vital signs", "Blood gas", "Chemistry", "Haematology", "Demographics and admission", "Time and target",
)
NEVER_CLEANED = {"SepsisLabel", "ICULOS"}
AGE_TOP_CODE = 100
AGE_TOP_CODE_RECODED = 90


@dataclass(frozen=True)
class QualityRule:
    rule_id: str
    title: str
    fields: tuple[str, ...]
    action: str
    rationale: str
    apply: Callable[[pd.DataFrame], dict[str, pd.Series]]
    """Return ``{column: boolean mask}`` of values to change; the action defines how."""


def _range_masks(table: pd.DataFrame) -> dict[str, pd.Series]:
    masks = {}
    for name, spec in FIELD_SPECS.items():
        if name not in table.columns or name in NEVER_CLEANED:
            continue
        values = table[name]
        if spec.allowed is not None:
            mask = values.notna() & ~values.isin(spec.allowed)
        else:
            mask = (values < spec.low) | (values > spec.high)
        if mask.any():
            masks[name] = mask
    return masks


def _calcium_ionised(table: pd.DataFrame) -> dict[str, pd.Series]:
    if "Calcium" not in table:
        return {}
    return {"Calcium": table["Calcium"].between(0.5, 2.0)}


def _inverted_pressure(table: pd.DataFrame) -> dict[str, pd.Series]:
    if not {"SBP", "DBP"} <= set(table.columns):
        return {}
    inverted = table["DBP"] > table["SBP"]
    return {column: inverted & table[column].notna() for column in ("SBP", "DBP", "MAP") if column in table}


def _bilirubin_direct_above_total(table: pd.DataFrame) -> dict[str, pd.Series]:
    if not {"Bilirubin_direct", "Bilirubin_total"} <= set(table.columns):
        return {}
    return {"Bilirubin_direct": table["Bilirubin_direct"] > table["Bilirubin_total"]}


def _age_top_code(table: pd.DataFrame) -> dict[str, pd.Series]:
    if "Age" not in table:
        return {}
    return {"Age": table["Age"] == AGE_TOP_CODE}


LAB_CATEGORIES = ("Blood gas", "Chemistry", "Haematology")
SETTING_FIELDS = {"FiO2"}


def _carried_forward_labs(table: pd.DataFrame) -> dict[str, pd.Series]:
    """Lab values identical to the same patient's value in the immediately preceding row."""
    if not {"patient_id", "hour_index"} <= set(table.columns):
        return {}
    previous_row = table["patient_id"].eq(table["patient_id"].shift()) & table["hour_index"].diff().eq(1)
    masks = {}
    for name, spec in FIELD_SPECS.items():
        if spec.category not in LAB_CATEGORIES or name in SETTING_FIELDS or name not in table:
            continue
        values = table[name]
        masks[name] = previous_row & values.notna() & values.eq(values.shift())
    return masks


QUALITY_RULES: tuple[QualityRule, ...] = (
    QualityRule(
        "Q1_calcium_ionised",
        "Calcium values between 0.5 and 2.0 look like ionised calcium",
        ("Calcium",),
        "set to missing",
        "Total calcium in mg/dL is about 8.5-10.5; a separate mode near 1.1 matches ionised calcium in mmol/L. "
        "The two measure different things and cannot be converted, so mixing them distorts the field.",
        _calcium_ionised,
    ),
    QualityRule(
        "Q2_plausible_range",
        "Values outside hard plausibility limits",
        tuple(name for name in FIELD_SPECS if name not in NEVER_CLEANED),
        "set to missing",
        "Values beyond physiological or instrument limits (e.g. FiO2 of 4000 or below 0.21, potassium 27.5, "
        "chloride 26) are entry or device errors. Removing them prevents single values dominating summaries, "
        "scaling, and distances.",
        _range_masks,
    ),
    QualityRule(
        "Q3_inverted_pressure",
        "Diastolic pressure above systolic in the same hour",
        ("SBP", "DBP", "MAP"),
        "set the row's SBP, DBP and MAP to missing",
        "DBP above SBP is impossible for one reading, so at least one value is wrong and it is not possible to "
        "tell which; MAP derives from both.",
        _inverted_pressure,
    ),
    QualityRule(
        "Q4_bilirubin_direct_above_total",
        "Direct bilirubin above total bilirubin",
        ("Bilirubin_direct",),
        "set direct bilirubin to missing",
        "Direct bilirubin is a component of total bilirubin and cannot exceed it.",
        _bilirubin_direct_above_total,
    ),
    QualityRule(
        "Q5_age_top_code",
        "Age recorded as 100",
        ("Age",),
        f"recode to {AGE_TOP_CODE_RECODED}",
        "Ages 90-99 are absent while 100 is common, consistent with de-identification top-coding of ages 90 "
        "and above. Recoding to 90 keeps these patients in the oldest group without implying they are 100.",
        _age_top_code,
    ),
    QualityRule(
        "Q6_lab_carry_forward",
        "Lab value identical to the previous hour's value",
        tuple(name for name, spec in FIELD_SPECS.items() if spec.category in LAB_CATEGORIES and name not in SETTING_FIELDS),
        "keep the first value of each run; set hourly repeats to missing",
        "At hospital A, 92-98% of consecutive-hour lab pairs (AST, BUN, HCO3, WBC) repeat exactly, against 2-60% at "
        "hospital B, so A's labs are copied forward at source. Copies make labs look measured more often at A, "
        "which changes missingness features by site. FiO2 is excluded because it is a setting that legitimately "
        "stays constant.",
        _carried_forward_labs,
    ),
)
RULESETS = {"none": (), "standard": tuple(rule.rule_id for rule in QUALITY_RULES)}


def rules_for(ruleset: str) -> tuple[QualityRule, ...]:
    if ruleset not in RULESETS:
        raise ValueError(f"ruleset must be one of {sorted(RULESETS)}")
    selected = set(RULESETS[ruleset])
    return tuple(rule for rule in QUALITY_RULES if rule.rule_id in selected)


def records_to_table(records: list[PatientRecord]) -> pd.DataFrame:
    """One long numeric table with ``patient_id``, ``source`` and ``hour_index`` columns."""
    lengths = [len(record.frame) for record in records]
    table = pd.concat([record.frame for record in records], ignore_index=True)
    for column in table.columns:
        if not pd.api.types.is_numeric_dtype(table[column]):
            table[column] = pd.to_numeric(table[column], errors="coerce")
    table.insert(0, "hour_index", np.concatenate([np.arange(length) for length in lengths]))
    table.insert(0, "source", np.repeat([record.source for record in records], lengths))
    table.insert(0, "patient_id", np.repeat([record.patient_id for record in records], lengths))
    return table


def _apply_to_table(table: pd.DataFrame, rules: tuple[QualityRule, ...]) -> tuple[pd.DataFrame, list[dict[str, Any]]]:
    """Apply rules in order to a copy of ``table`` and count what each changed."""
    cleaned = table.copy()
    audit = []
    for rule in rules:
        masks = rule.apply(cleaned)
        changed_rows = pd.Series(False, index=cleaned.index)
        by_field = {}
        for column, mask in masks.items():
            mask = mask.fillna(False).astype(bool)
            count = int(mask.sum())
            if not count:
                continue
            by_field[column] = count
            changed_rows |= mask
            if rule.rule_id == "Q5_age_top_code":
                cleaned.loc[mask, column] = AGE_TOP_CODE_RECODED
            else:
                cleaned.loc[mask, column] = np.nan
        patients = cleaned.loc[changed_rows, ["patient_id", "source"]].drop_duplicates("patient_id")
        audit.append({
            "rule_id": rule.rule_id,
            "title": rule.title,
            "action": rule.action,
            "rationale": rule.rationale,
            "values_changed": int(sum(by_field.values())),
            "values_changed_by_field": by_field,
            "rows_affected": int(changed_rows.sum()),
            "patients_affected": int(len(patients)),
            "patients_affected_by_source": {
                str(source): int(count) for source, count in patients["source"].value_counts().sort_index().items()
            },
        })
    return cleaned, audit


def evaluate_rules(table: pd.DataFrame, ruleset: str = "standard") -> list[dict[str, Any]]:
    """Counts each rule would change, applied in order, without returning cleaned data."""
    return _apply_to_table(table, rules_for(ruleset))[1]


def apply_quality_rules(
    records: list[PatientRecord],
    ruleset: str = "standard",
) -> tuple[list[PatientRecord], dict[str, Any]]:
    """Return cleaned copies of ``records`` and an audit of every change.

    Each record keeps its original columns and row count; ``SepsisLabel`` and
    ``ICULOS`` are never modified, so cohort membership and outcomes are
    unchanged by cleaning.
    """
    rules = rules_for(ruleset)
    audit: dict[str, Any] = {"ruleset": ruleset, "rules": []}
    if not rules or not records:
        return records, audit
    table = records_to_table(records)
    cleaned, audit["rules"] = _apply_to_table(table, rules)
    audit["values_changed"] = int(sum(rule["values_changed"] for rule in audit["rules"]))
    audit["observed_values"] = int(table.drop(columns=["patient_id", "source", "hour_index"]).notna().sum().sum())
    starts = np.cumsum([0, *(len(record.frame) for record in records[:-1])])
    result = []
    for record, start in zip(records, starts):
        columns = list(record.frame.columns)
        frame = cleaned.iloc[start:start + len(record.frame)][columns].reset_index(drop=True)
        result.append(PatientRecord(patient_id=record.patient_id, frame=frame, source=record.source))
    return result, audit

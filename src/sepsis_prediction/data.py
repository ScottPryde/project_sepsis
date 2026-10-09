"""PSV ingestion, schema checks, and fixed-window cohort construction."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import pandas as pd


TARGET_COLUMN = "SepsisLabel"
KNOWN_CLINICAL_COLUMNS = (
    "HR", "O2Sat", "Temp", "SBP", "MAP", "DBP", "Resp", "EtCO2",
    "BaseExcess", "HCO3", "FiO2", "pH", "PaCO2", "SaO2", "AST", "BUN",
    "Alkalinephos", "Calcium", "Chloride", "Creatinine", "Glucose", "Lactate",
    "Magnesium", "Phosphate", "Potassium", "Bilirubin_direct", "Bilirubin_total",
    "TroponinI", "Hct", "Hgb", "PTT", "WBC", "Fibrinogen", "Platelets",
    "Age", "Gender", "Unit1", "Unit2", "HospAdmTime",
)


@dataclass(frozen=True)
class PatientRecord:
    """A single source file and its stable patient identifier."""

    patient_id: str
    frame: pd.DataFrame


def validate_patient_frame(frame: pd.DataFrame, source: str = "<memory>") -> list[str]:
    """Return schema errors; absent clinical columns are permitted and become missing."""
    errors: list[str] = []
    if frame.columns.duplicated().any():
        duplicates = frame.columns[frame.columns.duplicated()].tolist()
        errors.append(f"{source}: duplicate columns: {duplicates}")
    if TARGET_COLUMN not in frame.columns:
        errors.append(f"{source}: required target column {TARGET_COLUMN!r} is missing")
        return errors

    labels = pd.to_numeric(frame[TARGET_COLUMN], errors="coerce")
    if labels.isna().any():
        errors.append(f"{source}: {TARGET_COLUMN} contains missing or non-numeric values")
    elif not labels.isin((0, 1)).all():
        errors.append(f"{source}: {TARGET_COLUMN} must contain only 0 and 1")

    if "ICULOS" in frame.columns:
        iculos = pd.to_numeric(frame["ICULOS"], errors="coerce")
        if iculos.isna().any():
            errors.append(f"{source}: ICULOS must contain numeric hourly indices when present")
        elif len(iculos) > 1 and not (iculos.diff().iloc[1:] == 1).all():
            errors.append(f"{source}: ICULOS must increase by exactly one per row")

    for column in KNOWN_CLINICAL_COLUMNS:
        if column in frame.columns:
            values = pd.to_numeric(frame[column], errors="coerce")
            invalid = frame[column].notna() & values.isna()
            if invalid.any():
                errors.append(f"{source}: clinical column {column!r} contains non-numeric values")
    return errors


def load_patient_file(path: str | Path) -> PatientRecord:
    """Read and validate one pipe-separated patient record."""
    source = Path(path)
    frame = pd.read_csv(source, sep="|")
    errors = validate_patient_frame(frame, str(source))
    if errors:
        raise ValueError("\n".join(errors))
    return PatientRecord(patient_id=source.stem, frame=frame)


def load_patient_files(directory: str | Path) -> list[PatientRecord]:
    """Load PSV files in deterministic filename order."""
    paths = sorted(Path(directory).glob("*.psv"), key=lambda path: path.name)
    if not paths:
        raise FileNotFoundError(f"No .psv files found in {directory}")
    return [load_patient_file(path) for path in paths]


def validate_records(records: Iterable[PatientRecord]) -> None:
    """Raise one concise error containing any invalid in-memory records."""
    errors = [
        error
        for record in records
        for error in validate_patient_frame(record.frame, record.patient_id)
    ]
    if errors:
        raise ValueError("\n".join(errors))
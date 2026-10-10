"""PSV ingestion, schema checks, and fixed-window cohort construction."""

from __future__ import annotations

import os
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd


TARGET_COLUMN = "SepsisLabel"
PARALLEL_LOAD_MIN_FILES = 2000
CACHE_KEY_COLUMNS = ("patient_id", "source")
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
    """A single source file, its stable patient identifier, and its source folder."""

    patient_id: str
    frame: pd.DataFrame
    source: str = ""


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
    return PatientRecord(patient_id=source.stem, frame=frame, source=source.parent.name)


def load_patient_files(
    directories: str | Path | Iterable[str | Path],
    workers: int | None = None,
) -> list[PatientRecord]:
    """Load PSV files from one or more folders in deterministic filename order.

    Each record's ``source`` is its folder name (for PhysioNet 2019,
    ``training_setA`` or ``training_setB``). Patient identifiers must be unique
    across folders. Large inputs are read in parallel; order is unchanged.
    """
    if isinstance(directories, (str, Path)):
        directories = [directories]
    paths: list[Path] = []
    for directory in directories:
        found = list(Path(directory).glob("*.psv"))
        if not found:
            raise FileNotFoundError(f"No .psv files found in {directory}")
        paths.extend(found)
    paths.sort(key=lambda path: path.name)
    stems = pd.Series([path.stem for path in paths])
    if stems.duplicated().any():
        raise ValueError(f"Duplicate patient identifiers across folders: {stems[stems.duplicated()].tolist()[:10]}")

    if workers is None:
        workers = min(os.cpu_count() or 1, 8) if len(paths) >= PARALLEL_LOAD_MIN_FILES else 1
    if workers <= 1:
        return [load_patient_file(path) for path in paths]
    with ProcessPoolExecutor(max_workers=workers) as pool:
        return list(pool.map(load_patient_file, paths, chunksize=250))


def write_record_cache(records: list[PatientRecord], path: str | Path) -> Path:
    """Write validated records to one Parquet table for fast reloading.

    The table has one row per hourly observation, keyed by ``patient_id`` and
    ``source``, in the records' order. Columns absent from a patient file are
    stored as missing, which the feature code already treats identically.
    """
    try:
        import pyarrow  # noqa: F401
    except ImportError as error:
        raise ImportError(
            "The Parquet cache needs pyarrow; install it with `python -m pip install -e '.[data]'`"
        ) from error
    if not records:
        raise ValueError("Cannot cache an empty record list")
    frames = []
    for record in records:
        frame = record.frame.copy()
        frame.insert(0, "source", record.source)
        frame.insert(0, "patient_id", record.patient_id)
        frames.append(frame)
    table = pd.concat(frames, ignore_index=True)
    value_columns = [column for column in table.columns if column not in CACHE_KEY_COLUMNS]
    table[value_columns] = table[value_columns].apply(pd.to_numeric, errors="coerce").astype("float64")
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    table.to_parquet(output, index=False)
    return output


def read_record_cache(path: str | Path) -> list[PatientRecord]:
    """Rebuild records written by ``write_record_cache`` without re-reading PSV files.

    Records were validated before caching, so they are not validated again.
    An ``ICULOS`` column that is entirely missing for a patient is dropped so the
    record matches its source file.
    """
    table = pd.read_parquet(path)
    if table.empty or not set(CACHE_KEY_COLUMNS) <= set(table.columns):
        raise ValueError(f"{path} is not a patient record cache")
    value_columns = [column for column in table.columns if column not in CACHE_KEY_COLUMNS]
    values = table[value_columns].to_numpy(dtype=float)
    patient_ids = table["patient_id"].astype(str).to_numpy()
    sources = table["source"].astype(str).to_numpy()
    starts = np.flatnonzero(np.r_[True, patient_ids[1:] != patient_ids[:-1]])
    ends = np.r_[starts[1:], len(patient_ids)]
    if pd.Series(patient_ids[starts]).duplicated().any():
        raise ValueError(f"{path} has non-contiguous patient rows")
    iculos_position = value_columns.index("ICULOS") if "ICULOS" in value_columns else None
    records: list[PatientRecord] = []
    for start, end in zip(starts, ends):
        frame = pd.DataFrame(values[start:end], columns=value_columns)
        if iculos_position is not None and np.isnan(values[start:end, iculos_position]).all():
            frame = frame.drop(columns="ICULOS")
        records.append(PatientRecord(patient_id=patient_ids[start], frame=frame, source=sources[start]))
    return records


def validate_records(records: Iterable[PatientRecord]) -> None:
    """Raise one concise error containing any invalid in-memory records."""
    errors = [
        error
        for record in records
        for error in validate_patient_frame(record.frame, record.patient_id)
    ]
    if errors:
        raise ValueError("\n".join(errors))
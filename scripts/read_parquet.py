"""Inspect and read Parquet files from the sepsis pipeline.

With no path it reads the project cache, data/cache/physionet2019.parquet,
relative to the repository root. It also accepts any other .parquet file or a
folder of them (read as one dataset).

Examples (from the repository root):

    python scripts/read_parquet.py
    python scripts/read_parquet.py --schema --stats
    python scripts/read_parquet.py --head 20 --columns HR MAP SepsisLabel
    python scripts/read_parquet.py --patient p000001
    python scripts/read_parquet.py --where "Age >= 65 and SepsisLabel == 1"
    python scripts/read_parquet.py --missing --describe
    python scripts/read_parquet.py --patient p000001 --to-csv p000001.csv
    python scripts/read_parquet.py "C:/path/to/other.parquet" --schema

Overview, --schema and --stats read only the file footers, so they are fast
even on large files. Other options load just the columns they need.

Requires pandas and pyarrow (pip install pandas pyarrow).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

try:
    import pandas as pd
    import pyarrow.dataset as ds
    import pyarrow.parquet as pq
except ImportError as error:  # pragma: no cover - friendly message only
    sys.exit(f"Missing dependency ({error.name}). Install with: pip install pandas pyarrow")

PATIENT_COLUMN_CANDIDATES = (
    "patient_id", "PatientID", "Patient_ID", "patientid", "patient", "pid", "subject_id", "record_id",
)
DEFAULT_PATH = Path(__file__).resolve().parents[1] / "data" / "cache" / "physionet2019.parquet"


def parquet_files(path: Path) -> list[Path]:
    if path.is_file():
        return [path]
    if path.is_dir():
        files = sorted(path.rglob("*.parquet"))
        if files:
            return files
        sys.exit(f"No .parquet files found under {path}")
    sys.exit(f"Path not found: {path}")


def human_bytes(size: int) -> str:
    value = float(size)
    for unit in ("B", "KB", "MB", "GB"):
        if value < 1024:
            return f"{value:.1f} {unit}"
        value /= 1024
    return f"{value:.1f} TB"


def show_overview(files: list[Path]) -> None:
    """Rows, columns, row groups and size, from file footers only."""
    total_rows = total_bytes = 0
    schema = None
    for file in files:
        meta = pq.ParquetFile(file).metadata
        total_rows += meta.num_rows
        total_bytes += file.stat().st_size
        schema = schema or pq.read_schema(file)
        if len(files) > 1:
            print(f"  {file.name}: {meta.num_rows:,} rows, {meta.num_row_groups} row groups, "
                  f"{human_bytes(file.stat().st_size)}")
    print(f"Files: {len(files)}   Rows: {total_rows:,}   Columns: {len(schema)}   "
          f"Size on disk: {human_bytes(total_bytes)}")


def show_schema(files: list[Path]) -> None:
    schema = pq.read_schema(files[0])
    width = max(len(name) for name in schema.names)
    print(f"Schema ({len(schema)} columns):")
    for field in schema:
        nullable = "" if field.nullable else "  not null"
        print(f"  {field.name:<{width}}  {field.type}{nullable}")
    metadata = schema.metadata or {}
    if b"pandas" in metadata:
        print("  (written by pandas; index information stored in metadata)")


def show_stats(files: list[Path]) -> None:
    """Per-column null counts and min/max from row-group statistics (no data read)."""
    nulls: dict[str, int] = {}
    mins: dict[str, object] = {}
    maxs: dict[str, object] = {}
    for file in files:
        meta = pq.ParquetFile(file).metadata
        for g in range(meta.num_row_groups):
            group = meta.row_group(g)
            for c in range(group.num_columns):
                column = group.column(c)
                name = column.path_in_schema
                stats = column.statistics
                if stats is None:
                    continue
                if stats.has_null_count:
                    nulls[name] = nulls.get(name, 0) + stats.null_count
                if stats.has_min_max:
                    mins[name] = stats.min if name not in mins else min(mins[name], stats.min)
                    maxs[name] = stats.max if name not in maxs else max(maxs[name], stats.max)
    if not nulls and not mins:
        print("No column statistics stored in these files.")
        return
    table = pd.DataFrame({"nulls": pd.Series(nulls), "min": pd.Series(mins), "max": pd.Series(maxs)})
    print("Footer statistics (whole dataset):")
    print(table.to_string())


def find_patient_column(columns: list[str], requested: str | None) -> str:
    if requested:
        if requested not in columns:
            sys.exit(f"Column '{requested}' not found. Available: {', '.join(columns)}")
        return requested
    for candidate in PATIENT_COLUMN_CANDIDATES:
        if candidate in columns:
            return candidate
    sys.exit(f"No patient ID column found among: {', '.join(columns)}. Pass --patient-column NAME.")


def load(files: list[Path], columns: list[str] | None, patient: list[str] | None,
         patient_column: str | None) -> pd.DataFrame:
    """Load only the needed columns, pushing any patient filter down to Parquet."""
    dataset = ds.dataset([str(f) for f in files], format="parquet")
    available = dataset.schema.names
    if columns:
        missing = [c for c in columns if c not in available]
        if missing:
            sys.exit(f"Unknown column(s): {', '.join(missing)}. Use --schema to list columns.")
    expression = None
    read_columns = columns
    if patient:
        id_column = find_patient_column(available, patient_column)
        expression = ds.field(id_column).isin(patient)
        if read_columns and id_column not in read_columns:
            read_columns = [id_column, *read_columns]
    return dataset.to_table(columns=read_columns, filter=expression).to_pandas()


def show_missing(frame: pd.DataFrame) -> None:
    share = frame.isna().mean().sort_values(ascending=False)
    print("Missing values by column (share of rows):")
    print((share * 100).round(1).astype(str).add("%").to_string())


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("path", type=Path, nargs="?", default=DEFAULT_PATH,
                        help=f"a .parquet file or a folder of them (default: {DEFAULT_PATH})")
    parser.add_argument("--schema", action="store_true", help="print column names and types")
    parser.add_argument("--stats", action="store_true", help="null counts and min/max from file footers")
    parser.add_argument("--head", type=int, metavar="N", help="print the first N rows (default 10 when loading)")
    parser.add_argument("--columns", nargs="+", metavar="COL", help="only read these columns")
    parser.add_argument("--patient", nargs="+", metavar="ID", help="only rows for these patient IDs")
    parser.add_argument("--patient-column", metavar="COL", help="patient ID column if not auto-detected")
    parser.add_argument("--where", metavar="EXPR", help='pandas query, e.g. "HR > 100 and SepsisLabel == 1"')
    parser.add_argument("--describe", action="store_true", help="summary statistics for numeric columns")
    parser.add_argument("--missing", action="store_true", help="share of missing values per column")
    parser.add_argument("--to-csv", type=Path, metavar="FILE", help="write the selected rows to CSV")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    print(f"Reading {args.path}")
    files = parquet_files(args.path)
    pd.set_option("display.width", 200)
    pd.set_option("display.max_columns", 50)

    show_overview(files)
    if args.schema:
        print()
        show_schema(files)
    if args.stats:
        print()
        show_stats(files)

    wants_data = any([args.head, args.columns, args.patient, args.where,
                      args.describe, args.missing, args.to_csv])
    if not wants_data:
        if not (args.schema or args.stats):
            print("\nAdd --schema, --stats, --head N, --patient ID, --where EXPR, "
                  "--describe, --missing or --to-csv FILE. See --help.")
        return 0

    frame = load(files, args.columns, args.patient, args.patient_column)
    if args.where:
        try:
            frame = frame.query(args.where)
        except Exception as error:
            sys.exit(f"Could not apply --where: {error}")
    print(f"\nSelected: {len(frame):,} rows x {frame.shape[1]} columns")

    if args.missing:
        print()
        show_missing(frame)
    if args.describe:
        print()
        print(frame.describe().T.to_string())
    if args.head or not (args.missing or args.describe or args.to_csv):
        print()
        print(frame.head(args.head or 10).to_string())
    if args.to_csv:
        args.to_csv.parent.mkdir(parents=True, exist_ok=True)
        frame.to_csv(args.to_csv, index=False)
        print(f"\nWrote {len(frame):,} rows to {args.to_csv}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
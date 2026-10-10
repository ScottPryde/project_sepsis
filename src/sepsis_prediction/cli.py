"""Command-line interface for validating data and running baseline experiments."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from sepsis_prediction.cnn import run_cnn_experiment
from sepsis_prediction.comparison import parse_run_arguments, write_run_comparison
from sepsis_prediction.data import TARGET_COLUMN, load_patient_files, read_record_cache, write_record_cache
from sepsis_prediction.eda import run_eda
from sepsis_prediction.features import (
    DEFAULT_HORIZON_HOURS,
    cohort_audit,
    create_examples,
    create_sequence_examples,
    summarise_cohort,
)
from sepsis_prediction.modeling import SPLIT_MODES, run_experiment
from sepsis_prediction.reporting import write_html_report, write_model_comparison
from sepsis_prediction.application import serve_application


def _add_data_source(parser: argparse.ArgumentParser, required: bool = True) -> None:
    source = parser.add_mutually_exclusive_group(required=required)
    source.add_argument(
        "--data-dir", type=Path, nargs="+",
        help="one or more folders of .psv files (e.g. data/raw/training_setA data/raw/training_setB)",
    )
    source.add_argument("--cache", type=Path, help="Parquet record cache written by the cache command")


def _add_split_options(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--split-mode", choices=SPLIT_MODES, default="pooled",
        help="pooled: random patient split; hospital: train on --train-sources, test on --test-sources",
    )
    parser.add_argument("--train-sources", nargs="+", default=(), help="source folder names used for fit and validation")
    parser.add_argument("--test-sources", nargs="+", default=(), help="source folder names used as the test set")


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="sepsis-pipeline")
    commands = parser.add_subparsers(dest="command", required=True)

    validate = commands.add_parser("validate", help="validate PSV files and report basic counts")
    _add_data_source(validate)
    validate.add_argument("--verbose", action="store_true", help="include patient filenames in the report")

    cache = commands.add_parser("cache", help="validate PSV files once and save them as one Parquet table")
    cache.add_argument("--data-dir", type=Path, nargs="+", required=True)
    cache.add_argument("--output", type=Path, default=Path("data/cache/records.parquet"))

    eda = commands.add_parser("eda", help="write dataset summaries and exploratory plots")
    _add_data_source(eda)
    eda.add_argument("--output-dir", type=Path, default=Path("eda"))

    run = commands.add_parser("run", help="build the cohort, train baselines, and save artifacts")
    _add_data_source(run)
    run.add_argument("--output-dir", type=Path, default=Path("artifacts"))
    run.add_argument("--test-size", type=float, default=0.2)
    run.add_argument("--random-state", type=int, default=42)
    run.add_argument(
        "--threshold",
        type=float,
        default=None,
        help="override validation-selected threshold for all models",
    )
    run.add_argument("--validation-size", type=float, default=0.2)
    run.add_argument("--target-sensitivity", type=float, default=0.8)
    run.add_argument("--synthetic-demo", action="store_true", help="label outputs as synthetic smoke-test results")
    run.add_argument(
        "--horizon-hours",
        type=int,
        default=DEFAULT_HORIZON_HOURS,
        help="outcome horizon after the six-hour look-back (default: 24)",
    )
    run.add_argument(
        "--models",
        nargs="+",
        choices=("logistic_regression", "random_forest", "xgboost"),
        default=("logistic_regression", "random_forest"),
    )
    run.add_argument("--cnn", action="store_true", help="also train the optional PyTorch 1D CNN")
    run.add_argument("--cnn-epochs", type=int, default=20)
    _add_split_options(run)

    serve = commands.add_parser("serve", help="serve the JavaScript pipeline application")
    _add_data_source(serve, required=False)
    serve.add_argument("--output-dir", type=Path, default=Path("outputs/synthetic_e2e"))
    serve.add_argument("--synthetic-demo", action="store_true", help="rerun the fabricated end-to-end demo")
    serve.add_argument("--host", default="127.0.0.1")
    serve.add_argument("--port", type=int, default=8765)
    serve.add_argument("--test-size", type=float, default=0.2)
    serve.add_argument("--validation-size", type=float, default=0.2)
    serve.add_argument("--random-state", type=int, default=42)
    serve.add_argument("--threshold", type=float, default=None)
    serve.add_argument("--target-sensitivity", type=float, default=0.8)
    serve.add_argument("--horizon-hours", type=int, default=DEFAULT_HORIZON_HOURS)
    serve.add_argument(
        "--models", nargs="+", choices=("logistic_regression", "random_forest", "xgboost"),
        default=("logistic_regression", "random_forest"),
    )
    serve.add_argument("--cnn", action="store_true")
    serve.add_argument("--cnn-epochs", type=int, default=20)
    _add_split_options(serve)

    compare = commands.add_parser("compare", help="summarise completed runs side by side")
    compare.add_argument(
        "--run", action="append", required=True, metavar="NAME=PATH",
        help="a completed run directory; repeat for each run (e.g. synthetic=outputs/synthetic_e2e_v2/run)",
    )
    compare.add_argument("--output-dir", type=Path, default=Path("outputs/comparison"))
    return parser


def _load_records(args: argparse.Namespace):
    if args.cache is not None:
        return read_record_cache(args.cache)
    return load_patient_files(args.data_dir)


def main() -> None:
    args = _parser().parse_args()
    if args.command == "serve":
        if not args.synthetic_demo and args.data_dir is None and args.cache is None:
            raise SystemExit("serve requires --data-dir or --cache unless --synthetic-demo is selected")
        serve_application(
            output_dir=args.output_dir,
            data_dir=args.data_dir,
            cache=args.cache,
            synthetic_demo=args.synthetic_demo,
            host=args.host,
            port=args.port,
            run_options={
                "test_size": args.test_size,
                "validation_size": args.validation_size,
                "random_state": args.random_state,
                "threshold": args.threshold,
                "target_sensitivity": args.target_sensitivity,
                "horizon_hours": args.horizon_hours,
                "models": args.models,
                "cnn": args.cnn,
                "cnn_epochs": args.cnn_epochs,
                "split_mode": args.split_mode,
                "train_sources": list(args.train_sources),
                "test_sources": list(args.test_sources),
            },
        )
        return

    if args.command == "compare":
        try:
            runs = parse_run_arguments(args.run)
        except ValueError as error:
            raise SystemExit(str(error)) from error
        path = write_run_comparison(runs, args.output_dir)
        print(json.dumps({"runs": {name: str(directory) for name, directory in runs.items()}, "report": str(path)}, indent=2))
        return

    if args.command == "cache":
        records = load_patient_files(args.data_dir)
        path = write_record_cache(records, args.output)
        print(json.dumps({
            "patients": len(records),
            "rows": sum(len(record.frame) for record in records),
            "sources": pd.Series([record.source for record in records]).value_counts().sort_index().to_dict(),
            "cache": str(path),
        }, indent=2))
        return

    records = _load_records(args)
    if args.command == "validate":
        report = {
            "patients": len(records),
            "rows": sum(len(record.frame) for record in records),
            "positive_patient_files": int(sum(
                record.frame[TARGET_COLUMN].eq(1).any() for record in records
            )),
        }
        if args.verbose:
            report["files"] = [record.patient_id for record in records]
        print(json.dumps(report, indent=2))
        return

    if args.command == "eda":
        summary = run_eda(records, args.output_dir)
        report_path = write_html_report(args.output_dir, records=records)
        print(json.dumps({
            "summary": summary,
            "output_dir": str(args.output_dir),
            "report": str(report_path),
        }, indent=2))
        return

    features, labels, groups, exclusions = create_examples(records, horizon_hours=args.horizon_hours)
    source_by_patient = {record.patient_id: record.source for record in records}
    sources = pd.Series([source_by_patient[patient] for patient in labels.index], index=labels.index, name="source")
    metrics = run_experiment(
        features,
        labels,
        groups,
        args.output_dir,
        models=tuple(args.models),
        test_size=args.test_size,
        random_state=args.random_state,
        threshold=args.threshold,
        horizon_hours=args.horizon_hours,
        validation_size=args.validation_size,
        target_sensitivity=args.target_sensitivity,
        split_mode=args.split_mode,
        sources=sources,
        train_sources=tuple(args.train_sources),
        test_sources=tuple(args.test_sources),
    )
    if args.synthetic_demo:
        config_path = args.output_dir / "run_config.json"
        run_config = json.loads(config_path.read_text(encoding="utf-8"))
        run_config["synthetic_demo"] = True
        config_path.write_text(json.dumps(run_config, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    if args.cnn:
        sequences, sequence_labels, sequence_groups, sequence_exclusions = create_sequence_examples(
            records, horizon_hours=args.horizon_hours,
        )
        if not sequence_labels.equals(labels) or not sequence_groups.equals(groups):
            raise RuntimeError("Tabular and sequence cohort definitions diverged")
        if sequence_exclusions != exclusions:
            raise RuntimeError("Tabular and sequence exclusion counts diverged")
        metrics["cnn_1d"] = run_cnn_experiment(
            sequences,
            sequence_labels,
            sequence_groups,
            args.output_dir,
            random_state=args.random_state,
            threshold=args.threshold,
            epochs=args.cnn_epochs,
            target_sensitivity=args.target_sensitivity,
        )
    comparison = write_model_comparison(args.output_dir)
    audit = cohort_audit(records, horizon_hours=args.horizon_hours)
    audit.to_csv(args.output_dir / "cohort_audit.csv", index=False)
    cohort_detail = summarise_cohort(audit, args.horizon_hours)
    cohort = {
        "eligible_patients": int(len(features)),
        "positive_outcomes": int(labels.sum()),
        "horizon_hours": args.horizon_hours,
        "exclusions": exclusions,
        "prevalence": cohort_detail["prevalence"],
        "loaded_patients": cohort_detail["loaded_patients"],
        "eligible_iculos_at_row_0": cohort_detail["eligible_iculos_at_row_0"],
        "by_source": cohort_detail["by_source"],
    }
    (args.output_dir / "cohort_summary.json").write_text(
        json.dumps(cohort, indent=2, sort_keys=True) + "\n", encoding="utf-8",
    )
    report_path = write_html_report(args.output_dir, records=records, features=features)
    print(json.dumps({
        "cohort": cohort,
        "metrics": metrics,
        "comparison": comparison.to_dict(orient="records"),
        "output_dir": str(args.output_dir),
        "report": str(report_path),
    }, indent=2))


if __name__ == "__main__":
    main()

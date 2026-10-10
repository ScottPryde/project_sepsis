"""Command-line interface for validating data and running baseline experiments."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from sepsis_prediction.cnn import run_cnn_experiment
from sepsis_prediction.data import TARGET_COLUMN, load_patient_files
from sepsis_prediction.eda import run_eda
from sepsis_prediction.features import DEFAULT_HORIZON_HOURS, create_examples, create_sequence_examples
from sepsis_prediction.modeling import run_experiment
from sepsis_prediction.reporting import write_html_report, write_model_comparison
from sepsis_prediction.application import serve_application


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="sepsis-pipeline")
    commands = parser.add_subparsers(dest="command", required=True)

    validate = commands.add_parser("validate", help="validate PSV files and report basic counts")
    validate.add_argument("--data-dir", type=Path, required=True)
    validate.add_argument("--verbose", action="store_true", help="include patient filenames in the report")

    eda = commands.add_parser("eda", help="write dataset summaries and exploratory plots")
    eda.add_argument("--data-dir", type=Path, required=True)
    eda.add_argument("--output-dir", type=Path, default=Path("eda"))

    run = commands.add_parser("run", help="build the cohort, train baselines, and save artifacts")
    run.add_argument("--data-dir", type=Path, required=True)
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

    serve = commands.add_parser("serve", help="serve the JavaScript pipeline application")
    serve.add_argument("--data-dir", type=Path)
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
    return parser


def main() -> None:
    args = _parser().parse_args()
    if args.command == "serve":
        if not args.synthetic_demo and args.data_dir is None:
            raise SystemExit("serve requires --data-dir unless --synthetic-demo is selected")
        serve_application(
            output_dir=args.output_dir,
            data_dir=args.data_dir,
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
            },
        )
        return

    records = load_patient_files(args.data_dir)
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
    cohort = {
        "eligible_patients": int(len(features)),
        "positive_outcomes": int(labels.sum()),
        "horizon_hours": args.horizon_hours,
        "exclusions": exclusions,
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
"""Side-by-side summary of completed runs, such as synthetic versus PhysioNet."""

from __future__ import annotations

import html
import json
from pathlib import Path
from typing import Any

import pandas as pd


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8")) if path.is_file() else {}


def _format_interval(value: Any, interval: list[Any] | None) -> str:
    if value is None:
        return "—"
    if not interval or interval[0] is None or interval[1] is None:
        return f"{value:.3f}"
    return f"{value:.3f} ({interval[0]:.3f}–{interval[1]:.3f})"


def summarise_runs(runs: dict[str, Path]) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return one cohort row per run and one result row per run and model."""
    cohort_rows: list[dict[str, Any]] = []
    model_rows: list[dict[str, Any]] = []
    for name, directory in runs.items():
        config = _read_json(directory / "run_config.json")
        cohort = _read_json(directory / "cohort_summary.json")
        evaluation = _read_json(directory / "model_evaluation.json")
        if not config or not evaluation:
            raise FileNotFoundError(f"{directory} is not a completed run directory")
        test = config.get("split_composition", {}).get("test", {})
        prevalence = evaluation.get("prevalence")
        cohort_rows.append({
            "run": name,
            "data": "synthetic" if config.get("synthetic_demo") else "PhysioNet 2019",
            "split_mode": config.get("split_mode", "pooled"),
            "train_sources": ", ".join(config.get("train_sources") or []) or "all",
            "test_sources": ", ".join(config.get("test_sources") or []) or "all",
            "loaded_patients": cohort.get("loaded_patients"),
            "eligible_patients": cohort.get("eligible_patients"),
            "cohort_prevalence": cohort.get("prevalence"),
            "excluded_incomplete_followup": cohort.get("exclusions", {}).get("incomplete_horizon_followup"),
            "excluded_positive_in_lookback": cohort.get("exclusions", {}).get("positive_in_lookback"),
            "test_patients": evaluation.get("n_test", test.get("patients")),
            "test_positives": evaluation.get("positives", test.get("positive_outcomes")),
            "test_prevalence": prevalence,
        })
        for model in evaluation.get("models", []):
            auprc = model.get("auprc")
            model_rows.append({
                "run": name,
                "model": model.get("model"),
                "auroc": model.get("auroc"),
                "auroc_ci_low": (model.get("auroc_ci") or [None, None])[0],
                "auroc_ci_high": (model.get("auroc_ci") or [None, None])[1],
                "auprc": auprc,
                "auprc_ci_low": (model.get("auprc_ci") or [None, None])[0],
                "auprc_ci_high": (model.get("auprc_ci") or [None, None])[1],
                "auprc_over_prevalence": auprc / prevalence if auprc is not None and prevalence else None,
                "sensitivity": model.get("operating_point", {}).get("sensitivity"),
                "specificity": model.get("operating_point", {}).get("specificity"),
                "ppv": model.get("operating_point", {}).get("ppv"),
            })
    return pd.DataFrame(cohort_rows), pd.DataFrame(model_rows)


def write_run_comparison(runs: dict[str, Path], output_dir: str | Path) -> Path:
    """Write ``comparison_cohorts.csv``, ``comparison_models.csv``, and ``comparison.html``."""
    if not runs:
        raise ValueError("At least one run is required")
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    cohorts, models = summarise_runs(runs)
    cohorts.to_csv(output / "comparison_cohorts.csv", index=False, float_format="%.6g")
    models.to_csv(output / "comparison_models.csv", index=False, float_format="%.6g")

    cohort_table = cohorts.to_html(
        index=False, border=0, na_rep="—", float_format=lambda value: f"{value:.3f}",
    )
    shown = pd.DataFrame({
        "Run": models["run"],
        "Model": models["model"],
        "AUROC (95% CI)": [
            _format_interval(row.auroc, [row.auroc_ci_low, row.auroc_ci_high]) for row in models.itertuples()
        ],
        "AUPRC (95% CI)": [
            _format_interval(row.auprc, [row.auprc_ci_low, row.auprc_ci_high]) for row in models.itertuples()
        ],
        "AUPRC ÷ prevalence": [
            "—" if pd.isna(value) else f"{value:.2f}×" for value in models["auprc_over_prevalence"]
        ],
        "Sensitivity": models["sensitivity"],
        "Specificity": models["specificity"],
        "PPV": models["ppv"],
    })
    model_table = shown.to_html(index=False, border=0, na_rep="—", float_format=lambda value: f"{value:.3f}")
    synthetic_runs = cohorts.loc[cohorts["data"] == "synthetic", "run"].tolist()
    synthetic_note = (
        f"<p class=\"note\">{html.escape(', '.join(synthetic_runs))}: fabricated records with planted signal. "
        "They check that the workflow runs end to end; their scores are not evidence about sepsis prediction.</p>"
        if synthetic_runs else ""
    )
    page = f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><title>Run comparison</title>
<style>
body {{ font-family: system-ui, sans-serif; margin: 2rem; color: #1f2933; }}
table {{ border-collapse: collapse; margin: 1rem 0 2rem; font-size: 0.9rem; }}
th, td {{ padding: 0.35rem 0.7rem; border-bottom: 1px solid #d9dee3; text-align: right; }}
th:first-child, td:first-child, th:nth-child(2), td:nth-child(2) {{ text-align: left; }}
.note {{ background: #fff6e0; padding: 0.6rem 0.9rem; border-left: 4px solid #d19a2a; }}
</style></head><body>
<h1>Run comparison</h1>
{synthetic_note}
<p>AUPRC should be read against test prevalence, which is what a model with no skill scores.
"AUPRC ÷ prevalence" shows how many times better than that each model ranks positives.
Intervals are paired bootstrap 95% intervals over test patients.</p>
<h2>Cohorts and test sets</h2>
{cohort_table}
<h2>Held-out results</h2>
{model_table}
<p>Educational, retrospective analysis. Not for clinical use.</p>
</body></html>
"""
    path = output / "comparison.html"
    path.write_text(page, encoding="utf-8")
    return path


def parse_run_arguments(values: list[str]) -> dict[str, Path]:
    """Parse ``name=path`` pairs, keeping order and rejecting duplicate names."""
    runs: dict[str, Path] = {}
    for value in values:
        name, separator, path = value.partition("=")
        if not separator or not name or not path:
            raise ValueError(f"Expected NAME=PATH, got {value!r}")
        if name in runs:
            raise ValueError(f"Duplicate run name {name!r}")
        runs[name] = Path(path)
    return runs

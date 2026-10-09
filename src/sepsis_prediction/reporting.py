"""Held-out model comparison tables and discrimination plots."""

from __future__ import annotations

import base64
import html
import json
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import precision_recall_curve, roc_curve

from sepsis_prediction.data import TARGET_COLUMN, PatientRecord
from sepsis_prediction.evaluation import compare_models


def write_model_comparison(output_dir: str | Path) -> pd.DataFrame:
    """Write metric table and ROC/PR plots from saved test predictions."""
    output = Path(output_dir)
    predictions = pd.read_csv(output / "test_predictions.csv")
    model_names = [
        column.removesuffix("_probability")
        for column in predictions.columns
        if column.endswith("_probability")
    ]
    if not model_names:
        raise ValueError("No model probabilities found in test_predictions.csv")

    table = pd.DataFrame([
        {
            "model": model_name,
            "auroc": _metric(output, model_name, "auroc"),
            "auprc_average_precision": _metric(output, model_name, "auprc_average_precision"),
            "sensitivity_recall": _metric(output, model_name, "sensitivity_recall"),
            "specificity": _metric(output, model_name, "specificity"),
            "f1": _metric(output, model_name, "f1"),
            "brier_score": _metric(output, model_name, "brier_score"),
        }
        for model_name in model_names
    ])
    table.to_csv(output / "model_comparison.csv", index=False, float_format="%.8g")

    labels = predictions["y_true"].to_numpy(dtype=int)
    has_both_classes = pd.Series(labels).nunique() == 2
    figure, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    if has_both_classes:
        for model_name in model_names:
            probabilities = predictions[f"{model_name}_probability"].to_numpy()
            false_positive_rate, true_positive_rate, _ = roc_curve(labels, probabilities)
            precision, recall, _ = precision_recall_curve(labels, probabilities)
            axes[0].plot(false_positive_rate, true_positive_rate, label=model_name)
            axes[1].plot(recall, precision, label=model_name)
        axes[0].plot([0, 1], [0, 1], color="#777777", linestyle="--", linewidth=1)
        axes[0].set(xlabel="False positive rate", ylabel="True positive rate", title="ROC")
        axes[1].set(xlabel="Recall", ylabel="Precision", title="Precision-recall")
        axes[0].legend(loc="lower right", frameon=False)
        axes[1].legend(loc="best", frameon=False)
    else:
        axes[0].text(0.5, 0.5, "ROC undefined: test set has one class", ha="center", va="center")
        axes[1].text(0.5, 0.5, "PR curve limited: test set has one class", ha="center", va="center")
        axes[0].set_axis_off()
        axes[1].set_axis_off()
    figure.tight_layout()
    figure.savefig(output / "model_curves.png", dpi=160)
    plt.close(figure)
    compare_models(output)
    return table


def _metric(output: Path, model_name: str, metric: str) -> float | None:
    import json

    content = json.loads((output / "metrics.json").read_text(encoding="utf-8"))
    value = content[model_name][metric]
    return float(value) if value is not None else None


def write_html_report(
    output_dir: str | Path,
    *,
    records: list[PatientRecord] | None = None,
    features: pd.DataFrame | None = None,
    title: str = "Early Sepsis Prediction | Experiment Report",
) -> Path:
    """Generate a self-contained report from available experiment artifacts."""
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    metrics = _read_json(output / "metrics.json", {})
    cohort = _read_json(output / "cohort_summary.json", {})
    dataset = _read_json(output / "dataset_summary.json", {})
    config = _read_json(output / "run_config.json", {})
    comparison = pd.read_csv(output / "model_comparison.csv") if (output / "model_comparison.csv").exists() else pd.DataFrame()

    sample_html = _render_sample(records, features)
    metrics_html = _render_metrics(metrics)
    comparison_html = _render_comparison(comparison)
    cohort_html = _render_cohort(cohort, dataset)
    feature_html = _render_feature_sample(features)
    charts_html = _render_charts(output, metrics)
    diagrams_html = _render_diagrams(config)
    evaluation = _read_json(output / "model_evaluation.json", {})
    methodology_html = _render_methodology(config, cohort, features)
    effective_html = _render_effective(evaluation)
    synthetic_notice = (
        '<div class="notice"><strong>Synthetic smoke test.</strong> This report uses fabricated data '
        'and demonstrates workflow execution only; its metrics are not evidence of clinical performance.</div>'
        if config.get("synthetic_demo") else ""
    )
    generated_at = pd.Timestamp.now(tz="UTC").strftime("%Y-%m-%d %H:%M UTC")

    document = f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<meta name="color-scheme" content="light">
<title>{html.escape(title)}</title>
<style>
:root {{ color-scheme: light; --ink:#1c2727; --muted:#52605f; --line:#d8dfdc; --paper:#f4f5f0; --white:#fff; --teal:#167b73; --coral:#c45042; --gold:#d89a38; --blue:#477d9e; --font: "Aptos", "Segoe UI", sans-serif; }}
* {{ box-sizing:border-box; }} body {{ margin:0; color:var(--ink); background:var(--paper); font-family:var(--font); line-height:1.5; }}
.masthead {{ padding:26px max(24px, calc((100vw - 1180px)/2)); color:#f8f7ef; background:linear-gradient(110deg,#123d3a,#1a6258 62%,#247c6c); border-bottom:5px solid var(--gold); }}
.eyebrow {{ text-transform:uppercase; font-size:12px; font-weight:700; letter-spacing:.12em; color:#bfe2d6; }} h1 {{ max-width:900px; margin:10px 0 6px; font-size:42px; line-height:1.08; }}
.masthead p {{ max-width:850px; margin:8px 0 0; color:#e1eee9; }} main {{ max-width:1180px; margin:0 auto; padding:24px; }}
.notice {{ padding:13px 16px; margin-bottom:24px; border-left:4px solid var(--coral); background:#fff4ef; color:#542d28; }}
section {{ padding:22px 0 26px; border-top:1px solid var(--line); }} h2 {{ margin:0 0 14px; font-size:22px; }} h3 {{ margin:0 0 10px; font-size:16px; }}
.summary-grid {{ display:grid; grid-template-columns:repeat(auto-fit,minmax(150px,1fr)); gap:10px; margin-bottom:20px; }}
.stat {{ min-height:92px; padding:14px 16px; background:var(--white); border-top:3px solid var(--teal); }} .stat strong {{ display:block; font-size:26px; line-height:1.15; }} .stat span {{ color:var(--muted); font-size:13px; }}
.chart-grid {{ display:grid; grid-template-columns:repeat(auto-fit,minmax(min(100%,430px),1fr)); gap:16px; }}
.panel {{ min-width:0; padding:16px; background:var(--white); border:1px solid var(--line); }} .panel img {{ display:block; width:100%; height:auto; }}
.table-wrap {{ width:100%; overflow-x:auto; border:1px solid var(--line); background:var(--white); }} table {{ width:100%; border-collapse:collapse; font-size:13px; white-space:nowrap; }} th,td {{ padding:9px 11px; text-align:left; border-bottom:1px solid #e6eae7; }} th {{ position:sticky; top:0; color:#35504d; background:#edf2ee; font-weight:700; }} tbody tr:nth-child(even) {{ background:#f8faf8; }}
.diagram {{ overflow-x:auto; padding:12px; background:#fff; border:1px solid var(--line); }} svg {{ display:block; max-width:100%; height:auto; }} .diagram-label {{ color:var(--muted); font-size:13px; margin:0 0 8px; }}
.muted {{ color:var(--muted); font-size:12px; }} .prose {{ max-width:82ch; }} .prose p {{ margin:0 0 10px; }} .verdict {{ padding:14px 16px; margin:12px 0; background:#fff; border-left:4px solid var(--teal); }} .caution {{ color:#8f3b32; }}
.rank-table td:last-child {{ white-space:normal; min-width:180px; }} details.detail {{ margin:18px 0; padding:12px 16px; background:#fff; border:1px solid var(--line); }} details.detail summary {{ cursor:pointer; font-weight:700; }}
.empty {{ color:var(--muted); padding:12px 0; }} .footer {{ border-top:1px solid var(--line); padding:18px 0; color:var(--muted); font-size:12px; }}
@media(max-width:600px) {{ main {{ padding:18px; }} .masthead {{ padding:22px 18px; }} h1 {{ font-size:30px; }} .panel {{ padding:12px; }} th,td {{ padding:8px; }} }}
@media print {{ body {{ background:white; }} .masthead {{ print-color-adjust:exact; }} section {{ break-inside:avoid; }} }}
</style>
</head>
<body>
<header class="masthead"><div class="eyebrow">PhysioNet 2019 · Applied ML study</div><h1>{html.escape(title)}</h1><p>Patient-level early-prediction experiment · six-hour look-back · generated {generated_at}</p></header>
<main>
<div class="notice"><strong>Research use only.</strong> Retrospective modelling results are not clinically validated. This report may contain patient-derived clinical values; handle and share it according to the dataset terms and your institution's privacy requirements.</div>
{synthetic_notice}
{cohort_html}
{methodology_html}
<section><h2>Model results</h2>{effective_html}<details class="detail"><summary>Additional model metrics</summary>{metrics_html}{comparison_html}</details><div class="chart-grid">{charts_html}</div></section>
<section><h2>Pipeline and data lineage</h2>{diagrams_html}</section>
<section><h2>Sample of source observations</h2><p class="diagram-label">First six rows only, matching the model input window; identifiers are replaced with report-local labels.</p>{sample_html}</section>
<section><h2>Sample engineered inputs</h2><p class="diagram-label">Descriptive feature values from the same fixed look-back, where available.</p>{feature_html}</section>
<div class="footer">Prediction design: input rows 0–5; outcome is any positive SepsisLabel from row 6 onward. Patients positive in the look-back or without outcome follow-up are excluded. IDs and SepsisLabel are not model inputs; ICULOS is excluded.</div>
</main>
</body></html>"""
    report_path = output / "report.html"
    report_path.write_text(document, encoding="utf-8")
    return report_path


def _read_json(path: Path, default: Any) -> Any:
    if not path.exists():
        return default
    return json.loads(path.read_text(encoding="utf-8"))


def _render_metrics(metrics: dict[str, dict[str, Any]]) -> str:
    if not metrics:
        return '<p class="empty">No trained model results are present yet. Run the training command to populate this section.</p>'
    metric_labels = {
        "auroc": "AUROC", "auprc_average_precision": "AUPRC", "sensitivity_recall": "Sensitivity",
        "specificity": "Specificity", "precision": "Precision", "f1": "F1", "brier_score": "Brier",
    }
    items = []
    for name, values in metrics.items():
        count = values.get("n", "—")
        items.append(f'<div class="stat"><strong>{html.escape(str(count))}</strong><span>{html.escape(name)} · test patients</span></div>')
        for key, label in metric_labels.items():
            value = values.get(key)
            shown = "—" if value is None else f"{float(value):.3f}"
            items.append(f'<div class="stat"><strong>{shown}</strong><span>{html.escape(name)} · {label}</span></div>')
    return '<div class="summary-grid">' + "".join(items) + "</div>"


def _render_comparison(table: pd.DataFrame) -> str:
    if table.empty:
        return ""
    rendered = table.to_html(
        index=False,
        border=0,
        classes="results-table",
        na_rep="—",
        float_format=lambda value: f"{value:.3f}",
    )
    return '<h3>Held-out comparison</h3><div class="table-wrap">' + rendered + '</div>'


def _render_cohort(cohort: dict[str, Any], dataset: dict[str, Any]) -> str:
    values = []
    if cohort:
        values.extend([
            ("Eligible patients", cohort.get("eligible_patients", "—")),
            ("Positive outcomes", cohort.get("positive_outcomes", "—")),
        ])
        for key, value in cohort.get("exclusions", {}).items():
            values.append((f"Excluded · {key.replace('_', ' ')}", value))
    if dataset:
        values.extend([
            ("Source patients", dataset.get("patient_count", "—")),
            ("Hourly observations", dataset.get("hourly_observation_count", "—")),
            ("Positive row-label fraction", _format_value(dataset.get("positive_label_fraction"))),
        ])
    if not values:
        return ""
    cards = "".join(
        f'<div class="stat"><strong>{html.escape(str(value))}</strong><span>{html.escape(label)}</span></div>'
        for label, value in values
    )
    return f'<section><h2>Study snapshot</h2><div class="summary-grid">{cards}</div></section>'


def _render_charts(output: Path, metrics: dict[str, dict[str, Any]]) -> str:
    sections = []
    curve_path = output / "model_curves.png"
    if curve_path.exists():
        encoded = base64.b64encode(curve_path.read_bytes()).decode("ascii")
        sections.append(f'<div class="panel"><h3>Discrimination curves</h3><img alt="ROC and precision-recall curves" src="data:image/png;base64,{encoded}"></div>')
    for filename, heading, alt in (
        ("model_forest.png", "Discrimination with uncertainty", "AUPRC and AUROC confidence intervals"),
        ("calibration.png", "Calibration", "Calibration curves by model"),
        ("decision_curve.png", "Decision curve", "Net benefit by model and alert threshold"),
        ("missingness.png", "Missingness by variable", "Clinical variable missingness chart"),
        ("record_lengths.png", "Patient record lengths", "Patient record length histogram"),
        ("labels_by_hour.png", "Label prevalence by hour", "Sepsis label fraction by hour index"),
    ):
        image_path = output / filename
        if image_path.exists():
            encoded = base64.b64encode(image_path.read_bytes()).decode("ascii")
            sections.append(f'<div class="panel"><h3>{heading}</h3><img alt="{alt}" src="data:image/png;base64,{encoded}"></div>')
    if metrics:
        metric_names = ("auroc", "auprc_average_precision", "sensitivity_recall", "specificity", "f1")
        rows = []
        for name, values in metrics.items():
            cells = []
            for metric in metric_names:
                value = values.get(metric)
                percent = 0 if value is None else max(0, min(1, float(value))) * 100
                cells.append(f'<div class="bar-track" title="{html.escape(metric)}: {_format_value(value)}"><span style="width:{percent:.1f}%;background:{_color_for_metric(metric)}"></span></div>')
            rows.append(f'<div class="metric-row"><strong>{html.escape(name)}</strong>{"".join(cells)}</div>')
        headers = "".join(f"<span>{metric.replace('_', ' ')}</span>" for metric in metric_names)
        sections.append(f'<div class="panel"><h3>Metric profile</h3><div class="metric-header"><span>Model</span>{headers}</div>{"".join(rows)}<p class="diagram-label">Bar length is scaled from 0 to 1; Brier score is shown in the result cards and table.</p></div>')
    if not sections:
        return '<div class="panel empty">Run an experiment to generate result charts.</div>'
    return '<style>.metric-header,.metric-row{display:grid;grid-template-columns:130px repeat(5,minmax(62px,1fr));gap:8px;align-items:center;margin:9px 0;font-size:11px}.metric-header{color:#52605f;text-transform:capitalize}.metric-row strong{font-size:12px}.bar-track{height:10px;background:#e9eeeb;border-radius:2px;overflow:hidden}.bar-track span{display:block;height:100%;border-radius:2px}</style>' + "".join(sections)


def _render_methodology(
    config: dict[str, Any],
    cohort: dict[str, Any],
    features: pd.DataFrame | None,
) -> str:
    if not config:
        return ""
    design = config.get("design", {})
    horizon = design.get("horizon_hours", cohort.get("horizon_hours", 24))
    specs = config.get("model_specs", {})
    model_rows = "".join(
        "<tr>"
        f"<td>{html.escape(name.replace('_', ' ').title())}</td>"
        f"<td>{html.escape(spec.get('input', '—'))}</td>"
        f"<td>{html.escape('; '.join(spec.get('preprocessing', [])))}</td>"
        f"<td>{html.escape(spec.get('estimator', '—'))}</td>"
        f"<td>{html.escape(spec.get('class_weighting', '—'))}</td></tr>"
        for name, spec in specs.items()
    )
    fit_count = config.get("n_fit", "—")
    validation_count = config.get("n_validation", "—")
    test_count = config.get("n_test", "—")
    feature_count = features.shape[1] if features is not None else "—"
    threshold_selection = config.get("threshold_selection", "validation threshold at target sensitivity")
    label_definition = design.get("label_definition", "PhysioNet label timing is documented in the project design")
    return f'''<section><h2>Methodology</h2><div class="prose">
<p>Each patient contributes one prediction from rows 0–5. The outcome is any positive SepsisLabel in rows 6–{5 + int(horizon)} inclusive, a {int(horizon)}-hour label horizon. Negative patients without complete follow-up through that window are excluded; positives observed inside it are retained.</p>
<p>{html.escape(str(label_definition))}. Because the label leads Sepsis-3 onset by six hours, this horizon describes label timing rather than a direct onset timestamp.</p>
<p>Tabular input has {feature_count} engineered features. Patient-level stratified partitions are fit ({fit_count}), validation ({validation_count}), and test ({test_count}). All learned preprocessing is fit on fit patients only. Threshold policy: {html.escape(str(threshold_selection))}. Class weighting is disabled across model families.</p>
</div><h3>Model specifications</h3><div class="table-wrap"><table><thead><tr><th>Model</th><th>Input</th><th>Preprocessing</th><th>Estimator</th><th>Class weighting</th></tr></thead><tbody>{model_rows}</tbody></table></div>
<p class="diagram-label">Retrospective single-dataset evaluation only; not validated for clinical use.</p></section>'''


def _render_effective(evaluation: dict[str, Any]) -> str:
    rows = evaluation.get("models", [])
    if not rows:
        return '<p class="empty">Run training to generate the held-out comparison.</p>'
    def show(value: Any) -> str:
        if value is None:
            return "—"
        try:
            return f"{float(value):.3f}"
        except (TypeError, ValueError):
            return "—"

    def show_ci(value: Any, interval: list[Any]) -> str:
        if value is None:
            return "—"
        if interval[0] is None or interval[1] is None:
            return show(value)
        return f"{show(value)} ({show(interval[0])}–{show(interval[1])})"

    header = ("<tr><th>Rank</th><th>Model</th><th>AUPRC (95% CI)</th><th>AUROC (95% CI)</th>"
              "<th>Sensitivity</th><th>Specificity</th><th>PPV</th><th>Alerts / 100</th>"
              "<th>Brier</th><th>Calibration slope</th><th>Comparison with top</th></tr>")
    body = []
    for row in rows:
        operating = row["operating_point"]
        calibration = row["calibration"]
        difference = row.get("vs_top")
        comparison = "Top ranked" if difference is None else (
            f"AUPRC difference {show(difference['auprc_difference'])}; 95% CI "
            f"{show(difference['auprc_difference_ci'][0])}–{show(difference['auprc_difference_ci'][1])}"
        )
        body.append(
            f"<tr><td>{row['rank']}</td><td>{html.escape(row['model'].replace('_', ' ').title())}</td>"
            f"<td>{show_ci(row['auprc'], row['auprc_ci'])}</td>"
            f"<td>{show_ci(row['auroc'], row['auroc_ci'])}</td>"
            f"<td>{show(operating['sensitivity'])}</td><td>{show(operating['specificity'])}</td>"
            f"<td>{show(operating['ppv'])}</td><td>{show(operating['alerts_per_100'])}</td>"
            f"<td>{show(calibration['brier'])}</td><td>{show(calibration['calibration_slope'])}</td>"
            f"<td>{html.escape(comparison)}</td></tr>"
        )
    caution = (
        f'<p class="caution">Only {evaluation.get("positives", 0)} positive test patients; '
        "intervals and rankings may be unstable.</p>"
        if int(evaluation.get("positives", 0)) < 30 else ""
    )
    threshold_source = str(evaluation.get("threshold_source", "validation patients"))
    threshold_sentence = (
        "Thresholds were selected on validation patients and applied unchanged to test patients."
        if threshold_source.startswith("validation")
        else "Operating points use the fixed user-provided threshold on test patients."
    )
    return (f'<div class="verdict"><strong>Ranked by AUPRC</strong>, with AUROC as a tie-break. '
            f'{threshold_sentence} '
            f'Uncertainty uses {evaluation.get("n_bootstrap", 0)} paired bootstrap resamples of the same test patients.</div>'
            f'<div class="table-wrap"><table class="rank-table"><thead>{header}</thead><tbody>{"".join(body)}</tbody></table></div>'
            '<p class="diagram-label">Intervals describe sampling uncertainty on this test set. An interval that includes zero means the observed difference is not clearly separated; it does not establish model equivalence.</p>'
            + caution)


def _color_for_metric(metric: str) -> str:
    colors = {"auroc": "#167b73", "auprc_average_precision": "#477d9e", "sensitivity_recall": "#c45042", "specificity": "#d89a38", "f1": "#745d92"}
    return colors[metric]


def _render_diagrams(config: dict[str, Any]) -> str:
    design = config.get("design", {})
    selected = set(config.get("models", []))
    specs = config.get("model_specs", {})
    available = ("logistic_regression", "random_forest", "xgboost", "cnn_1d")
    colors = {"logistic_regression": "#167b73", "random_forest": "#477d9e",
              "xgboost": "#d89a38", "cnn_1d": "#c45042"}
    names = {name: name.replace("_", " ").title() for name in available}
    headings = (("Input", 84), ("Representation", 254), ("Train-only transforms", 424),
                ("Estimator", 594), ("Validation threshold", 764), ("Test evaluation", 934))
    lane_parts = [
        '<defs><marker id="lane-arrow" markerWidth="8" markerHeight="8" refX="7" refY="4" orient="auto">'
        '<path d="M0,0 L8,4 L0,8 z" fill="#536966"/></marker></defs>',
        '<g font-family="Segoe UI, sans-serif" text-anchor="middle">',
        '<rect x="18" y="8" width="964" height="32" rx="4" fill="#eef2ed" stroke="#8b9a91"/>',
        '<text x="500" y="29" font-size="13" fill="#1c2727">Shared cohort: six-hour look-back · '
        f'{int(design.get("horizon_hours", 24))}-hour outcome horizon · stratified patient split</text>',
    ]
    for label, x in headings:
        lane_parts.append(f'<text x="{x}" y="58" font-size="11" fill="#52605f">{html.escape(label)}</text>')
    for index, name in enumerate(available):
        y = 72 + index * 66
        active = name in selected
        stroke = colors[name] if active else "#a9b3b0"
        fill = "#ffffff" if active else "#f5f6f4"
        dash = "" if active else ' stroke-dasharray="5 4"'
        spec = specs.get(name, {})
        if name == "cnn_1d":
            representation = "Ordered values + masks"
            preprocessing = "Median impute + z-score"
            estimator = "1D CNN, early stop"
        else:
            representation = "Engineered summaries"
            preprocessing = "; ".join(spec.get("preprocessing", ["Median imputation"]))
            preprocessing = preprocessing.replace(" (fit-patient fitted)", "")
            estimator = spec.get("estimator", names[name])
            if len(estimator) > 24:
                estimator = names[name]
        boxes = (
            (18, 132, "Rows 0–5"),
            (188, 132, representation),
            (358, 132, preprocessing),
            (528, 132, estimator),
            (698, 132, "Threshold from validation"),
            (868, 114, "Frozen threshold; held-out metrics"),
        )
        for x, width, text in boxes:
            lane_parts.append(f'<rect x="{x}" y="{y}" width="{width}" height="42" rx="4" '
                              f'fill="{fill}" stroke="{stroke}"{dash}/>')
            if x == 18:
                lane_parts.append(f'<text x="{x + width / 2}" y="{y + 17}" font-size="9" '
                                  f'font-weight="700" fill="{stroke}">{html.escape(names[name])}'
                                  f'<tspan x="{x + width / 2}" dy="13" font-weight="400" fill="#1c2727">'
                                  f'{html.escape(text)}</tspan></text>')
            else:
                lane_parts.append(f'<text x="{x + width / 2}" y="{y + 25}" font-size="10" fill="#1c2727">'
                                  f'{html.escape(text)}</text>')
        for x in (150, 320, 490, 660, 830):
            lane_parts.append(f'<path d="M{x} {y + 21} H{x + 30}" fill="none" stroke="#536966" '
                              'stroke-width="1.5" marker-end="url(#lane-arrow)"/>')
    lane_parts.append('</g>')
    pipeline_height = 72 + len(available) * 66
    pipeline = (f'<div class="diagram"><p class="diagram-label">Per-model pipeline; solid lanes ran, dashed lanes were not selected</p>'
                f'<svg viewBox="0 0 1000 {pipeline_height}" role="img" aria-label="Per-model pipeline lanes">'
                f'{"".join(lane_parts)}</svg></div>')
    lineage = '''<div class="diagram"><p class="diagram-label">Data lineage · predictors and target come from disjoint time ranges</p><svg viewBox="0 0 1040 190" role="img" aria-label="Data lineage diagram"><defs><marker id="arrow2" markerWidth="9" markerHeight="9" refX="7" refY="4.5" orient="auto"><path d="M0,0 L9,4.5 L0,9 z" fill="#536966"/></marker></defs><g font-family="Segoe UI, sans-serif" font-size="13" text-anchor="middle"><rect x="24" y="48" width="190" height="78" rx="5" fill="#eef2ed" stroke="#8b9a91"/><text x="119" y="78" fill="#1c2727"><tspan>Raw patient file</tspan><tspan x="119" dy="20">values + labels</tspan></text><rect x="300" y="24" width="235" height="66" rx="5" fill="#e4f1ed" stroke="#167b73" stroke-width="2"/><text x="417" y="51" fill="#1c2727"><tspan>Rows 0–5</tspan><tspan x="417" dy="19">clinical inputs only</tspan></text><rect x="300" y="108" width="235" height="58" rx="5" fill="#fbebe7" stroke="#c45042" stroke-width="2"/><text x="417" y="132" fill="#1c2727"><tspan>Row 6 onward</tspan><tspan x="417" dy="18">outcome labels only</tspan></text><rect x="642" y="48" width="174" height="78" rx="5" fill="#fff0d9" stroke="#d89a38" stroke-width="2"/><text x="729" y="78" fill="#1c2727"><tspan>Join by patient</tspan><tspan x="729" dy="20">after eligibility</tspan></text><rect x="880" y="48" width="142" height="78" rx="5" fill="#e9e6f0" stroke="#745d92" stroke-width="2"/><text x="951" y="78" fill="#1c2727"><tspan>Saved test</tspan><tspan x="951" dy="20">predictions</tspan></text><path d="M215 86 H290 M535 56 C580 56 590 70 632 80 M535 137 C580 137 590 105 632 94 M817 86 H870" fill="none" stroke="#536966" stroke-width="2" marker-end="url(#arrow2)"/><text x="520" y="184" fill="#52605f">Patient ID is lineage metadata, never an input feature · SepsisLabel is target-only</text></g></svg></div>'''
    metadata = f'<p class="diagram-label">Recorded design: {html.escape(str(design.get("lookback_hours", "six-hour look-back")))}; outcome: {html.escape(str(design.get("prediction_period", "bounded post-look-back label")))}; split unit: {html.escape(str(design.get("split_unit", "patient")))}.</p>'
    return pipeline + lineage + metadata


def _render_sample(
    records: list[PatientRecord] | None,
    features: pd.DataFrame | None,
) -> str:
    if not records:
        return '<p class="empty">Source rows are unavailable for this report. Generate it through the `run` or `eda` command to include a sample.</p>'
    if features is not None:
        eligible_ids = set(features.index.astype(str))
        records = [record for record in records if record.patient_id in eligible_ids]
    records = records[:5]
    if not records:
        return '<p class="empty">No eligible source observations are available for the sample.</p>'
    sample_columns = [column for column in ("HR", "O2Sat", "Temp", "SBP", "MAP", "Resp", TARGET_COLUMN) if any(column in record.frame.columns for record in records)]
    rows = []
    for patient_number, record in enumerate(records[:3], start=1):
        for hour, (_, row) in enumerate(record.frame.head(6).iterrows()):
            values = [f"Sample {patient_number}", str(hour)]
            values.extend(_format_value(row.get(column)) for column in sample_columns)
            rows.append(values)
    if not rows:
        return '<p class="empty">No input observations were available.</p>'
    headers = ["Sample patient", "Hour index", *sample_columns]
    table = pd.DataFrame(rows, columns=headers)
    return '<div class="table-wrap">' + table.to_html(index=False, escape=True, border=0) + '</div>'


def _render_feature_sample(features: pd.DataFrame | None) -> str:
    if features is None or features.empty:
        return '<p class="empty">Engineered features are unavailable for this report.</p>'
    preferred = [column for column in ("HR_latest", "HR_mean", "HR_delta", "HR_missing_fraction", "O2Sat_latest", "Temp_latest", "MAP_latest") if column in features.columns]
    sample = features.loc[:, preferred].head(5).copy()
    sample.insert(0, "sample_patient", [f"Sample {index}" for index in range(1, len(sample) + 1)])
    sample = sample.apply(lambda column: column.map(_format_value))
    return '<div class="table-wrap">' + sample.to_html(index=False, escape=True, border=0) + '</div>'


def _format_value(value: Any) -> str:
    if value is None or pd.isna(value):
        return "—"
    try:
        number = float(value)
    except (TypeError, ValueError):
        return html.escape(str(value))
    if number.is_integer():
        return str(int(number))
    return f"{number:.3g}"
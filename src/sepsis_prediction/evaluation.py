"""Held-out model comparison using thresholds frozen on validation patients."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score


def _metric_pair(y: np.ndarray, probabilities: np.ndarray) -> tuple[float | None, float | None]:
    if np.unique(y).size != 2:
        return None, None
    return float(average_precision_score(y, probabilities)), float(roc_auc_score(y, probabilities))


def _interval(values: np.ndarray) -> list[float | None]:
    finite = values[np.isfinite(values)]
    if finite.size < 20:
        return [None, None]
    low, high = np.percentile(finite, [2.5, 97.5])
    return [float(low), float(high)]


def _operating_point(y: np.ndarray, probabilities: np.ndarray, threshold: float) -> dict[str, float | None]:
    predicted = probabilities >= threshold
    positives = int(y.sum())
    true_positive = int((predicted & (y == 1)).sum())
    false_positive = int((predicted & (y == 0)).sum())
    alerts = int(predicted.sum())
    ppv = true_positive / alerts if alerts else 0.0
    return {
        "threshold": float(threshold),
        "sensitivity": true_positive / positives if positives else None,
        "specificity": 1 - false_positive / int((y == 0).sum()) if (y == 0).any() else None,
        "ppv": ppv,
        "alerts_per_100": 100 * alerts / len(y) if len(y) else None,
        "alerts_per_true_case": alerts / true_positive if true_positive else None,
    }


def _calibration(y: np.ndarray, probabilities: np.ndarray) -> dict[str, float | None]:
    brier = float(brier_score_loss(y, probabilities))
    slope = intercept = None
    if np.unique(y).size == 2 and np.std(probabilities) > 0:
        from sklearn.linear_model import LogisticRegression

        clipped = np.clip(probabilities, 1e-6, 1 - 1e-6)
        logits = np.log(clipped / (1 - clipped))
        fitted = LogisticRegression(C=1e6, max_iter=1000).fit(logits[:, None], y)
        slope = float(fitted.coef_[0, 0])
        intercept = float(fitted.intercept_[0])
    bins = np.array_split(np.argsort(probabilities), min(10, len(y))) if len(y) else []
    ece = sum(
        len(indices) / len(y) * abs(float(probabilities[indices].mean()) - float(y[indices].mean()))
        for indices in bins if len(indices)
    ) if len(y) else None
    return {
        "brier": brier,
        "calibration_slope": slope,
        "calibration_intercept": intercept,
        "ece_10_bin": float(ece) if ece is not None else None,
    }


def _net_benefit(y: np.ndarray, probabilities: np.ndarray, thresholds: np.ndarray) -> list[float]:
    n = len(y)
    values = []
    for threshold in thresholds:
        flagged = probabilities >= threshold
        tp = int((flagged & (y == 1)).sum())
        fp = int((flagged & (y == 0)).sum())
        values.append(tp / n - fp / n * threshold / (1 - threshold))
    return values


def compare_models(
    output_dir: str | Path,
    *,
    n_bootstrap: int = 1000,
) -> dict[str, Any]:
    """Write paired uncertainty, operating-point, calibration, and utility outputs."""
    output = Path(output_dir)
    predictions = pd.read_csv(output / "test_predictions.csv")
    config = json.loads((output / "run_config.json").read_text(encoding="utf-8"))
    y = predictions["y_true"].to_numpy(dtype=int)
    names = [column.removesuffix("_probability") for column in predictions
             if column.endswith("_probability")]
    probabilities = {name: predictions[f"{name}_probability"].to_numpy(dtype=float) for name in names}
    thresholds = config.get("model_thresholds", {})
    fallback_threshold = config.get("threshold_override")
    if fallback_threshold is None:
        fallback_threshold = 0.5
    seed = int(config.get("random_state", 42))
    rng = np.random.default_rng(seed)
    bootstrap: dict[str, dict[str, np.ndarray]] = {
        name: {"auprc": np.full(n_bootstrap, np.nan), "auroc": np.full(n_bootstrap, np.nan)}
        for name in names
    }
    if np.unique(y).size == 2:
        for index in range(n_bootstrap):
            sample = rng.integers(0, len(y), len(y))
            if np.unique(y[sample]).size != 2:
                continue
            for name in names:
                ap, auc = _metric_pair(y[sample], probabilities[name][sample])
                bootstrap[name]["auprc"][index] = ap
                bootstrap[name]["auroc"][index] = auc

    rows: list[dict[str, Any]] = []
    for name in names:
        ap, auc = _metric_pair(y, probabilities[name])
        threshold = float(thresholds.get(name, fallback_threshold))
        rows.append({
            "model": name,
            "auprc": ap,
            "auprc_ci": _interval(bootstrap[name]["auprc"]),
            "auroc": auc,
            "auroc_ci": _interval(bootstrap[name]["auroc"]),
            "operating_point": _operating_point(y, probabilities[name], threshold),
            "calibration": _calibration(y, probabilities[name]),
        })
    rows.sort(key=lambda row: (
        -1 if row["auprc"] is None else row["auprc"],
        -1 if row["auroc"] is None else row["auroc"],
    ), reverse=True)
    for rank, row in enumerate(rows, start=1):
        row["rank"] = rank
        row["vs_top"] = None
        if rank > 1 and rows[0]["auprc"] is not None and row["auprc"] is not None:
            best_name = rows[0]["model"]
            name = row["model"]
            differences = bootstrap[best_name]["auprc"] - bootstrap[name]["auprc"]
            row["vs_top"] = {
                "auprc_difference": rows[0]["auprc"] - row["auprc"],
                "auprc_difference_ci": _interval(differences),
            }

    curve_thresholds = np.round(np.arange(0.01, 0.51, 0.01), 2)
    prevalence = float(y.mean()) if len(y) else 0.0
    result = {
        "n_test": len(y),
        "positives": int(y.sum()),
        "prevalence": prevalence,
        "n_bootstrap": n_bootstrap,
        "bootstrap_seed": seed,
        "threshold_source": config.get("threshold_selection", "validation patients"),
        "target_sensitivity": float(config.get("target_sensitivity", 0.8)),
        "ranking_rule": "AUPRC, with AUROC as a tie-break",
        "models": rows,
        "decision_curve": {
            "thresholds": curve_thresholds.tolist(),
            "treat_all": (prevalence - (1 - prevalence) * curve_thresholds /
                          (1 - curve_thresholds)).tolist(),
            "models": {name: _net_benefit(y, probabilities[name], curve_thresholds) for name in names},
        },
    }
    (output / "model_evaluation.json").write_text(
        json.dumps(result, indent=2, allow_nan=False) + "\n", encoding="utf-8",
    )
    table = pd.DataFrame([{
        "rank": row["rank"], "model": row["model"], "auprc": row["auprc"],
        "auprc_ci_low": row["auprc_ci"][0], "auprc_ci_high": row["auprc_ci"][1],
        "auroc": row["auroc"], "auroc_ci_low": row["auroc_ci"][0],
        "auroc_ci_high": row["auroc_ci"][1],
        "threshold": row["operating_point"]["threshold"],
        "sensitivity": row["operating_point"]["sensitivity"],
        "specificity": row["operating_point"]["specificity"],
        "ppv": row["operating_point"]["ppv"],
        "alerts_per_100": row["operating_point"]["alerts_per_100"],
        "alerts_per_true_case": row["operating_point"]["alerts_per_true_case"],
        "brier": row["calibration"]["brier"],
        "calibration_slope": row["calibration"]["calibration_slope"],
        "ece_10_bin": row["calibration"]["ece_10_bin"],
    } for row in rows])
    table.to_csv(output / "model_ranking.csv", index=False, float_format="%.8g")
    _plot_outputs(output, y, probabilities, result)
    return result


def _plot_outputs(
    output: Path,
    y: np.ndarray,
    probabilities: dict[str, np.ndarray],
    result: dict[str, Any],
) -> None:
    colors = ("#167b73", "#477d9e", "#d89a38", "#c45042")
    figure, axes = plt.subplots(1, 2, figsize=(9, 4.2))
    for axis, metric in zip(axes, ("auprc", "auroc"), strict=True):
        for position, row in enumerate(result["models"]):
            value = row[metric]
            low, high = row[f"{metric}_ci"]
            if value is not None:
                axis.plot([low, high], [position, position], color=colors[position % len(colors)],
                          linewidth=2) if low is not None else None
                axis.plot(value, position, "o", color=colors[position % len(colors)])
        axis.set_yticks(range(len(result["models"])), [row["model"].replace("_", " ")
                                                       for row in result["models"]])
        axis.set_xlim(0, 1)
        axis.set_title(metric.upper() + " (95% CI)")
        axis.grid(axis="x", alpha=0.2)
    figure.tight_layout()
    figure.savefig(output / "model_forest.png", dpi=150)
    plt.close(figure)

    figure, axis = plt.subplots(figsize=(5.5, 4.2))
    axis.plot([0, 1], [0, 1], "--", color="#777777", linewidth=1)
    for position, (name, probabilities_for_model) in enumerate(probabilities.items()):
        indices = np.array_split(np.argsort(probabilities_for_model), min(10, len(y)))
        x = [probabilities_for_model[index].mean() for index in indices if len(index)]
        observed = [y[index].mean() for index in indices if len(index)]
        axis.plot(x, observed, marker="o", label=name.replace("_", " "),
                  color=colors[position % len(colors)])
    axis.set(xlim=(0, 1), ylim=(0, 1), xlabel="Mean predicted risk", ylabel="Observed rate",
             title="Test-set calibration")
    axis.legend(frameon=False)
    figure.tight_layout()
    figure.savefig(output / "calibration.png", dpi=150)
    plt.close(figure)

    decision = result["decision_curve"]
    thresholds = np.asarray(decision["thresholds"])
    figure, axis = plt.subplots(figsize=(5.5, 4.2))
    axis.plot(thresholds, decision["treat_all"], color="#777777", label="Alert all")
    axis.axhline(0, color="#333333", linewidth=1, label="Alert none")
    for position, (name, values) in enumerate(decision["models"].items()):
        axis.plot(thresholds, values, label=name.replace("_", " "),
                  color=colors[position % len(colors)])
    axis.set(xlabel="Alert threshold", ylabel="Net benefit", title="Decision curve")
    axis.legend(frameon=False)
    figure.tight_layout()
    figure.savefig(output / "decision_curve.png", dpi=150)
    plt.close(figure)
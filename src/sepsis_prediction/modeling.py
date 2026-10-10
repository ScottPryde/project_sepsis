"""Patient-level splitting, train-only preprocessing, and baseline evaluation."""

from __future__ import annotations

import json
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier

from sepsis_prediction.features import DEFAULT_HORIZON_HOURS


SPLIT_MODES = ("pooled", "hospital")

DESIGN = {
    "lookback_rows": 6,
    "lookback_hours": "hours 0-5 inclusive",
    "lookback_anchor": "first six rows of each record; ICULOS at row 0 is recorded for audit, not used to shift or exclude",
    "prediction_period": "any SepsisLabel=1 within the configured horizon after hour 5",
    "label_definition": "SepsisLabel switches on six hours before Sepsis-3 onset (PhysioNet 2019)",
    "exclude_positive_in_lookback": True,
    "exclude_records_shorter_than_rows": 6,
    "minimum_rows_for_observed_outcome": 7,
    "predictor_rows": "only rows 0-5",
    "excluded_predictors": ["SepsisLabel", "patient_id", "ICULOS"],
    "split_unit": "patient",
}

MODEL_SPECS = {
    "logistic_regression": {
        "input": "engineered tabular features",
        "preprocessing": ["median imputation (fit-patient fitted)", "standardisation (fit-patient fitted)"],
        "estimator": "L2 logistic regression, max_iter=2000",
        "class_weighting": "none",
    },
    "random_forest": {
        "input": "engineered tabular features",
        "preprocessing": ["median imputation (fit-patient fitted)"],
        "estimator": "300 trees, default depth",
        "class_weighting": "none",
    },
    "xgboost": {
        "input": "engineered tabular features",
        "preprocessing": ["median imputation (fit-patient fitted)"],
        "estimator": "300 boosted trees, max_depth=4, learning_rate=0.05, subsample=0.8, colsample_bytree=0.8",
        "class_weighting": "none",
    },
}


def make_pipeline(model_name: str, feature_names: list[str], random_state: int) -> Pipeline:
    """Create a model with all learned preprocessing inside its sklearn pipeline."""
    if model_name == "logistic_regression":
        numeric_pipeline = Pipeline([
            ("imputer", SimpleImputer(strategy="median", keep_empty_features=True)),
            ("scaler", StandardScaler()),
        ])
        estimator = LogisticRegression(max_iter=2000, random_state=random_state)
    elif model_name == "random_forest":
        numeric_pipeline = Pipeline([
            ("imputer", SimpleImputer(strategy="median", keep_empty_features=True)),
        ])
        estimator = RandomForestClassifier(
            n_estimators=300,
            n_jobs=1,
            random_state=random_state,
        )
    elif model_name == "xgboost":
        try:
            from xgboost import XGBClassifier
        except ImportError as error:
            raise ImportError(
                "XGBoost is optional; install it with `python -m pip install -e '.[xgboost]'`"
            ) from error
        numeric_pipeline = Pipeline([
            ("imputer", SimpleImputer(strategy="median", keep_empty_features=True)),
        ])
        estimator = XGBClassifier(
            n_estimators=300,
            max_depth=4,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            eval_metric="logloss",
            n_jobs=1,
            random_state=random_state,
        )
    else:
        raise ValueError(f"Unsupported model: {model_name}")

    preprocessing = ColumnTransformer(
        [("numeric", numeric_pipeline, feature_names)],
        remainder="drop",
        verbose_feature_names_out=False,
    )
    return Pipeline([("preprocess", preprocessing), ("classifier", estimator)])


def calculate_metrics(y_true: np.ndarray, probabilities: np.ndarray, threshold: float) -> dict[str, Any]:
    predictions = (probabilities >= threshold).astype(int)
    matrix = confusion_matrix(y_true, predictions, labels=[0, 1])
    true_negative, false_positive, _, _ = matrix.ravel()
    both_classes = np.unique(y_true).size == 2
    return {
        "threshold": float(threshold),
        "n": int(y_true.size),
        "positive_count": int(y_true.sum()),
        "accuracy": float(accuracy_score(y_true, predictions)),
        "precision": float(precision_score(y_true, predictions, zero_division=0)),
        "sensitivity_recall": float(recall_score(y_true, predictions, zero_division=0)),
        "specificity": float(true_negative / (true_negative + false_positive))
        if true_negative + false_positive else None,
        "f1": float(f1_score(y_true, predictions, zero_division=0)),
        "auroc": float(roc_auc_score(y_true, probabilities)) if both_classes else None,
        "auprc_average_precision": float(average_precision_score(y_true, probabilities))
        if both_classes else None,
        "brier_score": float(brier_score_loss(y_true, probabilities)),
        "confusion_matrix_labels_0_1": matrix.astype(int).tolist(),
    }


def _write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")


def _package_versions() -> dict[str, str | None]:
    packages = ("numpy", "pandas", "scikit-learn", "matplotlib", "joblib", "xgboost", "torch")
    versions: dict[str, str | None] = {}
    for package in packages:
        try:
            versions[package] = version(package)
        except PackageNotFoundError:
            versions[package] = None
    return versions


def _threshold_at_sensitivity(y_true: np.ndarray, probabilities: np.ndarray, target: float) -> float:
    if not 0 < target <= 1:
        raise ValueError("target_sensitivity must be in (0, 1]")
    positives = int(y_true.sum())
    if positives == 0 or np.unique(y_true).size < 2:
        raise ValueError("Validation patients must contain both outcome classes")
    for candidate in np.sort(np.unique(probabilities))[::-1]:
        sensitivity = int(((probabilities >= candidate) & (y_true == 1)).sum()) / positives
        if sensitivity >= target:
            return float(candidate)
    return 0.0


def split_patients(
    labels: pd.Series,
    groups: pd.Series,
    *,
    test_size: float = 0.2,
    validation_size: float = 0.2,
    random_state: int = 42,
    split_mode: str = "pooled",
    sources: pd.Series | None = None,
    train_sources: tuple[str, ...] = (),
    test_sources: tuple[str, ...] = (),
) -> dict[str, np.ndarray]:
    """Return fit, validation, and test row positions for one patient per row.

    ``pooled`` holds out a stratified random ``test_size`` share of all patients.
    ``hospital`` uses every patient from ``test_sources`` as the test set and
    draws fit and validation patients from ``train_sources`` only, so test
    patients come from a hospital the models never saw. In both modes the
    validation share is a stratified split of the non-test patients.
    """
    if split_mode not in SPLIT_MODES:
        raise ValueError(f"split_mode must be one of {SPLIT_MODES}")
    if not 0 < validation_size < 1:
        raise ValueError("validation_size must be between 0 and 1")
    if not len(labels) == len(groups) or not labels.index.equals(groups.index):
        raise ValueError("labels and groups must have matching indexes")
    if groups.astype(str).duplicated().any():
        raise ValueError("Expected exactly one cohort row per patient for stratified splitting")
    positions = np.arange(len(labels))

    if split_mode == "pooled":
        if not 0 < test_size < 1:
            raise ValueError("test_size must be between 0 and 1")
        test_splitter = StratifiedShuffleSplit(n_splits=1, test_size=test_size, random_state=random_state)
        train_validation_positions, test_positions = next(test_splitter.split(positions, labels))
    else:
        if sources is None or not sources.index.equals(labels.index):
            raise ValueError("hospital split mode needs a sources Series aligned to the labels")
        if not train_sources or not test_sources:
            raise ValueError("hospital split mode needs both train_sources and test_sources")
        if set(train_sources) & set(test_sources):
            raise ValueError("train_sources and test_sources must not overlap")
        source_values = sources.astype(str).to_numpy()
        unknown = (set(train_sources) | set(test_sources)) - set(source_values)
        if unknown:
            raise ValueError(f"Unknown sources {sorted(unknown)}; available: {sorted(set(source_values))}")
        train_validation_positions = positions[np.isin(source_values, train_sources)]
        test_positions = positions[np.isin(source_values, test_sources)]

    validation_splitter = StratifiedShuffleSplit(
        n_splits=1, test_size=validation_size, random_state=random_state + 1,
    )
    fit_relative, validation_relative = next(validation_splitter.split(
        train_validation_positions, labels.iloc[train_validation_positions],
    ))
    split = {
        "fit": train_validation_positions[fit_relative],
        "validation": train_validation_positions[validation_relative],
        "test": test_positions,
    }
    fit_groups = set(groups.iloc[split["fit"]].astype(str))
    validation_groups = set(groups.iloc[split["validation"]].astype(str))
    test_groups = set(groups.iloc[split["test"]].astype(str))
    if fit_groups & validation_groups or fit_groups & test_groups or validation_groups & test_groups:
        raise RuntimeError("Patient group overlap found between fit, validation, and test")
    for name, split_positions in split.items():
        if labels.iloc[split_positions].nunique() != 2:
            raise ValueError(f"{name.title()} split contains one outcome class; adjust split sizes or seed")
    return split


def _split_composition(
    labels: pd.Series,
    sources: pd.Series | None,
    split: dict[str, np.ndarray],
) -> dict[str, dict[str, Any]]:
    composition: dict[str, dict[str, Any]] = {}
    for name, positions in split.items():
        part = labels.iloc[positions]
        entry: dict[str, Any] = {
            "patients": int(len(part)),
            "positive_outcomes": int(part.sum()),
            "prevalence": float(part.mean()),
        }
        if sources is not None:
            entry["by_source"] = {
                str(source): int(count)
                for source, count in sources.iloc[positions].astype(str).value_counts().sort_index().items()
            }
        composition[name] = entry
    return composition


def run_experiment(
    features: pd.DataFrame,
    labels: pd.Series,
    groups: pd.Series,
    output_dir: str | Path,
    *,
    models: tuple[str, ...] = ("logistic_regression", "random_forest"),
    test_size: float = 0.2,
    random_state: int = 42,
    threshold: float | None = None,
    horizon_hours: int = DEFAULT_HORIZON_HOURS,
    validation_size: float = 0.2,
    target_sensitivity: float = 0.8,
    split_mode: str = "pooled",
    sources: pd.Series | None = None,
    train_sources: tuple[str, ...] = (),
    test_sources: tuple[str, ...] = (),
) -> dict[str, dict[str, Any]]:
    """Split patients, fit selected baselines, and write artifacts."""
    if horizon_hours < 1:
        raise ValueError("horizon_hours must be a positive integer")
    supported_models = {"logistic_regression", "random_forest", "xgboost"}
    if not models:
        raise ValueError("At least one model must be selected")
    unknown_models = set(models) - supported_models
    if unknown_models:
        raise ValueError(f"Unsupported models: {sorted(unknown_models)}")
    if not (len(features) == len(labels) == len(groups)):
        raise ValueError("features, labels, and groups must have equal lengths")
    if features.empty or features.columns.empty:
        raise ValueError("No feature rows or feature columns were provided")
    if not features.index.equals(labels.index) or not features.index.equals(groups.index):
        raise ValueError("features, labels, and groups must have matching indexes")
    if labels.nunique() != 2:
        raise ValueError("Training cohort must contain both outcome classes")

    split = split_patients(
        labels, groups,
        test_size=test_size, validation_size=validation_size, random_state=random_state,
        split_mode=split_mode, sources=sources,
        train_sources=tuple(train_sources), test_sources=tuple(test_sources),
    )
    fit_positions = split["fit"]
    validation_positions = split["validation"]
    test_positions = split["test"]
    train_positions = np.concatenate([fit_positions, validation_positions])

    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    feature_names = list(features.columns)
    y_fit = labels.iloc[fit_positions].to_numpy(dtype=int)
    y_validation = labels.iloc[validation_positions].to_numpy(dtype=int)
    y_test = labels.iloc[test_positions].to_numpy(dtype=int)
    validation_features = features.iloc[validation_positions]
    test_features = features.iloc[test_positions]
    metrics: dict[str, dict[str, Any]] = {}
    thresholds: dict[str, float] = {}
    importance_rows: list[dict[str, Any]] = []
    predictions = pd.DataFrame({
        "patient_id": groups.iloc[test_positions].astype(str).to_numpy(),
        "y_true": y_test,
    })
    validation_predictions = pd.DataFrame({
        "patient_id": groups.iloc[validation_positions].astype(str).to_numpy(),
        "y_true": y_validation,
    })

    for model_name in models:
        model = make_pipeline(model_name, feature_names, random_state)
        model.fit(features.iloc[fit_positions], y_fit)
        validation_probabilities = model.predict_proba(validation_features)[:, 1]
        probabilities = model.predict_proba(test_features)[:, 1]
        selected_threshold = threshold if threshold is not None else _threshold_at_sensitivity(
            y_validation, validation_probabilities, target_sensitivity,
        )
        thresholds[model_name] = float(selected_threshold)
        metrics[model_name] = calculate_metrics(y_test, probabilities, selected_threshold)
        validation_predictions[f"{model_name}_probability"] = validation_probabilities
        validation_predictions[f"{model_name}_prediction"] = (
            validation_probabilities >= selected_threshold
        ).astype(int)
        predictions[f"{model_name}_probability"] = probabilities
        predictions[f"{model_name}_prediction"] = (probabilities >= selected_threshold).astype(int)
        classifier = model.named_steps["classifier"]
        if hasattr(classifier, "coef_"):
            values = classifier.coef_[0]
            importance_type = "coefficient"
        else:
            values = classifier.feature_importances_
            importance_type = "feature_importance"
        importance_rows.extend({
            "model": model_name,
            "feature": feature_name,
            "importance_type": importance_type,
            "value": float(value),
            "absolute_value": float(abs(value)),
        } for feature_name, value in zip(feature_names, values, strict=True))
        joblib.dump(model, output / f"{model_name}.joblib")

    predictions.to_csv(output / "test_predictions.csv", index=False, float_format="%.12g")
    validation_predictions.to_csv(output / "validation_predictions.csv", index=False, float_format="%.12g")
    pd.DataFrame(importance_rows).sort_values(
        ["model", "absolute_value"], ascending=[True, False], ignore_index=True,
    ).to_csv(output / "feature_importance.csv", index=False, float_format="%.12g")
    _write_json(output / "metrics.json", metrics)
    _write_json(output / "feature_names.json", feature_names)
    _write_json(output / "split_patients.json", {
        "train": groups.iloc[train_positions].astype(str).tolist(),
        "fit": groups.iloc[fit_positions].astype(str).tolist(),
        "validation": groups.iloc[validation_positions].astype(str).tolist(),
        "test": groups.iloc[test_positions].astype(str).tolist(),
    })
    _write_json(output / "run_config.json", {
        "design": {**DESIGN, "horizon_hours": horizon_hours},
        "models": list(models),
        "model_specs": {name: MODEL_SPECS[name] for name in models},
        "random_state": random_state,
        "split_mode": split_mode,
        "train_sources": list(train_sources) if split_mode == "hospital" else None,
        "test_sources": list(test_sources) if split_mode == "hospital" else None,
        "split_composition": _split_composition(labels, sources, split),
        "test_size": test_size if split_mode == "pooled" else None,
        "validation_size": validation_size,
        "threshold_selection": (
            "fixed user override" if threshold is not None
            else "validation threshold at target sensitivity"
        ),
        "target_sensitivity": target_sensitivity,
        "threshold_override": threshold,
        "model_thresholds": thresholds,
        "package_versions": _package_versions(),
        "n_patients": int(len(features)),
        "n_train": int(len(train_positions)),
        "n_fit": int(len(fit_positions)),
        "n_validation": int(len(validation_positions)),
        "n_test": int(len(test_positions)),
    })
    return metrics
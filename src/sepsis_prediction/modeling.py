"""Patient-level splitting, train-only preprocessing, and baseline evaluation."""

from __future__ import annotations

import json
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
from sklearn.model_selection import GroupShuffleSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier


DESIGN = {
    "lookback_rows": 6,
    "lookback_hours": "hours 0-5 inclusive",
    "prediction_period": "any SepsisLabel=1 from hour 6 onward",
    "exclude_positive_in_lookback": True,
    "exclude_records_shorter_than_rows": 6,
    "minimum_rows_for_observed_outcome": 7,
    "predictor_rows": "only rows 0-5",
    "excluded_predictors": ["SepsisLabel", "patient_id", "ICULOS"],
    "split_unit": "patient",
}


def make_pipeline(model_name: str, feature_names: list[str], random_state: int) -> Pipeline:
    """Create a model with all learned preprocessing inside its sklearn pipeline."""
    if model_name == "logistic_regression":
        numeric_pipeline = Pipeline([
            ("imputer", SimpleImputer(strategy="median", keep_empty_features=True)),
            ("scaler", StandardScaler()),
        ])
        estimator = LogisticRegression(
            class_weight="balanced", max_iter=2000, random_state=random_state,
        )
    elif model_name == "random_forest":
        numeric_pipeline = Pipeline([
            ("imputer", SimpleImputer(strategy="median", keep_empty_features=True)),
        ])
        estimator = RandomForestClassifier(
            n_estimators=300,
            class_weight="balanced_subsample",
            n_jobs=1,
            random_state=random_state,
        )
    elif model_name == "xgboost":
        try:
            from xgboost import XGBClassifier
        except ImportError as error:
            raise ImportError(
                "XGBoost dependency is unavailable; reinstall the project with `python -m pip install -e .`"
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


def run_experiment(
    features: pd.DataFrame,
    labels: pd.Series,
    groups: pd.Series,
    output_dir: str | Path,
    *,
    models: tuple[str, ...] = ("logistic_regression", "random_forest"),
    test_size: float = 0.2,
    random_state: int = 42,
    threshold: float = 0.5,
) -> dict[str, dict[str, Any]]:
    """Split patients, fit selected baselines, and write artifacts."""
    if not 0 < test_size < 1:
        raise ValueError("test_size must be between 0 and 1")
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

    splitter = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=random_state)
    train_positions, test_positions = next(splitter.split(features, labels, groups))
    train_groups = set(groups.iloc[train_positions].astype(str))
    test_groups = set(groups.iloc[test_positions].astype(str))
    if train_groups & test_groups:
        raise RuntimeError("Patient group overlap found between train and test")
    if labels.iloc[train_positions].nunique() != 2:
        raise ValueError("Training split contains only one outcome class; adjust test_size or seed")

    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    feature_names = list(features.columns)
    y_train = labels.iloc[train_positions].to_numpy(dtype=int)
    y_test = labels.iloc[test_positions].to_numpy(dtype=int)
    test_features = features.iloc[test_positions]
    metrics: dict[str, dict[str, Any]] = {}
    importance_rows: list[dict[str, Any]] = []
    predictions = pd.DataFrame({
        "patient_id": groups.iloc[test_positions].astype(str).to_numpy(),
        "y_true": y_test,
    })

    for model_name in models:
        model = make_pipeline(model_name, feature_names, random_state)
        model.fit(features.iloc[train_positions], y_train)
        probabilities = model.predict_proba(test_features)[:, 1]
        metrics[model_name] = calculate_metrics(y_test, probabilities, threshold)
        predictions[f"{model_name}_probability"] = probabilities
        predictions[f"{model_name}_prediction"] = (probabilities >= threshold).astype(int)
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
    pd.DataFrame(importance_rows).sort_values(
        ["model", "absolute_value"], ascending=[True, False], ignore_index=True,
    ).to_csv(output / "feature_importance.csv", index=False, float_format="%.12g")
    _write_json(output / "metrics.json", metrics)
    _write_json(output / "feature_names.json", feature_names)
    _write_json(output / "split_patients.json", {
        "train": groups.iloc[train_positions].astype(str).tolist(),
        "test": groups.iloc[test_positions].astype(str).tolist(),
    })
    _write_json(output / "run_config.json", {
        "design": DESIGN,
        "models": list(models),
        "random_state": random_state,
        "test_size": test_size,
        "threshold": threshold,
        "n_patients": int(len(features)),
        "n_train": int(len(train_positions)),
        "n_test": int(len(test_positions)),
    })
    return metrics
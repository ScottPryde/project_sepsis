import json

import numpy as np
import pandas as pd

from sepsis_prediction.modeling import run_experiment
from sepsis_prediction.cnn import run_cnn_experiment


def make_synthetic_cohort() -> tuple[pd.DataFrame, pd.Series, pd.Series]:
    patient_ids = [f"p{index:03d}" for index in range(60)]
    labels = pd.Series([index % 3 == 0 for index in range(60)], index=patient_ids, dtype="int64")
    signal = np.array([float(index % 3 == 0) + (index % 5) * 0.03 for index in range(60)])
    features = pd.DataFrame({
        "signal": signal,
        "some_missing": [np.nan if index % 4 == 0 else float(index) for index in range(60)],
        "all_nan": np.nan,
    }, index=patient_ids)
    groups = pd.Series(patient_ids, index=patient_ids, name="patient_id", dtype="string")
    return features, labels, groups


def test_group_split_train_only_preprocessing_and_reproducible_artifacts(tmp_path) -> None:
    features, labels, groups = make_synthetic_cohort()
    first = tmp_path / "first"
    second = tmp_path / "second"

    selected_models = ("logistic_regression", "random_forest", "xgboost")
    metrics_first = run_experiment(
        features, labels, groups, first, random_state=123, models=selected_models,
    )
    metrics_second = run_experiment(
        features, labels, groups, second, random_state=123, models=selected_models,
    )

    split = json.loads((first / "split_patients.json").read_text(encoding="utf-8"))
    assert set(split["train"]).isdisjoint(split["test"])
    assert set(split["train"]) | set(split["test"]) == set(groups)
    assert metrics_first == metrics_second
    assert (first / "metrics.json").read_bytes() == (second / "metrics.json").read_bytes()
    assert (first / "test_predictions.csv").read_bytes() == (second / "test_predictions.csv").read_bytes()
    importance = pd.read_csv(first / "feature_importance.csv")
    assert set(importance["importance_type"]) == {"coefficient", "feature_importance"}
    assert set(importance["model"]) == set(selected_models)

    fitted_model = __import__("joblib").load(first / "logistic_regression.joblib")
    fitted_imputer = (
        fitted_model.named_steps["preprocess"]
        .named_transformers_["numeric"]
        .named_steps["imputer"]
    )
    train_ids = split["train"]
    expected_median = features.loc[train_ids, "some_missing"].median()
    assert fitted_imputer.statistics_[1] == expected_median
    assert fitted_imputer.statistics_[2] == 0
    assert set(metrics_first) == set(selected_models)

    for artifact in (
        "logistic_regression.joblib",
        "random_forest.joblib",
        "metrics.json",
        "feature_names.json",
        "split_patients.json",
        "run_config.json",
        "test_predictions.csv",
        "feature_importance.csv",
        "xgboost.joblib",
    ):
        assert (first / artifact).is_file()


def test_cnn_trains_on_shared_patient_split_and_writes_checkpoint(tmp_path) -> None:
    features, labels, groups = make_synthetic_cohort()
    output = tmp_path / "cnn"
    run_experiment(
        features,
        labels,
        groups,
        output,
        models=("logistic_regression",),
        random_state=31,
    )
    rng = np.random.default_rng(31)
    sequences = rng.normal(size=(len(labels), 6, 4)).astype(np.float32)
    sequences[0, 0, 0] = np.nan

    cnn_metrics = run_cnn_experiment(
        sequences,
        labels,
        groups,
        output,
        random_state=31,
        epochs=1,
    )

    assert cnn_metrics["n"] > 0
    assert (output / "cnn_1d.pt").is_file()
    predictions = pd.read_csv(output / "test_predictions.csv")
    assert "cnn_1d_probability" in predictions
    assert "cnn_1d_prediction" in predictions
    updated_metrics = json.loads((output / "metrics.json").read_text(encoding="utf-8"))
    assert "cnn_1d" in updated_metrics
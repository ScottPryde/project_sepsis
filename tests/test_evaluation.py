import json

import numpy as np
import pandas as pd

from sepsis_prediction.evaluation import compare_models


def test_comparison_uses_saved_validation_thresholds_and_writes_outputs(tmp_path) -> None:
    rng = np.random.default_rng(11)
    labels = np.tile([0, 0, 0, 1], 100)
    probabilities = np.clip(0.1 + labels * 0.55 + rng.normal(0, 0.12, len(labels)), 0, 1)
    weaker = np.clip(0.2 + labels * 0.1 + rng.normal(0, 0.15, len(labels)), 0, 1)
    pd.DataFrame({
        "patient_id": [f"p{index}" for index in range(len(labels))],
        "y_true": labels,
        "logistic_regression_probability": probabilities,
        "random_forest_probability": weaker,
    }).to_csv(tmp_path / "test_predictions.csv", index=False)
    (tmp_path / "run_config.json").write_text(json.dumps({
        "random_state": 5,
        "target_sensitivity": 0.8,
        "model_thresholds": {"logistic_regression": 0.45, "random_forest": 0.3},
    }), encoding="utf-8")

    result = compare_models(tmp_path, n_bootstrap=100)

    assert result["models"][0]["model"] == "logistic_regression"
    assert result["models"][0]["operating_point"]["threshold"] == 0.45
    assert result["threshold_source"] == "validation patients"
    for artifact in (
        "model_evaluation.json", "model_ranking.csv", "model_forest.png",
        "calibration.png", "decision_curve.png",
    ):
        assert (tmp_path / artifact).is_file()
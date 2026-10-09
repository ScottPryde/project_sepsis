"""Optional 1D CNN for fixed-length, ordered patient observation windows."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from sepsis_prediction.data import KNOWN_CLINICAL_COLUMNS
from sepsis_prediction.modeling import _threshold_at_sensitivity, _write_json, calculate_metrics


def run_cnn_experiment(
    sequences: np.ndarray,
    labels: pd.Series,
    groups: pd.Series,
    output_dir: str | Path,
    *,
    random_state: int = 42,
    threshold: float | None = None,
    epochs: int = 20,
    target_sensitivity: float = 0.8,
) -> dict[str, Any]:
    """Fit and evaluate a CNN using the tabular run's patient partitions."""
    try:
        import torch
        from torch import nn
    except ImportError as error:
        raise ImportError(
            "PyTorch is optional; install it with `python -m pip install -e '.[cnn]'`"
        ) from error

    if sequences.ndim != 3 or sequences.shape[0] != len(labels):
        raise ValueError("sequences must have shape (patients, hours, variables)")
    if not (len(labels) == len(groups)) or not labels.index.equals(groups.index):
        raise ValueError("labels and groups must have matching indexes")
    if labels.nunique() != 2:
        raise ValueError("CNN cohort must contain both outcome classes")
    if epochs < 1:
        raise ValueError("epochs must be a positive integer")

    output = Path(output_dir)
    split = json.loads((output / "split_patients.json").read_text(encoding="utf-8"))
    position_by_patient = {str(patient_id): position for position, patient_id in enumerate(groups)}
    try:
        fit_positions = np.array([position_by_patient[patient_id] for patient_id in split["fit"]])
        validation_positions = np.array([
            position_by_patient[patient_id] for patient_id in split["validation"]
        ])
        test_positions = np.array([position_by_patient[patient_id] for patient_id in split["test"]])
    except KeyError as error:
        raise ValueError("Saved patient split does not match the CNN cohort") from error
    if len(fit_positions) + len(validation_positions) + len(test_positions) != len(groups):
        raise ValueError("Saved patient split does not partition the CNN cohort")
    if labels.iloc[fit_positions].nunique() != 2 or labels.iloc[validation_positions].nunique() != 2:
        raise ValueError("CNN fit and validation partitions must contain both outcome classes")

    train_values = sequences[fit_positions].astype(np.float32, copy=True)
    validation_values = sequences[validation_positions].astype(np.float32, copy=True)
    test_values = sequences[test_positions].astype(np.float32, copy=True)
    train_mask = np.isnan(train_values).astype(np.float32)
    validation_mask = np.isnan(validation_values).astype(np.float32)
    test_mask = np.isnan(test_values).astype(np.float32)
    with np.errstate(invalid="ignore"):
        medians = np.nanmedian(train_values, axis=(0, 1))
    medians = np.nan_to_num(medians, nan=0.0, posinf=0.0, neginf=0.0)
    train_values = np.where(np.isnan(train_values), medians[None, None, :], train_values)
    test_values = np.where(np.isnan(test_values), medians[None, None, :], test_values)
    means = train_values.mean(axis=(0, 1))
    scales = train_values.std(axis=(0, 1))
    scales[scales == 0] = 1.0
    train_values = (train_values - means[None, None, :]) / scales[None, None, :]
    validation_values = np.where(np.isnan(validation_values), medians[None, None, :], validation_values)
    validation_values = (validation_values - means[None, None, :]) / scales[None, None, :]
    test_values = (test_values - means[None, None, :]) / scales[None, None, :]
    train_values = np.concatenate([train_values, train_mask], axis=2)
    validation_values = np.concatenate([validation_values, validation_mask], axis=2)
    test_values = np.concatenate([test_values, test_mask], axis=2)

    torch.manual_seed(random_state)
    np.random.seed(random_state)
    torch.set_num_threads(1)
    model = nn.Sequential(
        nn.Conv1d(sequences.shape[2] * 2, 32, kernel_size=3, padding=1),
        nn.ReLU(),
        nn.Conv1d(32, 32, kernel_size=3, padding=1),
        nn.ReLU(),
        nn.AdaptiveAvgPool1d(1),
        nn.Flatten(),
        nn.Linear(32, 1),
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    train_labels = labels.iloc[fit_positions].to_numpy(dtype=np.float32)
    validation_labels = labels.iloc[validation_positions].to_numpy(dtype=np.float32)
    loss_function = nn.BCEWithLogitsLoss()
    x_train = torch.from_numpy(train_values.transpose(0, 2, 1))
    y_train = torch.from_numpy(train_labels[:, None])
    x_validation = torch.from_numpy(validation_values.transpose(0, 2, 1))
    y_validation = torch.from_numpy(validation_labels[:, None])
    generator = torch.Generator().manual_seed(random_state)
    dataset = torch.utils.data.TensorDataset(x_train, y_train)
    loader = torch.utils.data.DataLoader(
        dataset, batch_size=min(32, len(dataset)), shuffle=True, generator=generator,
    )
    best_state: dict[str, Any] | None = None
    best_validation_loss = float("inf")
    best_epoch = 0
    epochs_without_improvement = 0
    patience = min(5, max(1, epochs // 4))
    for epoch in range(epochs):
        model.train()
        for batch_features, batch_labels in loader:
            optimizer.zero_grad(set_to_none=True)
            loss = loss_function(model(batch_features), batch_labels)
            loss.backward()
            optimizer.step()
        model.eval()
        with torch.no_grad():
            validation_loss = float(loss_function(model(x_validation), y_validation).item())
        if validation_loss < best_validation_loss - 1e-5:
            best_validation_loss = validation_loss
            best_epoch = epoch + 1
            best_state = {name: value.detach().clone() for name, value in model.state_dict().items()}
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= patience:
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        validation_tensor = torch.from_numpy(validation_values.transpose(0, 2, 1))
        validation_probabilities = torch.sigmoid(model(validation_tensor)).squeeze(1).numpy()
        test_tensor = torch.from_numpy(test_values.transpose(0, 2, 1))
        probabilities = torch.sigmoid(model(test_tensor)).squeeze(1).numpy()
    selected_threshold = threshold if threshold is not None else _threshold_at_sensitivity(
        validation_labels.astype(int), validation_probabilities, target_sensitivity,
    )
    y_test = labels.iloc[test_positions].to_numpy(dtype=int)
    metrics = calculate_metrics(y_test, probabilities, selected_threshold)

    output.mkdir(parents=True, exist_ok=True)
    model_path = output / "cnn_1d.pt"
    torch.save({
        "state_dict": model.state_dict(),
        "input_variables": list(KNOWN_CLINICAL_COLUMNS),
        "input_channels": "standardized values followed by missingness masks",
        "lookback_rows": int(sequences.shape[1]),
        "medians": medians,
        "means": means,
        "scales": scales,
        "random_state": random_state,
        "epochs": best_epoch,
        "best_validation_loss": best_validation_loss,
    }, model_path)

    prediction_path = output / "test_predictions.csv"
    predictions = pd.read_csv(prediction_path)
    test_patient_ids = groups.iloc[test_positions].astype(str).tolist()
    if predictions["patient_id"].astype(str).tolist() != test_patient_ids:
        raise RuntimeError("CNN and tabular test patient splits do not match")
    predictions["cnn_1d_probability"] = probabilities
    predictions["cnn_1d_prediction"] = (probabilities >= selected_threshold).astype(int)
    predictions.to_csv(prediction_path, index=False, float_format="%.12g")

    validation_path = output / "validation_predictions.csv"
    validation_predictions = pd.read_csv(validation_path)
    validation_patient_ids = groups.iloc[validation_positions].astype(str).tolist()
    if validation_predictions["patient_id"].astype(str).tolist() != validation_patient_ids:
        raise RuntimeError("CNN and tabular validation patient splits do not match")
    validation_predictions["cnn_1d_probability"] = validation_probabilities
    validation_predictions["cnn_1d_prediction"] = (
        validation_probabilities >= selected_threshold
    ).astype(int)
    validation_predictions.to_csv(validation_path, index=False, float_format="%.12g")

    metrics_path = output / "metrics.json"
    all_metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    all_metrics["cnn_1d"] = metrics
    _write_json(metrics_path, all_metrics)
    config_path = output / "run_config.json"
    config = json.loads(config_path.read_text(encoding="utf-8"))
    config["models"].append("cnn_1d")
    config["cnn_epochs"] = best_epoch
    config.setdefault("model_specs", {})["cnn_1d"] = {
        "input": "ordered six-hour values and missingness masks",
        "preprocessing": ["fit-patient median imputation", "fit-patient z-score"],
        "estimator": f"two Conv1d(32,k=3) blocks, global average pool, Adam; early stopping at epoch {best_epoch}",
        "class_weighting": "none",
    }
    config.setdefault("model_thresholds", {})["cnn_1d"] = float(selected_threshold)
    _write_json(config_path, config)
    return metrics
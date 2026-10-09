"""Optional 1D CNN for fixed-length, ordered patient observation windows."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit

from sepsis_prediction.data import KNOWN_CLINICAL_COLUMNS
from sepsis_prediction.modeling import _write_json, calculate_metrics


def run_cnn_experiment(
    sequences: np.ndarray,
    labels: pd.Series,
    groups: pd.Series,
    output_dir: str | Path,
    *,
    test_size: float = 0.2,
    random_state: int = 42,
    threshold: float = 0.5,
    epochs: int = 20,
) -> dict[str, Any]:
    """Fit and evaluate a small CNN; learned sequence transforms use train only."""
    try:
        import torch
        from torch import nn
    except ImportError as error:
        raise ImportError(
            "PyTorch dependency is unavailable; reinstall the project with `python -m pip install -e .`"
        ) from error

    if sequences.ndim != 3 or sequences.shape[0] != len(labels):
        raise ValueError("sequences must have shape (patients, hours, variables)")
    if not (len(labels) == len(groups)) or not labels.index.equals(groups.index):
        raise ValueError("labels and groups must have matching indexes")
    if labels.nunique() != 2:
        raise ValueError("CNN cohort must contain both outcome classes")
    if epochs < 1:
        raise ValueError("epochs must be a positive integer")

    splitter = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=random_state)
    train_positions, test_positions = next(splitter.split(sequences, labels, groups))
    train_groups = set(groups.iloc[train_positions].astype(str))
    test_groups = set(groups.iloc[test_positions].astype(str))
    if train_groups & test_groups:
        raise RuntimeError("Patient group overlap found between train and test")
    if labels.iloc[train_positions].nunique() != 2:
        raise ValueError("CNN training split contains only one outcome class; adjust seed or test size")

    train_values = sequences[train_positions].astype(np.float32, copy=True)
    test_values = sequences[test_positions].astype(np.float32, copy=True)
    with np.errstate(invalid="ignore"):
        medians = np.nanmedian(train_values, axis=(0, 1))
    medians = np.nan_to_num(medians, nan=0.0, posinf=0.0, neginf=0.0)
    train_values = np.where(np.isnan(train_values), medians[None, None, :], train_values)
    test_values = np.where(np.isnan(test_values), medians[None, None, :], test_values)
    means = train_values.mean(axis=(0, 1))
    scales = train_values.std(axis=(0, 1))
    scales[scales == 0] = 1.0
    train_values = (train_values - means[None, None, :]) / scales[None, None, :]
    test_values = (test_values - means[None, None, :]) / scales[None, None, :]

    torch.manual_seed(random_state)
    np.random.seed(random_state)
    torch.set_num_threads(1)
    model = nn.Sequential(
        nn.Conv1d(sequences.shape[2], 32, kernel_size=3, padding=1),
        nn.ReLU(),
        nn.Conv1d(32, 32, kernel_size=3, padding=1),
        nn.ReLU(),
        nn.AdaptiveAvgPool1d(1),
        nn.Flatten(),
        nn.Linear(32, 1),
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    train_labels = labels.iloc[train_positions].to_numpy(dtype=np.float32)
    positive_weight = float((train_labels == 0).sum() / max((train_labels == 1).sum(), 1))
    loss_function = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([positive_weight]))
    x_train = torch.from_numpy(train_values.transpose(0, 2, 1))
    y_train = torch.from_numpy(train_labels[:, None])
    generator = torch.Generator().manual_seed(random_state)
    dataset = torch.utils.data.TensorDataset(x_train, y_train)
    loader = torch.utils.data.DataLoader(
        dataset, batch_size=min(32, len(dataset)), shuffle=True, generator=generator,
    )
    model.train()
    for _ in range(epochs):
        for batch_features, batch_labels in loader:
            optimizer.zero_grad(set_to_none=True)
            loss = loss_function(model(batch_features), batch_labels)
            loss.backward()
            optimizer.step()

    model.eval()
    with torch.no_grad():
        test_tensor = torch.from_numpy(test_values.transpose(0, 2, 1))
        probabilities = torch.sigmoid(model(test_tensor)).squeeze(1).numpy()
    y_test = labels.iloc[test_positions].to_numpy(dtype=int)
    metrics = calculate_metrics(y_test, probabilities, threshold)

    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    model_path = output / "cnn_1d.pt"
    torch.save({
        "state_dict": model.state_dict(),
        "input_variables": list(KNOWN_CLINICAL_COLUMNS),
        "lookback_rows": int(sequences.shape[1]),
        "medians": medians,
        "means": means,
        "scales": scales,
        "random_state": random_state,
        "epochs": epochs,
    }, model_path)

    prediction_path = output / "test_predictions.csv"
    predictions = pd.read_csv(prediction_path)
    cnn_predictions = pd.DataFrame({
        "patient_id": groups.iloc[test_positions].astype(str).to_numpy(),
        "cnn_1d_probability": probabilities,
        "cnn_1d_prediction": (probabilities >= threshold).astype(int),
    })
    if predictions["patient_id"].astype(str).tolist() != cnn_predictions["patient_id"].tolist():
        raise RuntimeError("CNN and tabular test patient splits do not match")
    predictions["cnn_1d_probability"] = cnn_predictions["cnn_1d_probability"]
    predictions["cnn_1d_prediction"] = cnn_predictions["cnn_1d_prediction"]
    predictions.to_csv(prediction_path, index=False, float_format="%.12g")

    metrics_path = output / "metrics.json"
    all_metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    all_metrics["cnn_1d"] = metrics
    _write_json(metrics_path, all_metrics)
    config_path = output / "run_config.json"
    config = json.loads(config_path.read_text(encoding="utf-8"))
    config["models"].append("cnn_1d")
    config["cnn_epochs"] = epochs
    _write_json(config_path, config)
    return metrics
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from sklearn.metrics import adjusted_rand_score

from sepsis_prediction.cli import main
from sepsis_prediction.evaluation import score_with_bootstrap
from sepsis_prediction.unsupervised import (
    LABEL_STATEMENT,
    describe_against_outcome,
    fit_clusters,
    fit_embedding,
    fit_preprocessor,
    run_exploration,
    select_features,
    wilson_interval,
)


def planted_cohort(patients: int = 300, seed: int = 3) -> tuple[pd.DataFrame, pd.Series, np.ndarray]:
    """Three well-separated groups in HR/MAP/Temp summaries plus noise and missingness columns."""
    rng = np.random.default_rng(seed)
    groups = np.arange(patients) % 3
    centres = np.array([[70, 70, 36.5], [100, 70, 38.0], [85, 96, 37.25]])
    base = centres[groups] + rng.normal(0, [3, 3, 0.15], size=(patients, 3))
    index = [f"p{number:04d}" for number in range(patients)]
    features = pd.DataFrame({
        "HR_mean": base[:, 0], "MAP_mean": base[:, 1], "Temp_mean": base[:, 2],
        "HR_latest": base[:, 0] + rng.normal(0, 1, patients),
        "MAP_latest": base[:, 1] + rng.normal(0, 1, patients),
        "Temp_latest": base[:, 2] + rng.normal(0, 0.05, patients),
        "Age_latest": rng.normal(60, 1, patients),
        "Lactate_mean": np.where(rng.random(patients) < 0.2, np.nan, rng.normal(2, 0.2, patients)),
        "Age_missing_fraction": 0.0,
        "HR_mean_copy": base[:, 0],
    }, index=index)
    labels = pd.Series((groups == 1) & (rng.random(patients) < 0.6), index=index, dtype="int64")
    return features, labels, groups


def split_ids(index: pd.Index, seed: int = 0) -> dict[str, list[str]]:
    order = np.random.default_rng(seed).permutation(len(index))
    ids = index[order].tolist()
    cut_fit, cut_validation = int(0.6 * len(ids)), int(0.8 * len(ids))
    return {"fit": ids[:cut_fit], "validation": ids[cut_fit:cut_validation], "test": ids[cut_validation:]}


def test_planted_clusters_are_recovered() -> None:
    features, _, groups = planted_cohort()
    preprocessor = fit_preprocessor(features)
    matrix = preprocessor.transform(features)
    scores = fit_embedding(matrix, random_state=1).transform(matrix)

    selection, chosen = fit_clusters(scores, ("kmeans", "gmm"), (2, 6), random_state=1)

    assert chosen["kmeans"][0] == 3
    assert adjusted_rand_score(groups, chosen["kmeans"][1].predict(scores)) >= 0.8
    assert adjusted_rand_score(groups, chosen["gmm"][1].predict(scores)) >= 0.8
    assert selection["k"].tolist() == [2, 3, 4, 5, 6]


def test_preprocessor_learns_from_fit_rows_and_drops_constant_and_duplicate_columns() -> None:
    features, _, _ = planted_cohort()
    split = split_ids(features.index)
    fit = features.loc[split["fit"]]

    preprocessor = fit_preprocessor(fit)

    medians = dict(zip(preprocessor.input_columns, preprocessor.imputer.statistics_))
    assert medians["Lactate_mean"] == fit["Lactate_mean"].median()
    assert "Age_missing_fraction" in preprocessor.dropped["constant"]
    assert "HR_mean_copy" in preprocessor.dropped["duplicate"]
    assert "Age_missing_fraction" not in preprocessor.kept_columns
    assert select_features(features, "measured").columns.str.endswith("_missing_fraction").sum() == 0


def test_test_patients_never_influence_fitted_steps(tmp_path: Path) -> None:
    features, labels, _ = planted_cohort()
    split = split_ids(features.index)
    altered = features.copy()
    altered.loc[split["test"], ["HR_mean", "MAP_mean"]] *= 50

    run_exploration(features, labels, split, tmp_path / "original", stability_resamples=3, random_state=5)
    run_exploration(altered, labels, split, tmp_path / "altered", stability_resamples=3, random_state=5)

    for name in ("pca_explained_variance.csv", "k_selection.csv", "pca_top_loadings.csv"):
        original = (tmp_path / "original" / "explore" / "all" / name).read_bytes()
        assert original == (tmp_path / "altered" / "explore" / "all" / name).read_bytes(), name
    original_summary = json.loads((tmp_path / "original" / "explore" / "explore_summary.json").read_text())
    altered_summary = json.loads((tmp_path / "altered" / "explore" / "explore_summary.json").read_text())
    for method in ("kmeans", "gmm"):
        assert (original_summary["feature_sets"]["all"]["clusters"][method]["stability"]
                == altered_summary["feature_sets"]["all"]["clusters"][method]["stability"])


def test_shuffled_labels_leave_fitted_outputs_identical(tmp_path: Path) -> None:
    features, labels, _ = planted_cohort()
    split = split_ids(features.index)
    shuffled = pd.Series(np.random.default_rng(9).permutation(labels.to_numpy()), index=labels.index)

    run_exploration(features, labels, split, tmp_path / "true", stability_resamples=3, random_state=5)
    run_exploration(features, shuffled, split, tmp_path / "shuffled", stability_resamples=3, random_state=5)

    for feature_set in ("all", "measured"):
        true_dir = tmp_path / "true" / "explore" / feature_set
        shuffled_dir = tmp_path / "shuffled" / "explore" / feature_set
        true_embedding = pd.read_csv(true_dir / "embedding_test.csv").drop(columns="outcome")
        shuffled_embedding = pd.read_csv(shuffled_dir / "embedding_test.csv").drop(columns="outcome")
        pd.testing.assert_frame_equal(true_embedding, shuffled_embedding)
        true_scores = pd.read_csv(true_dir / "anomaly_scores.csv")["anomaly_score"]
        assert true_scores.equals(pd.read_csv(shuffled_dir / "anomaly_scores.csv")["anomaly_score"])
        assert (true_dir / "k_selection.csv").read_bytes() == (shuffled_dir / "k_selection.csv").read_bytes()


def test_same_seed_reproduces_every_output(tmp_path: Path) -> None:
    features, labels, _ = planted_cohort()
    split = split_ids(features.index)

    run_exploration(features, labels, split, tmp_path / "first", stability_resamples=3, random_state=11)
    run_exploration(features, labels, split, tmp_path / "second", stability_resamples=3, random_state=11)

    first_files = sorted(path.relative_to(tmp_path / "first") for path in (tmp_path / "first").rglob("*.*"))
    for relative in first_files:
        if relative.suffix == ".png":
            continue
        assert (tmp_path / "first" / relative).read_bytes() == (tmp_path / "second" / relative).read_bytes(), relative


def test_small_and_one_class_test_sets_do_not_crash() -> None:
    features, _, _ = planted_cohort(patients=12)
    labels = pd.Series(0, index=features.index, dtype="int64")

    table, test = describe_against_outcome(np.arange(12) % 2, labels, features)

    assert test["p_value"] is None and "skipped" in test
    assert table["sepsis_rate"].eq(0).all()
    assert wilson_interval(0, 0) == [None, None]
    lone = score_with_bootstrap(labels.to_numpy(), np.arange(12.0), n_bootstrap=50)
    assert lone["auroc"] is None and lone["auroc_ci"] == [None, None]
    sparse_labels = labels.copy()
    sparse_labels.iloc[0] = 1
    _, sparse_test = describe_against_outcome(np.arange(12) % 2, sparse_labels, features)
    assert "below" in sparse_test["skipped"]


def test_umap_is_optional(tmp_path: Path) -> None:
    features, labels, _ = planted_cohort()
    result = run_exploration(
        features, labels, split_ids(features.index), tmp_path, methods=("pca", "umap", "kmeans"),
        feature_sets=("all",), stability_resamples=2,
    )

    config = result["config"]
    assert config["umap_requested"] is True
    if importlib.util.find_spec("umap") is None:
        assert config["umap_available"] is False
        assert "not installed" in config["umap_note"]
        assert "UMAP1" not in pd.read_csv(tmp_path / "explore" / "all" / "embedding_test.csv")


def write_cohort(directory: Path, patients: int = 90) -> None:
    rng = np.random.default_rng(4)
    for number in range(patients):
        group = number % 3
        positive = group == 1 and number % 2 == 1
        labels = [0] * 9
        if positive:
            labels[7] = 1
        hr = [70 + 30 * (group == 1) + rng.normal(0, 2) for _ in range(9)]
        map_values = [85 - 15 * (group == 1) + 15 * (group == 2) + rng.normal(0, 2) for _ in range(9)]
        source = directory / ("training_setA" if number % 2 == 0 else "training_setB")
        source.mkdir(parents=True, exist_ok=True)
        pd.DataFrame({"HR": hr, "MAP": map_values, "SepsisLabel": labels}).to_csv(
            source / f"p{number % 2}{number:05d}.psv", sep="|", index=False,
        )


def test_cli_explore_reuses_run_split_writes_artifacts_and_report(tmp_path: Path, monkeypatch) -> None:
    write_cohort(tmp_path / "raw")
    folders = [str(tmp_path / "raw" / "training_setA"), str(tmp_path / "raw" / "training_setB")]
    output_dir = tmp_path / "run"
    monkeypatch.setattr(sys, "argv", [
        "sepsis-pipeline", "run", "--data-dir", *folders, "--output-dir", str(output_dir),
        "--models", "logistic_regression", "--horizon-hours", "3",
    ])
    main()
    supervised_report = (output_dir / "report.html").read_text(encoding="utf-8")
    assert "Unsupervised exploration" not in supervised_report

    monkeypatch.setattr(sys, "argv", [
        "sepsis-pipeline", "explore", "--data-dir", *folders, "--output-dir", str(output_dir),
        "--horizon-hours", "3", "--stability-resamples", "3", "--k-range", "2", "4",
    ])
    main()

    explore_dir = output_dir / "explore"
    config = json.loads((explore_dir / "explore_config.json").read_text(encoding="utf-8"))
    run_split = json.loads((output_dir / "split_patients.json").read_text(encoding="utf-8"))
    assert config["split_origin"].startswith("reused")
    assert config["n_test"] == len(run_split["test"])
    assert config["label_statement"] == LABEL_STATEMENT
    for feature_set in ("all", "measured"):
        set_dir = explore_dir / feature_set
        for name in (
            "embedding_test.csv", "embedding.png", "pca_explained_variance.csv", "pca_top_loadings.csv",
            "k_selection.csv", "k_selection.png", "cluster_profiles_kmeans.csv", "cluster_profiles_gmm.csv",
            "anomaly_scores.csv",
        ):
            assert (set_dir / name).is_file(), f"{feature_set}/{name}"
        embedding = pd.read_csv(set_dir / "embedding_test.csv")
        assert set(embedding["patient_id"]) == set(run_split["test"])
        summary = json.loads((explore_dir / "explore_summary.json").read_text(encoding="utf-8"))
        assert isinstance(summary["feature_sets"][feature_set]["clusters"]["kmeans"]["k_at_range_limit"], bool)
        profile = pd.read_csv(set_dir / "cluster_profiles_kmeans.csv")
        assert {"share_training_setA", "share_training_setB", "median_record_rows"} <= set(profile.columns)

    report = (output_dir / "report.html").read_text(encoding="utf-8")
    assert "Unsupervised exploration" in report
    assert LABEL_STATEMENT in report
    assert "Isolation Forest anomaly score" in report
    assert "labels for description only" in report
    assert "https://" not in report


def test_cli_explore_without_run_computes_split_and_rejects_horizon_mismatch(tmp_path: Path, monkeypatch) -> None:
    write_cohort(tmp_path / "raw")
    folders = [str(tmp_path / "raw" / "training_setA"), str(tmp_path / "raw" / "training_setB")]
    output_dir = tmp_path / "fresh"
    monkeypatch.setattr(sys, "argv", [
        "sepsis-pipeline", "explore", "--data-dir", *folders, "--output-dir", str(output_dir),
        "--horizon-hours", "3", "--stability-resamples", "2", "--methods", "pca", "kmeans",
        "--feature-sets", "measured",
    ])
    main()
    config = json.loads((output_dir / "explore" / "explore_config.json").read_text(encoding="utf-8"))
    assert config["split_origin"].startswith("computed")
    assert (output_dir / "explore" / "split_patients.json").is_file()
    assert not (output_dir / "explore" / "measured" / "anomaly_scores.csv").exists()

    (output_dir / "split_patients.json").write_text(
        (output_dir / "explore" / "split_patients.json").read_text(encoding="utf-8"), encoding="utf-8",
    )
    (output_dir / "run_config.json").write_text(json.dumps({"design": {"horizon_hours": 24}}), encoding="utf-8")
    monkeypatch.setattr(sys, "argv", [
        "sepsis-pipeline", "explore", "--data-dir", *folders, "--output-dir", str(output_dir), "--horizon-hours", "3",
    ])
    with pytest.raises(SystemExit, match="horizon-hours 24"):
        main()

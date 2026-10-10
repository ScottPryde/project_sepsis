"""Label-blind structure in the six-hour look-back, described against the outcome afterwards.

Every learned step (imputation, scaling, PCA, UMAP, clustering, Isolation
Forest) is fitted on fit-split patients using features only. The outcome is
joined at the end, on test patients, to describe what was found. Cluster
stability is measured on validation patients.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import chi2_contingency
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.ensemble import IsolationForest
from sklearn.impute import SimpleImputer
from sklearn.metrics import adjusted_rand_score, silhouette_score
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler

from sepsis_prediction.evaluation import score_with_bootstrap


EXPLORE_METHODS = ("pca", "umap", "kmeans", "gmm", "isolation_forest")
DEFAULT_METHODS = ("pca", "kmeans", "gmm", "isolation_forest")
FEATURE_SETS = ("all", "measured")
MISSINGNESS_SUFFIXES = ("_missing_fraction", "_any_missing")
KEY_VARIABLES = {
    "HR": "HR_mean", "MAP": "MAP_mean", "Temp": "Temp_mean", "Resp": "Resp_mean",
    "O2Sat": "O2Sat_mean", "Lactate": "Lactate_mean", "WBC": "WBC_mean", "Age": "Age_latest",
}
PCA_VARIANCE = 0.9
PCA_MAX_COMPONENTS = 20
SILHOUETTE_SAMPLE = 5000
UNSTABLE_ARI = 0.6
GMM_REG_COVAR = 0.05
SCALED_CLIP = 10.0
MIN_EXPECTED_COUNT = 5
LABEL_STATEMENT = "Labels were used only to describe results, never to fit these methods."
SURFACE = "#fcfcfb"
INK_MUTED = "#52605f"
NEGATIVE_GREY = "#a9b3b0"
POSITIVE_CORAL = "#c45042"
TEAL = "#167b73"
GOLD = "#d89a38"


def select_features(features: pd.DataFrame, feature_set: str) -> pd.DataFrame:
    """``all`` keeps every engineered feature; ``measured`` drops missingness indicators."""
    if feature_set not in FEATURE_SETS:
        raise ValueError(f"feature_set must be one of {FEATURE_SETS}")
    if feature_set == "all":
        return features
    return features.loc[:, [column for column in features.columns if not column.endswith(MISSINGNESS_SUFFIXES)]]


@dataclass
class Preprocessor:
    """Fit-patient median imputation, removal of constant and duplicate columns, then scaling."""

    input_columns: list[str]
    kept_columns: list[str]
    imputer: SimpleImputer
    scaler: StandardScaler
    dropped: dict[str, list[str]] = field(default_factory=dict)

    def transform(self, frame: pd.DataFrame) -> np.ndarray:
        """Impute and scale, then cap at +/-``SCALED_CLIP`` standard deviations.

        No physiologic range checks run upstream, so a single implausible entry
        (e.g. FiO2 recorded as 4000) would otherwise dominate PCA and distances.
        """
        imputed = pd.DataFrame(
            self.imputer.transform(frame[self.input_columns]), columns=self.input_columns, index=frame.index,
        )
        return np.clip(self.scaler.transform(imputed[self.kept_columns]), -SCALED_CLIP, SCALED_CLIP)

    def clipped_share(self, frame: pd.DataFrame) -> dict[str, float]:
        imputed = pd.DataFrame(
            self.imputer.transform(frame[self.input_columns]), columns=self.input_columns, index=frame.index,
        )
        beyond = np.abs(self.scaler.transform(imputed[self.kept_columns])) > SCALED_CLIP
        return {"cells": float(beyond.mean()), "patients": float(beyond.any(axis=1).mean())}


def fit_preprocessor(fit_features: pd.DataFrame) -> Preprocessor:
    """Learn preprocessing from fit patients only."""
    columns = list(fit_features.columns)
    imputer = SimpleImputer(strategy="median", keep_empty_features=True).fit(fit_features)
    imputed = pd.DataFrame(imputer.transform(fit_features), columns=columns)
    constant = [column for column in columns if imputed[column].std(ddof=0) == 0]
    remaining = imputed.drop(columns=constant)
    duplicate = remaining.columns[remaining.T.duplicated()].tolist()
    kept = [column for column in remaining.columns if column not in set(duplicate)]
    if not kept:
        raise ValueError("No informative features remain after removing constant and duplicate columns")
    scaler = StandardScaler().fit(remaining[kept])
    return Preprocessor(columns, kept, imputer, scaler, {"constant": constant, "duplicate": duplicate})


def fit_embedding(fit_matrix: np.ndarray, random_state: int) -> PCA:
    """PCA keeping components up to 90% explained variance, capped at 20."""
    limit = min(fit_matrix.shape)
    full = PCA(n_components=limit, random_state=random_state).fit(fit_matrix)
    cumulative = np.cumsum(full.explained_variance_ratio_)
    components = int(min(PCA_MAX_COMPONENTS, np.searchsorted(cumulative, PCA_VARIANCE) + 1, limit))
    return PCA(n_components=max(components, min(2, limit)), random_state=random_state).fit(fit_matrix)


def fit_umap(fit_scores: np.ndarray, random_state: int) -> Any | None:
    """Optional 2-D UMAP of the fit PCA scores; returns None if umap-learn is absent."""
    try:
        import umap
    except ImportError:
        return None
    return umap.UMAP(n_neighbors=30, min_dist=0.1, random_state=random_state).fit(fit_scores)


def _make_clusterer(method: str, k: int, random_state: int) -> KMeans | GaussianMixture:
    if method == "kmeans":
        return KMeans(n_clusters=k, n_init=10, random_state=random_state)
    if method == "gmm":
        # Median imputation of sparse labs piles many patients onto one value; a larger
        # covariance floor stops the mixture spending components on those point masses.
        return GaussianMixture(
            n_components=k, covariance_type="full", reg_covar=GMM_REG_COVAR, random_state=random_state,
        )
    raise ValueError(f"Unknown clustering method {method!r}")


def fit_clusters(
    fit_scores: np.ndarray,
    methods: tuple[str, ...],
    k_range: tuple[int, int],
    random_state: int,
) -> tuple[pd.DataFrame, dict[str, tuple[int, Any]]]:
    """Fit each method for every k; pick k by silhouette (k-means) and BIC (Gaussian mixture)."""
    low, high = k_range
    if not 2 <= low <= high:
        raise ValueError("k_range must satisfy 2 <= low <= high")
    high = min(high, len(fit_scores) - 1)
    sample = min(SILHOUETTE_SAMPLE, len(fit_scores))
    rows: list[dict[str, Any]] = []
    fitted: dict[str, dict[int, Any]] = {method: {} for method in methods}
    for k in range(low, high + 1):
        row: dict[str, Any] = {"k": k}
        for method in methods:
            model = _make_clusterer(method, k, random_state).fit(fit_scores)
            fitted[method][k] = model
            if method == "kmeans":
                labels = model.labels_
                row["kmeans_inertia"] = float(model.inertia_)
                row["kmeans_silhouette"] = (
                    float(silhouette_score(fit_scores, labels, sample_size=sample, random_state=random_state))
                    if np.unique(labels).size > 1 else None
                )
            else:
                row["gmm_bic"] = float(model.bic(fit_scores))
        rows.append(row)
    selection = pd.DataFrame(rows)
    chosen: dict[str, tuple[int, Any]] = {}
    if "kmeans" in methods:
        k = int(selection.loc[selection["kmeans_silhouette"].astype(float).idxmax(), "k"])
        chosen["kmeans"] = (k, fitted["kmeans"][k])
    if "gmm" in methods:
        k = int(selection.loc[selection["gmm_bic"].idxmin(), "k"])
        chosen["gmm"] = (k, fitted["gmm"][k])
    return selection, chosen


def cluster_stability(
    method: str,
    k: int,
    reference: Any,
    fit_scores: np.ndarray,
    holdout_scores: np.ndarray,
    resamples: int,
    random_state: int,
) -> dict[str, Any]:
    """Refit on bootstrap resamples of fit patients; compare assignments of held-out patients by ARI."""
    reference_labels = reference.predict(holdout_scores)
    rng = np.random.default_rng(random_state)
    scores = []
    for index in range(resamples):
        sample = rng.integers(0, len(fit_scores), len(fit_scores))
        refit = _make_clusterer(method, k, random_state + index + 1).fit(fit_scores[sample])
        scores.append(adjusted_rand_score(reference_labels, refit.predict(holdout_scores)))
    values = np.array(scores, dtype=float)
    mean = float(values.mean()) if values.size else None
    return {
        "resamples": resamples,
        "evaluated_on": "validation patients",
        "mean_ari": mean,
        "min_ari": float(values.min()) if values.size else None,
        "unstable": bool(mean is not None and mean < UNSTABLE_ARI),
        "threshold": UNSTABLE_ARI,
    }


def fit_anomaly(fit_matrix: np.ndarray, random_state: int) -> IsolationForest:
    return IsolationForest(n_estimators=300, contamination="auto", random_state=random_state).fit(fit_matrix)


def anomaly_scores(model: IsolationForest, matrix: np.ndarray) -> np.ndarray:
    """Higher means more unusual (``score_samples`` is higher for normal points, so it is negated)."""
    return -model.score_samples(matrix)


def wilson_interval(positives: int, total: int, z: float = 1.959964) -> list[float | None]:
    if total == 0:
        return [None, None]
    rate = positives / total
    denominator = 1 + z ** 2 / total
    centre = (rate + z ** 2 / (2 * total)) / denominator
    half = z * np.sqrt(rate * (1 - rate) / total + z ** 2 / (4 * total ** 2)) / denominator
    return [float(max(0.0, centre - half)), float(min(1.0, centre + half))]


def describe_against_outcome(
    assignments: np.ndarray,
    outcome: pd.Series,
    raw_features: pd.DataFrame,
    sources: pd.Series | None = None,
    record_lengths: pd.Series | None = None,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Per-cluster size, sepsis rate with Wilson interval, key medians, missingness, and source mix.

    Receives test patients only. Returns the table (sorted by sepsis rate) and
    a chi-square test of rate differences across clusters, skipped when any
    expected cell count is below five or only one outcome class is present.
    """
    frame = pd.DataFrame({"cluster": assignments, "outcome": outcome.to_numpy()}, index=outcome.index)
    missing_columns = [column for column in raw_features.columns if column.endswith("_missing_fraction")]
    source_names = sorted(sources.astype(str).unique()) if sources is not None else []
    rows: list[dict[str, Any]] = []
    for cluster, members in frame.groupby("cluster", sort=True):
        positives = int(members["outcome"].sum())
        interval = wilson_interval(positives, len(members))
        row: dict[str, Any] = {
            "cluster": int(cluster),
            "patients": int(len(members)),
            "share_of_test": len(members) / len(frame),
            "sepsis_outcomes": positives,
            "sepsis_rate": positives / len(members),
            "rate_ci_low": interval[0],
            "rate_ci_high": interval[1],
        }
        member_features = raw_features.loc[members.index]
        for name, column in KEY_VARIABLES.items():
            row[f"median_{name}"] = (
                float(member_features[column].median()) if column in member_features else None
            )
        row["mean_missing_fraction"] = (
            float(member_features[missing_columns].to_numpy().mean()) if missing_columns else None
        )
        if record_lengths is not None:
            row["median_record_rows"] = float(record_lengths.loc[members.index].median())
        for source in source_names:
            row[f"share_{source}"] = float((sources.loc[members.index].astype(str) == source).mean())
        rows.append(row)
    table = pd.DataFrame(rows).sort_values(["sepsis_rate", "cluster"], ascending=[False, True], ignore_index=True)

    test: dict[str, Any] = {"test": "chi-square, clusters x outcome", "statistic": None, "p_value": None}
    contingency = pd.crosstab(frame["cluster"], frame["outcome"])
    if contingency.shape[0] < 2 or contingency.shape[1] < 2:
        test["skipped"] = "fewer than two clusters or one outcome class in the test set"
    else:
        statistic, p_value, dof, expected = chi2_contingency(contingency.to_numpy())
        if expected.min() < MIN_EXPECTED_COUNT:
            test["skipped"] = f"an expected cell count is below {MIN_EXPECTED_COUNT}"
        else:
            test.update({"statistic": float(statistic), "p_value": float(p_value), "dof": int(dof)})
    return table, test


def _plot_embedding(coordinates: np.ndarray, outcome: np.ndarray, axis_labels: tuple[str, str], title: str, path: Path) -> None:
    figure, axis = plt.subplots(figsize=(7.5, 6))
    figure.patch.set_facecolor(SURFACE)
    axis.set_facecolor(SURFACE)
    negative = outcome == 0
    axis.scatter(
        coordinates[negative, 0], coordinates[negative, 1], s=6, c=NEGATIVE_GREY, alpha=0.35,
        linewidths=0, marker="o", label=f"No sepsis in horizon (n={int(negative.sum())})", rasterized=True,
    )
    axis.scatter(
        coordinates[~negative, 0], coordinates[~negative, 1], s=22, c=POSITIVE_CORAL, alpha=0.9,
        edgecolors=SURFACE, linewidths=0.6, marker="^", label=f"Sepsis in horizon (n={int((~negative).sum())})",
    )
    low = np.percentile(coordinates, 0.5, axis=0)
    high = np.percentile(coordinates, 99.5, axis=0)
    padding = (high - low) * 0.05
    low, high = low - padding, high + padding
    outside = int((~((coordinates >= low) & (coordinates <= high)).all(axis=1)).sum())
    axis.set_xlim(low[0], high[0])
    axis.set_ylim(low[1], high[1])
    axis.set(xlabel=axis_labels[0], ylabel=axis_labels[1], title=title)
    axis.title.set_color("#1c2727")
    axis.title.set_fontsize(11)
    if outside:
        axis.text(
            0.99, 0.01, f"{outside} extreme patients outside this view (central 99% shown)",
            transform=axis.transAxes, ha="right", va="bottom", fontsize=8, color=INK_MUTED,
        )
    for spine in ("top", "right"):
        axis.spines[spine].set_visible(False)
    for spine in ("left", "bottom"):
        axis.spines[spine].set_color("#c9d1ce")
    axis.tick_params(colors=INK_MUTED, labelsize=9)
    axis.xaxis.label.set_color(INK_MUTED)
    axis.yaxis.label.set_color(INK_MUTED)
    axis.legend(loc="best", frameon=False, fontsize=9, labelcolor="#1c2727")
    figure.tight_layout()
    figure.savefig(path, dpi=160)
    plt.close(figure)


def _plot_k_selection(selection: pd.DataFrame, chosen: dict[str, tuple[int, Any]], path: Path) -> None:
    panels = [
        (method, column, label, rule)
        for method, column, label, rule in (
            ("kmeans", "kmeans_silhouette", "k-means silhouette (higher is better)", "highest silhouette"),
            ("gmm", "gmm_bic", "Gaussian mixture BIC (lower is better)", "lowest BIC"),
        )
        if method in chosen
    ]
    figure, axes = plt.subplots(1, len(panels), figsize=(5.2 * len(panels), 3.8), squeeze=False)
    figure.patch.set_facecolor(SURFACE)
    for axis, (method, column, label, rule) in zip(axes[0], panels):
        axis.set_facecolor(SURFACE)
        values = selection[column].astype(float)
        axis.plot(selection["k"], values, color=TEAL, linewidth=2, marker="o", markersize=5)
        k = chosen[method][0]
        chosen_value = float(selection.loc[selection["k"] == k, column].iloc[0])
        axis.scatter([k], [chosen_value], s=140, facecolors="none", edgecolors=GOLD, linewidths=2.2, zorder=3)
        span = float(values.max() - values.min()) or 1.0
        upper = (chosen_value - float(values.min())) / span > 0.5
        rightmost = k == int(selection["k"].max())
        axis.annotate(
            f"chosen k={k}\n({rule})", (k, chosen_value), textcoords="offset points",
            xytext=(-14 if rightmost else 14, -30 if upper else 22),
            ha="right" if rightmost else "left", va="center", fontsize=9, color="#1c2727",
            bbox={"boxstyle": "round,pad=0.25", "facecolor": SURFACE, "edgecolor": "#d8dfdc"},
            zorder=4,
        )
        axis.set(xlabel="Number of clusters (k)", title=label)
        axis.title.set_fontsize(10)
        axis.set_xticks(selection["k"])
        axis.grid(axis="y", color="#e6eae7", linewidth=0.8)
        for spine in ("top", "right"):
            axis.spines[spine].set_visible(False)
        axis.tick_params(colors=INK_MUTED, labelsize=9)
    figure.tight_layout()
    figure.savefig(path, dpi=160)
    plt.close(figure)


def _write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")


def run_exploration(
    features: pd.DataFrame,
    labels: pd.Series,
    split: dict[str, list[str]],
    output_dir: str | Path,
    *,
    sources: pd.Series | None = None,
    record_lengths: pd.Series | None = None,
    methods: tuple[str, ...] = DEFAULT_METHODS,
    k_range: tuple[int, int] = (2, 8),
    feature_sets: tuple[str, ...] = FEATURE_SETS,
    stability_resamples: int = 50,
    random_state: int = 42,
    split_origin: str = "computed",
    run_settings: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Run every selected method for each feature set and write ``<output_dir>/explore``."""
    unknown = set(methods) - set(EXPLORE_METHODS)
    if unknown:
        raise ValueError(f"Unknown methods {sorted(unknown)}; choose from {EXPLORE_METHODS}")
    if not features.index.equals(labels.index):
        raise ValueError("features and labels must share one patient index")
    for name in ("fit", "validation", "test"):
        missing = set(split.get(name, [])) - set(features.index)
        if missing or not split.get(name):
            raise ValueError(f"Split {name!r} is empty or names patients outside the cohort")
    fit_ids, validation_ids, test_ids = split["fit"], split["validation"], split["test"]
    cluster_methods = tuple(method for method in ("kmeans", "gmm") if method in methods)
    explore_dir = Path(output_dir) / "explore"
    explore_dir.mkdir(parents=True, exist_ok=True)
    umap_requested = "umap" in methods
    umap_available = False
    summary: dict[str, Any] = {"feature_sets": {}}

    for feature_set in feature_sets:
        set_dir = explore_dir / feature_set
        set_dir.mkdir(exist_ok=True)
        selected = select_features(features, feature_set)
        preprocessor = fit_preprocessor(selected.loc[fit_ids])
        fit_matrix = preprocessor.transform(selected.loc[fit_ids])
        validation_matrix = preprocessor.transform(selected.loc[validation_ids])
        test_matrix = preprocessor.transform(selected.loc[test_ids])
        y_test = labels.loc[test_ids].astype(int)
        set_summary: dict[str, Any] = {
            "input_features": len(preprocessor.input_columns),
            "used_features": len(preprocessor.kept_columns),
            "dropped_constant": preprocessor.dropped["constant"],
            "dropped_duplicate": preprocessor.dropped["duplicate"],
            "clipped_fit": preprocessor.clipped_share(selected.loc[fit_ids]),
            "clipped_test": preprocessor.clipped_share(selected.loc[test_ids]),
        }

        pca = fit_embedding(fit_matrix, random_state)
        fit_scores = pca.transform(fit_matrix)
        validation_scores = pca.transform(validation_matrix)
        test_scores = pca.transform(test_matrix)
        variance = pd.DataFrame({
            "component": [f"PC{index + 1}" for index in range(pca.n_components_)],
            "explained_variance_ratio": pca.explained_variance_ratio_,
            "cumulative": np.cumsum(pca.explained_variance_ratio_),
        })
        variance.to_csv(set_dir / "pca_explained_variance.csv", index=False, float_format="%.6g")
        loading_rows = []
        for component in range(min(3, pca.n_components_)):
            weights = pd.Series(pca.components_[component], index=preprocessor.kept_columns)
            for feature, weight in weights.reindex(weights.abs().sort_values(ascending=False).index)[:10].items():
                loading_rows.append({"component": f"PC{component + 1}", "feature": feature, "loading": float(weight)})
        pd.DataFrame(loading_rows).to_csv(set_dir / "pca_top_loadings.csv", index=False, float_format="%.6g")
        pc1 = score_with_bootstrap(y_test.to_numpy(), test_scores[:, 0], seed=random_state)
        set_summary["pca"] = {
            "components": int(pca.n_components_),
            "explained_variance": float(variance["cumulative"].iloc[-1]),
            "pc1_auroc": pc1["auroc"],
            "pc1_auroc_ci": pc1["auroc_ci"],
            "pc1_auroc_direction_free": None if pc1["auroc"] is None else max(pc1["auroc"], 1 - pc1["auroc"]),
        }

        embedding = pd.DataFrame(
            test_scores[:, :2], index=pd.Index(test_ids, name="patient_id"), columns=["PC1", "PC2"],
        )
        view = ("PC1", "PC2")
        if umap_requested:
            reducer = fit_umap(fit_scores, random_state)
            if reducer is not None:
                umap_available = True
                coordinates = reducer.transform(test_scores)
                embedding["UMAP1"], embedding["UMAP2"] = coordinates[:, 0], coordinates[:, 1]
                view = ("UMAP1", "UMAP2")
        embedding["outcome"] = y_test.to_numpy()
        if sources is not None:
            embedding["source"] = sources.loc[test_ids].to_numpy()

        _plot_embedding(
            embedding[list(view)].to_numpy(), y_test.to_numpy(), view,
            f"Test patients, {'UMAP' if view[0] == 'UMAP1' else 'PCA'} of {feature_set} features (fit without labels)",
            set_dir / "embedding.png",
        )

        if cluster_methods:
            selection, chosen = fit_clusters(fit_scores, cluster_methods, k_range, random_state)
            selection.to_csv(set_dir / "k_selection.csv", index=False, float_format="%.6g")
            _plot_k_selection(selection, chosen, set_dir / "k_selection.png")
            set_summary["clusters"] = {}
            for method, (k, model) in chosen.items():
                assignments = model.predict(test_scores)
                embedding[f"{method}_cluster"] = assignments
                profile, test = describe_against_outcome(
                    assignments, y_test, features.loc[test_ids],
                    None if sources is None else sources.loc[test_ids],
                    None if record_lengths is None else record_lengths.loc[test_ids],
                )
                profile.to_csv(set_dir / f"cluster_profiles_{method}.csv", index=False, float_format="%.6g")
                set_summary["clusters"][method] = {
                    "k": k,
                    "selection_rule": "highest silhouette" if method == "kmeans" else "lowest BIC",
                    "k_at_range_limit": bool(k in (int(selection["k"].min()), int(selection["k"].max()))),
                    "stability": cluster_stability(
                        method, k, model, fit_scores, validation_scores, stability_resamples, random_state,
                    ),
                    "outcome_rate_test": test,
                }

        if "isolation_forest" in methods:
            forest = fit_anomaly(fit_matrix, random_state)
            scores = anomaly_scores(forest, test_matrix)
            pd.DataFrame({
                "patient_id": test_ids, "anomaly_score": scores, "outcome": y_test.to_numpy(),
            }).to_csv(set_dir / "anomaly_scores.csv", index=False, float_format="%.8g")
            set_summary["isolation_forest"] = score_with_bootstrap(y_test.to_numpy(), scores, seed=random_state)

        embedding.reset_index().to_csv(set_dir / "embedding_test.csv", index=False, float_format="%.6g")
        summary["feature_sets"][feature_set] = set_summary

    config = {
        "methods": list(methods),
        "feature_sets": list(feature_sets),
        "k_range": list(k_range),
        "stability_resamples": stability_resamples,
        "random_state": random_state,
        "split_origin": split_origin,
        "n_fit": len(fit_ids),
        "n_validation": len(validation_ids),
        "n_test": len(test_ids),
        "umap_requested": umap_requested,
        "umap_available": umap_available,
        "umap_note": (
            None if not umap_requested or umap_available
            else "umap-learn is not installed; PCA views only. Install with the explore extra."
        ),
        "pca": {"variance_target": PCA_VARIANCE, "max_components": PCA_MAX_COMPONENTS},
        "silhouette_sample": SILHOUETTE_SAMPLE,
        "scaled_value_cap_sd": SCALED_CLIP,
        "gmm_reg_covar": GMM_REG_COVAR,
        "isolation_forest": {"n_estimators": 300, "contamination": "auto"},
        "fitted_on": "fit patients, features only",
        "described_on": "test patients",
        "label_statement": LABEL_STATEMENT,
        **(run_settings or {}),
    }
    _write_json(explore_dir / "explore_config.json", config)
    _write_json(explore_dir / "explore_summary.json", summary)
    return {"config": config, "summary": summary}

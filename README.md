# Early Sepsis Prediction Study

A reproducible workflow for dataset inspection, early-window cohort construction, tabular and optional sequence models, and held-out patient-level evaluation using the PhysioNet/Computing in Cardiology Challenge 2019 PSV files. Clinical data is external and is never committed here.

## Prediction Design

Each PSV file represents one patient, with hourly observations in file order. The project uses rows at hours 0-5 inclusive as a fixed six-hour look-back and predicts whether any `SepsisLabel=1` occurs in the next 24 hours by default (rows 6-29). In PhysioNet 2019, `SepsisLabel` switches on six hours before Sepsis-3 onset, so the outcome is the dataset's early label, not a direct onset-time estimate. Patients labeled positive in any of rows 0-5 are excluded. Positive patients observed within the horizon are retained even if the record ends early; negative patients must have complete horizon follow-up. Set another horizon with `--horizon-hours`.

Features use only those six look-back rows. Dynamic inputs receive latest value, mean, minimum, maximum, population standard deviation, first-to-last observed delta, missing fraction, and an any-missing indicator. Static inputs (`Age`, `Gender`, `Unit1`, `Unit2`, `HospAdmTime`) receive only latest value and missingness indicators. Missing schema columns and all-NaN windows retain a stable feature schema and are handled by the fit-patient imputer. `SepsisLabel`, the filename-derived patient identifier, and `ICULOS` are never predictors. `ICULOS`, when present, is checked for one-hour increments but excluded as a predictor.

Patients are stratified into fit, validation, and test partitions before model fitting. Every learned preprocessing step is fitted on fit patients only. Thresholds are selected on validation patients and frozen before test evaluation; the target sensitivity defaults to 80%. Logistic Regression, Random Forest, and XGBoost use the same feature rows and split. CNN training is opt-in, uses the same patient partitions and missingness-mask channels, and early-stops using validation loss.

The tabular workflow remains the default. Install the optional XGBoost or PyTorch extra only when selecting those models.

## Install

Python 3.10 or newer is required.

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -e ".[dev]"
```

Install the optional model libraries only when needed: `python -m pip install -e ".[xgboost]"` for XGBoost and `python -m pip install -e ".[cnn]"` for the CNN.

To install into the project virtual environment explicitly, run `.\.venv\Scripts\python.exe -m pip install -e ".[dev,cnn,xgboost]"`. In VS Code, select `.venv\Scripts\python.exe` as the workspace interpreter; installing with a different interpreter places packages in a different environment.

## Data Setup

Obtain the PhysioNet Challenge 2019 dataset from <https://physionet.org/content/challenge-2019/1.0.0/> and follow its access and usage terms. `python data/download_data.py` fetches and checksums both hospitals into `data/raw/training_setA` (hospital A, 20,336 files) and `data/raw/training_setB` (hospital B, 20,000 files). Keep downloaded data outside version control; `data/` and common output directories are ignored.

`--data-dir` accepts one or more folders of `.psv` files, one patient per file. Each patient is tagged with its folder name as its `source` (hospital), and patient identifiers must be unique across folders. The validator requires a `SepsisLabel` column containing numeric 0/1 values. Missing clinical input columns are allowed and become all-missing features; present clinical values must be numeric. If present, `ICULOS` must increase by one per row. Input files are processed in sorted filename order for deterministic cohort construction; folders with 2,000 or more files are read in parallel. `validate` prints counts by default; add `--verbose` to list patient filenames.

Reading 40,336 small files takes several minutes, so validate them once into a Parquet cache (needs `python -m pip install -e ".[data]"`) and pass `--cache` to every other command:

```powershell
sepsis-pipeline cache --data-dir .\data\raw\training_setA .\data\raw\training_setB --output .\data\cache\physionet2019.parquet
```

Records are validated before caching and not re-validated on load. Rebuild the cache whenever the raw files change.

## Run

Validate the files and get basic counts:

```powershell
sepsis-pipeline validate --cache .\data\cache\physionet2019.parquet
```

Create exploratory summaries and figures from the raw loaded dataset, before cohort filtering, including per-hospital counts:

```powershell
sepsis-pipeline eda --cache .\data\cache\physionet2019.parquet --output-dir .\outputs\physionet\eda
```

Build the early-prediction cohort, train both tabular baselines, evaluate held-out patient groups, and save artifacts:

```powershell
sepsis-pipeline run --cache .\data\cache\physionet2019.parquet --output-dir .\outputs\physionet\pooled --random-state 42 --test-size 0.2
```

By default (`--split-mode pooled`) patients from all sources are split at random. To test generalisation across hospitals, train on one and test on every eligible patient from the other:

```powershell
sepsis-pipeline run --cache .\data\cache\physionet2019.parquet --output-dir .\outputs\physionet\a_to_b --split-mode hospital --train-sources training_setA --test-sources training_setB
```

In hospital mode, fit and validation patients come only from `--train-sources`; `--test-size` is ignored.

### Unsupervised exploration

`explore` looks for structure in the six-hour look-back without the outcome, then describes what it finds against the outcome:

```powershell
sepsis-pipeline explore --cache .\data\cache\physionet2019.parquet --output-dir .\outputs\physionet\pooled
```

Point `--output-dir` at a completed `run` directory to reuse its `split_patients.json`, so exploration describes the same test patients as the supervised models; `--horizon-hours` must match that run. Without one, `explore` makes the same seeded split (`--split-mode` options apply) and saves it under `explore/`.

All steps are fitted on fit patients using features only: median imputation and scaling (constant and duplicate columns dropped, scaled values capped at ±10 SD because no range checks run upstream and PhysioNet contains implausible entries such as FiO2 = 4000), PCA to 90% variance (at most 20 components), k-means (k chosen by silhouette) and a Gaussian mixture (k chosen by BIC) for `--k-range 2 8`, and a 300-tree Isolation Forest. Cluster stability is the mean adjusted Rand index of `--stability-resamples 50` bootstrap refits, compared on validation patients; below 0.6 is flagged unstable. Test patients are then assigned, and each cluster's sepsis rate (Wilson 95% interval), key-variable medians, missingness, hospital share, and median record length are reported. The Isolation Forest score is evaluated as an unsupervised baseline with bootstrap AUROC and AUPRC.

`--feature-sets all measured` (the default) runs everything twice: once on all features and once without missingness indicators, because which tests were ordered can dominate the structure. `--methods` adds `umap` for a picture-only 2-D view when `pip install -e ".[explore]"` is installed; without it, PCA views are used and the config says so.

Outputs go to `<output-dir>/explore/`: `explore_config.json`, `explore_summary.json`, and per feature set `embedding_test.csv`, `embedding.png`, `pca_explained_variance.csv`, `pca_top_loadings.csv`, `k_selection.csv`, `k_selection.png`, `cluster_profiles_kmeans.csv`, `cluster_profiles_gmm.csv`, and `anomaly_scores.csv`. The run's `report.html` is regenerated with an "Unsupervised exploration" section. Clusters are statistical groupings, not clinical phenotypes.

Summarise completed runs side by side, for example the synthetic demo against the real-data runs:

```powershell
sepsis-pipeline compare --run synthetic=.\outputs\synthetic_e2e_v2\run --run pooled=.\outputs\physionet\pooled --run a_to_b=.\outputs\physionet\a_to_b --run b_to_a=.\outputs\physionet\b_to_a --output-dir .\outputs\comparison
```

This writes `comparison_cohorts.csv`, `comparison_models.csv`, and a self-contained `comparison.html`. Each model's AUPRC is shown with its bootstrap interval and as a multiple of test prevalence, which is what a no-skill model scores.

The look-back window is always the first six rows of each file. In PhysioNet 2019 about a fifth of records start after ICU hour 1, mostly in hospital A, so those rows are not always the first six ICU hours. `ICULOS` at row 0 is recorded per patient in `cohort_audit.csv` and summarised in `cohort_summary.json`; it is never a predictor.

The outcome horizon defaults to 24 hours. For a different endpoint, pass `--horizon-hours 12` or `--horizon-hours 48`. By default, each model's threshold is selected on validation patients to reach 80% sensitivity; `--validation-size` and `--target-sensitivity` adjust that procedure. `--threshold` explicitly overrides it with a common fixed threshold.

To run the whole workflow without downloading the clinical dataset, use the fabricated synthetic demo:

```powershell
python .\scripts\run_synthetic_demo.py
```

Install `python -m pip install -e ".[dev,cnn,xgboost]"` before running the demo. It writes 160 fabricated patients into two source folders with PhysioNet-like features: varied record lengths, sparse laboratory values, records that start after ICU hour 1, and patients who trigger every cohort exclusion. It runs validation, EDA, all tabular models, and one CNN epoch, then checks the report at `outputs/synthetic_e2e/run/report.html` and fails if any exclusion rule or source goes unexercised. Its generated labels and measurements are not clinical data or evidence.

Choose models with `--models logistic_regression random_forest xgboost`. Add `--cnn --cnn-epochs 20` to train the PyTorch CNN. The CNN and tabular models share one stratified patient split; CNN early stopping uses validation loss. Run `python -m sepsis_prediction.cli` in place of the installed command when working from a source checkout.

Start the JavaScript application from the project environment with `.\.venv\Scripts\python.exe -m sepsis_prediction.cli serve --synthetic-demo --output-dir .\outputs\synthetic_e2e`. For clinical data, use `.\.venv\Scripts\python.exe -m sepsis_prediction.cli serve --cache .\data\cache\physionet2019.parquet --output-dir .\outputs\physionet\app`; the split-mode flags work here too. Open the printed local URL. Each model has its own run button, elapsed timer, results, and model-specific charts; runs are serialized and refresh the relevant panel on completion. The full generated report remains available inside the application. The server binds only to loopback.

## Outputs

- The `serve` command opens the JavaScript pipeline application. Its model panels can run Logistic Regression, Random Forest, XGBoost, or the CNN independently, with completion timing, metrics, feature signals, and model-specific charts. A full generated report remains available in the application and as `report.html`.
- The report describes the PhysioNet 2019 PSV datasource, sorted file ingestion, required target checks, numeric field validation, optional fields, and `ICULOS` continuity checks. Physiologic range validation and unit normalization are not performed.
- `cohort_summary.json`: eligible outcome counts, prevalence, configured horizon, and exclusion counts (including incomplete negative follow-up), overall and per source, plus the distribution of `ICULOS` at row 0.
- `cohort_audit.csv`: one row per loaded patient with source, record length, `ICULOS` at row 0, cohort status or exclusion reason, and outcome.
- `metrics.json`: AUROC, average precision/AUPRC, sensitivity, specificity, precision, F1, accuracy, Brier score, validation-selected threshold, and confusion matrix. AUROC/AUPRC are `null` if the test split has only one class.
- `model_comparison.csv` and `model_curves.png`: held-out comparison table and ROC/precision-recall plots.
- `model_evaluation.json`, `model_ranking.csv`, `model_forest.png`, `calibration.png`, and `decision_curve.png`: head-to-head ranking, paired bootstrap uncertainty, frozen-threshold operating points, calibration, and decision analysis.
- `feature_importance.csv`: Logistic Regression coefficients and tree feature importances. These are model associations, not causal or clinical effects.
- `test_predictions.csv`: held-out patient IDs, labels, probabilities, and thresholded predictions.
- `split_patients.json` and `validation_predictions.csv`: fit/validation/test patient membership and validation outputs for audit and reproducibility.
- `feature_names.json` and `run_config.json`: feature schema, split mode and parameters, each split's size, prevalence and source mix (`split_composition`), explicit prediction design, and package versions.
- `logistic_regression.joblib` and `random_forest.joblib`: fitted preprocessing-plus-model pipelines.
- `xgboost.joblib`: fitted XGBoost pipeline when selected.
- `cnn_1d.pt`: CNN weights and train-fitted imputation/scaling values when `--cnn` is selected.
- App-driven per-model runs are isolated under `<output-dir>/models/<model>/` so one model refresh does not overwrite another model's artifacts.

The `eda` command writes `dataset_summary.json`, `missingness.csv`, `record_lengths.csv`, `labels_by_hour.csv`, corresponding PNG plots, and its own `report.html`. EDA summaries are descriptive and include raw loaded records, not only the model cohort.

Reusing the same input files, software versions, options, and random seed produces the same split, metrics, and predictions. Model files are intended for this educational experiment, not deployment.

## Tests

Tests create temporary synthetic PSV files and do not download or require clinical data. They cover cohort timing, feature/sequence alignment, patient splitting, train-only preprocessing, reproducibility, CLI artifacts, and EDA outputs:

```powershell
python -m pytest
```

The plan for PhysioNet integration and unsupervised exploration is in [docs/plan_physionet_and_unsupervised.md](docs/plan_physionet_and_unsupervised.md).

## Clinical-Use Caveat

This is an educational, retrospective modelling project, not a clinically validated decision-support tool. The fixed-window label setup, dataset cohort, missingness, and evaluation split limit what results mean. Reported associations are not causal or clinical interpretations, and the metrics must not be used to guide patient care.
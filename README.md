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

Obtain the PhysioNet Challenge 2019 dataset from <https://physionet.org/content/challenge-2019/1.0.0/> and follow its access and usage terms. Point the commands at a directory containing `.psv` files, one patient per file. Keep downloaded data outside version control; `data/` and common output directories are ignored.

The validator requires a `SepsisLabel` column containing numeric 0/1 values. Missing clinical input columns are allowed and become all-missing features; present clinical values must be numeric. If present, `ICULOS` must increase by one per row. Input files are processed in sorted filename order for deterministic cohort construction. `validate` prints counts by default; add `--verbose` to list patient filenames.

## Run

Validate the files and get basic counts:

```powershell
sepsis-pipeline validate --data-dir .\data\train
```

Create exploratory summaries and figures from the raw loaded dataset, before cohort filtering:

```powershell
sepsis-pipeline eda --data-dir .\data\train --output-dir .\eda
```

Build the early-prediction cohort, train both tabular baselines, evaluate held-out patient groups, and save artifacts:

```powershell
sepsis-pipeline run --data-dir .\data\train --output-dir .\artifacts --random-state 42 --test-size 0.2
```

The outcome horizon defaults to 24 hours. For a different endpoint, pass `--horizon-hours 12` or `--horizon-hours 48`. By default, each model's threshold is selected on validation patients to reach 80% sensitivity; `--validation-size` and `--target-sensitivity` adjust that procedure. `--threshold` explicitly overrides it with a common fixed threshold.

To run the whole workflow without downloading the clinical dataset, use the fabricated synthetic demo:

```powershell
python .\scripts\run_synthetic_demo.py
```

Install `python -m pip install -e ".[dev,cnn,xgboost]"` before running the demo. It runs validation, EDA, all tabular models, one CNN epoch, and checks the report at `outputs/synthetic_e2e/run/report.html`. Its generated labels and measurements are not clinical data or evidence.

Choose models with `--models logistic_regression random_forest xgboost`. Add `--cnn --cnn-epochs 20` to train the PyTorch CNN. The CNN and tabular models share one stratified patient split; CNN early stopping uses validation loss. Run `python -m sepsis_prediction.cli` in place of the installed command when working from a source checkout.

Start the JavaScript application from the project environment with `.\.venv\Scripts\python.exe -m sepsis_prediction.cli serve --synthetic-demo --output-dir .\outputs\synthetic_e2e`. For clinical data, use `.\.venv\Scripts\python.exe -m sepsis_prediction.cli serve --data-dir .\data\train --output-dir .\artifacts`. Open the printed local URL. Each model has its own run button, elapsed timer, results, and model-specific charts; runs are serialized and refresh the relevant panel on completion. The full generated report remains available inside the application. The server binds only to loopback.

## Outputs

- The `serve` command opens the JavaScript pipeline application. Its model panels can run Logistic Regression, Random Forest, XGBoost, or the CNN independently, with completion timing, metrics, feature signals, and model-specific charts. A full generated report remains available in the application and as `report.html`.
- The report describes the PhysioNet 2019 PSV datasource, sorted file ingestion, required target checks, numeric field validation, optional fields, and `ICULOS` continuity checks. Physiologic range validation and unit normalization are not performed.
- `cohort_summary.json`: eligible outcome counts, configured horizon, and exclusion counts, including incomplete negative follow-up.
- `metrics.json`: AUROC, average precision/AUPRC, sensitivity, specificity, precision, F1, accuracy, Brier score, validation-selected threshold, and confusion matrix. AUROC/AUPRC are `null` if the test split has only one class.
- `model_comparison.csv` and `model_curves.png`: held-out comparison table and ROC/precision-recall plots.
- `model_evaluation.json`, `model_ranking.csv`, `model_forest.png`, `calibration.png`, and `decision_curve.png`: head-to-head ranking, paired bootstrap uncertainty, frozen-threshold operating points, calibration, and decision analysis.
- `feature_importance.csv`: Logistic Regression coefficients and tree feature importances. These are model associations, not causal or clinical effects.
- `test_predictions.csv`: held-out patient IDs, labels, probabilities, and thresholded predictions.
- `split_patients.json` and `validation_predictions.csv`: fit/validation/test patient membership and validation outputs for audit and reproducibility.
- `feature_names.json` and `run_config.json`: feature schema, split parameters, explicit prediction design, and package versions.
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

The project plan and architecture/data-flow diagram are in [docs/experimental_design.md](docs/experimental_design.md).

## Clinical-Use Caveat

This is an educational, retrospective modelling project, not a clinically validated decision-support tool. The fixed-window label setup, dataset cohort, missingness, and evaluation split limit what results mean. Reported associations are not causal or clinical interpretations, and the metrics must not be used to guide patient care.
# Early Sepsis Prediction Study

A reproducible workflow for dataset inspection, early-window cohort construction, tabular and optional sequence models, and held-out patient-level evaluation using the PhysioNet/Computing in Cardiology Challenge 2019 PSV files. Clinical data is external and is never committed here.

## Prediction Design

Each PSV file represents one patient, with hourly observations in file order. The project uses rows at hours 0-5 inclusive as a fixed six-hour look-back and predicts whether any `SepsisLabel=1` occurs from hour 6 onward. Patients labeled positive in any of rows 0-5 are excluded, as are records shorter than six rows and records with no post-look-back observation. At least seven rows are therefore required to observe an outcome label.

Features use only those six look-back rows. For each known clinical input, deterministic features include latest observed value, mean, minimum, maximum, population standard deviation, first-to-last observed delta, missing fraction, and an any-missing indicator. Missing schema columns and all-NaN windows retain a stable feature schema and are handled by the train-fitted imputer. `SepsisLabel`, the filename-derived patient identifier, and `ICULOS` are never predictors. `ICULOS` is deliberately excluded: the fixed six-row design already establishes the observation period, and elapsed ICU time could otherwise encode care-process timing.

The cohort is split by patient before model fitting. Every learned imputation and scaling step is fitted on training patients only. Logistic Regression, Random Forest, and XGBoost use the same feature rows and patient split. CNN training is opt-in and uses the corresponding ordered six-hour observations and the same deterministic split; its variable-wise imputation and scaling are fitted on training patients only. The default probability threshold is 0.5; the threshold, seed, cohort design, metrics, and patient split are saved with the run.

PyTorch and XGBoost are installed with the standard project dependencies. The tabular workflow remains the default; select XGBoost with `--models` and use `--cnn` to add sequence-model training.

## Install

Python 3.10 or newer is required.

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -e ".[dev]"
```

## Data Setup

Obtain the PhysioNet Challenge 2019 dataset from <https://physionet.org/content/challenge-2019/1.0.0/> and follow its access and usage terms. Point the commands at a directory containing `.psv` files, one patient per file. Keep downloaded data outside version control; `data/` and common output directories are ignored.

The validator requires a `SepsisLabel` column containing numeric 0/1 values. Missing clinical input columns are allowed and become all-missing features; present clinical values must be numeric. Input files are processed in sorted filename order for deterministic cohort construction.

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

To run the whole workflow without downloading the clinical dataset, use the fabricated synthetic demo:

```powershell
python .\scripts\run_synthetic_demo.py
```

It runs validation, EDA, all tabular models, one CNN epoch, and checks the report at `outputs/synthetic_e2e/run/report.html`. Its generated labels and measurements are not clinical data or evidence.

Choose models with `--models logistic_regression random_forest xgboost`. Add `--cnn --cnn-epochs 20` to train the PyTorch CNN. Run `python -m sepsis_prediction.cli` in place of the installed command when working from a source checkout.

## Outputs

- `report.html`: self-contained visual report with held-out results and charts, raw six-hour sample observations, engineered-feature sample, and pipeline/data-lineage diagrams. It embeds patient-derived values; share and store it according to dataset and institutional privacy requirements.
- `cohort_summary.json`: eligible outcome counts and exclusion counts, including records without outcome follow-up.
- `metrics.json`: AUROC, average precision/AUPRC, sensitivity, specificity, precision, F1, accuracy, Brier score, threshold, and confusion matrix. AUROC/AUPRC are `null` if the test split has only one class.
- `model_comparison.csv` and `model_curves.png`: held-out comparison table and ROC/precision-recall plots.
- `feature_importance.csv`: Logistic Regression coefficients and tree feature importances. These are model associations, not causal or clinical effects.
- `test_predictions.csv`: held-out patient IDs, labels, probabilities, and thresholded predictions.
- `split_patients.json`: train/test patient membership for audit and reproducibility.
- `feature_names.json` and `run_config.json`: feature schema, split parameters, and explicit prediction design.
- `logistic_regression.joblib` and `random_forest.joblib`: fitted preprocessing-plus-model pipelines.
- `xgboost.joblib`: fitted XGBoost pipeline when selected.
- `cnn_1d.pt`: CNN weights and train-fitted imputation/scaling values when `--cnn` is selected.

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
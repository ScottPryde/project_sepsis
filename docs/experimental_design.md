# Experimental Design and Project Plan

## Data Flow

```mermaid
flowchart TD
    A[PhysioNet patient PSV files] --> B[Schema validation and raw EDA]
    B --> C[Fixed cohort: rows 0-5 as input]
    C --> D{Any positive label in rows 0-5?}
    D -->|Yes| X[Exclude: prevalent label]
    D -->|No| E{Any observed row from hour 6 onward?}
    E -->|No| Y[Exclude: outcome unavailable]
    E -->|Yes| F[Outcome: positive label within configured horizon]
    F --> G{Positive in horizon or complete follow-up?}
    G -->|No, incomplete negative| Z[Exclude: incomplete horizon follow-up]
    G -->|Yes| H[Eligible bounded-horizon cohort]
    H --> I[Stratified fit/validation/test split]
    I --> J[Tabular summaries]
    I --> K[Ordered six-hour tensor]
    J --> L[Train-only imputation/scaling]
    K --> M[Train-only imputation/scaling]
    L --> N[Logistic Regression / Random Forest / XGBoost]
    M --> O[1D CNN, opt-in training]
    N --> P[Held-out predictions and metrics]
    O --> P
    P --> Q[Comparison report and model artifacts]
```

## Evaluation Protocol

- The fixed six-row look-back and configured finite outcome horizon are shared between tabular and sequence representations. Negative examples require complete follow-up; positives observed inside the horizon remain eligible.
- PhysioNet 2019 `SepsisLabel` switches on six hours before Sepsis-3 onset. The prediction target and interpretation account for this label shift.
- Patient IDs are groups, not predictors. A deterministic stratified patient split creates fit, validation, and test partitions, shared across all selected models.
- Imputation and scaling parameters are learned from fit patients only. Validation patients select thresholds and control CNN early stopping; test patients are used once for final evaluation.
- Static demographic/context variables use their value and missingness indicators only; they are not expanded into redundant temporal statistics.
- By default each model's highest threshold reaching the target sensitivity (80%) is selected on validation patients, recorded, and applied unchanged to the test patients. `--threshold` is an explicit fixed-threshold override.
- AUROC and average precision are reported only where both outcome classes occur in the test set. Sensitivity, specificity, precision, F1, accuracy, Brier score, and confusion matrix are also saved.
- The CNN receives observation masks as channels, alongside imputed values. It still differs from tabular models in both input representation and model family; a performance difference cannot be attributed to architecture alone.
- This is retrospective educational analysis, not a clinically validated decision-support system. Challenge utility and direct comparison with challenge submissions are not implemented.

## Indicative Eight-Week Plan

```mermaid
gantt
    title Indicative Project Schedule
    dateFormat YYYY-MM-DD
    axisFormat Week %W
    section Design and data
    Confirm design and acquire data :a1, 2026-10-05, 7d
    Dataset validation and EDA     :a2, after a1, 14d
    section Implementation
    Cohort and preprocessing       :b1, 2026-10-19, 14d
    Tabular baselines              :b2, 2026-11-02, 14d
    Optional XGBoost and CNN       :b3, 2026-11-16, 14d
    section Evaluation and report
    Evaluation and interpretation  :c1, 2026-11-30, 7d
    Final analysis and write-up    :c2, 2026-12-07, 14d
```

The calendar dates above are illustrative placeholders and should be shifted to the module's confirmed submission schedule.

| Work package | Activities | Indicative timing |
|---|---|---|
| Design and dataset setup | Confirm timing rule, obtain data, validate schema | Week 1 |
| Exploratory analysis | Record lengths, class balance, missingness, distributions | Weeks 1-2 |
| Cohort and preprocessing | Leakage checks, feature engineering, patient split | Weeks 2-3 |
| Tabular baselines | Logistic Regression and Random Forest | Weeks 3-4 |
| Boosted trees | XGBoost and interpretation | Weeks 4-5 |
| Temporal model | Sequence preparation and 1D CNN | Weeks 5-6 |
| Evaluation | Comparison table, plots, robustness review | Weeks 6-7 |
| Reporting | Limitations, results, reproducibility | Weeks 7-8 |

## Reproduction Record

Each model run writes `run_config.json` with package versions, `split_patients.json`, `feature_names.json`, validation and held-out predictions, metrics, and model files. The same random seed does not guarantee bit-identical results across different library versions or hardware backends.

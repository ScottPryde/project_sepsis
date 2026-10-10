# Comparative Machine-Learning Approaches for Early Sepsis Prediction

## 1. Project overview

**Working title:** Comparative Machine-Learning Approaches for Early Sepsis Prediction from ICU Time-Series Data

**Project type:** Applied machine-learning implementation and comparative evaluation

**Dataset:** PhysioNet/Computing in Cardiology Challenge 2019 — Early Prediction of Sepsis from Clinical Data

**Primary objective:** Demonstrate the implementation and evaluation of several established machine-learning methods on a common clinical prediction problem, with particular attention to preprocessing and the representation of longitudinal data.

This project is not intended to invent a novel algorithm or outperform published challenge entries. It will use an open clinical dataset to demonstrate a reproducible end-to-end workflow: data inspection, cleaning, missing-data handling, feature engineering, model training, evaluation and interpretation.

## 2. Background and rationale

Sepsis is a time-critical condition. Intensive care unit (ICU) records contain observations such as vital signs and laboratory results collected at different times and frequencies. These data are often incomplete, and their temporal patterns may contain useful information.

The PhysioNet 2019 dataset provides hourly clinical observations and a time-dependent sepsis label. It is suitable for a practical comparison because it supports both conventional tabular modelling and models that process sequential data.

The project will compare models using a consistent prediction task and evaluation procedure. It will also document the consequences of key preprocessing choices rather than treating preprocessing as an invisible step.

## 3. Research question

**Primary question**

> How do different machine-learning approaches compare for early prediction of sepsis from longitudinal ICU clinical data?

**Supporting question**

> How does representing clinical observations as engineered tabular features, rather than as a temporal sequence, affect predictive performance?

The work is a comparative methods study. Its contribution is the implementation, controlled comparison and critical interpretation of established approaches, not algorithmic novelty.

## 4. Dataset

The project will use the **PhysioNet/Computing in Cardiology Challenge 2019 dataset**, available from PhysioNet:

https://physionet.org/content/challenge-2019/1.0.0/

The dataset contains approximately 40,000 ICU patient records. Patient records are supplied as `.psv` files with hourly observations. The columns cover 40 input variables and one target variable, `SepsisLabel`.

### 4.1 Variable groups

| Group | Example fields |
|---|---|
| Vital signs | `HR`, `O2Sat`, `Temp`, `SBP`, `MAP`, `DBP`, `Resp`, `EtCO2` |
| Blood gas and oxygenation | `BaseExcess`, `HCO3`, `FiO2`, `pH`, `PaCO2`, `SaO2` |
| Biochemistry and electrolytes | `AST`, `BUN`, `Alkalinephos`, `Calcium`, `Chloride`, `Creatinine`, `Glucose`, `Lactate`, `Magnesium`, `Phosphate`, `Potassium`, `Bilirubin_direct`, `Bilirubin_total`, `TroponinI` |
| Haematology and coagulation | `Hct`, `Hgb`, `PTT`, `WBC`, `Fibrinogen`, `Platelets` |
| Demographic and care context | `Age`, `Gender`, `Unit1`, `Unit2`, `HospAdmTime`, `ICULOS` |
| Target | `SepsisLabel` |

The dataset documentation should be treated as the authoritative source for definitions, units and label construction. `SepsisLabel` is the target and must not be used as an input feature.

## 5. Scope and prediction design

The project will define a clear prediction timestamp and use only information available at or before that timestamp. A practical starting design is to assess risk from an initial clinical observation window and compare predictions against the dataset's time-dependent sepsis labels. The exact window and label rule will be documented before model training.

The project will avoid using the complete patient record to predict an earlier outcome, as that would introduce future-information leakage. The handling of the challenge's sepsis-label timing will be described explicitly.

`ICULOS` (hours since ICU admission) will not automatically be treated as a routine predictor. Its inclusion will be considered only if it is justified by the prediction design and applied consistently without leaking future information.

This is an educational modelling study, not a clinically validated decision-support tool.

## 6. Data-processing and preprocessing plan

Preprocessing is a central part of the demonstration. The pipeline will retain a record of the raw data and produce reproducible intermediate datasets.

### Stage 1 — Ingestion and validation

- Load patient `.psv` files into pandas DataFrames.
- Confirm column names, data types, row counts and patient-record lengths.
- Check that the target is present and has expected values.
- Identify duplicate records, malformed values and unexpected ranges.
- Preserve a patient identifier for splitting and traceability, but exclude it from model features.

### Stage 2 — Exploratory data analysis

- Summarise the number of patients and hourly observations.
- Examine the distribution of record lengths and sepsis labels.
- Calculate missingness by variable and, where useful, over time.
- Visualise missingness and selected clinical-variable distributions.
- Inspect plausible ranges and potential outliers, distinguishing data errors from genuine clinical extremes.

### Stage 3 — Missing-data handling

Clinical measurements are not collected at every hour. Missingness may reflect measurement practice as well as technical absence.

The project will compare or demonstrate appropriate strategies, for example:

1. **Forward fill with a defined limit** for selected time-series measurements, so old values are not carried forward indefinitely.
2. **Median imputation** fitted on training data for remaining missing numeric values.
3. **Missingness indicators** for selected variables, allowing the model to distinguish an imputed value from an observed measurement.

The chosen strategies will be documented and applied without allowing information from the test set to influence imputation values. Any comparison of imputation strategies will be treated as a small, controlled experiment rather than an exhaustive search.

### Stage 4 — Scaling and encoding

- Scale numeric features for Logistic Regression and the neural network where appropriate.
- Fit scaling parameters on training data only.
- Confirm that binary variables are encoded consistently.
- Apply the same input definitions to comparable models wherever feasible.

Tree-based models generally do not require numeric feature scaling, which can itself be noted in the comparison.

### Stage 5 — Feature engineering for tabular models

Within a predefined look-back window, create patient-time examples using features such as:

- latest available value;
- mean, minimum and maximum;
- variability, such as standard deviation;
- simple change or trend over time;
- selected missingness indicators.

Features will be generated using only observations available by the prediction timestamp. The feature set will be kept manageable and documented.

### Stage 6 — Sequence preparation for the temporal model

For the 1D CNN, retain hourly order rather than collapsing all observations into summary statistics. Construct fixed-length windows with consistent variable ordering, using a documented method for padding, truncation and missing values.

This enables a comparison between **engineered tabular representation** and **temporally ordered input**. The project will make clear that this comparison changes both the representation and the model family; results therefore cannot attribute any difference solely to the algorithm.

## 7. Models to implement

| Model | Input representation | Purpose |
|---|---|---|
| Logistic Regression | Engineered tabular features | Interpretable baseline |
| Random Forest | Engineered tabular features | Non-linear tree ensemble |
| XGBoost | Engineered tabular features | Gradient-boosted tree benchmark |
| 1D CNN | Ordered hourly sequences | Demonstrate deep learning on temporal clinical data |

The three tabular models will use the same core prediction examples and feature set to make comparisons as fair as possible. The CNN will use the corresponding available observation windows in sequence form.

The project will start with a working baseline before adding the next model. If time or computing constraints arise, a well-evaluated comparison of the first three models takes priority over an unfinished CNN.

## 8. Experimental design and evaluation

### 8.1 Data splitting

- Split at patient level so that observations from the same patient cannot appear in both training and test sets.
- Use a fixed random seed and record the split.
- Use validation data or cross-validation within the training data for model selection.
- Fit imputation, scaling, feature selection and other learned preprocessing steps using training data only.
- Do not repeatedly tune models against the final test set.

### 8.2 Metrics

Because sepsis prediction is an imbalanced classification problem, accuracy alone is insufficient. Report a selection of:

- **AUROC** — discrimination across classification thresholds.
- **AUPRC** — precision-recall performance under class imbalance.
- **Sensitivity/recall** — proportion of positive cases detected.
- **Specificity** — proportion of negative cases correctly identified.
- **F1 score** — balance between precision and recall at a stated threshold.
- **Calibration or Brier score**, if feasible — quality of predicted probabilities.
- **Challenge utility score**, if the official evaluation implementation can be reproduced correctly.

Report confusion matrices and the selected decision threshold. If the prediction setup permits it, also describe how early predictions occur relative to the labelled onset. Metrics from this project should not be compared directly with published challenge scores unless the evaluation protocol, split, prediction timing and scoring method are comparable.

### 8.3 Interpretation

- Examine Logistic Regression coefficients with appropriate caution.
- Review feature importance or permutation importance for tree-based models.
- Optionally use SHAP for a selected model if time permits.
- Discuss whether predictive signals are clinically plausible, without interpreting model associations as causal effects.

## 9. Planned outputs

1. A documented dataset-ingestion and validation script.
2. Exploratory data-analysis figures, including missingness summaries.
3. A reproducible preprocessing pipeline.
4. A feature-engineering pipeline for tabular models.
5. A sequence-preparation pipeline for the CNN, if completed.
6. Trained model scripts or notebooks.
7. A comparison table and figures using common evaluation metrics.
8. A short analysis of preprocessing choices, model complexity, interpretability and limitations.
9. A README explaining the environment, data setup and steps to reproduce the experiments.
10. Week-5 intermediate design materials: architecture/data-flow diagram, project plan/Gantt chart, dataset description and evidence of work in progress.

## 10. Proposed software and tools

- **Python** — implementation language.
- **pandas / NumPy** — data loading and manipulation.
- **Matplotlib / Seaborn** — visualisation.
- **scikit-learn** — preprocessing, Logistic Regression, Random Forest, splitting and metrics.
- **XGBoost** — gradient-boosted trees.
- **PyTorch** — 1D CNN.
- **SHAP** — optional model interpretation.
- **Jupyter notebooks** — exploratory analysis and reporting.

The core pipeline does not require a database or complex deployment infrastructure. The priority is a clear, reproducible experimental workflow.

## 11. Indicative workflow and Gantt outline

The timing should be adjusted to the module's actual submission dates.

| Work package | Activities | Indicative order |
|---|---|---|
| 1. Define design | Confirm prediction window, label handling, metrics and split strategy | Week 1 |
| 2. Dataset setup | Obtain data, inspect files, validate schema and create data dictionary | Week 1 |
| 3. Exploratory analysis | Missingness, class balance, record lengths, distributions | Weeks 1–2 |
| 4. Preprocessing | Implement imputation, indicators, scaling and leakage checks | Weeks 2–3 |
| 5. Baseline models | Logistic Regression and Random Forest | Weeks 3–4 |
| 6. Boosted trees | Implement and evaluate XGBoost | Weeks 4–5 |
| 7. Temporal model | Prepare sequences and implement a simple 1D CNN | Weeks 5–6 |
| 8. Evaluation | Compare models, test robustness and generate plots | Weeks 6–7 |
| 9. Write-up | Discuss results, limitations and reproducibility | Weeks 7–8 |

### Week-5 intermediate submission

The intermediate design can include:

- project objective and research question;
- dataset and variable groups;
- architecture/data-flow diagram;
- Gantt chart;
- initial dataset-inspection or missingness-analysis script;
- preliminary figures or a documented preprocessing plan;
- model-comparison and evaluation plan.

If implementation is still early, the plan and a small verified dataset-inspection script are suitable evidence of progress.

## 12. Risks and mitigations

| Risk | Mitigation |
|---|---|
| Dataset labels are misunderstood | Use the official documentation and explicitly document the label/prediction-time relationship. |
| Future information leaks into features | Define prediction timestamps first; restrict all features to data available at that time. |
| Patient observations cross data splits | Split by patient, not by hourly row. |
| Imputation or scaling leaks test information | Fit preprocessing only on training data and apply the fitted transformation to validation/test data. |
| Class imbalance makes accuracy misleading | Report AUPRC, sensitivity, specificity and other relevant metrics. |
| CNN takes too much time | Prioritise a complete tabular comparison and treat the CNN as an extension. |
| Published results use different protocols | Describe protocol differences and avoid claiming direct superiority from incomparable scores. |
| Scope becomes too broad | Keep the primary experiment to four models and a small number of justified preprocessing choices. |

## 13. Expected discussion

The report will discuss:

- how much missingness varies across clinical variables;
- how imputation and missingness indicators affect the modelling workflow;
- whether engineered summary and trend features perform differently from sequence input;
- the balance between predictive performance and interpretability;
- the computational and implementation cost of the models;
- the limitations of retrospective ICU data, label definitions and a single dataset.

No outcome is assumed in advance. The value of the project is in conducting and documenting the comparison correctly, including results that do not favour the more complex model.

## 14. Literature and positioning

The PhysioNet 2019 challenge has already been studied using methods including signature-based models, gradient-boosted trees, Random Forests, recurrent neural networks and convolutional neural networks. This project will not claim novelty from applying those algorithms to the dataset.

Instead, it will focus on a manageable and reproducible comparison with transparent preprocessing and evaluation. Published challenge papers and the official evaluation code can inform the methodology:

- Dataset and challenge: https://physionet.org/content/challenge-2019/1.0.0/
- Challenge papers: https://physionet.org/content/challenge-2019/1.0.0/papers/
- Challenge evaluation code: https://github.com/physionetchallenges/evaluation-2019

Before final submission, the literature review should verify relevant papers' exact experimental setups and reported metrics.

## 15. Summary

This project demonstrates a complete applied machine-learning workflow on an open clinical dataset. Its central comparison is between established tabular models and a temporal deep-learning model, supported by explicit data-quality analysis, missing-data handling, feature engineering, leakage prevention and consistent evaluation.

The intended result is not a new clinical algorithm, but a technically sound, reproducible and critically evaluated comparison of methods.

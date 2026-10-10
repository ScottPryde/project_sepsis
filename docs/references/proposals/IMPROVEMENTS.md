# Improvement Plan

This plan follows a review of the pipeline's design, code and tests. The engineering is sound: learned preprocessing is fitted on training patients only, patient overlap is checked, ICULOS is excluded with a stated reason, runs are recorded, and the cohort timing is tested. The issues below are mainly in the prediction design and evaluation, which decide what the results mean.

Items are ordered by how much they change the interpretation of results. Each one states the problem, the change, where it goes in the code, and how to tell it is done.

## Status

| # | Item | Priority | Status |
|---|---|---|---|
| 0 | Head-to-head model comparison, methodology section and per-model lineage diagram in the report | — | Done in this change |
| 1 | Document the six-hour label shift | High | To do |
| 2 | Fixed outcome horizon with complete follow-up | High | To do |
| 3 | Consistent class weighting and a validation-chosen threshold | High | To do |
| 4 | Validation split for threshold selection and CNN early stopping | High | To do |
| 5 | Cross-hospital validation (set A versus set B) | High | To do |
| 6 | Give the CNN the same missingness information | Medium | To do |
| 7 | Repeated splits or cross-validation for stable rankings | Medium | To do |
| 8 | Permutation importance or SHAP instead of impurity importance | Medium | To do |
| 9 | Engineering and packaging fixes | Low | To do |

---

## 0. Model comparison in the report (done)

The report previously listed metrics at a fixed 0.5 threshold. Because the models use different class weighting, those numbers compared different operating points rather than different models.

New module `src/sepsis_prediction/evaluation.py`, called from `write_model_comparison`, now produces the following from `test_predictions.csv`, so every model is judged on the same patients:

- **Ranking.** Models are ranked by AUPRC with AUROC as a tie-break, using paired percentile bootstrap 95% CIs (1,000 resamples, seeded from the run's random state).
- **Distinguishability.** Each model's AUPRC and AUROC difference from the top model is bootstrapped. A model is called worse only when the interval excludes zero. The verdict distinguishes "worse on both metrics" from "worse on AUPRC only".
- **Matched-sensitivity operating point.** At 80% sensitivity the report gives specificity, PPV, alerts per 100 patients, alerts per true case and missed cases per 100 patients.
- **Calibration.** It reports calibration-in-the-large, the logistic calibration slope, 10-bin ECE and a decile calibration curve.
- **Clinical utility.** It plots decision-curve net benefit against treat-all and treat-none, for thresholds from 1% to 50%.
- **Outputs.** The module writes `model_evaluation.json`, `model_ranking.csv`, `model_forest.png`, `calibration.png` and `decision_curve.png`.

The report also has:

- a **Methodology** section built from `run_config.json`, including per-model specs now recorded by `MODEL_SPECS` in `modeling.py` and the CNN block in `cnn.py`;
- a **pipeline diagram with one lane per model**, showing each model's representation, train-only preprocessing and estimator, with dashed lanes for models not selected in the run.

The fixed-threshold metrics remain in a collapsible block, labelled as not like-for-like. Tests are in `tests/test_evaluation.py`.

**Known limitation.** The 80%-sensitivity threshold is chosen on the test patients, so those columns describe the models and are not an estimate of deployed performance. Item 4 removes this limitation.

---

## 1. Document the six-hour label shift

**Problem.** In the PhysioNet 2019 data, `SepsisLabel` already turns on six hours before the Sepsis-3 onset time (t ≥ t_sepsis − 6). Excluding patients labelled positive in rows 0–5 therefore removes everyone with onset before about hour 12. A positive outcome means onset at about hour 12 or later. The proposal's risk table names this exact risk ("dataset labels are misunderstood"), but the README and design doc never state it.

**Change.** State it in `README.md` under Prediction Design, in `docs/experimental_design.md`, and in the `DESIGN` dict in `modeling.py`, for example as `"label_definition": "SepsisLabel=1 from t_sepsis-6h (PhysioNet shift)"`. The report methodology already explains it.

**Done when.** A reader of the README can say what onset time a positive outcome corresponds to.

## 2. Fixed outcome horizon with complete follow-up

**Problem.** The outcome is "any positive label for the rest of the record". That window varies from 1 hour to weeks:

- A patient with 7 rows is counted as a confirmed negative after 1 hour of follow-up.
- Longer stays have more chances to turn positive, so the model partly learns length of stay, a proxy for acuity, rather than impending sepsis.
- Results cannot be stated as "risk of sepsis in the next N hours".

**Change.**
- Add a `--horizon-hours` option, defaulting to 24, so the outcome is any positive label in rows 6 to 6 + H − 1.
- Exclude negatives whose record ends before the horizon closes, as a new exclusion reason `incomplete_horizon_followup`.
- Keep positives that fall within the horizon even if the record ends early.
- Record the horizon in `DESIGN` and `run_config.json`.
- Run H = 12, 24 and 48 and report all three. This also delivers the "how early" analysis promised in proposal section 8.2.

**Where.** `create_patient_example` and the exclusion bookkeeping in `features.py`, the CLI arguments, the `DESIGN` dict, and the cohort timing tests.

**Done when.** The tests cover a negative with short follow-up (excluded), a positive just inside the horizon (kept as positive), and a positive just outside it (negative if follow-up is complete).

## 3. Consistent class weighting and a validation-chosen threshold

**Problem.** Logistic Regression, Random Forest and the CNN are class-weighted, but XGBoost is not (it has no `scale_pos_weight`). At a fixed 0.5 threshold with low prevalence, unweighted models can show near-zero sensitivity for reasons unrelated to how well they rank patients. On the synthetic sample, Random Forest and XGBoost both scored 0 sensitivity at 0.5 despite similar AUROC. Reweighting also inflates predicted risks, which worsens the Brier score and the calibration slope.

**Change.** Pick one policy and apply it to every model:
- **Option A (preferred for probabilities that mean something):** remove all class weighting, then choose the threshold on validation data.
- **Option B:** weight every model, including `scale_pos_weight` for XGBoost, and add a post-hoc calibration step (`CalibratedClassifierCV` or isotonic calibration on validation data).

Record the policy in `MODEL_SPECS`.

**Done when.** The "Class weighting" column in the report's model table reads the same for every model.

## 4. Validation split for threshold selection and CNN early stopping

**Problem.** There is one train/test split and no validation data, although proposal section 8.1 commits to one. The CNN trains for a fixed number of epochs with no early stopping. Any threshold other than 0.5 has to be picked on the test set.

**Change.**
- Split the training patients again into fit and validation sets (for example 80/20, stratified, seeded).
- Use the validation set to choose each model's 80%-sensitivity threshold and to early-stop the CNN on validation loss or AUPRC.
- Apply the frozen thresholds to the test set, and switch the report's operating-point columns to those thresholds.
- Save `split_patients.json` with `fit`, `validation` and `test` keys.

**Done when.** The report states "threshold chosen on validation patients". The caveat in item 0 can then be deleted, and the test-set sensitivity at the frozen threshold is reported honestly, even if it falls below 80%.

## 5. Cross-hospital validation

**Problem.** The data come from two hospitals: `training_setA` (`p0…` files) and `training_setB` (`p1…` files). A random split mixes them, so the results say nothing about transfer to a new site. That is the most important question for any deployment.

**Change.**
- Allow `--data-dir` to take several directories, or add `--site-from-filename`.
- Record each patient's site.
- Add `--split site` to train on A and test on B, then the reverse.
- Report both directions next to the random split.

**Done when.** The report shows three result blocks (random, A→B and B→A) and the methodology names the split strategy.

## 6. Give the CNN the same missingness information

**Problem.** The tabular models receive `missing_fraction` and `any_missing` features. The CNN receives median-imputed values with no indication of which were measured. In ICU data, whether a test was ordered is often strongly predictive, so the CNN–tabular comparison currently mixes "temporal order" with "less information". XGBoost can also handle NaN natively, and imputing before it removes that signal.

**Change.**
- Add one binary mask channel per variable to the CNN input, doubling the channels, and optionally a "hours since last observation" channel.
- Add an XGBoost variant without the imputer.
- Update the report methodology text, which currently states that the CNN has no missingness flags.

**Done when.** The CNN input description in `MODEL_SPECS` lists mask channels, and the representation comparison can be described as "same information, different structure".

## 7. Repeated splits or cross-validation

**Problem.** Rankings from one split are fragile when the test set has few positive patients. The report now warns when there are fewer than 30.

**Change.** Add `--repeats K` to run K seeded stratified splits, or stratified 5-fold cross-validation on the training data. Report the mean and spread of AUPRC and AUROC per model, and how often each model ranks first.

**Done when.** `model_evaluation.json` has a `repeats` block and the report shows rank stability.

## 8. Better feature attribution

**Problem.** `feature_importance.csv` uses impurity-based importances for the tree models. These favour continuous, high-cardinality features and are computed on training data.

**Change.** Compute `sklearn.inspection.permutation_importance` on the test patients using AUPRC as the score, for every tabular model. Optionally add SHAP for the top model, as planned in the proposal. Keep the existing caution that importances are associations, not causes.

**Done when.** The report shows a top-15 permutation-importance chart per model.

## 9. Engineering and packaging fixes

- **Make PyTorch optional.** `torch` is a hard dependency even though the CNN is opt-in, and it adds several GB to installs. Move it to `[project.optional-dependencies] cnn = ["torch>=2.2"]`, and consider the same for `xgboost`.
- **Drop constant features.** The static variables (Age, Gender, Unit1, Unit2, HospAdmTime) get min, max, std and delta features, about 30 constant columns. Emit only the value and a missing flag for these.
- **Remove duplicated logic.** The exclusion-reason logic appears in both `create_examples` and `create_sequence_examples`. Have `create_patient_example` return the reason and share it.
- **Stratify the split.** With one row per patient, `GroupShuffleSplit` is an unstratified random split. `StratifiedShuffleSplit` keeps test prevalence stable and is still patient-level.
- **Validate hour contiguity.** Rows are assumed to be consecutive hours. Use ICULOS (still never as a predictor) in `validate_patient_frame` to check that it increases by 1 per row.
- **Decouple the CNN step.** `run_cnn_experiment` reads and rewrites the tabular `test_predictions.csv`, `metrics.json` and `run_config.json`. Have each model return its predictions and let the CLI write the artifacts once.
- **Trim `validate` output.** It prints every filename, about 40,000 lines on the full data. Print counts, and list files only with `--verbose`.
- **Cache loaded data.** Every command reloads all PSV files. Cache them as one parquet file keyed on the file list and modification times.
- **Label the synthetic demo.** Its positives are separable by construction, so its metrics are near perfect. Add a banner to its report saying it is a smoke test, not a result.
- **Record package versions.** Write `pip freeze` or the key package versions into `run_config.json`, as the design doc already recommends.

## Suggested order

If time is short, do 1, 2, 3 and 4 together, because they define the outcome and the operating point. Then do 5, which is cheap and gives the most informative result. Items 6 and 7 make the model comparison trustworthy. Item 8 and the item 9 fixes can follow.

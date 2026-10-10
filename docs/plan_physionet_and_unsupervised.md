# Plan: PhysioNet 2019 Data Integration and Unsupervised Exploration

10 Oct 2026 · Scott Pryde

This plan covers two pieces of work that ship in order:

- **Part A** runs the existing pipeline on the real PhysioNet 2019 training data (sets A and B, already downloaded to `data/raw/`) and compares it with the synthetic demo.
- **Part B** adds the `explore` command from the unsupervised handover plan, corrected against the current code (see [Corrections to the handover plan](#corrections-to-the-handover-plan)).

Part B depends on Part A. Its hospital-share risk check needs the hospital tag added in A1. Its 15-minute acceptance target needs the load speed-up in A2. Its split reuse needs the split helper extracted in A4.

Out of scope for both parts: changes to the supervised models themselves, clinical interpretation, and deployment.

---

## Facts this plan is based on

These come from profiling all 40,336 raw files against the current cohort rules (6-row look-back, 24-hour horizon).

| | Synthetic demo | PhysioNet A + B |
|---|---|---|
| Patients | 72 | 40,336 (A: 20,336, B: 20,000) |
| Hourly rows | 2,160 | 1,552,210 |
| Rows per patient | always 30 | median 38; 13,716 below 30 |
| Patients ever positive | 25% | 7.3% (A 8.8%, B 5.7%) |
| Eligible cohort | 72, no exclusions | 27,164 |
| Excluded: incomplete negative follow-up | 0 | 12,439 (31%) |
| Excluded: positive in rows 0-5 | 0 | 733 |
| Cohort prevalence | 25% | **2.8%** (A 3.6%, B 2.0%) |
| Variables with any data | 6 vitals, fully observed | all 39; labs 82-99% missing |
| Expected test set (20%) | 15 patients, 4 positive | about 5,400 patients, about 150 positive |
| Supervised AUROC | 1.0 (by construction) | not yet run |

Other findings:

- All files pass the existing validator: 41 columns each, all numeric, no `ICULOS` gaps.
- **Row 0 is not always ICU hour 1.** `ICULOS` starts after 1 for 22% of patients. Among eligible patients that is 36% in set A and 6% in set B, with a maximum offset of 304 h.
- Parallel reading of the files took 181 s. The pipeline reads them serially, and every app model button triggers a fresh load.
- The pipeline reads one flat folder; `data/train` does not exist yet.
- `docs/experimental_design.md` was deleted in commit `13d5c18`, but the README still links to it. `IMPROVEMENTS.md` does not exist in the repo.

---

## Part A: Integrate the PhysioNet data

### A0. Housekeeping (0.5 day)

- Create branch `feature/physionet-integration`.
- Remove the README link to `docs/experimental_design.md`, and drop the handover plan's references to `IMPROVEMENTS.md`. Neither file is recreated.

### A1. Multi-folder loading with a hospital tag (1 day)

- `load_patient_files` accepts one or more directories; `--data-dir` becomes `nargs="+"`.
- Add `source: str` to `PatientRecord`, set from the parent folder name (`training_setA` / `training_setB`). The existing single-folder use keeps working, with the folder name as the source.
- Patient IDs stay as the filename stem. A and B do not clash (`p0…` vs `p1…`). Add a duplicate-ID check anyway.
- Carry `source` through `create_examples` as a returned Series. It is not a predictor.
- Result: no `--flat` copy of 40k files is needed.

### A2. Load speed and caching (1 day)

- Parallel file reading in `load_patient_files` (process pool, deterministic order preserved).
- New `sepsis-pipeline cache --data-dir … --output data/cache/physionet2019.pkl`. It writes one long table (patient_id, source, row, columns) after validation. Every command gets `--cache` as an alternative to `--data-dir`.
  - Decision: pandas pickle (no new dependency; the file is locally generated and trusted) or Parquet (needs `pyarrow` as an optional extra). Recommended: Parquet if `pyarrow` installs cleanly, otherwise pickle.
- Vectorise `eda.run_eda`. The per-column × per-record loop does about 1.6M `to_numeric` calls on the real data. Concatenate once and group instead.
- The app's per-model runs pass the cache path so each button does not reload 40k files.

### A3. Look-back window definition (0.5 day, needs a decision)

The current design uses rows 0-5. For 36% of eligible set-A patients those rows are not ICU hours 1-6.

- **Option 1 (recommended):** keep rows 0-5, which matches what the PhysioNet records contain at prediction time. Record `ICULOS` at row 0 per patient as an audit field (not a predictor) and report its distribution by hospital.
- **Option 2:** require `ICULOS` at row 0 to be ≤ some limit (e.g. 2), and add a new exclusion reason `late_record_start`.
- Either way, write the choice into `DESIGN` and `run_config.json`.

### A4. Evaluation modes: pooled and cross-hospital (1.5 days)

- Extract the split logic from `run_experiment` (`modeling.py` around line 206) into a public `split_patients(labels, groups, …)`. It returns the fit/validation/test ID lists that are already written to `split_patients.json`. Part B reuses this.
- New flag `--split-mode pooled|hospital`.
  - `pooled` (default): the current stratified random split across A + B. Write each split's hospital mix into `split_patients.json`.
  - `hospital`: `--train-sources training_setA --test-sources training_setB`. Fit and validation patients are drawn from the training hospital only, and the test set is the whole other hospital's eligible cohort.
- The CNN uses the same split helper.

### A5. Reporting for imbalanced, real data (1 day)

- Cohort table with exclusions and prevalence **per hospital**.
- AUPRC shown against the prevalence baseline (no-skill line on the PR plot, and "×baseline" in the table).
- A short note on the 31% follow-up exclusion: the cohort is ICU stays of at least 30 rows unless sepsis occurs earlier.
- The data-source section says PhysioNet rather than synthetic when `synthetic_demo` is false (already partly handled).

### A6. More realistic synthetic generator (1 day)

The synthetic demo stays as the smoke test and CI fixture, not as a benchmark. Today it never triggers any exclusion branch and leaves 33 of 39 variables empty. Extend `scripts/run_synthetic_demo.py` and the test fixtures to include:

- variable record lengths, so some patients fail follow-up or are too short
- some patients positive in rows 0-5
- sparse lab columns with realistic missingness
- `ICULOS` starting above 1 for some records
- two source folders, so A1 and A4 are exercised
- prevalence near 5-10%, so the threshold and AUPRC code meet imbalance

Keep the existing seed so runs stay deterministic.

### A7. Synthetic vs real comparison (1 day)

- New `sepsis-pipeline compare --run synthetic=outputs/synthetic_e2e/run --run pooled=outputs/physionet/pooled …`.
- Writes `comparison.csv` and a self-contained `comparison.html`. It has one column per run: patients, eligible, prevalence, exclusions, missingness summary, and AUROC/AUPRC with CIs and the prevalence baseline.
- The page states that the synthetic results verify the workflow only.

### A8. Real-data runs (0.5 day plus compute)

Output under `outputs/physionet/` (already git-ignored):

| Run | Purpose |
|---|---|
| `eda` | Raw dataset description, both hospitals |
| `pooled` (LR, RF, XGBoost, CNN 20 epochs) | Main results |
| `a_to_b`, `b_to_a` (tabular models) | Generalisation across hospitals |
| `pooled_h12` | Sensitivity to the 31% follow-up exclusion |
| `compare` | Synthetic vs real summary |

### Part A acceptance

- [ ] `validate`, `eda` and `run` work on `data/raw/training_setA data/raw/training_setB` without a flat copy.
- [ ] A cached run starts training within 30 s of launch.
- [ ] Pooled and both cross-hospital runs complete; reports show per-hospital cohorts and prevalence baselines.
- [ ] The synthetic demo triggers every exclusion reason and both sources.
- [ ] Existing 16 tests pass, plus new tests for multi-folder loading, the split helper and hospital split mode.

**Part A estimate: about 8 developer-days.**

---

## Part B: Unsupervised exploration

The handover plan "Unsupervised Exploration for project_sepsis" (10 Oct 2026) is accepted in substance: methods M1-M4, deliverables D1-D7, report integration and the test list. Implement it with the corrections below.

### Corrections to the handover plan

| Handover plan says | Actual state | Change |
|---|---|---|
| 312 engineered features | 287 (34 dynamic × 8 + 5 static × 3) | Don't hard-code a count; read `feature_names.json` |
| `modeling.split_patients` exists | Split is inline in `run_experiment` | Extract it in A4 first |
| Split is train/test | Split is **fit / validation / test** | Fit all unsupervised steps on **fit** patients. Use validation patients for the stability check (assign them and compare with a refit). Describe outcomes on test only |
| Reuse `evaluation.py` helpers for AUROC/AUPRC/bootstrap | Those helpers are private (`_metric_pair`, `_interval`); only `compare_models` is public | Make `metric_pair` and `bootstrap_interval` public, or score via `compare_models` |
| Drop constant features generated for static variables | Statics have no mean/min/max/std/delta. On real data `Age`, `Gender`, `HospAdmTime` are never missing, so their 6 missingness features are constant. `Unit1/Unit2_any_missing` duplicates `_missing_fraction` | Drop zero-variance features on fit patients generically. Also drop exact duplicate columns |
| IMPROVEMENTS item 2: open-ended outcome horizon | Already fixed: 24-hour horizon with complete negative follow-up | Close that open decision and its risk row. Replace the risk with "31% short-stay exclusion"; report median record length per cluster anyway |
| Read `IMPROVEMENTS.md`, update `docs/experimental_design.md` | Neither file exists | Resolved by A0 |
| Download with `--flat`; validate `data/train` | Data is in `data/raw/training_setA` and `training_setB` | Use multi-folder loading (A1) |
| "Existing 12 tests still pass" | 16 tests | Update the count |
| Report each cluster's share from each hospital | Records carry no hospital tag | Provided by A1 |
| Full run under 15 min on a laptop | Reading files alone took 3 min in parallel. Silhouette is O(n²) on about 16k fit patients and is repeated for 7 values of k | Use the A2 cache. Compute silhouette on a fixed seeded sample (e.g. 5,000). Stability resamples refit k-means only at the chosen k |
| Isolation Forest score vs outcome | `score_samples` is higher for *normal* points | Negate it so higher = more anomalous before computing AUROC |
| Report must contain no `https://` | Confirmed: the current report source has none | Keep the test as written |

### Additional design points

- **Split modes.** `explore` follows the `--split-mode` of the run whose `split_patients.json` it reuses. In `hospital` mode, clusters are fitted on one hospital and test patients from the other are assigned to them. This directly tests whether the subgroups carry over.
- **Missingness sensitivity.** Run `--feature-set measured` by default alongside `all` on the real data. Missingness is the main expected confounder (labs 82-99% missing).
- **Small clusters.** With about 150 test positives, require an expected count ≥ 5 per cell for the chi-square test. Otherwise report Fisher-Monte Carlo or skip the test and say so.
- **Open decision, Isolation Forest placement.** Show it only in the new section, with the top supervised model as a reference line. Don't add it as a row in the supervised comparison table, which ranks fitted classifiers.
- **Open decision, cluster membership as a feature.** Defer it. If adopted, fit it inside the supervised fit split.

### Work breakdown

| WP | Work | Days | Depends on |
|---|---|---|---|
| B1 | Shared preprocessing, split reuse, `explore` CLI skeleton, `explore_config.json` | 1.5 | A1, A2, A4 |
| B2 | PCA and optional UMAP, `explore` extra in `pyproject.toml` | 1 | B1 |
| B3 | k-means / GMM, k selection, stability, cluster profiles incl. hospital share | 2 | B2 |
| B4 | Isolation Forest baseline, public evaluation helpers | 0.5 | B1 |
| B5 | Report section and exploration lane in the diagram | 1.5 | B3, B4 |
| B6 | Tests, docs, real-data runs (pooled and A→B) | 1.5 | B5 |
| B7 | Sequence autoencoder (stretch) | 3 | B6 |

**Part B estimate: about 8 developer-days, plus 3 for the stretch goal.**

### Part B acceptance

The handover plan's automated test list, plus:

- [ ] Hospital-split exploration runs (fit on A, describe on B).
- [ ] The cluster table includes hospital share and median record length.
- [ ] `measured`-only results are reported alongside `all`.
- [ ] Full real-data `explore` runs in under 15 minutes from the cache.

---

## Sequence and total

```
A0 ─ A1 ─ A2 ─┬─ A3 (decision) ─ A4 ─ A5 ─ A8 ─ A7
              └─ A6 (parallel with A3-A5)
                         A4 ──────────────── B1 ─ B2 ─ B3 ─┬─ B5 ─ B6 ─ (B7)
                                             B1 ─ B4 ──────┘
```

- Part A: about 8 days. Part B: about 8 days (+3 stretch). Total about 16 developer-days; about 12 elapsed with two people, since A6 and B4 can run in parallel.
- Use one branch per part (`feature/physionet-integration`, then `feature/unsupervised-explore` off it), merging A before B3 starts.

## Decisions (resolved 10 Oct 2026)

1. **A3:** keep the look-back as the first six rows. Record `ICULOS` at row 0 per patient for audit only.
2. **A2:** the cache is Parquet, with `pyarrow` in a new `data` optional extra.
3. **A0:** remove references to `docs/experimental_design.md` and `IMPROVEMENTS.md` rather than recreating them.
4. **Part B:** the Isolation Forest goes only in the new report section; cluster membership as a supervised feature is deferred.

## Progress

Branch `feature/physionet-integration`:

- [x] A0: README link to the deleted design doc removed
- [x] A1: `--data-dir` takes several folders; records carry `source`; duplicate IDs rejected
- [x] A2: parallel loading, `cache` command and `--cache` on every command, vectorised features and EDA, `data` extra
- [x] A3: `ICULOS` at row 0 in `cohort_audit.csv` and `cohort_summary.json`; anchor recorded in `DESIGN`
- [x] A4: public `split_patients`, `--split-mode pooled|hospital`, split composition in `run_config.json`, app passes the options through
- [x] A5: per-hospital cohort table, prevalence, and late-start count in the report; no-skill line on the PR plot; `auprc_over_prevalence` in `model_comparison.csv`
- [x] A6: synthetic demo now has 160 patients in two sources, triggers all four exclusions, has sparse labs, late starts, and 15% prevalence, and fails if any of that stops being exercised
- [x] A7: `compare` command writing CSVs and a self-contained `comparison.html`
- [ ] A8: pooled, A→B and B→A done (tabular models); still to run: CNN on pooled, `pooled_h12` sensitivity run

Branch `feature/unsupervised-explore` (off `256e0d2`):

- [x] B1: `explore` command; reuses `split_patients.json` (and checks the horizon matches) or makes the same seeded split; fit-only preprocessing that drops constant and duplicate columns and caps scaled values at ±10 SD
- [x] B2: PCA (90% variance, at most 20 components); optional UMAP with a fallback recorded in the config; `explore` extra
- [x] B3: k-means (silhouette) and Gaussian mixture (BIC); stability as bootstrap ARI on validation patients; cluster profiles with Wilson intervals, key medians, missingness, hospital share, record length; chi-square only when expected counts are at least 5; k at the edge of the range is flagged
- [x] B4: Isolation Forest scored with the new public `evaluation.score_with_bootstrap`
- [x] B5: "Unsupervised exploration" report section and an exploration lane in the pipeline diagram
- [x] B6: `tests/test_unsupervised.py` (planted clusters, fit-only, label blindness, reproducibility, shared split, one-class, UMAP optional, CLI artifacts, report); README; real-data run
- [ ] B7: sequence autoencoder (stretch, not started)

Branch `feature/data-quality` (off `72607d6`):

- [x] `review` command and `data_review.py`: field-by-field review page (introduction with test rationale and pipeline diagram, recommendations, collapsible field sections, cross-field and record-level checks, rule impact)
- [x] `quality.py`: field specs with hard limits and rules Q1-Q6, applied by default in `eda`, `run`, `explore`; audited in `quality_audit.json`, `run_config.json`, and the report
- [x] Application: "Data review and quality rules" section, `/data-review.html`, `/api/review`, and a Run data review job
- [x] End-to-end rerun on PhysioNet with the standard rules; pre-rule results kept in `outputs/physionet/no_rules/`

Measured on the full data: building the cache takes 400 s once (16 MB file); `validate` from the cache takes 35 s, and `eda` 58 s.

## Data and citation

PhysioNet/CinC Challenge 2019, version 1.0.0, open access: <https://physionet.org/content/challenge-2019/1.0.0/>. Cite as the project page asks. Data and outputs stay out of version control (`data/` and `outputs/` are git-ignored).

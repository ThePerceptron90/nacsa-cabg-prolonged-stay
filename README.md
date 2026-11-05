# Staged prediction of prolonged length of stay after isolated CABG

Code for the paper **"Prolonged length of stay after coronary artery bypass grafting: staged, interpretable prediction
model development and temporal evaluation in a national registry"** (Soleimani H, Dastjerdi P, Ebrahimi P, Comanici M,
Rajakaruna C, Mohamed Ahmed E, Angelini G, Fudulu D; submitted to *European Heart Journal - Digital Health*).
Archived version: [TO ADD: Zenodo DOI]. How to cite: see `CITATION.cff`.

Analysis code for a study developing and temporally validating staged models
(pre-operative, intra-operative and early post-operative) that predict prolonged
post-operative length of stay among patients discharged alive after isolated
coronary artery bypass grafting, using the UK National Adult Cardiac Surgery
Audit (NACSA).

**This repository contains code only. It contains no patient data.**

## Data access

The analysis uses an extract of NACSA, held by the National Institute for
Cardiovascular Outcomes Research (NICOR). The data cannot be shared by the
authors. Researchers can apply for access to NICOR directly.

## Study design (summary)

- Cohort: isolated CABG records in NACSA, 1996-2019.
- Development: 2012-2016. Temporal validation: 2017-2019. Records from
  1996-2011 are used only in a pre-specified training-period comparison.
- Primary outcome: post-operative stay over 10 whole days among patients
  discharged alive (in-hospital deaths excluded).
- Sensitivity outcomes: stay over 10 days or in-hospital death (composite), and
  stay over 10 days among all patients.
- Three prediction moments: pre-operative, intra-operative (applied at the end of surgery) and early
  post-operative (day 1-2); a fourth, complication-aware model is a ceiling analysis only.

## Decision numbers in the code

Comments in the scripts cite numbered decisions ("decision 87"). They refer to the project's decisions log, kept with
the reasons for every analytic choice; the decisions that shape the analysis are described in the protocol
(`docs/protocol/`) and the Supplementary Methods, and changes after the protocol freeze are listed there (S15).

## Repository layout

```
scripts/     one commented .py script per analysis step (readable version)
notebooks/   the same steps as Colab / Jupyter notebooks (generated from scripts/)
code/        shared helper module(s) imported by every step from Step 4 on
tools/       utilities, e.g. converting a script into a notebook
```

## Steps

| Step | Script | What it does |
|---|---|---|
| 1 | `01_first_look` | One-line summary of every column; counts by year and country |
| 2 | `02_deep_audit` | Outcome checks, missingness by year, cross-tabulations of overlapping fields, plausibility |
| 3 | `03_clean` | Cleaning rules (each logged with records changed), derived variables, clean analysis file |
| 4 | `04_cohort_profile` | Flow counts, Table 1 and full baseline tables with SMDs, missingness, category counts, predictor associations and VIF, outcome rates by year, sex and age, draft variable dictionary |
| 5 | `05_predictor_spec` | Predictor sets for the four models, rule checks (missingness, sparse levels), excluded candidates with reasons, regression terms per model, variable dictionary (ST1) |
| 6a | `06a_sample_size` | Minimum sample size for development (Riley 2020) and validation (Riley 2021), anchored on EuroSCORE II's C-statistic, with pessimistic and optimistic scenarios (ST7) |
| 6b | `06b_missing_data` | Evidence for the missing-data plan: missingness by year, outcome by missing status, blank vs recorded IABP and post-op fields, co-missingness, complete cases per model (no imputation here; the imputer is built at Step 8) |
| 8 | `08_infrastructure` | Shared modelling tools in `code/nacsa_modelling.py`, each checked: 8.1 stage loader (patients, predictors, outcome at 7/10/14 days, extras kept apart); 8.2 fixed folds (5 outer, 75/25 inner and final splits, seed 20261007, one set for all models, stages and outcomes); 8.3a stage-ordered deterministic imputer; 8.3b model input recipes (coding, restricted cubic splines, scaling); 8.4a lowess calibration curve and ICI/E50/E90; 8.4b AUC, PR-AUC, calibration intercept and slope, Brier, net benefit, paired bootstrap; 8.4c top 10%/20% classification; 8.4d shared figures (`code/nacsa_plots.py`; examples on made-up data in `docs/figure_style/`); 8.5 model wrappers M0-M5 with search spaces in `config/search_spaces.json` and the calibration step; 8.5f tuning harness (`code/nacsa_tuning.py`: Optuna TPE, one seeded study per model, stage and round, saved as JSON). Timing test in `08b_timing_test`. All 173 checks pass; Step 8 complete (8 Oct 2026) |
| 8.6 | `08b_timing_test` | Timing test (decision 135): one real round (outer round 1, complication-aware stage, 2012-16 only), searches side by side; time only, no performance. Projection for the main analysis about 4 h on 4 cores plus about 1.3 h of bootstraps (upper bound); round-1 searches kept in `outputs/tuning/primary/` for reuse |
| 9 | `09_baseline_models` | Baseline models M0 EuroSCORE II, M1 logistic, M2 ridge splines through the 5 outer rounds, all stages (decisions 137-139): out-of-fold risks saved on the machine only; fitted imputers and recipes saved for Step 10 with input fingerprints; technical checks only, no performance measures until all six models are built. Ridge penalty at the grid's smallest value in all 20 rounds (effectively unpenalised, as expected) |
| 10 | `10_ml_models` | Machine-learning models M3 XGBoost, M4 random forest, M5 neural network through the 5 outer rounds, all stages: Step 9 inputs reloaded and verified by fingerprint (320/320), 60 searches of 30 trials (2.1 h, 4 processes), winners refitted with their calibration step; out-of-fold risks saved on the machine only; 7/7 technical checks; no performance computed yet (decision 139) |
| 10b | `10b_internal_scoring` | First performance results (all six models built first, decision 139): 2012-16 out-of-fold scoring with 2,000 paired bootstrap resamples (ST9), AUC per outer fold, decision 88 headline selection with the decision 140 safety rule, internal stage comparisons, development figures SF15-SF18. Headline model: XGBoost at all three prediction stages |
| 11 | `11_final_models` | Final models (6 algorithms x 4 stages) tuned on the final split and fitted on all 2012-16 patients; locked with sha256 fingerprints (`docs/locks/final_models_sha256.txt`); model files kept off GitHub (decision 142); 5/5 checks; no performance on 2012-16 |
| 12 | `12_temporal_validation` | Locked final models (fingerprints checked) applied once to 2017-19: scoring overall and by year (2,000 paired bootstrap), stage verdicts (decision 87, safety rule decision 140), updating scenario (recalibrated on 2017, tested on 2018-19; decision 97), validation figures (versions A/B, calibration axis 0-70%) |
| 14 | `14_training_periods` | Training-period comparison (decision 18): sets B (2012-16, all-era predictors) and C (1996-2016, all-era) for ridge and XGBoost, tuned and locked like Step 11 (fingerprints in `docs/locks/`), tested once on 2017-19 with set A (main models); paired bootstrap, descriptive |
| 15 | `15_explainability` | Explainability (Step 15): exact TreeSHAP for XGBoost (pre-op, end of surgery, early post-op; 5,000 sampled 2017-19 patients; per variable and pre-specified clinical groups; beeswarm and dependence plots, plus top-10 dependence plots per stage, SF12b, decision 144) and ridge odds ratios with spline curves from the locked models |
| 16a | `16a_outcome_sensitivity` | Sensitivity S1-S4 (decisions 99, 145): composite outcome, all patients, cut-offs over 7 and over 14 days; XGBoost re-tuned and EuroSCORE II recalibrated per stage, locked (`docs/locks/sensitivity_models_sha256.txt`), tested once on 2017-19 with paired bootstrap; 29/29 checks |
| 16b | `16b_mice_sensitivity` | Sensitivity S5 (decisions 78, 146): proper multiple imputation (bootstrap + random draws, `MiceImputer`, worker `code/nacsa_mice.py`) for M1 and M2 at all four stages; 10 imputations, 48 for the complication-aware stage by the decision 78 FMI rule; Rubin's rules; paired comparison with the main models; 5/5 checks |
| 16c | `16c_complete_case_sensitivity` | Sensitivity S6 (decisions 74, 147): all six models re-developed on complete cases (Step 11 method), locked, tested on complete-case 2017-19 patients alongside the main models; 11/11 checks |
| 16d | `16d_sensitivity_summary` | ST16 (main analysis and S1-S6 in one table) and SF19 forest plot of 2017-19 AUC (decision 148) |
| 17 | `17_fairness_subgroups` | Fairness and subgroups (decisions 36, 149, 150): XGBoost, ridge splines and EuroSCORE II by sex, age band and urgency on 2017-19 (AUC, O/E, calibration, error rates at a single top-10%/20% cut-off), 2,000 resamples, differences from reference groups; SF13a subgroups, SF13b calibration by year; 6/6 checks |
| 18 | `18_final_exhibits` | Final exhibits (decisions 151-153, 159, 164, 168, 170, 173-175): Tables 1-3 (Table 3 = stage verdicts with AUC and net-benefit differences, paired CIs from the Step 12 replicates) and Figures 1-6 (Figure 5 = SHAP importance and beeswarm, Figure 6 = ridge splines vs SHAP dependence, both from Step 15; Figures 2-4 redrawn with panel letters from the Step 12 risks), scripted graphical abstract (four take-home messages, each checked against the outputs); SF5-SF8 redrawn from the Step 12 risks; moment labels pre-operative, intra-operative, early post-operative applied to every table (relabel); every supplementary table and figure with final numbers, compact estimate (95% CI) tables, in one Word and PDF supplement (`supplementary_material.docx/.pdf`, ST8c last) and `main_text_exhibits.docx/.pdf`; SF1-SF3 from the clean file (1996-2019), new SF9/SF10; reporting checklists with manuscript locations: ST18 TRIPOD+AI (+ for Abstracts), ST22 RECORD, ST23 CODE-EHR, and ST19 PROBAST+AI (JSON in `docs/checklists`, `docs/supplement`; CSV is blocked by .gitignore); Supplementary Methods (S1-S16; S15 = protocol deviations, S16 = further TRIPOD+AI items) from `docs/supplement/extended_methods.md` with numbered references (`key_references.json`) and the ST21 glossary; re-run after Step 19 to add page numbers. Needs Steps 3 (cleaning tables from Drive), 4, 5, 6a outputs |

Later steps (cohort profile, predictor specification, modelling, validation,
explainability, sensitivity analyses) are added as the analysis progresses.

## How to run (original Google Colab workflow)

1. Put the data extract in `MyDrive/nacsa/raw/` on Google Drive.
2. Open a notebook in Google Colab (File > Open notebook > GitHub, or upload it).
3. Steps 1-3: set `DATA_FILE` in the first code cell. From Step 4: put
   `code/nacsa_common.py` in `MyDrive/nacsa/code/` first (every later step
   imports it).
4. Run the cells top to bottom.
5. Outputs are written to `MyDrive/nacsa/outputs/<step>/`; the clean analysis
   file to `MyDrive/nacsa/processed/`.

Shared outputs follow privacy rules: counts from 1 to 9 are shown as "<10" and
no minimum or maximum values are reported.

Notebooks are generated from the scripts with:

```
python tools/py_to_ipynb.py scripts/03_clean.py notebooks/03_clean.ipynb
```

## Software

Python 3.13 with the packages in `requirements.txt` (versions are pinned as the
analysis progresses).

## Reproducing the analysis

1. Obtain the NACSA extract from NICOR and place the cleaned file at `$NACSA_DIR/processed/nacsa_clean.csv.gz`
   (Step 3 builds it from the raw extract).
2. Install the pinned packages: `pip install -r requirements.txt` (Python 3.13).
3. Run the steps in order: `NACSA_DIR=<data folder> NACSA_CODE_DIR=<this repo>/code python3 scripts/<step>.py`.
   Each step writes aggregate outputs to `$NACSA_DIR/outputs/<step>/`; patient-level working files stay in
   `$NACSA_DIR/work/` and are never shared. All random steps use fixed seeds.
4. `scripts/18_final_exhibits.py` rebuilds every table and figure of the paper and its supplement;
   `docs/manuscript/build_ms.py` builds the manuscript text.

The 2017-2019 evaluation data load only with `allow_validation=True`, in the evaluation steps (12-14 and later).
The protocol (`docs/protocol/`) was frozen on 7 October 2026 before any model was fitted; later changes are listed in
the Supplementary Methods (S15).

## Manuscript and submission files

- `docs/manuscript/`: manuscript text (`ms_text.py`) and Word build (`build_ms.py`; `REVIEW=1` places the figures,
  `SUBMIT=1` drops the author notes).
- `docs/submission/`: cover letter and submission-package builders.
- `docs/checklists/`: TRIPOD+AI, RECORD and CODE-EHR checklists (built into the supplement).

## Licence

MIT (see `LICENSE`).

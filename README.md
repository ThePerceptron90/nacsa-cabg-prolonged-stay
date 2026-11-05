# Staged prediction of prolonged length of stay after coronary artery bypass grafting

[![Licence: MIT](https://img.shields.io/badge/licence-MIT-blue.svg)](LICENSE)
![Python 3.13](https://img.shields.io/badge/python-3.13-blue.svg)
![Reporting: TRIPOD+AI](https://img.shields.io/badge/reporting-TRIPOD%2BAI-green.svg)
![Paper: under review](https://img.shields.io/badge/paper-under%20review-orange.svg)

Analysis code for the paper:

> Soleimani H, Dastjerdi P, Ebrahimi P, Comanici M, Rajakaruna C, Mohamed Ahmed E, Angelini G, Fudulu D.
> **Prolonged length of stay after coronary artery bypass grafting: staged, interpretable prediction model
> development and temporal evaluation in a national registry.** Submitted to *European Heart Journal - Digital Health*.

Archived version (Zenodo): DOI to be added at release. See [`CITATION.cff`](CITATION.cff) for how to cite.

**This repository contains code only. It contains no patient data and no patient-level results.**

---

## The study at a glance

| | |
|---|---|
| **Question** | How well can a hospital stay of more than 10 days after isolated coronary artery bypass grafting be predicted, and does machine learning do better than regression? |
| **Data** | UK National Adult Cardiac Surgery Audit (NACSA), held by the National Institute for Cardiovascular Outcomes Research (NICOR) |
| **Development** | Operations in 2012-2016 (75,814 patients discharged alive) |
| **Temporal evaluation** | Operations in 2017-2019 (32,104 patients), used once, after the models were locked |
| **Prediction moments** | Pre-operative; intra-operative (applied at the end of surgery); early post-operative (day 1-2). A fourth, complication-aware model estimates the ceiling of achievable accuracy. |
| **Models** | Recalibrated EuroSCORE II (benchmark), logistic regression, ridge regression with restricted cubic splines, XGBoost, random forest and a multilayer perceptron |
| **Evaluation** | Discrimination, calibration, net benefit (decision curves), drift by year, a model-updating scenario, SHAP explanations, subgroup (fairness) analyses and six sensitivity analyses |
| **Reporting** | TRIPOD+AI, RECORD and CODE-EHR checklists; PROBAST+AI self-assessment |

The protocol and statistical analysis plan were frozen on 7 October 2026, before any model was fitted
([`docs/protocol/`](docs/protocol/)). Changes made afterwards are listed with their reasons in the Supplementary Methods (S15).

## Repository structure

```
scripts/      one commented Python script per analysis step (the main, readable version)
notebooks/    the same steps as Jupyter notebooks, generated from scripts/
code/         shared modules: data loading and privacy rules (nacsa_common.py), modelling (nacsa_modelling.py),
              tuning (nacsa_tuning.py), plots (nacsa_plots.py), bootstrap scoring (nacsa_scoring.py) and
              multiple imputation (nacsa_mice.py)
config/       hyperparameter search spaces, fixed before tuning
docs/
  protocol/      frozen protocol (v1.0) and the current version (v2.1)
  supplement/    Supplementary Methods text, references and supplementary reference tables
  checklists/    TRIPOD+AI, RECORD and CODE-EHR checklists (built into the supplement)
  manuscript/    manuscript text and the script that builds the Word file
  submission/    cover letter and submission-package builders
  locks/         SHA-256 fingerprints of every locked model (proof they were not changed after locking)
  figure_style/  example figures drawn from made-up data, used to agree the figure style
tools/        utility that turns a script into a notebook
```

## Analysis pipeline

Each step is one script. Steps run in order; each reads the outputs of earlier steps.

| Step | Script | What it does |
|---|---|---|
| 1-2 | `01_first_look`, `02_deep_audit` | Audit of the raw extract: completeness by year, outcome checks, consistency and plausibility |
| 3 | `03_clean` | Cleaning rules (each logged), derived variables, the clean analysis file |
| 4 | `04_cohort_profile` | Patient flow, Table 1, missing data, predictor associations, outcome rates over time |
| 5 | `05_predictor_spec` | Predictor sets for each prediction moment and the variable dictionary |
| 6a-6b | `06a_sample_size`, `06b_missing_data` | Minimum sample size (Riley et al.) and the evidence behind the missing-data plan |
| 8 | `08_infrastructure`, `08b_timing_test` | Shared modelling tools, each tested on made-up data; run-time test |
| 9-10b | `09_baseline_models`, `10_ml_models`, `10b_internal_scoring` | Nested cross-validation on 2012-2016: regression and machine-learning models, internal performance, choice of the headline model |
| 11 | `11_final_models` | Final models fitted on all 2012-2016 patients and locked with fingerprints |
| 12 | `12_temporal_validation` | Locked models applied once to 2017-2019: performance, stage-to-stage comparisons, drift by year, updating scenario |
| 14 | `14_training_periods` | Does adding older (1996-2011) data help? |
| 15 | `15_explainability` | SHAP explanations of XGBoost and ridge-regression curves |
| 16a-16d | `16a`-`16d` sensitivity scripts | Alternative outcome definitions, multiple imputation, complete cases, summary |
| 17 | `17_fairness_subgroups` | Performance by sex, age band and urgency of operation |
| 18 | `18_final_exhibits` | Every table and figure of the paper and its supplement, the supplement as Word and PDF, and the graphical abstract |

## Reproducing the analysis

**1. Get the data.** The data cannot be shared by the authors. Researchers can apply to NICOR for NACSA data.

**2. Install the software** (Python 3.13):

```bash
pip install -r requirements.txt
```

Converting the supplement from Word to PDF also needs LibreOffice.

**3. Tell the scripts where things are.** Two settings are given in front of each command:

- `NACSA_DIR`: your data folder. Put the clean data file at `$NACSA_DIR/processed/nacsa_clean.csv.gz` (Step 3 creates it from the raw extract).
- `NACSA_CODE_DIR`: the `code/` folder of this repository, so every script can find the shared modules.

**4. Run the steps in order**, for example:

```bash
NACSA_DIR=/path/to/data NACSA_CODE_DIR=/path/to/this/repo/code python3 scripts/12_temporal_validation.py
```

Each step writes its **aggregate** results (tables, figures, logs) to `$NACSA_DIR/outputs/<step>/`. Files that
describe individual patients (for example, each patient's predicted risk) go to `$NACSA_DIR/work/` and are never shared.
Every random step uses a fixed seed, so a rerun gives the same numbers.

**5. Build the paper's exhibits.** `scripts/18_final_exhibits.py` rebuilds every table and figure and the
supplement; `docs/manuscript/build_ms.py` builds the manuscript Word file from its text.

## Safeguards built into the code

- **Evaluation lock:** the 2017-2019 patients load only with an explicit switch (`allow_validation=True`), used only
  after the models were locked, so no later-period outcome influenced any modelling choice.
- **Locked models:** every final model was fingerprinted (SHA-256, `docs/locks/`) before evaluation.
- **Small-number suppression:** shared outputs show counts of 1-9 as "<10" (with secondary suppression), hide rates
  based on them, and never report minimum or maximum values, following the data provider's rules.
- **No data in the repository:** `.gitignore` blocks data and output files.

## Citation

If you use this code, please cite the paper and the archived software (see [`CITATION.cff`](CITATION.cff)).

## Licence

MIT, see [`LICENSE`](LICENSE).

## Acknowledgements

We thank NICOR and the cardiac surgical units contributing to the National Adult Cardiac Surgery Audit. The analysis
code was written and run with the help of an AI assistant (Claude, Anthropic) under the authors' direction, as
disclosed in the paper. Comments in the scripts cite numbered decisions from the project's decisions log; the decisions
that shape the analysis are described in the protocol and the Supplementary Methods.

## Contact

Hamidreza Soleimani, Bristol Heart Institute, University of Bristol (ec26229@bristol.ac.uk).

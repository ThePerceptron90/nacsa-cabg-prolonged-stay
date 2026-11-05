# =============================================================================
# NACSA PLoS project  |  Step 8: Shared modelling infrastructure
# -----------------------------------------------------------------------------
# HOW TO USE THIS FILE
#   Claude runs this file in its own workspace (decision 56). To run it
#   yourself in Colab, copy ONE block at a time (each starts with "# %%") into
#   a cell, or open the .ipynb version, and run top to bottom.
# =============================================================================

# %% [markdown]
# # Step 8: Shared modelling infrastructure
#
# **What this step does:** it builds and checks the parts every model shares,
# so all models get the same patients, folds, preprocessing and metrics. The
# shared code lives in `code/nacsa_modelling.py`; this script runs its checks
# and writes aggregate check tables. The analysis plan was frozen on
# 7 Oct 2026 (decision 100c); nothing here changes a method.
#
# Parts (added as each is agreed):
#
# | Part | What | Status |
# |---|---|---|
# | 8.1 | Stage loader `get_stage_data()` (decision 111) | done |
# | 8.2 | Fixed cross-validation folds (decision 112) | done |
# | 8.3 | Shared preprocessing: 8.3a imputer (decisions 113, 114); 8.3b model inputs (decision 115) | done |
# | 8.4 | Tested metric functions: 8.4a calibration curve, ICI/E50/E90 (decision 116); 8.4b AUC, PR-AUC, CITL, slope, Brier, net benefit, paired bootstrap (decision 117); 8.4c top 10%/20% classification (decision 118); 8.4d shared figures (decisions 119-121) | this version |
# | 8.5 | Model wrappers and tuning: 8.5a M0, M1, M2 (decisions 123, 124); 8.5b XGBoost fitting step (decision 125); 8.5c calibration step (decision 126); 8.5d random forest (decision 127) | this version (M5 and Optuna harness to come) |
# | 8.6 | Timing test | to come |
#
# **Outputs** (in `outputs/08_infrastructure/`):
#
# | File | What it shows |
# |---|---|
# | `stage_data_check.csv` | patients, events and predictor count per stage and outcome cut-off (development only) |
# | `stage_missing_check.csv` | % missing per predictor and year in X, compared with Step 6b |
# | `folds_check.csv` | per outer fold: patients, primary events and rate, in-hospital deaths, status not recorded, % per year |
# | `inner_split_check.csv` | per round and the final split: train/tune patients, events and rate |
# | `imputation_mix.csv` | per predictor with gaps: share of each level among recorded vs filled values (median for numbers), round-1 learning part |
# | `model_input_columns.csv` | number of model columns per stage and recipe, against the Step 5 counts |
# | `spline_knots.csv` | spline knots per numeric predictor (round-1 learning part) |
# | `calibration_metric_test.csv` | ICI/E50/E90/Emax on made-up data with known answers |
# | `bootstrap_test.csv` | paired-bootstrap AUC example on made-up data (two models and a copy) |
# | `performance_example.csv` | every measure from `performance()` on made-up staircase data (shows the output format) |
# | `figures/example_*.pdf/png` | the four curve plots on made-up data (ROC versions A and B), for style review |
# | `log.txt` | run details, fold-file fingerprint and the result of every check |
#
# Privacy: aggregate only; counts of 1-9 show as "<10"; rates hidden when
# events or non-events are 1-9. Only 2012-16 patients are loaded (decision 25).

# %% [markdown]
# ## 1. Set up
#
# **What you should see:** no output.

# %%
import importlib
import os
import sys
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd

pd.set_option("display.width", 250)
pd.set_option("display.max_columns", 30)

try:
    from google.colab import drive
    drive.mount("/content/drive")
except ImportError:
    pass

CODE_DIR = os.environ.get("NACSA_CODE_DIR", "/content/drive/MyDrive/nacsa/code")
if CODE_DIR not in sys.path:
    sys.path.insert(0, CODE_DIR)
import nacsa_common as nc
import nacsa_modelling as nm
importlib.reload(nc)
importlib.reload(nm)

OUT_DIR = nc.OUTPUTS_DIR / "08_infrastructure"
OUT_DIR.mkdir(parents=True, exist_ok=True)

LOG = []
CHECKS = []


def log(msg):
    print(msg)
    LOG.append(msg)


def check(name, ok, detail=""):
    """Record a pass/fail check; stop the script at the first failure."""
    CHECKS.append((name, ok))
    log(f"[{'PASS' if ok else 'FAIL'}] {name}{': ' + detail if detail else ''}")
    if not ok:
        raise AssertionError(f"Check failed: {name}")


# %% [markdown]
# ## 2. Part 8.1: the stage loader
#
# `get_stage_data(stage)` returns X (the stage's predictors), y (the outcome)
# and extras (EuroSCORE II, year, sex, age band, urgency, record id; never
# predictors). We load the development data once and build every stage from it.
#
# **Checks:**
# 1. Development data: 75,814 patients and 12,023 events (Step 4).
# 2. Every stage has the agreed number of predictors (26 / 34 / 36 / 40).
# 3. No outcome, stay, EuroSCORE II, year or id column in X.
# 4. Same patients and same outcome in every stage.
# 5. Missingness per predictor and year matches Step 6b (so the Step 5
#    recoding is applied exactly once).
# 6. The validation lock still stops 2017-19 without `allow_validation=True`.
# 7. The 7- and 14-day outcomes (S3, S4) keep the same patients and nest
#    correctly (every >14 is also >10, every >10 is also >7).

# %%
dev = nc.load_clean(period="development")

EXPECTED_N, EXPECTED_EVENTS = 75_814, 12_023
EXPECTED_P = {"preop": 26, "end_of_surgery": 34, "early_postop": 36, "complication_aware": 40}
NON_PREDICTORS = {"y", "ES2", "Year", "period", "record_id", "los_days", "PostLOS", "Mortality",
                  "age_band", "Country"} | set(nc.OUTCOMES.values())

data = {stage: nm.get_stage_data(stage, df=dev) for stage in nm.STAGES}

rows = []
for stage, (X, y, extras) in data.items():
    check(f"{stage}: {EXPECTED_N:,} patients", len(X) == EXPECTED_N == len(y) == len(extras),
          f"{len(X):,}")
    check(f"{stage}: {EXPECTED_EVENTS:,} events", int(y.sum()) == EXPECTED_EVENTS, f"{int(y.sum()):,}")
    check(f"{stage}: {EXPECTED_P[stage]} predictors", X.shape[1] == EXPECTED_P[stage], f"{X.shape[1]}")
    check(f"{stage}: no non-predictor column in X", not (NON_PREDICTORS & set(X.columns)))
    check(f"{stage}: X, y and extras aligned", X.index.equals(y.index) and X.index.equals(extras.index))
    rows.append({"stage": stage, "outcome_cutoff_days": 10, "n_patients": len(X),
                 "n_events": int(y.sum()), "n_predictors": X.shape[1]})

ref_X, ref_y, ref_extras = data["preop"]
for stage, (X, y, extras) in data.items():
    check(f"{stage}: same patients and outcome as preop",
          extras["record_id"].equals(ref_extras["record_id"]) and y.equals(ref_y))
    check(f"{stage}: earlier-stage predictors come first, unchanged",
          X.iloc[:, :ref_X.shape[1]].equals(ref_X))

# %% [markdown]
# ### Missingness compared with Step 6b
#
# Step 6b reported % missing per predictor and year. The loader's X must give
# the same numbers (to one decimal place), which shows the Step 5 recoding
# (salvage merged into emergency; "no diseased vessels" set to missing) is
# applied once and only once. The text is compared exactly, using the same
# helper as Step 6b ("<10 records" when 1-9 are missing). Step 6b listed only
# predictors with some missing values; every other predictor must show 0.0.

# %%
X_full, _, ex_full = data["complication_aware"]
miss = X_full.isna().groupby(ex_full["Year"]).sum().T              # counts per predictor x year
n_year = ex_full["Year"].value_counts().sort_index()

step6b_file = nc.OUTPUTS_DIR / "06b_missing_data" / "missing_by_year.csv"
step6b = pd.read_csv(step6b_file, dtype=str).set_index("code")
years = [str(y) for y in range(2012, 2017)]

comparison, mismatches = [], []
for var in X_full.columns:
    for yr in years:
        n_miss, n = int(miss.loc[var, int(yr)]), int(n_year[int(yr)])
        ours = nc.pct_missing_text(n_miss, n)          # same helper Step 6b used
        # Step 6b tabulated only predictors with any missing value; the rest must be 0.0
        theirs = step6b.loc[var, yr] if var in step6b.index else "0.0 (not in 6b: never missing)"
        same = ours == theirs.split(" (")[0]
        comparison.append({"variable": var, "year": yr, "loader_pct_missing": ours,
                           "step6b_pct_missing": theirs, "match": same})
        if not same:
            mismatches.append(f"{var} {yr}: loader {ours} vs 6b {theirs}")
comparison = pd.DataFrame(comparison)
log(f"Predictors never missing in 2012-16 (absent from the Step 6b table): "
    f"{sorted(set(X_full.columns) - set(step6b.index))}")
check("missingness per predictor and year matches Step 6b exactly", not mismatches,
      f"{len(comparison)} predictor-year cells compared" + (": " + "; ".join(mismatches[:10]) if mismatches else ""))

# %% [markdown]
# ### The validation lock

# %%
try:
    nm.get_stage_data("preop", period="validation", verbose=False)
    locked = False
except PermissionError:
    locked = True
check("validation lock stops 2017-19 without allow_validation=True", locked)

# %% [markdown]
# ### Sensitivity outcomes over 7 and over 14 days (S3, S4)

# %%
y_by_cut = {}
for cut in (7, 14):
    X, y, extras = nm.get_stage_data("preop", cutoff=cut, df=dev)
    y_by_cut[cut] = y
    check(f"cutoff {cut}: same {EXPECTED_N:,} patients", len(y) == EXPECTED_N)
    rows.append({"stage": "preop (all stages share the outcome)", "outcome_cutoff_days": cut,
                 "n_patients": len(y), "n_events": int(y.sum()), "n_predictors": X.shape[1]})
y_by_cut[10] = ref_y
check("outcomes nest: >14 days implies >10 implies >7",
      bool(((y_by_cut[14] <= y_by_cut[10]) & (y_by_cut[10] <= y_by_cut[7])).all()))

try:
    nm.get_stage_data("preop", outcome="composite", cutoff=7, verbose=False)
    guarded = False
except ValueError:
    guarded = True
check("other cut-offs refused for non-survivor outcomes", guarded)

# %% [markdown]
# ## 3. Part 8.2: fixed cross-validation folds (decision 112)
#
# One fold assignment for every model, stage and outcome. It covers all
# 2012-16 patients usable in any outcome definition (survivors with a known
# stay; plus, for S1 and S2, in-hospital deaths and patients whose discharge
# status was not recorded) and is balanced on four groups: survivor with long
# stay, survivor with short stay, died in hospital, status not recorded.
# Seed 20261007 for the outer folds, seed + k for the inner 75/25 split of round
# k, seed + 6 for the final split. The fold file is patient-level and stays in
# `processed/`; only the aggregate checks below are shared.
#
# **Checks:**
# 1. Population = 75,814 survivors + in-hospital deaths + status not
#    recorded; every patient in exactly one outer fold.
# 2. Folds of near-equal size with the same event rate and share of deaths.
# 3. In every round the scoring fold is never "train" or "tune"; the inner
#    split is 75/25 with the same event rate.
# 4. Same file when rebuilt, and when the patients come in a different order.
# 5. Every primary-analysis patient (all stages) and every S1 and S2 patient
#    finds its fold.
# 6. Year mix per fold is similar (checked, not forced).

# %%
import hashlib

pop = nm.fold_population()
n_deaths, n_unknown = int((pop["stratum"] == 2).sum()), int((pop["stratum"] == 3).sum())
check("fold population = primary patients + in-hospital deaths + status not recorded",
      len(pop) == EXPECTED_N + n_deaths + n_unknown and int((pop["stratum"] < 2).sum()) == EXPECTED_N,
      f"{len(pop):,} = {EXPECTED_N:,} + {n_deaths:,} deaths + {n_unknown:,} status not recorded")

folds = nm.make_folds(pop)
check("every patient in exactly one outer fold 1-5",
      folds["record_id"].is_unique and folds["outer_fold"].between(1, nm.N_OUTER).all())

fold_path = nm.save_folds(folds)
sha = hashlib.sha256(fold_path.read_bytes()).hexdigest()
log(f"Fold file: {fold_path.name}, {len(folds):,} rows, sha256 {sha}")

again = nm.make_folds(pop)
shuffled = nm.make_folds(pop.sample(frac=1, random_state=1))
check("rebuilding gives identical folds", again.equals(folds))
check("input order does not change the folds", shuffled.equals(folds))

# Per outer fold: sizes, events (primary), deaths, year mix
fx = folds.merge(nc.load_clean(period="development", outcome=None, verbose=False)[["record_id", "Year"]],
                 on="record_id", how="left")
check("all fold patients are 2012-16 patients", fx["Year"].between(2012, 2016).all())
fold_rows = []
for k in range(1, nm.N_OUTER + 1):
    f = fx[fx["outer_fold"] == k]
    prim = f[f["stratum"] < 2]
    row = {"outer_fold": k, "patients_all": len(f), "patients_primary": len(prim),
           "events_primary": int((prim["stratum"] == 1).sum()),
           "event_rate_primary_pct": nc.rate_text(int((prim["stratum"] == 1).sum()), len(prim)),
           "deaths_in_hospital": nc.suppress(int((f["stratum"] == 2).sum())),
           "status_not_recorded": nc.suppress(int((f["stratum"] == 3).sum()))}
    for yr in range(2012, 2017):
        row[f"pct_{yr}"] = round(100 * (f["Year"] == yr).mean(), 1)
    fold_rows.append(row)
fold_table = pd.DataFrame(fold_rows)
print(fold_table.to_string(index=False))
sizes = fold_table["patients_all"]
rates = fold_table["events_primary"] / fold_table["patients_primary"]
check("outer folds differ in size by at most 1 patient", sizes.max() - sizes.min() <= 1)
check("primary event rate within 0.1 points across folds", 100 * (rates.max() - rates.min()) < 0.1,
      f"{100 * rates.min():.2f}% to {100 * rates.max():.2f}%")
year_spread = max(fold_table[f"pct_{y}"].max() - fold_table[f"pct_{y}"].min() for y in range(2012, 2017))
check("year mix similar across folds (largest spread under 1.5 points)", year_spread < 1.5,
      f"largest spread {year_spread:.1f} points")

# Inner and final splits
inner_rows = []
for col in [f"round_{k}" for k in range(1, nm.N_OUTER + 1)] + ["final"]:
    if col != "final":
        k = int(col.split("_")[1])
        check(f"{col}: scoring fold is only 'score' and nothing else is",
              ((folds[col] == "score") == (folds["outer_fold"] == k)).all())
    learn = folds[folds[col] != "score"]
    share_tune = (learn[col] == "tune").mean()
    for role in ["train", "tune"]:
        r = learn[(learn[col] == role) & (learn["stratum"] < 2)]
        ev = int((r["stratum"] == 1).sum())
        inner_rows.append({"split": col, "role": role, "patients_primary": len(r), "events_primary": ev,
                           "event_rate_primary_pct": nc.rate_text(ev, len(r))})
    rate_tr = (learn.loc[learn[col] == "train", "stratum"] == 1).sum() / ((learn[col] == "train") & (learn["stratum"] < 2)).sum()
    rate_tu = (learn.loc[learn[col] == "tune", "stratum"] == 1).sum() / ((learn[col] == "tune") & (learn["stratum"] < 2)).sum()
    check(f"{col}: tune share 25% and same event rate in train and tune",
          abs(share_tune - 0.25) < 0.001 and abs(rate_tr - rate_tu) < 0.001,
          f"tune {100 * share_tune:.1f}%, event rate {100 * rate_tr:.2f}% vs {100 * rate_tu:.2f}%")
inner_table = pd.DataFrame(inner_rows)
print(inner_table.to_string(index=False))

# Every analysis finds its folds
for stage, (X, y, extras) in data.items():
    fa = nm.attach_folds(extras, folds)
    check(f"{stage}: every primary patient has a fold", fa["outer_fold"].notna().all() and len(fa) == len(X))
for label, oc in [("S1 (composite)", "composite"), ("S2 (all patients)", "all")]:
    _, _, ex_s = nm.get_stage_data("preop", outcome=oc, verbose=False)
    check(f"{label} patients all have a fold", nm.attach_folds(ex_s, folds)["outer_fold"].notna().all(),
          f"{len(ex_s):,} patients")

# %% [markdown]
# ## 4. Part 8.3a: the imputer (decisions 77, 81-85, 113, 114)
#
# Fitted on the learning part of round 1 (outer folds 2-5) and applied to the
# scoring fold (fold 1), exactly as it will be used in cross-validation. The
# full (complication-aware) imputer covers every block; a pre-op-only imputer
# is fitted alongside to prove stage order.
#
# **Checks:**
# 1. No gaps left, in the fitted patients and in the new (scoring) patients.
# 2. Recorded values are never changed.
# 3. Every filled category is a level seen in training; every filled number
#    lies within the training range.
# 4. Stage order: pre-op values are identical whether the imputer covers only
#    the pre-op stage or all stages (decision 85).
# 5. Deterministic: fitting twice gives identical results.
# 6. The outcome is not used: shuffling y changes nothing (decision 81).
# 7. The fitted imputer holds no patient rows (deployable, decision 77).
#
# **Also reported:** rounds needed per block; for every predictor, the mix of
# recorded values vs filled values (how far "most likely category" shifts
# things, decision 113); fit time.

# %%
import time

Xc, yc, exc = data["complication_aware"]
fa = nm.attach_folds(exc, folds)
learn = (fa["round_1"] != "score").to_numpy()
X_learn, X_score, y_learn = Xc[learn], Xc[~learn], yc[learn]

t0 = time.time()
imp_full = nm.StageImputer("complication_aware").fit(X_learn)
fit_seconds = time.time() - t0
log(f"Full imputer fitted on {learn.sum():,} patients in {fit_seconds:.0f} s; rounds per block "
    f"{imp_full.rounds_used_}; settled within {nm.MAX_ROUNDS} rounds: {imp_full.converged_}")
F_learn, F_score = imp_full.transform(X_learn), imp_full.transform(X_score)

check("no gaps left (fitted and new patients)", not F_learn.isna().any().any() and not F_score.isna().any().any())
check("recorded values unchanged",
      all((F.loc[X[v].notna(), v] == X.loc[X[v].notna(), v]).all()
          for F, X in [(F_learn, X_learn), (F_score, X_score)] for v in X.columns))

bad = []
for v in Xc.columns:
    for F, X in [(F_learn, X_learn), (F_score, X_score)]:
        filled = F.loc[X[v].isna(), v]
        if filled.empty:
            continue
        if nc.VARIABLES[v]["type"] == "continuous":
            if (filled < X_learn[v].min()).any() or (filled > X_learn[v].max()).any():
                bad.append(v)
        elif not set(filled.unique()) <= set(X_learn[v].dropna().unique()):
            bad.append(v)
check("filled categories are training levels; filled numbers within training range", not bad, str(sorted(set(bad))))

pre_cols = nc.predictors("preop")
imp_pre = nm.StageImputer("preop").fit(X_learn[pre_cols])
check("stage order: pre-op values identical in pre-op-only and full imputer",
      imp_pre.transform(X_learn[pre_cols]).equals(F_learn[pre_cols])
      and imp_pre.transform(X_score[pre_cols]).equals(F_score[pre_cols]))

imp_pre2 = nm.StageImputer("preop").fit(X_learn[pre_cols])
check("deterministic: refitting gives identical values",
      imp_pre2.transform(X_score[pre_cols]).equals(imp_pre.transform(X_score[pre_cols])))

y_shuffled = y_learn.sample(frac=1, random_state=1).to_numpy()
imp_pre3 = nm.StageImputer("preop").fit(X_learn[pre_cols], y_shuffled)
check("outcome not used: shuffled y gives identical values",
      imp_pre3.transform(X_score[pre_cols]).equals(imp_pre.transform(X_score[pre_cols])))


def holds_rows(obj, n, depth=0):
    """True if any stored array/table (searched inside the fitted imputer) has one row per patient."""
    if depth > 4:
        return False
    if isinstance(obj, (pd.DataFrame, pd.Series, np.ndarray)):
        return obj.ndim > 0 and len(obj) >= n
    if isinstance(obj, dict):
        return any(holds_rows(x, n, depth + 1) for x in obj.values())
    if isinstance(obj, (list, tuple)):
        return any(holds_rows(x, n, depth + 1) for x in obj)
    if hasattr(obj, "__dict__"):
        return any(holds_rows(x, n, depth + 1) for x in vars(obj).values())
    return False


check("fitted imputer holds no patient rows", not holds_rows(imp_full, min(len(X_learn), 1000)))

# Recorded vs filled mix, learning part (aggregate)
mix_rows = []
miss_learn = X_learn.isna()
for v in Xc.columns:
    n_filled = int(miss_learn[v].sum())
    if n_filled == 0:
        continue
    rec, fil = X_learn.loc[~miss_learn[v], v], F_learn.loc[miss_learn[v], v]
    if nc.VARIABLES[v]["type"] == "continuous":
        mix_rows.append({"variable": v, "level": "median", "n_filled": nc.suppress(n_filled),
                         "recorded": round(float(rec.median()), 1),
                         "filled": round(float(fil.median()), 1) if n_filled >= nc.SMALL_CELL else "suppressed"})
        continue
    levels = nc.MODEL_LEVELS.get(v) or nc.VARIABLES[v].get("levels") or {}
    for lv in sorted(rec.unique(), key=str):
        k_fil = int((fil == lv).sum())
        mix_rows.append({"variable": v, "level": levels.get(lv, lv), "n_filled": nc.suppress(n_filled),
                         "recorded": f"{100 * (rec == lv).mean():.1f}%",
                         "filled": (f"{100 * k_fil / n_filled:.1f}%"
                                    if n_filled >= nc.SMALL_CELL and not nc.is_small(k_fil)
                                    and not nc.is_small(n_filled - k_fil) else "suppressed")})
mix_table = pd.DataFrame(mix_rows)
print(mix_table.to_string(index=False))

# %% [markdown]
# ## 5. Part 8.3b: model inputs after imputation (decisions 101, 115)
#
# Four recipes: "linear" (M1), "spline" (M2), "tree" (M3, M4), "mlp" (M5).
# Fitted on the imputed learning part of round 1, applied to fold 1.
#
# **Checks:**
# 1. Column counts per stage match Step 5 / decision 73: 42/51/53/58 (M1) and
#    48/61/63/68 (M2), apart from logged duplicate-knot merges.
# 2. Splines: straight beyond the outer knots, continuous at the knots, and
#    the same curves as an independent natural cubic spline (scipy).
# 3. Scaling: mean 0 and SD 1 in the training patients; new patients scaled
#    with the training values.
# 4. One-hot: each unordered predictor's columns add to exactly 1; an unseen
#    level stops with an error.
# 5. Tree recipe: numbers equal the imputed numbers; MLP has the same columns
#    as the tree recipe.
# 6. No patient rows stored.

# %%
from scipy.interpolate import CubicSpline
from scipy.special import expit, logit
from statsmodels.nonparametric.smoothers_lowess import lowess

EXPECTED_COLS = {"linear": {"preop": 42, "end_of_surgery": 51, "early_postop": 53, "complication_aware": 58},
                 "spline": {"preop": 48, "end_of_surgery": 61, "early_postop": 63, "complication_aware": 68}}
col_rows, knot_rows = [], []
encoders = {}
for stage in nm.STAGES:
    cols = nc.predictors(stage)
    for recipe in nm.RECIPES:
        enc = nm.ModelInputs(recipe).fit(F_learn[cols])
        encoders[(stage, recipe)] = enc
        n = len(enc.feature_names_)
        merged = sum(enc.knots_dropped_.values()) if recipe == "spline" else 0
        row = {"stage": stage, "recipe": recipe, "columns": n,
               "expected_step5": EXPECTED_COLS.get(recipe, {}).get(stage, ""), "knots_merged": ""}
        if recipe in EXPECTED_COLS:
            row["knots_merged"] = merged
            check(f"{stage} {recipe}: {n} columns = Step 5 count {EXPECTED_COLS[recipe][stage]} - merged knots {merged}",
                  n == EXPECTED_COLS[recipe][stage] - merged)
        col_rows.append(row)
        if recipe == "spline" and stage == "complication_aware":
            for c, kn in enc.knots_.items():
                knot_rows.append({"predictor": c, "knots": ", ".join(f"{k:g}" for k in kn),
                                  "duplicate_knots_dropped": enc.knots_dropped_[c]})
                if enc.knots_dropped_[c]:
                    log(f"Spline for {c}: {enc.knots_dropped_[c]} duplicate knot(s) dropped, "
                        f"knots {', '.join(f'{k:g}' for k in kn)}")
col_table, knot_table = pd.DataFrame(col_rows), pd.DataFrame(knot_rows)
print(col_table.to_string(index=False)); print(knot_table.to_string(index=False))

# Splines, on the real age knots
age_knots = encoders[("preop", "spline")].knots_["Age"]
lo, hi = age_knots[0], age_knots[-1]
xs = np.linspace(lo, hi, 801)
B = np.column_stack([np.ones_like(xs), nm.rcs_basis(xs, age_knots)])
target = CubicSpline(age_knots, [0.3, -1.0, 2.0, 0.5], bc_type="natural")(xs)
coef = np.linalg.lstsq(B, target, rcond=None)[0]
check("spline basis reproduces an independent natural cubic spline", np.abs(B @ coef - target).max() < 1e-8,
      f"largest difference {np.abs(B @ coef - target).max():.1e}")
tails = [nm.rcs_basis(np.linspace(lo - 30, lo - 0.01, 60), age_knots),
         nm.rcs_basis(np.linspace(hi + 0.01, hi + 30, 60), age_knots)]
check("spline straight beyond the outer knots", max(np.abs(np.diff(T, 2, axis=0)).max() for T in tails) < 1e-9)
eps = 1e-7
check("spline continuous at every knot",
      max(np.abs(nm.rcs_basis([k - eps], age_knots) - nm.rcs_basis([k + eps], age_knots)).max() for k in age_knots) < 1e-6)

# Scaling
for recipe in ("linear", "spline", "mlp"):
    enc = encoders[("complication_aware", recipe)]
    Zl, Zs = enc.transform(F_learn[nc.predictors("complication_aware")]), enc.transform(F_score[nc.predictors("complication_aware")])
    check(f"{recipe}: training columns mean 0 and SD 1",
          np.abs(Zl.mean()).max() < 1e-8 and np.abs(Zl.std(ddof=0) - 1).max() < 1e-8,
          f"constant columns: {enc.constant_columns_}")
    raw = enc._encode(F_score[nc.predictors("complication_aware")])
    check(f"{recipe}: new patients scaled with training values",
          np.allclose(Zs.to_numpy(), (raw.to_numpy() - enc.mean_) / enc.sd_))

# One-hot and tree inputs
enc_t = encoders[("complication_aware", "tree")]
Zt = enc_t.transform(F_score[nc.predictors("complication_aware")])
for c in [v for v in nc.predictors("complication_aware") if nc.VARIABLES[v]["type"] == "nominal"]:
    check(f"tree: one-hot columns of {c} add to 1", (Zt[[n for n in Zt.columns if n.startswith(c + "=")]].sum(axis=1) == 1).all())
num = [v for v in nc.predictors("complication_aware") if nc.VARIABLES[v]["type"] == "continuous"]
check("tree: numbers equal the imputed numbers", np.array_equal(Zt[num].to_numpy(), F_score[num].to_numpy(float)))
check("mlp has exactly the tree columns",
      encoders[("complication_aware", "mlp")].feature_names_ == enc_t.feature_names_)
bad_level = F_score[nc.predictors("preop")].copy()
bad_level.iloc[0, bad_level.columns.get_loc("PreopHR")] = 99
try:
    encoders[("preop", "tree")].transform(bad_level)
    raised = False
except ValueError:
    raised = True
check("an unseen category level stops with an error", raised)
check("no patient rows stored in any fitted recipe",
      not any(holds_rows(e, min(len(F_learn), 1000)) for e in encoders.values()))

# %% [markdown]
# ## 6. Part 8.4a: calibration curve and ICI / E50 / E90 (decision 116)
#
# Tested on made-up data with known answers (no model is fitted here):
#
# 1. The lowess settings reproduce R's lowess(iter = 0): compared with an
#    independent, plain implementation of the same algorithm (local straight
#    line, tricube weights, nearest 2/3 of points) on 600 points.
# 2. The speed setting (delta = 1% of the range) changes the curve by a
#    negligible amount on 20,000 points.
# 3. Perfectly calibrated risks give ICI close to 0.
# 4. Known miscalibration: predictions at 80% of the true risk give an ICI
#    close to the true value (mean of 0.2 x true risk), and E50 / E90 close to
#    their true values.
# 5. Time for one ICI on 32,000 patients (the size of 2017-19), to plan the
#    2,000 bootstrap resamples.

# %%
rng = np.random.default_rng(20261007)


def plain_lowess(x, y, frac):
    """Textbook lowess, one pass (iter = 0): weighted straight line through the
    nearest frac x n points, tricube weights. Written independently as a check."""
    n = len(x)
    r = int(np.ceil(frac * n))
    out = np.empty(n)
    for i in range(n):
        d = np.abs(x - x[i])
        h = np.sort(d)[r - 1]
        w = np.clip(1 - (d / h) ** 3, 0, None) ** 3 if h > 0 else (d == 0).astype(float)
        W = w.sum()
        xm, ym = (w * x).sum() / W, (w * y).sum() / W
        sxx = (w * (x - xm) ** 2).sum()
        b = (w * (x - xm) * (y - ym)).sum() / sxx if sxx > 0 else 0.0
        out[i] = ym + b * (x[i] - xm)
    return out


xs = rng.uniform(0.01, 0.6, 600)
ys = rng.binomial(1, xs).astype(float)
ref = plain_lowess(xs, ys, nm.LOWESS_FRAC)
ours = lowess(ys, xs, frac=nm.LOWESS_FRAC, it=0, delta=0.0, return_sorted=False)
check("lowess settings match an independent plain implementation", np.abs(ours - ref).max() < 1e-8,
      f"largest difference {np.abs(ours - ref).max():.1e}")

p_true = rng.beta(2, 9, 20_000)                                  # risks shaped roughly like ours (mean ~0.18)
y_sim = rng.binomial(1, p_true)
full = lowess(y_sim, p_true, frac=nm.LOWESS_FRAC, it=0, delta=0.0, return_sorted=False)
fast = nm.calibration_curve(y_sim, p_true)
check("speed setting (delta) changes the curve negligibly", np.abs(full - fast).max() < 1e-3,
      f"largest change {np.abs(full - fast).max():.1e}")

n_big = 200_000
p_big = rng.beta(2, 9, n_big)
y_big = rng.binomial(1, p_big)
good = nm.calibration_errors(y_big, p_big)
check("perfectly calibrated risks: ICI close to 0", good["ICI"] < 0.005, f"ICI {good['ICI']:.4f}")
q_big = 0.8 * p_big
bad = nm.calibration_errors(y_big, q_big)
true_gap = 0.2 * p_big
truth = {"ICI": true_gap.mean(), "E50": np.median(true_gap), "E90": np.quantile(true_gap, 0.9)}
check("known miscalibration: ICI, E50 and E90 close to the truth",
      all(abs(bad[k] - truth[k]) < 0.005 for k in truth),
      ", ".join(f"{k} {bad[k]:.4f} vs true {truth[k]:.4f}" for k in truth))

p32 = rng.beta(2, 9, 32_000)
y32 = rng.binomial(1, p32)
t0 = time.time()
for _ in range(5):
    nm.calibration_errors(y32, p32)
ici_seconds = (time.time() - t0) / 5
log(f"One ICI on 32,000 patients takes {ici_seconds:.2f} s; 2,000 bootstrap resamples about "
    f"{2000 * ici_seconds / 60:.0f} min per model and stage")
calib_test = pd.DataFrame([
    {"test": "perfectly calibrated (n=200,000)", **{k: round(v, 4) for k, v in good.items()}},
    {"test": "predictions at 80% of true risk (n=200,000)", **{k: round(v, 4) for k, v in bad.items()}},
    {"test": "truth for the line above", **{k: round(float(v), 4) for k, v in truth.items()}, "Emax": ""},
])
print(calib_test.to_string(index=False))

# %% [markdown]
# ## 7. Part 8.4b: the other measures and the paired bootstrap (decisions 86, 87, 96, 98, 117)
#
# Each measure is checked on made-up data with a known answer:
#
# 1. AUC equals the share of (event, non-event) pairs ranked correctly
#    (ties count half), counted one pair at a time.
# 2. PR-AUC (average precision) equals a hand calculation (0.8333) and, for
#    random risks, the event rate.
# 3. Calibration-in-the-large and slope recover values built into the data
#    (0.4 and 0.7), and the fast version equals statsmodels to 8 decimals.
# 4. Brier and scaled Brier equal a hand calculation; predicting the event
#    rate for everyone gives scaled Brier 0.
# 5. Net benefit equals a hand calculation; a perfect model's net benefit is
#    the event rate; "treat all" equals flagging everyone.
# 6. Bootstrap: identical on rerun; a model compared with itself gives a
#    difference of exactly 0; the AUC's bootstrap standard error is close to
#    the Hanley-McNeil formula; pairing narrows the interval of a difference.
# 7. Time for 2,000 resamples of every measure on 32,000 patients.

# %%
from itertools import product

# 1. AUC by counting pairs
ya = rng.binomial(1, 0.3, 300)
pa = np.round(rng.uniform(0, 1, 300), 2)                     # rounding creates ties
pos, neg = pa[ya == 1], pa[ya == 0]
pairs_auc = np.mean([(a > b) + 0.5 * (a == b) for a, b in product(pos, neg)])
check("AUC equals the share of correctly ranked pairs", abs(nm.performance(ya, pa)["AUC"] - pairs_auc) < 1e-12,
      f"{pairs_auc:.4f}")

# 2. PR-AUC
hand = nm.performance(np.array([1, 0, 1, 0]), np.array([0.9, 0.8, 0.7, 0.1]))["PR_AUC"]
check("PR-AUC equals hand calculation 0.8333", abs(hand - (0.5 * 1 + 0.5 * 2 / 3)) < 1e-12, f"{hand:.4f}")
y_r = rng.binomial(1, 0.159, 200_000)
pr_random = nm.performance(y_r, rng.uniform(0, 1, 200_000))["PR_AUC"]
check("PR-AUC of random risks is about the event rate", abs(pr_random - y_r.mean()) < 0.005,
      f"{pr_random:.4f} vs event rate {y_r.mean():.4f}")

# 3. Calibration-in-the-large and slope
lp_t = logit(rng.beta(2, 9, 200_000))
y_shift = rng.binomial(1, expit(lp_t + 0.4))
y_flat = rng.binomial(1, expit(0.7 * lp_t))
c_shift = nm.calibration_intercept_slope(y_shift, expit(lp_t))
c_flat = nm.calibration_intercept_slope(y_flat, expit(lp_t))
check("calibration-in-the-large recovers 0.4", abs(c_shift["CITL"] - 0.4) < 0.03, f"{c_shift['CITL']:.3f}")
check("calibration slope recovers 0.7", abs(c_flat["slope"] - 0.7) < 0.03, f"{c_flat['slope']:.3f}")
ref = nm.calibration_intercept_slope_statsmodels(y_shift, expit(lp_t))
check("fast calibration fit equals statsmodels to 8 decimals",
      all(abs(c_shift[k] - ref[k]) < 1e-8 for k in ref))

# 4. Brier
b = nm.brier_scores(np.array([1, 0, 1, 0]), np.array([0.9, 0.2, 0.6, 0.1]))
check("Brier and scaled Brier equal hand calculation (0.055, 0.78)",
      abs(b["Brier"] - 0.055) < 1e-12 and abs(b["scaled_Brier"] - 0.78) < 1e-12)
check("predicting the event rate gives scaled Brier 0",
      abs(nm.brier_scores(y_r, np.full(len(y_r), y_r.mean()))["scaled_Brier"]) < 1e-12)

# 5. Net benefit
nb_hand = nm.net_benefit(np.array([1, 1, 0, 0, 0]), np.array([0.5, 0.05, 0.3, 0.1, 0.02]), 0.2)
check("net benefit equals hand calculation 0.15", abs(nb_hand - 0.15) < 1e-12, f"{nb_hand:.4f}")
check("perfect model's net benefit is the event rate",
      all(abs(nm.net_benefit(y_r, y_r.astype(float), t) - y_r.mean()) < 1e-12 for t in nm.NB_THRESHOLDS))
check("treat-all equals flagging everyone",
      all(abs(nm.net_benefit_treat_all(y_r, t) - nm.net_benefit(y_r, np.ones(len(y_r)), t)) < 1e-12
          for t in nm.NB_THRESHOLDS))

# 6. Bootstrap (on 32,000 made-up patients, two correlated models)
y_b = rng.binomial(1, p32)
model_a = expit(logit(p32) + rng.normal(0, 0.3, len(p32)))
model_b = expit(logit(p32) + rng.normal(0, 0.6, len(p32)))
auc_only = lambda yy, pp: float(nm.roc_auc_score(yy, pp))
t0 = time.time()
tab1, reps1 = nm.paired_bootstrap(y_b, {"A": model_a, "B": model_b, "A_copy": model_a}, auc_only,
                                  pairs=[("A", "B"), ("A", "A_copy")])
auc_boot_seconds = time.time() - t0
tab2, _ = nm.paired_bootstrap(y_b, {"A": model_a, "B": model_b, "A_copy": model_a}, auc_only,
                              pairs=[("A", "B"), ("A", "A_copy")])
check("bootstrap identical on rerun", tab1.equals(tab2))
same = tab1[tab1["what"] == "A - A_copy"].iloc[0]
check("a model compared with itself: difference exactly 0", same["ci_low"] == 0 == same["ci_high"])
auc_a = tab1.loc[tab1["what"] == "A", "estimate"].iloc[0]
n1, n0 = int(y_b.sum()), int(len(y_b) - y_b.sum())
q1, q2 = auc_a / (2 - auc_a), 2 * auc_a ** 2 / (1 + auc_a)
se_hm = np.sqrt((auc_a * (1 - auc_a) + (n1 - 1) * (q1 - auc_a ** 2) + (n0 - 1) * (q2 - auc_a ** 2)) / (n1 * n0))
se_boot = reps1["A"]["value"].std(ddof=1)
check("bootstrap SE of AUC close to Hanley-McNeil formula (within 15%)", abs(se_boot / se_hm - 1) < 0.15,
      f"bootstrap {se_boot:.4f}, formula {se_hm:.4f}")
diff_row = tab1[tab1["what"] == "A - B"].iloc[0]
unpaired_width = 1.96 * np.sqrt(reps1["A"]["value"].var() + reps1["B"]["value"].var()) * 2
check("pairing narrows the interval of a difference",
      (diff_row["ci_high"] - diff_row["ci_low"]) < unpaired_width,
      f"paired width {diff_row['ci_high'] - diff_row['ci_low']:.4f} vs unpaired {unpaired_width:.4f}")

# 7. Time for the full set of measures
t0 = time.time()
for b_ in range(50):
    idx = nm.bootstrap_indices(len(y_b), b_)
    nm.performance(y_b[idx], model_a[idx])
full_seconds = (time.time() - t0) / 50
log(f"All measures on 32,000 patients: {full_seconds:.3f} s per resample; 2,000 resamples about "
    f"{2000 * full_seconds / 60:.1f} min per model and dataset (AUC-only bootstrap of 3 models: "
    f"{auc_boot_seconds:.0f} s)")
boot_example = tab1.assign(**{c: tab1[c].round(4) for c in ["estimate", "ci_low", "ci_high"]})
print(boot_example.to_string(index=False))

# %% [markdown]
# ## 8. Part 8.4c: classification at the top 10% / 20% of risk (decisions 98, 118)
#
# Patients tied at the cut-off share the last places equally, so exactly 10%
# (20%) are flagged for every model. Checks on made-up data:
#
# 1. With no ties, the flags are the ordinary top-k flags.
# 2. With ties (a staircase of risks, as isotonic calibration produces), the
#    flags still add up to exactly 10% / 20% of patients.
# 3. The result equals the average of many random tie-breaks.
# 4. A small hand example: sensitivity, specificity, PPV and NPV all 4/9.
# 5. For contrast: how much more than 10% "flag all tied patients" would flag.

# %%
p_cont = rng.beta(2, 9, 1000)
w = nm.flag_weights(p_cont, 0.10)
top_k = np.zeros(1000)
top_k[np.argsort(-p_cont)[:100]] = 1
check("no ties: ordinary top-10% flags", np.array_equal(w, top_k))

stairs = np.round(rng.beta(2, 9, 32_104) * 20) / 20            # risks on a 5%-step staircase
for s in nm.TOP_SHARES:
    ws = nm.flag_weights(stairs, s)
    check(f"ties: flags add up to exactly {int(100 * s)}% of patients", abs(ws.sum() - s * len(stairs)) < 1e-6,
          f"{ws.sum():.1f} of {len(stairs):,}; {int(((ws > 0) & (ws < 1)).sum()):,} patients share the last places")

y_st = rng.binomial(1, stairs)
stairs_k = stairs[:1000]
y_k = y_st[:1000]
exact = nm.classification_from_weights(y_k, nm.flag_weights(stairs_k, 0.10))
sims = []
for r in range(4000):
    tiebreak = np.random.default_rng(r).random(1000)
    order = np.lexsort((tiebreak, -stairs_k))                  # by risk, ties in random order
    f = np.zeros(1000)
    f[order[:100]] = 1
    sims.append(nm.classification_from_weights(y_k, f)["sensitivity"])
check("equals the average of 4,000 random tie-breaks", abs(np.mean(sims) - exact["sensitivity"]) < 0.002,
      f"sensitivity {exact['sensitivity']:.4f} vs random average {np.mean(sims):.4f}")

hand = nm.classification_from_weights(np.array([1, 0, 1, 0, 0, 1]),
                                      nm.flag_weights(np.array([0.9, 0.8, 0.5, 0.5, 0.5, 0.1]), 0.5))
check("hand example: all four measures 4/9", all(abs(v - 4 / 9) < 1e-12 for v in hand.values()))

cut10 = np.sort(stairs)[::-1][int(np.ceil(0.1 * len(stairs))) - 1]
log(f"For contrast: flagging every patient tied at the cut-off would flag "
    f"{100 * (stairs >= cut10).mean():.1f}% instead of 10% on the staircase example")
perf_example = pd.DataFrame([(k, f"{v:.4f}" if isinstance(v, float) else f"{v:,}")
                             for k, v in nm.performance(y_st, stairs).items()],
                            columns=["measure", "value (made-up staircase data)"])
print(perf_example.to_string(index=False))

# %% [markdown]
# ## 9. Part 8.4d: shared figures (decisions 119-121)
#
# The four curve plots are drawn on made-up data (three stages, all six model
# slots) and checked against the tested numbers:
#
# 1. The area under each drawn ROC curve equals the AUC (within 0.002; the
#    tips resting on counts of 1-9 are not drawn).
# 2. No drawn ROC or precision-recall point rests on a count of 1-9.
# 3. The drawn calibration curve is exactly the curve ICI is computed from,
#    drawn only between each model's 1st and 99th percentile (decision 121);
#    a warning is raised when a model's 99th percentile exceeds the fixed axis (0-70%, decision 141).
# 4. The drawn decision curves equal net benefit at 10, 15, 20 and 30%.
# 5. Every figure is written as PDF and PNG (ROC in versions A and B).
#
# The made-up figures are saved in `outputs/08_infrastructure/figures/` and a
# copy in the repository (`docs/figure_style/`) so the style can be reviewed.

# %%
import nacsa_plots as npl
importlib.reload(npl)
import statsmodels.api as sm
from sklearn.metrics import roc_auc_score

FIG_DIR = OUT_DIR / "figures"
FIG_DIR.mkdir(exist_ok=True)
fake = {}
for i, stage in enumerate(["preop", "end_of_surgery", "early_postop"]):
    lp_f = logit(rng.beta(2, 10, 32_000)) * (1 + 0.15 * i)
    y_f = rng.binomial(1, expit(lp_f))
    preds_f = {m: expit(lp_f * s + rng.normal(0, sd, len(lp_f)) + b) for m, (s, sd, b) in
               {"M0": (0.6, 0.6, 0.3), "M1": (1, 0.45, 0), "M2": (1, 0.35, 0), "M3": (1.1, 0.33, 0.05),
                "M4": (0.8, 0.4, -0.1), "M5": (1, 0.4, 0.1)}.items()}
    ci_f = {m: (roc_auc_score(y_f, q), roc_auc_score(y_f, q) - 0.008, roc_auc_score(y_f, q) + 0.008)
            for m, q in preds_f.items()}
    fake[stage] = (y_f, preds_f, ci_f)

drawn = {}
_, drawn["roc_B"] = npl.stage_figure("roc", {s: (y, p, {"ci": c}) for s, (y, p, c) in fake.items()},
                                     FIG_DIR / "example_roc_B", title="Made-up data: ROC, all models (version B)")
pick = ["M0", "M2", "M3"]
_, drawn["roc_A"] = npl.stage_figure("roc", {s: (y, {m: p[m] for m in pick}, {"ci": {m: c[m] for m in pick}})
                                            for s, (y, p, c) in fake.items()},
                                     FIG_DIR / "example_roc_A", title="Made-up data: ROC, three models (version A)")
for kind in ["pr", "calibration", "decision"]:
    _, drawn[kind] = npl.stage_figure(kind, {s: (y, p) for s, (y, p, c) in fake.items()},
                                      FIG_DIR / f"example_{kind}", title=f"Made-up data: {kind}")

worst = max(abs(np.trapezoid(v[1], v[0]) - roc_auc_score(fake[s][0], fake[s][1][m]))
            for s in fake for m, v in drawn["roc_B"][s].items() if not m.startswith("_"))
check("area under each drawn ROC curve equals the AUC", worst < 0.002, f"largest difference {worst:.5f}")

small = 0
for s, (y, p, c) in fake.items():
    n1, n0 = int(y.sum()), int(len(y) - y.sum())
    for m, (fpr, tpr) in [(k, v) for k, v in drawn["roc_B"][s].items() if not k.startswith("_")]:
        cnt = np.concatenate([tpr * n1, fpr * n0, n1 - tpr * n1, n0 - fpr * n0])
        small += int(np.sum((np.rint(cnt) >= 1) & (np.rint(cnt) < nc.SMALL_CELL)))
    for m, (rec, prec) in drawn["pr"][s].items():
        tp = rec * n1
        flagged = tp / prec
        cnt = np.concatenate([tp, flagged - tp, n1 - tp, flagged])
        small += int(np.sum((np.rint(cnt) >= 1) & (np.rint(cnt) < nc.SMALL_CELL)))
check("no drawn ROC or precision-recall point rests on a count of 1-9", small == 0, f"{small} such points")

ok_cal, ok_trim = True, True
for s in fake:
    for m, val in drawn["calibration"][s].items():
        if m.startswith("_"):
            continue
        x, curve = val
        p_m = fake[s][1][m]
        full = nm.calibration_curve(fake[s][0], p_m)[np.argsort(p_m)]
        xs = np.sort(p_m)
        lo, hi = np.percentile(p_m, npl.CALIBRATION_TRIM)
        keep = (xs >= lo) & (xs <= hi)
        ok_cal &= np.array_equal(curve, full[keep])
        ok_trim &= bool(x.min() >= lo and x.max() <= hi)
check("drawn calibration curve is exactly the ICI curve (within the drawn range)", ok_cal)
check("calibration curves drawn only between the 1st and 99th percentile", ok_trim)
warned = sorted({w.split(":")[0] for s in fake for w in drawn["calibration"][s]["_warnings"]})
log(f"Calibration axis warning (made-up data, as intended for a deliberately miscalibrated model): {warned}")
check("axis warning raised for a model whose 99th percentile exceeds the axis limit", "EuroSCORE II" in warned)
legend_fits = [drawn[v][s]["_legend"] for v in ("roc_A", "roc_B") for s in fake]
check("ROC legends inside the panels touch no model curve",
      not any(f["overlap"] for f in legend_fits),
      "font sizes " + ", ".join(f"{f['fontsize']:g}" for f in legend_fits))
check("legends use model names only (no M-codes)",
      all(not s["label"].startswith("M") or s["label"].startswith("MLP") for s in npl.MODEL_STYLE.values()))

ok_nb = True
for s, (y, p, c) in fake.items():
    for m, (tt, nb) in drawn["decision"][s].items():
        for thr in nm.NB_THRESHOLDS:
            k = int(np.argmin(np.abs(tt - thr)))
            ok_nb &= bool(np.isclose(nb[k], nm.net_benefit(y, p[m], thr)))
check("drawn decision curves equal net benefit at 10/15/20/30%", ok_nb)

files = [FIG_DIR / f"example_{k}.{ext}" for k in ["roc_A", "roc_B", "pr", "calibration", "decision"]
         for ext in ("pdf", "png")]
check("every example figure written as PDF and PNG", all(f.exists() and f.stat().st_size > 0 for f in files))
STYLE_DIR = Path(CODE_DIR).parent / "docs" / "figure_style"
STYLE_DIR.mkdir(parents=True, exist_ok=True)
for f in FIG_DIR.glob("example_*.png"):
    (STYLE_DIR / f.name).write_bytes(f.read_bytes())
log(f"Example figures (made-up data) copied to {STYLE_DIR.name}/ for style review")

# %% [markdown]
# ## 10. Part 8.5a: model wrappers M0, M1, M2 (decisions 93, 103, 123, 124)
#
# Made-up data only (the timing test on real data is part 8.6). Checks:
#
# 1. Ridge (M2): the fitted coefficients satisfy the penalised score equations
#    (average residual x column = lambda x coefficient; intercept residuals sum to 0).
# 2. Logistic (M1): coefficients equal statsmodels' unpenalised fit.
# 3. Tuning works: with plenty of clean data the chosen lambda is small; with
#    few, noisy patients it is clearly larger.
# 4. Edge rule: with pure noise the best lambda is the largest, and the grid is
#    extended upward once.
# 5. EuroSCORE II (M0): recalibration recovers an intercept and slope built
#    into the data; with missing values (decision 124) the recalibration uses
#    recorded patients only and missing ones get the training median.
# 6. Tuning is deterministic; time of a full ridge grid at real size.

# %%
def make_data(n, p_signal, p_noise, seed, strength=0.5, beta_seed=None):
    """Made-up standardised predictors and outcomes; patients from `seed`, true
    coefficients from `beta_seed` (same beta_seed = same underlying model)."""
    r = np.random.default_rng(seed)
    Z = r.standard_normal((n, p_signal + p_noise))
    rb = np.random.default_rng(seed if beta_seed is None else beta_seed + 10_000)
    beta = np.r_[rb.normal(0, strength, p_signal), np.zeros(p_noise)]
    yy = r.binomial(1, expit(-1.7 + Z @ beta))
    return Z, yy


Zb, yb = make_data(20_000, 10, 10, 1)
ridge = nm.RidgeModel()
for lam in (0.1, 0.001):
    f = ridge.fit(Zb, yb, {"lambda": lam})
    mu = f.predict_proba(Zb)[:, 1]
    grad = Zb.T @ (yb - mu) / len(yb)
    check(f"ridge score equations hold at lambda {lam}",
          np.allclose(grad, lam * f.coef_[0], atol=1e-7) and abs(np.mean(yb - mu)) < 1e-7,
          f"largest gap {np.abs(grad - lam * f.coef_[0]).max():.1e}; mean residual {np.mean(yb - mu):.1e}")

f1 = nm.LogisticModel().fit(Zb, yb)
ref = sm.GLM(yb, sm.add_constant(Zb), family=sm.families.Binomial()).fit()
check("logistic (M1) equals statsmodels' unpenalised fit",
      np.allclose(np.r_[f1.intercept_, f1.coef_[0]], ref.params, atol=1e-6))

Ztr, ytr = make_data(40_000, 20, 5, 2, beta_seed=1)
Ztu, ytu = make_data(15_000, 20, 5, 3, beta_seed=1)      # new patients, same underlying model
big, _ = ridge.tune(Ztr, ytr, Ztu, ytu)
Zs, ys = make_data(300, 3, 60, 4, strength=0.4, beta_seed=2)
Zs2, ys2 = make_data(300, 3, 60, 5, strength=0.4, beta_seed=2)
small, _ = ridge.tune(Zs, ys, Zs2, ys2)
check("tuning: plenty of clean data -> small lambda; few noisy patients -> larger lambda",
      big["lambda"] <= 0.001 and small["lambda"] > 10 * big["lambda"],
      f"chosen {big['lambda']:.2g} ({big['edge']}) vs {small['lambda']:.2g}")

rn = np.random.default_rng(6)
Zn, yn = rn.standard_normal((2000, 30)), rn.binomial(1, 0.16, 2000)
Zn2, yn2 = rn.standard_normal((2000, 30)), rn.binomial(1, 0.16, 2000)
noise, noise_log = ridge.tune(Zn, yn, Zn2, yn2)
check("edge rule: pure noise -> largest lambda -> grid extended once",
      noise["edge"] == "largest: grid extended" and len(noise_log) == len(nm.RIDGE_LAMBDAS) + len(nm.RIDGE_EXTENSION),
      f"chosen {noise['lambda']:.3g}, {len(noise_log)} values tried")

es2 = 100 * rng.beta(1.2, 60, 30_000)                     # made-up EuroSCORE II, %
y_es = rng.binomial(1, expit(1.2 + 0.6 * logit(es2 / 100)))
m0 = nm.EuroScoreModel().fit(es2, y_es)
check("EuroSCORE II recalibration recovers intercept 1.2 and slope 0.6",
      abs(m0.intercept_[0] - 1.2) < 0.15 and abs(m0.coef_[0][0] - 0.6) < 0.05,
      f"{m0.intercept_[0]:.2f}, {m0.coef_[0][0]:.2f}")
es2_gap = es2.copy()
es2_gap[:55] = np.nan                                    # 55 missing, as in the real data
m0_gap = nm.EuroScoreModel()
f_gap = m0_gap.fit(es2_gap, y_es)
f_rec = nm.EuroScoreModel().fit(es2[55:], y_es[55:])
pred_gap = m0_gap.predict(f_gap, es2_gap)
n_filled = m0_gap.n_filled_                              # read before any further predict() call
check("missing EuroSCORE II: recalibration uses recorded patients only",
      np.allclose(np.r_[f_gap.intercept_, f_gap.coef_[0]], np.r_[f_rec.intercept_, f_rec.coef_[0]]))
check("missing EuroSCORE II: filled with the training median, every patient predicted",
      n_filled == 55 and not np.isnan(pred_gap).any()
      and np.allclose(pred_gap[:55], m0_gap.predict(f_gap, np.full(55, np.median(es2[55:])))),
      f"{n_filled} filled with median {f_gap.es2_median_:.2f}%")

again, _ = ridge.tune(Ztr, ytr, Ztu, ytu)
check("ridge tuning deterministic", again == big)
Zr, yr = make_data(45_000, 40, 27, 7, beta_seed=3)
Zr2, yr2 = make_data(15_000, 40, 27, 8, beta_seed=3)
t0 = time.time()
ridge.tune(Zr, yr, Zr2, yr2)
log(f"Full ridge grid (30 values) at real size (45,000 x 67 columns): {time.time() - t0:.0f} s")

# %% [markdown]
# ## 11. Part 8.5b: XGBoost (M3) fitting step (decision 125)
#
# Made-up data at real size (45,000 training and 15,000 tuning patients, 52
# columns as in the tree recipe), with a curved and an interaction effect that
# a straight-line model cannot capture. Checks:
#
# 1. The search space in config/search_spaces.json is the one agreed.
# 2. Early stopping works: a fast learning rate stops early; a slow one needs
#    many more trees, never more than 2,000.
# 3. The refit uses the number of trees found in the trial.
# 4. Deterministic with the fixed seed; risks strictly between 0 and 1.
# 5. Sanity: on this data XGBoost beats plain logistic regression (it can learn
#    the curve and the interaction).
# 6. Time per fit, for the timing test.

# %%
space = nm.SEARCH_SPACES["M3_xgboost"]
check("XGBoost search space in config matches decision 125",
      space["tuned"]["learning_rate"] == {"low": 0.01, "high": 0.3, "log": True}
      and space["tuned"]["max_depth"]["low"] == 2 and space["tuned"]["max_depth"]["high"] == 10
      and space["tuned"]["min_child_weight"] == {"low": 1, "high": 100, "log": True}
      and space["tuned"]["reg_lambda"] == {"low": 0.01, "high": 10, "log": True}
      and space["fixed"]["n_estimators_max"] == 2000 and space["fixed"]["early_stopping_rounds"] == 50)

rx = np.random.default_rng(11)
bx = np.r_[rx.normal(0, 0.4, 20), np.zeros(32)]
def xdata(n, seed):
    r = np.random.default_rng(seed)
    Z = r.standard_normal((n, 52))
    lp = -1.7 + Z @ bx + 0.5 * Z[:, 0] * Z[:, 1] + 0.4 * (Z[:, 2] ** 2 - 1)
    return Z, r.binomial(1, expit(lp))
Zx, yx = xdata(45_000, 12)
Zxt, yxt = xdata(15_000, 13)
Zxs, yxs = xdata(15_000, 14)                              # separate scoring patients
xgbm = nm.XGBoostModel()
fast = dict(learning_rate=0.3, max_depth=6, min_child_weight=1, subsample=1.0, colsample_bytree=1.0, reg_lambda=1.0)
slow = dict(learning_rate=0.01, max_depth=4, min_child_weight=10, subsample=0.8, colsample_bytree=0.8, reg_lambda=1.0)
t0 = time.time()
m_fast, n_fast = xgbm.fit_trial(Zx, yx, Zxt, yxt, fast)
t_fast = time.time() - t0
t0 = time.time()
m_slow, n_slow = xgbm.fit_trial(Zx, yx, Zxt, yxt, slow)
t_slow = time.time() - t0
check("early stopping: fast learning rate stops early, slow one needs more trees (max 2,000)",
      n_fast < 200 and n_fast < n_slow <= 2000, f"{n_fast} vs {n_slow} trees")
refit = xgbm.fit(np.vstack([Zx, Zxt]), np.r_[yx, yxt], {"params": slow, "n_trees": n_slow})
check("refit uses the number of trees found in the trial", refit.get_booster().num_boosted_rounds() == n_slow)
p1 = xgbm.predict(refit, Zxs)
p2 = xgbm.predict(xgbm.fit(np.vstack([Zx, Zxt]), np.r_[yx, yxt], {"params": slow, "n_trees": n_slow}), Zxs)
check("XGBoost deterministic with fixed seed; risks strictly between 0 and 1",
      np.array_equal(p1, p2) and p1.min() > 0 and p1.max() < 1)
auc_xgb = roc_auc_score(yxs, p1)
auc_lr = roc_auc_score(yxs, nm.LogisticModel().predict(nm.LogisticModel().fit(np.vstack([Zx, Zxt]), np.r_[yx, yxt]), Zxs))
check("sanity: XGBoost beats plain logistic regression on curved/interaction data", auc_xgb > auc_lr + 0.005,
      f"AUC {auc_xgb:.3f} vs {auc_lr:.3f} (made-up data)")
log(f"XGBoost fit at real size: {t_fast:.1f} s ({n_fast} trees, learning rate 0.3) to {t_slow:.1f} s "
    f"({n_slow} trees, learning rate 0.01)")

# %% [markdown]
# ## 12. Part 8.5c: the calibration step (decisions 93, 126)
#
# Made-up data. A deliberately distorted model (its risks squeezed towards the
# middle, as random forests often do) and a well-calibrated one. Checks:
#
# 1. The two halves of the inner 25% are disjoint, cover it, and have the same
#    event rate.
# 2. Why the halves matter: scoring isotonic on the patients it was fitted on
#    looks better than it really is (lower log loss than the cross-fitted score).
# 3. For the distorted model, Platt and isotonic beat "none"; for the
#    well-calibrated model, "none" is about as good (calibration adds little).
# 4. Final fit: out-of-fold predictions come from copies not trained on those
#    patients (5 folds, all patients covered); the calibrated final model is
#    better calibrated (lower ICI) on new patients than the uncalibrated one.
# 5. Deterministic.

# %%
class Squeezed(nm.LogisticModel):
    """Made-up distorted model: logistic regression whose risks are squeezed."""
    def predict(self, fitted, Z):
        return expit(0.45 * logit(super().predict(fitted, Z)) - 0.9)


Zc, yc = make_data(40_000, 15, 5, 21, beta_seed=4)
Zct, yct = make_data(15_000, 15, 5, 22, beta_seed=4)
Zcs, ycs = make_data(20_000, 15, 5, 23, beta_seed=4)
a_half, b_half = nm.tune_halves(yct)
check("tuning halves disjoint, complete, same event rate",
      not set(a_half) & set(b_half) and len(a_half) + len(b_half) == len(yct)
      and abs(yct[a_half].mean() - yct[b_half].mean()) < 0.001)

sq = Squeezed()
p_sq = sq.predict(sq.fit(Zc, yc), Zct)
in_sample_iso = nm._safe_log_loss(yct, nm.Calibrator("isotonic").fit(p_sq, yct).transform(p_sq))
cross_iso = nm.crossfit_calibration_loss(p_sq, yct, "isotonic")
check("same-patient isotonic score is optimistic; cross-fitting removes that", in_sample_iso < cross_iso,
      f"log loss {in_sample_iso:.4f} (fitted and scored on same patients) vs {cross_iso:.4f} (cross-fitted)")

loss_sq = {m: nm.crossfit_calibration_loss(p_sq, yct, m) for m in nm.CALIBRATION_METHODS}
lr_w = nm.LogisticModel()
p_ok = lr_w.predict(lr_w.fit(Zc, yc), Zct)
loss_ok = {m: nm.crossfit_calibration_loss(p_ok, yct, m) for m in nm.CALIBRATION_METHODS}
check("distorted model: Platt and isotonic beat none",
      loss_sq["platt"] < loss_sq["none"] and loss_sq["isotonic"] < loss_sq["none"],
      ", ".join(f"{k} {v:.4f}" for k, v in loss_sq.items()))
check("well-calibrated model: none about as good (within 0.001)",
      loss_ok["none"] <= min(loss_ok.values()) + 0.001,
      ", ".join(f"{k} {v:.4f}" for k, v in loss_ok.items()))

final_iso = nm.fit_with_calibration(sq, Zc, yc, {}, "isotonic")
check("final fit: 5 out-of-fold groups covering every patient",
      final_iso.oof_index_ is not None and sorted(set(final_iso.oof_index_)) == list(range(5))
      and len(final_iso.oof_index_) == len(yc))
final_none = nm.fit_with_calibration(sq, Zc, yc, {}, "none")
ici_cal = nm.calibration_errors(ycs, final_iso.predict(Zcs))["ICI"]
ici_raw = nm.calibration_errors(ycs, final_none.predict(Zcs))["ICI"]
check("final calibrated model better calibrated on new patients (lower ICI)", ici_cal < ici_raw,
      f"ICI {ici_cal:.4f} calibrated vs {ici_raw:.4f} uncalibrated")
again_cal = nm.fit_with_calibration(sq, Zc, yc, {}, "isotonic")
check("calibration step deterministic",
      np.array_equal(again_cal.predict(Zcs), final_iso.predict(Zcs))
      and nm.crossfit_calibration_loss(p_sq, yct, "platt") == loss_sq["platt"])

# %% [markdown]
# ## 13. Part 8.5d: random forest (M4) (decision 127)
#
# Made-up data, 10,000 training patients (real-size timing is part 8.6). Checks:
# 1. The search space in config/search_spaces.json is the one agreed.
# 2. Deterministic with the fixed seed; risks between 0 and 1.
# 3. Works with the calibration step: random forests tend to squeeze risks, and
#    the calibrated forest is better calibrated (lower ICI) on new patients.

# %%
rf_space = nm.SEARCH_SPACES["M4_random_forest"]
check("random forest search space in config matches decision 127",
      rf_space["tuned"]["max_features"] == {"low": 0.05, "high": 0.5}
      and rf_space["tuned"]["min_samples_leaf"] == {"low": 1, "high": 100, "log": True, "type": "int"}
      and rf_space["tuned"]["max_samples"] == {"low": 0.3, "high": 1.0}
      and rf_space["fixed"]["n_estimators"] == 500 and rf_space["fixed"]["bootstrap"] is True)

Zf, yf = make_data(10_000, 15, 10, 31, beta_seed=5)
Zfs, yfs = make_data(10_000, 15, 10, 32, beta_seed=5)
rfm = nm.RandomForestModel()
rf_set = {"params": dict(max_features=0.3, min_samples_leaf=5, max_samples=0.7)}
t0 = time.time()
f_rf = rfm.fit(Zf, yf, rf_set)
rf_seconds = time.time() - t0
p_rf = rfm.predict(f_rf, Zfs)
check("random forest deterministic; risks between 0 and 1",
      np.array_equal(p_rf, rfm.predict(rfm.fit(Zf, yf, rf_set), Zfs)) and p_rf.min() >= 0 and p_rf.max() <= 1)
cal_rf = nm.fit_with_calibration(rfm, Zf, yf, rf_set, "platt")
ici_rf_raw = nm.calibration_errors(yfs, p_rf)["ICI"]
ici_rf_cal = nm.calibration_errors(yfs, cal_rf.predict(Zfs))["ICI"]
check("calibrated random forest better calibrated on new patients", ici_rf_cal < ici_rf_raw,
      f"ICI {ici_rf_cal:.4f} (Platt) vs {ici_rf_raw:.4f} (none)")
log(f"Random forest fit (10,000 made-up patients, 500 trees, 30% of predictors): {rf_seconds:.1f} s")

# %% [markdown]
# ## 14. Part 8.5e: neural network (M5) (decision 128)
#
# scikit-learn's multilayer perceptron; weight decay (alpha) and our own early
# stopping on log loss (scikit-learn's built-in rule watches accuracy). Made-up
# data with a curved effect, 20,000 training patients. Checks:
# 1. The search space in config/search_spaces.json is the one agreed.
# 2. Early stopping keeps the best epoch: the returned model's log loss on the
#    inner 25% is the best seen (a later epoch counts only if it gains at least
#    0.0001), and training stopped 10 epochs after it.
# 3. Refitting on the same patients for the recorded number of epochs gives
#    exactly the trial model (same seed, same batch order), so fit() and
#    fit_trial() agree; risks between 0 and 1.
# 4. Weight decay works: a larger alpha gives smaller weights.
# 5. The network finds the curved effect that logistic regression cannot
#    (lower log loss on new patients), and works with the calibration step.

# %%
mlp_space = nm.SEARCH_SPACES["M5_mlp"]
check("neural network search space in config matches decision 128",
      mlp_space["tuned"]["n_layers"] == {"low": 1, "high": 4, "type": "int"}
      and mlp_space["tuned"]["n_neurons"] == {"low": 16, "high": 512, "log": True, "type": "int"}
      and mlp_space["tuned"]["learning_rate"] == {"low": 1e-5, "high": 1e-2, "log": True}
      and mlp_space["tuned"]["batch_size"] == {"choices": [256, 512, 1024]}
      and (mlp_space["tuned"]["alpha"]["low"], mlp_space["tuned"]["alpha"]["high"]) == (1e-4, 10)
      and mlp_space["fixed"]["max_epochs"] == 200 and mlp_space["fixed"]["patience"] == 10)


def make_curved(n, seed):
    """Made-up data whose true risk has a U-shaped effect of the first predictor."""
    r = np.random.default_rng(seed)
    Z = r.standard_normal((n, 10))
    yy = r.binomial(1, expit(-2.2 + 0.6 * Z[:, 1] + 0.4 * Z[:, 2] + 0.8 * (Z[:, 0] ** 2 - 1)))
    return Z, yy


Zm, ym = make_curved(20_000, 41)
Zmu, ymu = make_curved(7_000, 42)            # inner 25%
Zms, yms = make_curved(10_000, 43)           # new patients
mlp = nm.MLPModel()
mlp_params = dict(n_layers=2, n_neurons=64, learning_rate=1e-3, batch_size=256, alpha=1.0)
t0 = time.time()
f_mlp, n_ep = mlp.fit_trial(Zm, ym, Zmu, ymu, mlp_params)
mlp_seconds = time.time() - t0
hist = f_mlp.tune_losses_
loss_now = nm._safe_log_loss(ymu, mlp.predict(f_mlp, Zmu))
# a later epoch counts as better only if it gains at least 0.0001 (decision 128)
gain = mlp.fixed["min_improvement"]
check("early stopping keeps the best epoch's weights",
      np.isclose(loss_now, hist[n_ep - 1]) and all(h > hist[n_ep - 1] - gain for h in hist[n_ep:])
      and all(hist[i] < min(hist[:i]) - gain for i in [n_ep - 1] if i > 0)
      and (len(hist) == n_ep + 10 or len(hist) == 200),
      f"best epoch {n_ep} of {len(hist)}; log loss {loss_now:.4f}")
f_re = mlp.fit(Zm, ym, {"params": mlp_params, "n_epochs": n_ep})
p_mlp = mlp.predict(f_re, Zms)
check("refit for the recorded epochs reproduces the trial model; risks between 0 and 1",
      all(np.array_equal(a, b) for a, b in zip(f_re.coefs_, f_mlp.coefs_)) and p_mlp.min() >= 0 and p_mlp.max() <= 1)
w_small = sum((c ** 2).sum() for c in mlp.fit(Zm, ym, {"params": {**mlp_params, "alpha": 1e-4}, "n_epochs": 10}).coefs_)
w_big = sum((c ** 2).sum() for c in mlp.fit(Zm, ym, {"params": {**mlp_params, "alpha": 10.0}, "n_epochs": 10}).coefs_)
check("larger weight decay gives smaller weights", w_big < w_small,
      f"sum of squared weights {w_big:.1f} (alpha 10) vs {w_small:.1f} (alpha 0.0001)")
ll_mlp = nm._safe_log_loss(yms, p_mlp)
ll_lr = nm._safe_log_loss(yms, nm.LogisticModel().fit(Zm, ym).predict_proba(Zms)[:, 1])
check("neural network finds the curved effect logistic regression misses", ll_mlp < ll_lr,
      f"log loss on new patients {ll_mlp:.4f} (network) vs {ll_lr:.4f} (logistic)")
cal_mlp = nm.fit_with_calibration(mlp, Zm, ym, {"params": mlp_params, "n_epochs": n_ep}, "platt")
p_cal = cal_mlp.predict(Zms)
check("neural network works with the calibration step", p_cal.min() >= 0 and p_cal.max() <= 1,
      f"ICI {nm.calibration_errors(yms, p_cal)['ICI']:.4f} (Platt) vs {nm.calibration_errors(yms, p_mlp)['ICI']:.4f} (none)")
log(f"Neural network trial (20,000 made-up patients, 2 x 64 neurons, {len(hist)} epochs): {mlp_seconds:.1f} s")

# %% [markdown]
# ## 15. Part 8.5f: tuning harness (decisions 129-132)
#
# `code/nacsa_tuning.py` runs one study (one model, one stage, one round): 30
# Optuna TPE trials one after another with a fixed seed (129), no pruner (130),
# default TPE settings (131), saved as one JSON file (132). Made-up data, 4,000
# training and 1,500 tuning patients (real-size timing is part 8.6, not run here).
# Checks:
# 1. Each study has its own fixed seed: 72 different seeds, identical when asked again.
# 2. The trial count comes from config (30) and the winner is the lowest score.
# 3. A rerun gives exactly the same trials and winner.
# 4. Studies run side by side in 2 processes give exactly the sequential results.
#    Each runs as a separate Python program: the first version forked copies of
#    this process, which froze because XGBoost's thread pool was already in use.
# 5. A study already saved is loaded, not rerun, and its file is not changed.
# 6. The winning settings refit with the calibration step; risks between 0 and 1.
# 7. Ridge grid saved in the same format, same choice as RidgeModel.tune.
# 8. Saved files hold no patient-level data; the harness never loads data itself
#    (the validation lock stays with the caller).

# %%
import json
import shutil
import nacsa_tuning as nt
importlib.reload(nt)

seeds = {(st, rd, m): nt.study_seed(st, rd, m) for st in nm.STAGES
         for rd in [f"round_{k}" for k in range(1, 6)] + ["final"] for m in ("M3", "M4", "M5")}
check("each study has its own fixed seed",
      len(set(seeds.values())) == len(seeds) == 72
      and all(nt.study_seed(*k) == v for k, v in seeds.items()), f"{len(set(seeds.values()))} different seeds")

TUNE_DIR = OUT_DIR / "tuning_test"
shutil.rmtree(TUNE_DIR, ignore_errors=True)
Zh, yh = make_data(4_000, 10, 5, 51, beta_seed=6)
Zhu, yhu = make_data(1_500, 10, 5, 52, beta_seed=6)
t0 = time.time()
rec = nt.run_study("M3", "preop", "round_1", Zh, yh, Zhu, yhu, TUNE_DIR)
xgb_study_seconds = time.time() - t0
losses = [t["log_loss"] for t in rec["trials"]]
check("trial count from config (30); winner is the lowest score",
      rec["n_trials"] == len(rec["trials"]) == nm.SEARCH_SPACES["M3_xgboost"]["trials"] == 30
      and rec["best"]["log_loss"] == min(losses) and rec["trials"][rec["best"]["number"]]["log_loss"] == min(losses),
      f"best trial {rec['best']['number']}, log loss {min(losses):.4f}; calibration {rec['best']['calibration']}")

rec_again = nt.tune_optuna(nm.XGBoostModel(), Zh, yh, Zhu, yhu, rec["seed"])
same = lambda a, b: [(t["params"], t["steps"], t["log_loss"]) for t in a["trials"]] == \
                    [(t["params"], t["steps"], t["log_loss"]) for t in b["trials"]]
check("rerun gives exactly the same trials and winner", same(rec, rec_again) and rec_again["best"]["params"] == rec["best"]["params"])

jobs = [dict(model="M3", stage=st, round_name="round_2", Z_tr=Zh, y_tr=yh, Z_tu=Zhu, y_tu=yhu,
             out_dir=TUNE_DIR / "parallel", n_trials=10) for st in ("preop", "end_of_surgery")]
par = nt.run_studies(jobs, n_processes=2)
seq = [nt.run_study(**{**j, "out_dir": TUNE_DIR / "sequential"}) for j in jobs]
check("studies side by side in 2 processes equal the sequential results; no temporary arrays left",
      all(same(a, b) for a, b in zip(par, seq)) and par[0]["seed"] != par[1]["seed"]
      and not list((TUNE_DIR / "parallel").glob(".job_*")))

path = nt.study_path(TUNE_DIR, "M3", "preop", "round_1")
before = path.read_bytes()
t0 = time.time()
loaded = nt.run_study("M3", "preop", "round_1", Zh, yh, Zhu, yhu, TUNE_DIR)
check("a saved study is loaded, not rerun; file unchanged",
      loaded.get("loaded_from_file") is True and path.read_bytes() == before and time.time() - t0 < 1 and same(loaded, rec))

best = rec["best"]
final = nm.fit_with_calibration(nm.XGBoostModel(), np.vstack([Zh, Zhu]), np.r_[yh, yhu], best["settings"], best["calibration"])
p_final = final.predict(Zhu)
check("winning settings refit with the calibration step; risks between 0 and 1",
      p_final.min() >= 0 and p_final.max() <= 1 and final.calibrator.method == best["calibration"])

rec_r = nt.run_study("M2", "preop", "round_1", Zh, yh, Zhu, yhu, TUNE_DIR)
check("ridge grid saved in the same format; same choice as RidgeModel.tune",
      rec_r["best"]["settings"]["lambda"] == nm.RidgeModel().tune(Zh, yh, Zhu, yhu)[0]["lambda"]
      and len(rec_r["trials"]) >= 30)


def longest_list(x):
    if isinstance(x, dict):
        return max([longest_list(v) for v in x.values()] + [0])
    if isinstance(x, list):
        return max([len(x)] + [longest_list(v) for v in x])
    return 0


saved = sorted(TUNE_DIR.rglob("*.json"))
src = Path(nt.__file__).read_text()
check("saved files hold no patient-level data; harness never loads data",
      all(longest_list(json.loads(f.read_text())) <= 40 for f in saved)
      and "get_stage_data" not in src and "allow_validation" not in src and "read_csv" not in src,
      f"{len(saved)} study files, largest {max(f.stat().st_size for f in saved):,} bytes")
log(f"XGBoost study (30 trials, 4,000 made-up patients): {xgb_study_seconds:.1f} s (timing on real data is part 8.6)")

# %% [markdown]
# ## 16. Part 8.5g: fit-failure fallbacks (decision 133)
#
# Every logistic-type fit (M0, M1, M2, the Platt calibrator and the imputer's
# category models) goes through `fit_logistic()`: newton-cholesky with 500
# iterations, then 5,000, then lbfgs with 10,000 (fixed in config). A failed
# tuning trial is recorded and the search goes on. Checks:
# 1. The fallback chain in config is the one agreed.
# 2. No retry was needed anywhere in the checks above (real data included).
# 3. A fit that fails its first attempt succeeds at the next, with the same answer.
# 4. If every attempt fails, the failure is raised and logged.
# 5. Imputer rule 8: if its models cannot be fitted, the variable is filled with
#    its training most common value, logged; no gaps left (real pre-op data).
# 6. Tuning: failed trials are recorded with their message, still count toward
#    the budget, and the winner comes from completed trials; if every trial
#    fails, the study is marked failed (rule 5).

# %%
fb = nm.SEARCH_SPACES["fit_fallbacks"]
check("fallback chain in config matches decision 133",
      fb["regression_attempts"] == [{"solver": "newton-cholesky", "max_iter": 500},
                                    {"solver": "newton-cholesky", "max_iter": 5000},
                                    {"solver": "lbfgs", "max_iter": 10000}])
check("no retry needed anywhere in the checks above", len(nm.FIT_EVENTS) == 0,
      f"{len(nm.FIT_EVENTS)} retries or failures")

Zg, yg = make_data(5_000, 10, 5, 61, beta_seed=7)
short_first = [{"solver": "newton-cholesky", "max_iter": 1}, {"solver": "newton-cholesky", "max_iter": 5000},
               {"solver": "lbfgs", "max_iter": 10000}]
m_retry = nm.fit_logistic(None, Zg, yg, where="test: short first attempt", attempts=short_first)
m_direct = nm.fit_logistic(None, Zg, yg, where="test: direct")
check("a fit that fails its first attempt succeeds at the next, same answer",
      nm.FIT_EVENTS[-1]["succeeded_at"] == 2 and "did not converge" in nm.FIT_EVENTS[-1]["reasons"][0]
      and np.allclose(m_retry.coef_, m_direct.coef_, atol=1e-8))

try:
    nm.fit_logistic(None, Zg, yg, where="test: all short",
                    attempts=[{"solver": "newton-cholesky", "max_iter": 1}, {"solver": "lbfgs", "max_iter": 1}])
    all_failed_raised = False
except nm.FitFailed:
    all_failed_raised = True
check("if every attempt fails, the failure is raised and logged",
      all_failed_raised and nm.FIT_EVENTS[-1]["succeeded_at"] is None and len(nm.FIT_EVENTS[-1]["reasons"]) == 2)

saved_attempts = fb["regression_attempts"]
fb["regression_attempts"] = [{"solver": "lbfgs", "max_iter": 1}]       # force every category model to fail
try:
    Xg = X_learn[pre_cols].iloc[:8_000]
    imp_fb = nm.StageImputer("preop").fit(Xg)
    Fg = imp_fb.transform(Xg)
finally:
    fb["regression_attempts"] = saved_attempts
cat_gaps = [v for v in pre_cols if Xg[v].isna().any() and nc.VARIABLES[v]["type"] != "continuous"]
check("imputer rule 8: unfittable variables filled with their training most common value; no gaps",
      sorted(imp_fb.fallbacks_) == sorted(set(imp_fb.fallbacks_)) and set(imp_fb.fallbacks_) == set(cat_gaps)
      and not Fg.isna().any().any()
      and all((Fg.loc[Xg[v].isna(), v] == imp_fb.start_[v]).all() for v in imp_fb.fallbacks_),
      f"{len(set(imp_fb.fallbacks_))} category predictors with gaps fell back: {sorted(set(imp_fb.fallbacks_))}")


class FlakyXGB(nm.XGBoostModel):
    """Test wrapper: deep trees 'fail'."""
    def fit_trial(self, Z_tr, y_tr, Z_tu, y_tu, params):
        if params["max_depth"] > 6:
            raise ValueError("test failure for deep trees")
        return super().fit_trial(Z_tr, y_tr, Z_tu, y_tu, params)


class BrokenXGB(nm.XGBoostModel):
    def fit_trial(self, *args):
        raise ValueError("test failure")


rec_f = nt.tune_optuna(FlakyXGB(), Zh, yh, Zhu, yhu, seed=1, n_trials=12)
failed = [t for t in rec_f["trials"] if t["state"] == "FAIL"]
check("failed trials recorded, counted in the budget; winner from completed trials",
      len(rec_f["trials"]) == 12 and rec_f["n_failed"] == len(failed) > 0 and rec_f["status"] == "ok"
      and all("test failure" in t["error"] for t in failed) and rec_f["best"]["params"]["max_depth"] <= 6,
      f"{len(failed)} of 12 trials failed; winner max_depth {rec_f['best']['params']['max_depth']}")
rec_b = nt.tune_optuna(BrokenXGB(), Zh, yh, Zhu, yhu, seed=1, n_trials=5)
check("every trial failed: study marked failed (rule 5)",
      rec_b["status"] == "failed" and rec_b["best"] is None and rec_b["n_failed"] == 5)

# %% [markdown]
# ## 17. Part 8.5h: calibration choice inside each trial (decision 134)
#
# Every trial trains its model once and scores all three calibration options on
# the inner-25% predictions (two-halves cross-fitting, decision 126); the best is
# kept, and options equal when rounded to 3 decimals count as tied, the simpler
# winning (none, then Platt, then isotonic; decision 109 rule 7). Checks:
# 1. Calibration is no longer a sampled setting in config; the rule is recorded.
# 2. The tie rule on hand-made scores.
# 3. Well-calibrated risks keep 'none'; distorted risks get Platt or isotonic.
# 4. In a saved study, every completed trial has all three scores, kept the option
#    the rule picks, and was scored with that option's log loss.

# %%
check("calibration not sampled for M3-M5; tie rule recorded in config",
      all("calibration" not in nm.SEARCH_SPACES[k]["tuned"] for k in ("M3_xgboost", "M4_random_forest", "M5_mlp"))
      and nm.SEARCH_SPACES["calibration_step"]["options_simplest_first"] == ["none", "platt", "isotonic"])
check("tie rule: equal to 3 decimals -> simpler option wins",
      nm.pick_calibration({"none": 0.40012, "platt": 0.40008, "isotonic": 0.40001}) == "none"
      and nm.pick_calibration({"none": 0.4012, "platt": 0.4004, "isotonic": 0.4003}) == "platt"
      and nm.pick_calibration({"none": 0.45, "platt": 0.41, "isotonic": 0.40}) == "isotonic")
p_good = nm.LogisticModel().fit(Zh, yh).predict_proba(Zhu)[:, 1]
m_good, _, l_good = nm.choose_calibration(p_good, yhu)
m_bad, _, l_bad = nm.choose_calibration(p_good ** 2.5, yhu)
check("well-calibrated risks keep 'none'; distorted risks are recalibrated",
      m_good == "none" and m_bad in ("platt", "isotonic"),
      f"good: {m_good} ({', '.join(f'{k} {v:.4f}' for k, v in l_good.items())}); "
      f"distorted: {m_bad} ({', '.join(f'{k} {v:.4f}' for k, v in l_bad.items())})")
done = [t for t in rec["trials"] if t["state"] == "COMPLETE"]
check("every trial scored all three options and kept the one the rule picks",
      len(done) == 30 and all(set(t["calibration_losses"]) == set(nm.CALIBRATION_METHODS)
                              and t["calibration"] == nm.pick_calibration(t["calibration_losses"])
                              and t["log_loss"] == t["calibration_losses"][t["calibration"]] for t in done),
      "kept: " + ", ".join(f"{m} {sum(t['calibration'] == m for t in done)}" for m in nm.CALIBRATION_METHODS))

# %% [markdown]
# ## 18. Save outputs
#
# Event rates are shown only when events and non-events are both 10 or more.

# %%
summary = pd.DataFrame(rows)
summary["event_rate_pct"] = [nc.rate_text(e, n) for e, n in zip(summary["n_events"], summary["n_patients"])]
summary["n_patients"] = [nc.suppress(n) for n in summary["n_patients"]]
summary["n_events"] = [nc.suppress(n) for n in summary["n_events"]]
summary.to_csv(OUT_DIR / "stage_data_check.csv", index=False)
comparison.to_csv(OUT_DIR / "stage_missing_check.csv", index=False)
fold_table.to_csv(OUT_DIR / "folds_check.csv", index=False)
inner_table.to_csv(OUT_DIR / "inner_split_check.csv", index=False)
mix_table.to_csv(OUT_DIR / "imputation_mix.csv", index=False)
col_table.to_csv(OUT_DIR / "model_input_columns.csv", index=False)
knot_table.to_csv(OUT_DIR / "spline_knots.csv", index=False)
calib_test.to_csv(OUT_DIR / "calibration_metric_test.csv", index=False)
boot_example.to_csv(OUT_DIR / "bootstrap_test.csv", index=False)
perf_example.to_csv(OUT_DIR / "performance_example.csv", index=False)
print(summary.to_string(index=False))

log(f"Checks passed: {sum(ok for _, ok in CHECKS)} of {len(CHECKS)}")
log(f"Run finished {datetime.now():%Y-%m-%d %H:%M}; pandas {pd.__version__}, numpy {np.__version__}")
(OUT_DIR / "log.txt").write_text("\n".join(LOG) + "\n")

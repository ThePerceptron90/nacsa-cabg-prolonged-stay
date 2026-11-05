# =============================================================================
# NACSA PLoS project  |  Step 6b: Missing data (supporting tables)
# -----------------------------------------------------------------------------
# HOW TO USE THIS FILE
#   Claude runs this file in its own workspace (decision 56). To run it
#   yourself in Colab, copy ONE block at a time (each starts with "# %%") into
#   a cell, or open the .ipynb version, and run top to bottom.
# =============================================================================

# %% [markdown]
# # Step 6b: Missing data - the evidence behind the plan
#
# **What this step does:** it produces the tables that support the agreed
# missing-data plan (decisions 74-85, 6 Oct 2026). It does NOT impute
# anything: the imputer is built into the modelling code at Step 8.
#
# **The plan, in one paragraph.** Patients with an unknown outcome are
# excluded. Missing predictor values are filled by deterministic single
# regression imputation (chained equations, one dataset), using the other
# predictors only, never the outcome, in stage order (a gap is filled only from
# variables available at its own stage or earlier). The imputer is fitted on
# the training data of every fold and finally on all 2012-16 data, then applied
# unchanged to 2017-19. Blank IABP and post-operative fields are imputed like
# everything else. No missing indicators; tree models receive the same imputed
# data. Sensitivity analyses: MICE with 10 imputations (regression models
# only) and complete-case analysis.
#
# **Outputs** (in `outputs/06b_missing_data/`):
#
# | File | What it shows | Supports |
# |---|---|---|
# | `missing_by_year.csv` | % missing per predictor and year, 2012-2019 | decisions 75, 83 (recording drifts over time) |
# | `missing_outcome_by_status.csv` | long-stay rate when a predictor is missing vs recorded | decisions 81, 83; limitations |
# | `missing_iabp_comparison.csv` | blank IABP vs recorded no vs recorded yes | decision 80 |
# | `missing_iabp_imputation_check.csv` | how many blanks an imputer would call "IABP" | decisions 80, 82 |
# | `missing_postop_comparison.csv` | blank vs no vs yes for return to theatre, dialysis, stroke | decision 79 |
# | `missing_comissingness.csv` | how often two predictors are missing together | decision 79 |
# | `missing_complete_cases.csv` | % of patients with no missing predictor, per model | decision 74 |
# | `log.txt` | run details | |
#
# Privacy: counts of 1-9 show as "<10"; rates are hidden when events or
# non-events are 1-9; no minimum or maximum values.

# %% [markdown]
# ## 1. Set up and load
#
# Development patients (2012-16) are the main population. Validation patients
# (2017-19) are loaded only to describe how much is missing per year and per
# model (description is allowed under the validation lock, decision 25); their
# outcomes are not used anywhere in this script.
#
# **What you should see:** two `load_clean:` lines.

# %%
import importlib
import os
import sys
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
importlib.reload(nc)

OUT_DIR = nc.OUTPUTS_DIR / "06b_missing_data"
OUT_DIR.mkdir(parents=True, exist_ok=True)

dev = nc.prepare_for_model(nc.load_clean(period="development", outcome="survivors"))
val = nc.prepare_for_model(nc.load_clean(period="validation", outcome="survivors", allow_validation=True))
val = val.drop(columns=["y"])          # make sure no validation outcome can be used below

# Every predictor used by any main model (the complication-aware set contains all of them)
ALL_PREDICTORS = list(dict.fromkeys(v for m in nc.PREDICTOR_SETS for v in nc.predictors(m)))
MISSING_ANY = [v for v in ALL_PREDICTORS if dev[v].isna().any() or val[v].isna().any()]
print(f"\n{len(ALL_PREDICTORS)} predictors, {len(MISSING_ANY)} with any missing value")


def pct(x):
    return round(100 * float(x), 1)


# %% [markdown]
# ## 2. Missingness by year (2012-2019)
#
# If missingness changes a lot between years, a model that learned "missing
# means X" in 2012-16 would be wrong in 2017-19. This is one reason for no
# missing indicators (decision 83) and no native missing handling in the tree
# models (decision 75).
#
# **What you should see:** IABP falling from about 18% to about 9%; myocardial
# protection rising from about 3% to about 10%.

# %%
both = pd.concat([dev[["Year"] + MISSING_ANY], val[["Year"] + MISSING_ANY]])
YEARS = sorted(both["Year"].unique())
# pct_missing_text shows "<10 records" when only 1-9 are missing, so a small
# percentage in a small year (2019 is partial) cannot reveal a count of 1-9
by_year = pd.DataFrame({
    str(yr): [nc.pct_missing_text(both.loc[both["Year"] == yr, v].isna().sum(),
                                  (both["Year"] == yr).sum()) for v in MISSING_ANY]
    for yr in YEARS
})
by_year.insert(0, "variable", [nc.label(v) for v in MISSING_ANY])
by_year["code"] = MISSING_ANY
by_year.to_csv(OUT_DIR / "missing_by_year.csv", index=False)
print(by_year.to_string(index=False))

# %% [markdown]
# ## 3. Long-stay rate when a predictor is missing vs recorded (development)
#
# If gaps were unrelated to the patient and the hospital, the two rates would
# be about the same. A difference means missingness carries information (most
# likely hospital recording practice, which the extract cannot identify). This
# supports decision 83 (no missing indicators: harmful under outcome-related
# missingness) and is reported as a limitation.

# %%
rows = []
for v in MISSING_ANY:
    miss = dev[v].isna()
    rows.append({
        "variable": nc.label(v),
        "% missing (development)": nc.pct_missing_text(miss.sum(), len(dev)),
        "patients missing": nc.suppress(miss.sum()),
        "long stay % if missing": nc.rate_text(dev.loc[miss, "y"].sum(), miss.sum()),
        "long stay % if recorded": nc.rate_text(dev.loc[~miss, "y"].sum(), (~miss).sum()),
        "code": v,
    })
outcome_by_status = pd.DataFrame(rows)
outcome_by_status.to_csv(OUT_DIR / "missing_outcome_by_status.csv", index=False)
print(outcome_by_status.to_string(index=False))

# %% [markdown]
# ## 4. Blank IABP vs recorded "no" vs recorded "yes" (development)
#
# The three IABP variables come from one field (IABP timing), so they are
# blank together. If a blank meant "no IABP", blank patients should look like
# recorded "no" patients. They do; but that is also what we would see if gaps
# were random among patients with a similar risk profile, so it does not prove
# "blank = no" (decision 80).
#
# The last row shows what the blank group would look like if gaps were
# completely random (the recorded patients' average): the blanks are clearly
# lower-risk than that.

# %%
IABP = ["iabp_preop", "iabp_intraop", "iabp_postop"]
blank = dev["iabp_intraop"].isna()
any_yes = dev[IABP].sum(axis=1) > 0
groups = {
    "Blank": blank,
    "Recorded no": ~blank & ~any_yes,
    "Recorded yes (any timing)": ~blank & any_yes,
    "All recorded (expected if gaps were completely random)": ~blank,
}
CHARS = [
    ("Cardiogenic shock, %", lambda d: d["CardiogenicShock"] == 1, "CardiogenicShock"),
    ("Critical pre-operative state, %", lambda d: d["CPS"] == 1, "CPS"),
    ("Emergency or salvage, %", lambda d: d["Urgency"] == 3, "Urgency"),
    ("Impaired LV function, %", lambda d: d["lv_impaired"] == 1, "lv_impaired"),
]
rows = []
for name, g in groups.items():
    d = dev[g]
    row = {"group": name, "patients": nc.suppress(len(d))}
    for lab, fn, col in CHARS:
        known = d[col].notna()
        row[lab] = nc.rate_text(fn(d[known]).sum(), known.sum(), digits=2)
    row["EuroSCORE II, mean %"] = round(d["ES2"].mean(), 2)
    row["Long stay, %"] = nc.rate_text(d["y"].sum(), len(d))
    rows.append(row)
iabp_table = pd.DataFrame(rows)
iabp_table.to_csv(OUT_DIR / "missing_iabp_comparison.csv", index=False)
print(iabp_table.to_string(index=False))

# %% [markdown]
# ## 5. How many blank patients would an imputer call "IABP"? (illustrative)
#
# A quick logistic model, fitted on recorded patients only, predicts each IABP
# variable from the pre-operative risk profile. Applied to the blanks:
#
# - **expected number** = sum of predicted probabilities, i.e. roughly what
#   imputation WITH random scatter would assign (the MICE sensitivity analysis);
# - **most likely value is "yes"** = patients with probability above 0.5, i.e.
#   what deterministic imputation would assign (the main analysis, decision 82).
#
# This is an illustration for decision 80, not the final imputer (Step 8 builds
# that, with the full stage-ordered predictor sets, and reports its own counts).
# Other missing values in the risk-profile variables are set to "no" here only
# for this check.

# %%
from sklearn.linear_model import LogisticRegression

X = pd.DataFrame({
    "log_es2": np.log(dev["ES2"].clip(lower=0.5)),
    "cps": dev["CPS"], "shock": dev["CardiogenicShock"],
    "emergency": (dev["Urgency"] == 3).astype(float), "urgent": (dev["Urgency"] == 2).astype(float),
    "lv_impaired": dev["lv_impaired"], "age_per_10y": dev["Age"] / 10,
    "recent_mi": dev["RecentMI"], "unstable_angina": dev["UA"],
}).fillna(0)
rows = []
for v in IABP:
    y_rec = dev.loc[~blank, v]
    fit = LogisticRegression(max_iter=2000).fit(X[~blank], y_rec)
    p = fit.predict_proba(X[blank])[:, 1]
    rows.append({
        "variable": nc.label(v),
        "recorded yes": nc.n_pct(y_rec.sum(), len(y_rec), digits=2),
        "blank patients": nc.suppress(blank.sum()),
        "expected yes among blanks (with scatter)": nc.suppress(round(p.sum())),
        "blanks whose most likely value is yes (deterministic)": nc.suppress((p > 0.5).sum()),
        "yes if blanks recoded as no": 0,
        "code": v,
    })
iabp_check = pd.DataFrame(rows)
iabp_check.to_csv(OUT_DIR / "missing_iabp_imputation_check.csv", index=False)
print(iabp_check.to_string(index=False))

# %% [markdown]
# ## 6. Blank vs no vs yes for the post-operative fields (development)
#
# For return to theatre (one source field, so its four variables are blank
# together), new dialysis and post-operative stroke: if a blank meant "no",
# the long-stay rate among blanks would match "recorded no"; if gaps were
# random, it would match "all recorded". With event rates of 1-4% the two are
# too close to tell apart, so these fields are imputed (decision 79).

# %%
rows = []
for v in ["RTT", "PostopDialysis", "PostopCVA"]:
    b, no, yes = dev[v].isna(), dev[v] == 0, dev[v] > 0
    rows.append({
        "variable": nc.label(v),
        "% blank": nc.pct_missing_text(b.sum(), len(dev)),
        "yes among recorded, %": nc.rate_text(yes.sum(), (~b).sum()),
        "long stay %: blank": nc.rate_text(dev.loc[b, "y"].sum(), b.sum()),
        "long stay %: recorded no": nc.rate_text(dev.loc[no, "y"].sum(), no.sum()),
        "long stay %: recorded yes": nc.rate_text(dev.loc[yes, "y"].sum(), yes.sum()),
        "long stay %: all recorded": nc.rate_text(dev.loc[~b, "y"].sum(), (~b).sum()),
        "code": v,
    })
postop_table = pd.DataFrame(rows)
postop_table.to_csv(OUT_DIR / "missing_postop_comparison.csv", index=False)
print(postop_table.to_string(index=False))

# %% [markdown]
# ## 7. Co-missingness (development)
#
# Row = patients missing that predictor; each cell = % of them also missing
# the column predictor. Shown for predictors missing in at least 1% of
# development patients. It shows, for example, that the return-to-theatre
# variables are always blank together (one source field) while dialysis and
# stroke overlap with them only partly (not whole forms left empty).

# %%
COMISS = [v for v in MISSING_ANY if dev[v].isna().mean() >= 0.01]
m = dev[COMISS].isna().astype(int)
pairs = m.T @ m                                   # patients missing both
co = (pairs / m.sum().values[:, None] * 100).round(0).astype(int).astype(object)
co[(pairs > 0) & (pairs < nc.SMALL_CELL)] = f"<{nc.SMALL_CELL} patients"   # privacy rule
co.index = [nc.short(v) for v in COMISS]
co.columns = [nc.short(v) for v in COMISS]
co.to_csv(OUT_DIR / "missing_comissingness.csv")
print(co.to_string())

# %% [markdown]
# ## 8. Complete cases per model (development and validation)
#
# The share of patients with every predictor recorded: the population of the
# complete-case sensitivity analysis (decision 74), and the reason complete-case
# analysis cannot be the main method.

# %%
rows = []
for model, label_ in nc.MODEL_LABELS.items():
    preds = nc.predictors(model)
    cd, cv = dev[preds].notna().all(axis=1), val[preds].notna().all(axis=1)
    rows.append({
        "model": label_,
        "predictors": len(preds),
        "complete cases, development": nc.n_pct(cd.sum(), len(dev)),
        "complete cases, validation": nc.n_pct(cv.sum(), len(val)),
    })
complete = pd.DataFrame(rows)
complete.to_csv(OUT_DIR / "missing_complete_cases.csv", index=False)
print(complete.to_string(index=False))

# %% [markdown]
# ## 9. Log

# %%
clean_name = (nc.CLEAN_FILE if nc.CLEAN_FILE.exists() else nc.CLEAN_FILE_CSV).name
log_lines = [
    "Script         : 06b_missing_data.py",
    f"Run at         : {datetime.now().isoformat(timespec='seconds')}",
    f"Clean file     : {clean_name} (schema check passed)",
    f"Development    : {len(dev)} patients, {int(dev['y'].sum())} events",
    f"Validation     : {len(val)} patients (missingness only; outcomes not used)",
    f"Predictors     : {len(ALL_PREDICTORS)}, of which {len(MISSING_ANY)} have missing values",
    f"Python {sys.version.split()[0]} | pandas {pd.__version__} | numpy {np.__version__} "
    f"| scikit-learn {__import__('sklearn').__version__}",
]
(OUT_DIR / "log.txt").write_text("\n".join(log_lines))
print("\n" + "\n".join(log_lines))

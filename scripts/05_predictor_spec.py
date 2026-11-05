# =============================================================================
# NACSA PLoS project  |  Step 5: Predictor specification
# -----------------------------------------------------------------------------
# HOW TO USE THIS FILE
#   Claude runs this file in its own workspace (decision 56). To run it
#   yourself in Colab, copy ONE block at a time (each starts with "# %%") into
#   a cell, or open the .ipynb version, and run top to bottom.
# =============================================================================

# %% [markdown]
# # Step 5: Predictor specification
#
# **What this step does:** it fixes, in code, which variables each model may
# use, and checks them against the rules we agreed. Nothing here looks at the
# outcome: every choice was made on clinical grounds, timing, overlap and data
# quality (decision 22).
#
# The decisions themselves were made in conversation (masterplan decisions 51
# and 57-68) and are written into `nacsa_common.py`, section 4b. This script:
#
# 1. applies the two modelling rules (salvage merged into emergency; "zero
#    diseased vessels" set to missing);
# 2. checks every predictor against the pre-specified rules: at most 30%
#    missing, and at least 100 development patients in every level;
# 3. writes the tables the paper needs: the predictors in each model, the
#    variables left out and why, the number of regression terms per model
#    (needed for the sample-size calculation in Step 6), and the updated
#    variable dictionary (ST1) with its "models using" column filled in.
#
# **Outputs** (in `outputs/05_predictor_spec/`):
#
# | File | What it shows | Paper |
# |---|---|---|
# | `predictor_sets.csv` | every model input, which models use it, % missing, smallest level | ST1 / methods |
# | `not_model_inputs.csv` | candidates left out of all models, with the reason | Supplement |
# | `model_summary.csv` | per model: predictors, regression terms, % complete cases | Methods, Step 6 |
# | `rule_checks.csv` | the 30% missing and 100-patient checks, pass or fail | Supplement |
# | `all_era_sets.csv`, `all_era_rule_checks.csv` | all-era sets for training sets B and C, checked on 1996-2016 | Supplement |
# | `variable_dictionary.csv` | ST1 with "models using" and "handling" filled in | ST1 |
# | `log.txt` | run details | |

# %% [markdown]
# ## 1. Set up and load
#
# Same as Step 4. In Colab, Google Drive is mounted; in Claude's workspace the
# `NACSA_DIR` setting points to the local copy of the data.
#
# Only **development** patients (2012-2016) are used for the rule checks,
# because the rules are part of building the models and validation data must
# never influence modelling choices (decision 25). Validation patients are
# loaded only for the "% missing" column of the variable dictionary, which is
# description.
#
# **What you should see:** two `load_clean:` lines.

# %%
import importlib
import os
import sys
from datetime import datetime

import numpy as np
import pandas as pd

pd.set_option("display.max_rows", 200)
pd.set_option("display.width", 250)

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

OUT_DIR = nc.OUTPUTS_DIR / "05_predictor_spec"
OUT_DIR.mkdir(parents=True, exist_ok=True)

full = nc.load_clean(period="all", outcome=None, allow_validation=True)
dev = nc.prepare_for_model(nc.load_clean(period="development", outcome="survivors"))
val = nc.prepare_for_model(nc.load_clean(period="validation", outcome="survivors", allow_validation=True))

MODELS = list(nc.PREDICTOR_SETS)
ALL_INPUTS = nc.predictors("complication_aware")
print(f"\nModel inputs in total: {len(ALL_INPUTS)}; candidates left out: {len(nc.NOT_MODEL_INPUTS)}")

# %% [markdown]
# ## 2. Levels used for modelling
#
# For categorical predictors we need the levels as the models will see them.
# For urgency and diseased vessels these differ from the descriptive tables
# (decision 65), so they come from `nc.MODEL_LEVELS`; for everything else
# from the variable registry.
#
# This block also confirms the two rules worked: no salvage code and no
# "zero vessels" code should remain.
#
# **What you should see:** two lines saying 0 remaining.

# %%
def model_levels(var):
    if var in nc.MODEL_LEVELS:
        return nc.MODEL_LEVELS[var]
    return nc.VARIABLES[var].get("levels", {})


print("Salvage codes remaining (should be 0):", int((dev["Urgency"] == 4).sum()))
print("Zero-vessel codes remaining (should be 0):", int((dev["CADExtent"] == 0).sum()))

# %% [markdown]
# ## 3. Check the pre-specified rules (decision 68)
#
# For every model input, in development patients:
#
# - **% missing** must be 30 or less;
# - for categorical predictors, the **smallest level** must have at least 100
#   patients.
#
# The script stops with an error if any predictor fails, so a rule can't be
# broken silently. The counts used are patient counts only.
#
# **What you should see:** the check table and `All predictors pass both rules.`

# %%
rows = []
for v in ALL_INPUTS:
    meta = nc.VARIABLES[v]
    pct_missing = 100 * dev[v].isna().mean()
    row = {"variable": nc.label(v), "type": meta["type"], "stage": meta["stage"],
           "% missing (development)": round(pct_missing, 1),
           "missing rule (<= 30%)": "pass" if pct_missing <= nc.MAX_PCT_MISSING else "FAIL"}
    if meta["type"] != "continuous":
        counts = [int((dev[v] == code).sum()) for code in model_levels(v)]
        smallest = min(counts)
        row["levels"] = len(counts)
        row["smallest level (patients)"] = smallest           # always >= 100 when the rule passes
        row["level rule (>= 100)"] = "pass" if smallest >= nc.MIN_LEVEL_PATIENTS else "FAIL"
    else:
        row["levels"] = ""
        row["smallest level (patients)"] = ""
        row["level rule (>= 100)"] = "not applicable"
    row["code"] = v
    rows.append(row)
checks = pd.DataFrame(rows)
checks.to_csv(OUT_DIR / "rule_checks.csv", index=False)
print(checks.drop(columns="code").to_string(index=False))

failed = checks[(checks["missing rule (<= 30%)"] == "FAIL") | (checks["level rule (>= 100)"] == "FAIL")]
assert failed.empty, f"These predictors break a pre-specified rule:\n{failed.to_string(index=False)}"
print("\nAll predictors pass both rules.")

# %% [markdown]
# ## 4. Predictors per model and regression terms
#
# **`predictor_sets.csv`**: one row per model input, with a column per model
# saying whether that model uses it.
#
# **`model_summary.csv`**: for each model, how many predictors, how many
# **regression terms** they need, and what % of development patients have every
# value recorded.
#
# *Regression terms* matter for the sample-size calculation in Step 6. A
# yes/no predictor needs one term; a categorical predictor with k levels needs
# k - 1 terms (one per level compared with the reference); a continuous
# predictor needs one term if entered as a straight line, and 3 if entered
# with a restricted cubic spline with 4 knots (allowing a curve). We report
# both, so Step 6 can use the honest, larger number.
#
# **What you should see:** a four-row summary table.

# %%
pred_rows = []
for v in ALL_INPUTS:
    row = {"variable": nc.label(v), "type": nc.VARIABLES[v]["type"], "stage": nc.VARIABLES[v]["stage"]}
    for m in MODELS:
        row[nc.MODEL_LABELS[m]] = "yes" if v in nc.PREDICTOR_SETS[m] else ""
    row["% missing (development)"] = round(100 * dev[v].isna().mean(), 1)
    row["code"] = v
    pred_rows.append(row)
pd.DataFrame(pred_rows).to_csv(OUT_DIR / "predictor_sets.csv", index=False)

SPLINE_TERMS = 3      # restricted cubic spline with 4 knots = 3 terms


def n_terms(var, splines):
    t = nc.VARIABLES[var]["type"]
    if t == "continuous":
        return SPLINE_TERMS if splines else 1
    if t == "binary":
        return 1
    return len(model_levels(var)) - 1          # ordinal and nominal: dummy-coded (decision 29)


summary = []
for m in MODELS:
    vs = nc.PREDICTOR_SETS[m]
    summary.append({
        "model": nc.MODEL_LABELS[m],
        "predictors": len(vs),
        "continuous": sum(nc.VARIABLES[v]["type"] == "continuous" for v in vs),
        "regression terms (straight lines)": sum(n_terms(v, False) for v in vs),
        "regression terms (with splines)": sum(n_terms(v, True) for v in vs),
        "% development patients with every value": round(100 * dev[vs].notna().all(axis=1).mean(), 1),
    })
summary = pd.DataFrame(summary)
summary.to_csv(OUT_DIR / "model_summary.csv", index=False)
print(summary.to_string(index=False))

# %% [markdown]
# ## 5. All-era sets for the training-period comparison (decision 69)
#
# The training-period comparison asks whether adding older patients
# (1996-2011) helps. It needs predictors recorded in every year, so four
# inputs recorded only from 2011 are replaced or removed (creatinine and
# dialysis status by creatinine above 200; the two IABP variables removed).
#
# | Training set | Years | Predictors |
# |---|---|---|
# | A | 2012-2016 | full sets (the main models) |
# | B | 2012-2016 | all-era sets |
# | C | 1996-2016 | all-era sets |
#
# The same two rules are checked, this time on 1996-2016 patients (training
# set C, the widest set that will use these predictors). The modelling rules
# (salvage merge, zero vessels missing) apply here too.
#
# **What you should see:** `All all-era predictors pass both rules.` and the
# summary rows for the all-era sets.

# %%
ERA_MODELS = list(nc.ALL_ERA_SETS)
setc = nc.prepare_for_model(nc.load_clean(period=["older", "development"], outcome="survivors"))
setc = setc[setc["Year"].between(*nc.ALL_ERA_YEARS)]
ALL_ERA_INPUTS = nc.predictors("early_postop", all_era=True)

era_rows = []
for v in ALL_ERA_INPUTS:
    meta = nc.VARIABLES[v]
    pct_missing = 100 * setc[v].isna().mean()
    row = {"variable": nc.label(v), "type": meta["type"], "stage": meta["stage"],
           "% missing (1996-2016)": round(pct_missing, 1),
           "missing rule (<= 30%)": "pass" if pct_missing <= nc.MAX_PCT_MISSING else "FAIL"}
    if meta["type"] != "continuous":
        smallest = min(int((setc[v] == code).sum()) for code in model_levels(v))
        row["smallest level (patients)"] = smallest
        row["level rule (>= 100)"] = "pass" if smallest >= nc.MIN_LEVEL_PATIENTS else "FAIL"
    else:
        row["smallest level (patients)"] = ""
        row["level rule (>= 100)"] = "not applicable"
    row["code"] = v
    era_rows.append(row)
era_checks = pd.DataFrame(era_rows)
era_checks.to_csv(OUT_DIR / "all_era_rule_checks.csv", index=False)
era_failed = era_checks[(era_checks["missing rule (<= 30%)"] == "FAIL")
                        | (era_checks["level rule (>= 100)"] == "FAIL")]
assert era_failed.empty, f"All-era predictors breaking a rule:\n{era_failed.to_string(index=False)}"
print("All all-era predictors pass both rules.")

pd.DataFrame([{"variable": nc.label(v), "stage": nc.VARIABLES[v]["stage"],
               **{nc.MODEL_LABELS[m]: "yes" if v in nc.ALL_ERA_SETS[m] else "" for m in ERA_MODELS},
               "code": v} for v in ALL_ERA_INPUTS]).to_csv(OUT_DIR / "all_era_sets.csv", index=False)

era_summary = pd.DataFrame([{
    "model": nc.MODEL_LABELS[m] + " [all-era set]",
    "predictors": len(nc.ALL_ERA_SETS[m]),
    "continuous": sum(nc.VARIABLES[v]["type"] == "continuous" for v in nc.ALL_ERA_SETS[m]),
    "regression terms (straight lines)": sum(n_terms(v, False) for v in nc.ALL_ERA_SETS[m]),
    "regression terms (with splines)": sum(n_terms(v, True) for v in nc.ALL_ERA_SETS[m]),
    "% development patients with every value": round(100 * dev[nc.ALL_ERA_SETS[m]].notna().all(axis=1).mean(), 1),
} for m in ERA_MODELS])
summary = pd.concat([summary, era_summary], ignore_index=True)
summary.to_csv(OUT_DIR / "model_summary.csv", index=False)
print(era_summary.to_string(index=False))

# %% [markdown]
# ## 6. Candidates left out of all models
#
# Each one with the reason and the masterplan decision number. These stay in
# the data and in the descriptive tables (Step 4); they just don't enter any
# model.
#
# **What you should see:** 13 rows.

# %%
left_out = pd.DataFrame([{"variable": nc.label(v), "reason (decision)": r, "code": v}
                         for v, r in nc.NOT_MODEL_INPUTS.items()])
left_out.to_csv(OUT_DIR / "not_model_inputs.csv", index=False)
print(left_out.drop(columns="code").to_string(index=False))

# %% [markdown]
# ## 7. Variable dictionary (ST1), now with "models using" and "handling"
#
# Same layout as the Step 4 draft, with two columns filled in:
#
# - **models using**: which of the four models use the variable (M1-M3, S),
#   and which use it in the all-era version for training sets B and C (B/C);
# - **handling**: how it enters the models (or why it doesn't). Missing
#   values in model inputs are imputed inside the modelling pipeline
#   (decisions 27 and 40); the details are set in Step 6.
#
# **What you should see:** the first rows of the dictionary.

# %%
SHORT_MODEL = {"preop": "M1", "end_of_surgery": "M2", "early_postop": "M3", "complication_aware": "S"}


def levels_text(var):
    meta = nc.VARIABLES[var]
    if "levels" not in meta:
        return meta["label"].split(", ")[-1] if ", " in meta["label"] else ""
    text = "; ".join(f"{k} = {v}" for k, v in meta["levels"].items())
    if var in nc.MODEL_LEVELS:
        text += " | modelling: " + "; ".join(f"{k} = {v}" for k, v in nc.MODEL_LEVELS[var].items())
    return text


def handling(var):
    if var in nc.NOT_MODEL_INPUTS:
        in_all_era = any(var in vs for vs in nc.ALL_ERA_SETS.values())
        prefix = "Not in the main models; all-era sets only: " if in_all_era else "Not a model input: "
        return prefix + nc.NOT_MODEL_INPUTS[var]
    if var == "ES2":
        return "Benchmark (recalibrated EuroSCORE II, model M0); not a model input"
    t = nc.VARIABLES[var]["type"]
    how = {"continuous": "continuous (spline in regression)", "binary": "yes/no",
           "ordinal": "ordered categories (dummy-coded in regression)",
           "nominal": "categories (dummy-coded in regression)"}[t]
    extra = {"Urgency": "; salvage merged into emergency (65)",
             "CADExtent": "; 'none' set to missing (65)"}.get(var, "")
    return f"Model input, {how}{extra}; missing values imputed"


DICT_VARS = nc.variables(["pre", "intra", "post"]) + ["ES2"]
present_by_year = full.groupby("Year")[DICT_VARS].apply(lambda d: d.notna().mean())
dict_rows = []
for v in DICT_VARS:
    meta = nc.VARIABLES[v]
    recorded = present_by_year.index[present_by_year[v] >= 0.5]
    ref = meta.get("ref", "")
    using = [SHORT_MODEL[m] for m in MODELS if v in nc.PREDICTOR_SETS[m]]
    using += [f"B/C {SHORT_MODEL[m]}" for m in ERA_MODELS if v in nc.ALL_ERA_SETS[m]]
    dict_rows.append({
        "code_name": v,
        "label": meta["label"],
        "source_or_derivation": meta["source"],
        "type": meta["type"],
        "units_or_levels": levels_text(v),
        "reference_level": meta["levels"].get(ref, ref) if "levels" in meta else "",
        "stage": meta["stage"],
        "models_using": ", ".join(using) if using else "none",
        "first_year_recorded": int(recorded.min()) if len(recorded) else "",
        "pct_missing_development": nc.pct_missing_text(dev[v].isna().sum(), len(dev)),
        "pct_missing_validation": nc.pct_missing_text(val[v].isna().sum(), len(val)),
        "handling": handling(v),
    })
dictionary = pd.DataFrame(dict_rows)
dictionary.to_csv(OUT_DIR / "variable_dictionary.csv", index=False)
print("M1 = pre-operative, M2 = end of surgery, M3 = early post-operative, S = complication-aware;"
      " B/C = all-era version in training sets B and C\n")
print(dictionary[["code_name", "label", "models_using", "handling"]].head(15).to_string(index=False))

# %% [markdown]
# ## 8. Log and list of files

# %%
clean_name = (nc.CLEAN_FILE if nc.CLEAN_FILE.exists() else nc.CLEAN_FILE_CSV).name
log_lines = [
    "Script         : 05_predictor_spec.py",
    f"Run at         : {datetime.now().isoformat(timespec='seconds')}",
    f"Clean file     : {clean_name} ({len(full)} rows; schema check passed)",
    f"Development    : {len(dev)} patients (rules checked here)",
    f"Model inputs   : " + "; ".join(f"{m} {len(nc.PREDICTOR_SETS[m])}" for m in MODELS),
    f"All-era sets   : " + "; ".join(f"{m} {len(nc.ALL_ERA_SETS[m])}" for m in ERA_MODELS)
    + f" (checked on {len(setc)} patients, 1996-2016)",
    f"Python {sys.version.split()[0]} | pandas {pd.__version__} | numpy {np.__version__}",
]
(OUT_DIR / "log.txt").write_text("\n".join(log_lines))
print("\n".join(log_lines))
print("\nSaved files:")
for f in sorted(OUT_DIR.iterdir()):
    print("  ", f.name)

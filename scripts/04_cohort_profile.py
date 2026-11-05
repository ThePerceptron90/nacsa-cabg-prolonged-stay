# =============================================================================
# NACSA PLoS project  |  Step 4: Cohort profile
# -----------------------------------------------------------------------------
# HOW TO USE THIS FILE IN COLAB
#   The file is split into blocks. Each block starts with a line "# %%".
#   Copy ONE block at a time into a Colab cell, read the comments, then run it
#   with Shift + Enter. Go top to bottom. (Or use the .ipynb version.)
#
#   Before the first run, put nacsa_common.py in MyDrive/nacsa/code/.
# =============================================================================

# %% [markdown]
# # Step 4: Cohort profile
#
# **What this step does:** it *describes* the patients. It fits no model and
# makes no decision. Everything it produces is a count, a percentage or a
# summary, so all outputs are safe to share (small counts are hidden).
#
# **What it produces** (all in `MyDrive/nacsa/outputs/04_cohort_profile/`):
#
# | File | What it shows | Paper |
# |---|---|---|
# | `flow_counts.csv` | records -> exclusions -> analysed, per period | Figure 1 numbers |
# | `table1_main.csv` | key characteristics, development vs validation, with SMDs | Table 1 |
# | `table_baseline_full.csv` | every candidate variable, same layout | Supplement |
# | `table_by_outcome.csv` | development patients with vs without prolonged stay | ST6 |
# | `table_deaths_vs_analysed.csv` | who option B excludes | Supplement |
# | `missing_overall.csv`, `missing_by_year.csv/.png`, `missing_by_sex_age.csv` | missing data | ST5 |
# | `category_counts.csv` | patients in every level of every category | ST2 |
# | `associations_matrix.csv/.png`, `associations_top_pairs.csv`, `vif.csv` | overlap between predictors | SF4 |
# | `outcome_by_year.csv/.png` | outcome rates 1996-2019 | SF3 |
# | `outcome_by_sex_age.csv` | outcome rate by sex and age band | TRIPOD+AI 20b |
# | `variable_dictionary_draft.csv` | first draft of the variable dictionary | ST1 |
# | `participants_events.csv` | patients and events per analysis | ST20 |
# | `log.txt` | when it ran, on which file, package versions | |
#
# **Which patients?** Unless a table says otherwise, the analysed cohort is the
# primary analysis (option B): patients **discharged alive** with a known
# length of stay. The validation patients (2017-2019) are loaded only to be
# *described* here, which is allowed by the validation lock (decision 25).

# %% [markdown]
# ## 1. Set up
#
# Three things happen:
#
# 1. Google Drive is mounted (as in Steps 1-3). When Claude runs the same code
#    in its own workspace there is no Drive to mount, so this is skipped.
# 2. Python is told where `nacsa_common.py` lives (`sys.path` is the list of
#    folders Python searches when you write `import ...`), then imports it as
#    `nc`. From now on, `nc.something` means "something from the shared file".
# 3. `importlib.reload` makes sure that, if you replace `nacsa_common.py` with a
#    newer version, Colab uses the new one without restarting.
#
# **What you should see:** `nacsa_common loaded from .../code/nacsa_common.py`
# and the output folder.

# %%
import importlib
import sys
from datetime import datetime
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

pd.set_option("display.max_rows", 300)
pd.set_option("display.max_columns", 30)
pd.set_option("display.width", 250)

try:                                   # in Colab: mount Google Drive
    from google.colab import drive
    drive.mount("/content/drive")
except ImportError:                    # outside Colab (Claude's workspace): nothing to mount
    pass

import os
CODE_DIR = os.environ.get("NACSA_CODE_DIR", "/content/drive/MyDrive/nacsa/code")
if CODE_DIR not in sys.path:
    sys.path.insert(0, CODE_DIR)
import nacsa_common as nc
importlib.reload(nc)
print("nacsa_common loaded from", nc.__file__)

OUT_DIR = nc.OUTPUTS_DIR / "04_cohort_profile"
OUT_DIR.mkdir(parents=True, exist_ok=True)
print("Outputs go to", OUT_DIR)

# %% [markdown]
# ## 2. Load the data
#
# We load three views of the same clean file, always through `nc.load_clean`:
#
# - `full`: every record 1996-2019, whatever its outcome. Used for the flow
#   diagram and the outcome rates by year.
# - `dev`: development patients (2012-2016) analysed for the primary outcome.
# - `val`: validation patients (2017-2019) analysed for the primary outcome.
#   Note `allow_validation=True`: without it the loader refuses (try it once
#   if you're curious: you'll get the VALIDATION LOCK message).
#
# Each call first checks the file still has 377,301 rows and the same 63
# columns, then adds the Step 4 variables: `diabetes_any`, `stroke_any`,
# `smoking_ever`, `outcome_all` (sensitivity A), `age_band`, and renames the
# composite outcome to `outcome_composite`. It also creates `y`, the outcome of
# the analysis (1 = stay over 10 days).
#
# **What you should see:** three `load_clean:` lines. Development should show
# about 75,800 analysed with about 12,000 events; validation about 32,100 and
# 4,600.

# %%
full = nc.load_clean(period="all", outcome=None, allow_validation=True)
dev = nc.load_clean(period="development", outcome="survivors")
val = nc.load_clean(period="validation", outcome="survivors", allow_validation=True)

PREDICTORS = nc.variables(["pre", "intra", "post"])          # every candidate predictor
ALL_VARS = PREDICTORS + ["ES2"]                               # plus the benchmark
print(f"\n{len(PREDICTORS)} candidate predictors: "
      f"{len(nc.variables('pre'))} pre-op, {len(nc.variables('intra'))} intra-op, "
      f"{len(nc.variables('post'))} post-op")

# %% [markdown]
# ## 3. Flow counts (numbers for Figure 1)
#
# For each period we count, in order: all records; records excluded because
# the patient **died in hospital** (option B); records excluded because the
# stay or the survival status is **unknown**; and the patients **analysed**,
# with the number and rate of prolonged stays.
#
# The last rows give the same counts for the two sensitivity outcomes, so
# Figure 1 can show all three.
#
# **What you should see:** a table with three columns (older, development,
# validation).

# %%
def flow_for(d):
    died = d["Mortality"] == 1
    alive = d["Mortality"] == 0
    status_unknown = d["Mortality"].isna()
    analysed = d["outcome_survivors"].notna()
    out = {
        "All records": len(d),
        "Excluded: died in hospital": int(died.sum()),
        "Excluded: survival status unknown": int(status_unknown.sum()),
        "Excluded: discharged alive, length of stay unknown": int((alive & ~analysed).sum()),
        "Analysed (primary, discharged alive)": int(analysed.sum()),
        "Events: stay over 10 days (primary)": int((d["outcome_survivors"] == 1).sum()),
        "Sensitivity C (composite): analysed": int(d["outcome_composite"].notna().sum()),
        "Sensitivity C (composite): events": int((d["outcome_composite"] == 1).sum()),
        "Sensitivity A (all patients): analysed": int(d["outcome_all"].notna().sum()),
        "Sensitivity A (all patients): events": int((d["outcome_all"] == 1).sum()),
    }
    return out


flow_raw = pd.DataFrame({p: flow_for(full[full["period"] == p]) for p in nc.PERIODS})

# check the arithmetic: the exclusions and the analysed patients must add up to all records
parts = flow_raw.iloc[1:5].sum()
assert (parts == flow_raw.loc["All records"]).all(), "Flow counts do not add up"

flow = flow_raw.copy().astype(object)
for p in flow.columns:
    flow[p] = [nc.suppress(v) for v in flow_raw[p]]
for name, n_row, e_row in [("Rate, primary (%)", "Analysed (primary, discharged alive)",
                            "Events: stay over 10 days (primary)"),
                           ("Rate, composite (%)", "Sensitivity C (composite): analysed",
                            "Sensitivity C (composite): events"),
                           ("Rate, all patients (%)", "Sensitivity A (all patients): analysed",
                            "Sensitivity A (all patients): events")]:
    flow.loc[name] = [nc.rate_text(flow_raw.loc[e_row, p], flow_raw.loc[n_row, p]) for p in flow.columns]
flow.to_csv(OUT_DIR / "flow_counts.csv")
print(flow.to_string())

# %% [markdown]
# ## 4. Table helpers
#
# These functions build the descriptive tables. You don't need to follow every
# line; what matters is what they do:
#
# **Continuous variables** are shown either as mean (SD) or as median [IQR].
# We decide once, from the development cohort: if the distribution is clearly
# skewed (absolute skewness above 1, e.g. bypass time, creatinine), the median
# [IQR] is used, otherwise the mean (SD). The same choice is used in every
# table, and the `statistic` column says which one it is.
#
# **Yes/no variables** show the number and % of "yes". **Categories** show one
# row per level. Percentages are of patients with a known value; the
# `% missing` columns say how many were unknown.
#
# **SMD (standardised mean difference)** measures how different two groups are,
# in units that don't depend on sample size. Unlike a p-value, it doesn't
# become "significant" just because there are 100,000 patients. Rule of thumb:
# below 0.1 the groups are essentially similar; 0.1-0.2 is a small difference;
# above 0.2 is worth a sentence in the paper. For categories with several
# levels, one SMD summarises the whole variable (the Yang and Dalton method).
#
# **Privacy:** counts of 1-9 become "<10". If that happens to exactly one level
# of a variable, the next smallest level is hidden too and shown as
# "suppressed" (it is not a small number), so the small one can't be worked
# out by subtraction.
#
# **What you should see:** `Statistic chosen for ... continuous variables`.

# %%
def level_map(var, *series):
    """Levels of a categorical variable in display order, plus any code the data
    has that the registry doesn't know (it would be flagged, not hidden)."""
    levels = dict(nc.VARIABLES[var].get("levels", {}))
    for s in series:
        for code in pd.unique(s.dropna()):
            if code not in levels:
                levels[code] = f"Unexpected code {code}"
                print(f"  NOTE: {var} has a code not in the registry: {code}")
    return levels


def smd_continuous(a, b):
    a, b = a.dropna().astype(float), b.dropna().astype(float)
    if len(a) < 2 or len(b) < 2:
        return np.nan
    pooled = np.sqrt((a.var() + b.var()) / 2)
    return abs(a.mean() - b.mean()) / pooled if pooled > 0 else 0.0


def smd_categorical(a, b, levels):
    """Yang and Dalton SMD; for two levels it equals the usual binary SMD."""
    a, b = a.dropna(), b.dropna()
    if len(a) == 0 or len(b) == 0:
        return np.nan
    pa = np.array([(a == lv).mean() for lv in levels])
    pb = np.array([(b == lv).mean() for lv in levels])
    if len(levels) == 2:
        p1, p2 = pa[1], pb[1]
        den = np.sqrt((p1 * (1 - p1) + p2 * (1 - p2)) / 2)
        return abs(p1 - p2) / den if den > 0 else 0.0
    pa, pb = pa[1:], pb[1:]
    S = (np.diag(pa) - np.outer(pa, pa) + np.diag(pb) - np.outer(pb, pb)) / 2
    d = pa - pb
    return float(np.sqrt(max(d @ np.linalg.pinv(S) @ d, 0.0)))


# one choice of statistic per continuous variable, made on the development cohort
STAT = {}
for v in nc.variables(types="continuous", include_benchmark=True):
    skew = dev[v].dropna().astype(float).skew()
    STAT[v] = "median [IQR]" if abs(skew) > 1 else "mean (SD)"
print(f"Statistic chosen for {len(STAT)} continuous variables:")
print(pd.Series(STAT).to_string())


def summarise_continuous(s, stat):
    s = s.dropna().astype(float)
    if len(s) < nc.SMALL_CELL:
        return f"<{nc.SMALL_CELL} values"
    if stat == "median [IQR]":
        q = s.quantile([0.25, 0.5, 0.75])
        return f"{q[0.5]:.1f} [{q[0.25]:.1f}-{q[0.75]:.1f}]"
    return f"{s.mean():.1f} ({s.std():.1f})"


def describe(groups, var_list, smd_between=None):
    """Build a descriptive table.

    groups       {column title: DataFrame}, in the order the columns should appear
    var_list     variable names (from the registry)
    smd_between  two column titles to compare with an SMD, or None
    """
    names = list(groups)
    rows = [{"variable": "Patients, n", "level": "", "statistic": "n",
             **{g: f"{len(d):,}" for g, d in groups.items()}}]
    for var in var_list:
        meta = nc.VARIABLES[var]
        kind = meta["type"]
        miss = {f"% missing, {g}": nc.pct_missing_text(d[var].isna().sum(), len(d)) for g, d in groups.items()}
        smd = np.nan
        if kind == "continuous":
            if smd_between:
                smd = smd_continuous(groups[smd_between[0]][var], groups[smd_between[1]][var])
            rows.append({"variable": meta["label"], "level": "", "statistic": STAT[var],
                         **{g: summarise_continuous(d[var], STAT[var]) for g, d in groups.items()},
                         "SMD": smd, **miss})
            continue
        levels = level_map(var, *[d[var] for d in groups.values()])
        if smd_between:
            smd = smd_categorical(groups[smd_between[0]][var], groups[smd_between[1]][var], list(levels))
        if kind == "binary":
            row = {"variable": meta["label"], "level": "", "statistic": "n (%)"}
            for g, d in groups.items():
                known = d[var].notna().sum()
                row[g] = nc.n_pct((d[var] == 1).sum(), known)
            rows.append({**row, "SMD": smd, **miss})
            continue
        # ordinal or nominal: a header row, then one row per level
        rows.append({"variable": meta["label"], "level": "", "statistic": "n (%)",
                     **{g: "" for g in names}, "SMD": smd, **miss})
        shown = {}
        for g, d in groups.items():
            counts = [int((d[var] == lv).sum()) for lv in levels]
            known = int(d[var].notna().sum())
            sup = nc.suppress_counts(counts)
            shown[g] = [c if isinstance(c, str) else nc.n_pct(c, known) for c in sup]
        for i, label in enumerate(levels.values()):
            rows.append({"variable": "", "level": label, "statistic": "",
                         **{g: shown[g][i] for g in names}})
    table = pd.DataFrame(rows)
    if "SMD" in table:
        table["SMD"] = table["SMD"].map(lambda x: "" if pd.isna(x) else f"{x:.2f}")
    return table.fillna("")


print("\nTable helpers ready.")

# %% [markdown]
# ## 5. Table 1 and the full baseline table (development vs validation)
#
# **Table 1 (main text)** follows the EuroSCORE II framework, the variables a
# cardiac surgeon expects to see, plus a few operative ones (the "hybrid"
# approach we agreed). **The full table** (supplement) lists every candidate
# predictor and the EuroSCORE II score itself.
#
# Both compare the development and validation cohorts. Large SMDs here are not
# a problem to fix; they describe how practice changed over time (e.g. fewer
# off-pump operations), which the temporal validation is designed to test.
#
# **What you should see:** Table 1 printed below.

# %%
TABLE1_VARS = ["Age", "Gender", "BMI", "diabetes_any", "Hypertension", "smoking_ever",
               "PulmonaryDisease", "PVD", "stroke_any", "NeuroDys", "PrevSurgery", "crcl",
               "PreopDialysis", "CCS", "NYHA", "RecentMI", "lv_impaired", "PAP", "CPS", "Urgency",
               "CADExtent", "PCI", "PumpCase", "DistalGrafts", "ES2"]

cohorts = {"Development 2012-2016": dev, "Validation 2017-2019": val}
pair = ("Development 2012-2016", "Validation 2017-2019")

table1 = describe(cohorts, TABLE1_VARS, smd_between=pair)
table1.to_csv(OUT_DIR / "table1_main.csv", index=False)

baseline = describe(cohorts, ALL_VARS, smd_between=pair)
baseline.to_csv(OUT_DIR / "table_baseline_full.csv", index=False)

print(table1.to_string(index=False))

# %% [markdown]
# ## 6. Development patients by outcome (ST6)
#
# The same layout, comparing development patients who stayed 10 days or less
# with those who stayed longer. This shows which characteristics go with a long
# stay. It is **description only**: it is not used to choose predictors
# (decision 22 forbids outcome-driven selection).
#
# **What you should see:** the first rows of the table.

# %%
by_outcome = describe({"Stay 10 days or less": dev[dev["y"] == 0], "Stay over 10 days": dev[dev["y"] == 1]},
                      ALL_VARS, smd_between=("Stay 10 days or less", "Stay over 10 days"))
by_outcome.to_csv(OUT_DIR / "table_by_outcome.csv", index=False)
print(by_outcome.head(25).to_string(index=False))

# %% [markdown]
# ## 7. Who does option B leave out?
#
# The primary analysis excludes patients who died in hospital. This table
# compares them with the analysed patients (2012-2019 together, so the counts
# are large enough to show). We expect the deaths to be older and sicker
# (shock, salvage operations, ventilation): this table shows by how much, and
# answers a reviewer's first question about option B.
#
# **What you should see:** the first rows of the table.

# %%
modern = full[full["period"].isin(["development", "validation"])]
deaths_table = describe({"Analysed (discharged alive)": modern[modern["outcome_survivors"].notna()],
                         "Died in hospital (excluded)": modern[modern["Mortality"] == 1]},
                        ALL_VARS, smd_between=("Analysed (discharged alive)", "Died in hospital (excluded)"))
deaths_table.to_csv(OUT_DIR / "table_deaths_vs_analysed.csv", index=False)
print(deaths_table.head(25).to_string(index=False))

# %% [markdown]
# ## 8. Missing data (ST5)
#
# Three views:
#
# 1. **Overall:** % missing per variable, development vs validation.
# 2. **By year:** % missing per variable for each year 2012-2019, as a table and
#    a heatmap (darker = more missing). A variable that is complete in some
#    years and empty in others was introduced or retired, which matters for the
#    models.
# 3. **By sex and age band** (development): TRIPOD+AI asks whether data quality
#    differs between groups (items 7 and 20b). If, say, creatinine is missing
#    far more often in older patients, imputation has to account for it.
#
# Remember the registry convention: many yes/no fields were filled with 0 in
# the extract, so they show 0% missing even if they were left blank (decision
# 16). Low missingness is not always real completeness.
#
# **What you should see:** the overall table, then the heatmap.

# %%
miss_overall = pd.DataFrame({
    "variable": [nc.label(v) for v in ALL_VARS],
    "stage": [nc.VARIABLES[v]["stage"] for v in ALL_VARS],
    "% missing, development": [nc.pct_missing_text(dev[v].isna().sum(), len(dev)) for v in ALL_VARS],
    "% missing, validation": [nc.pct_missing_text(val[v].isna().sum(), len(val)) for v in ALL_VARS],
    "code": ALL_VARS,
})
miss_overall.to_csv(OUT_DIR / "missing_overall.csv", index=False)
print(miss_overall.to_string(index=False))

analysed_modern = pd.concat([dev, val])
miss_year = analysed_modern.groupby("Year")[ALL_VARS].apply(lambda d: d.isna().mean() * 100).T.round(1)
(miss_year.set_axis([nc.label(v) for v in miss_year.index]).rename_axis("variable")
 .assign(code=miss_year.index).to_csv(OUT_DIR / "missing_by_year.csv"))

fig, ax = plt.subplots(figsize=(8, 0.28 * len(ALL_VARS) + 1.5))
im = ax.imshow(miss_year.values, aspect="auto", cmap="Greys", vmin=0, vmax=100)
ax.set_xticks(range(miss_year.shape[1]), [int(c) for c in miss_year.columns])
ax.set_yticks(range(miss_year.shape[0]), [nc.short(v) for v in miss_year.index], fontsize=7)
ax.axvline(4.5, color="tab:red", lw=1)        # boundary between 2016 and 2017
ax.set_title("% missing by year (analysed patients); red line = development | validation")
fig.colorbar(im, ax=ax, label="% missing", shrink=0.3)
fig.tight_layout()
fig.savefig(OUT_DIR / "missing_by_year.png", dpi=200)
plt.show()

sex_label = dev["Gender"].map({0: "Male", 1: "Female"})
rows = {}
for name, mask in [("Male", sex_label == "Male"), ("Female", sex_label == "Female")] + \
                  [(f"Age {b}", dev["age_band"] == b) for b in nc.AGE_BAND_LABELS]:
    sub = dev[mask]
    rows[name] = [nc.pct_missing_text(sub[v].isna().sum(), len(sub)) for v in ALL_VARS]
miss_sex_age = pd.DataFrame(rows, index=pd.Index([nc.label(v) for v in ALL_VARS], name="variable"))
miss_sex_age["code"] = ALL_VARS
miss_sex_age.to_csv(OUT_DIR / "missing_by_sex_age.csv")
print("\nSaved missing_by_sex_age.csv")

# %% [markdown]
# ## 9. Patients in every category (ST2)
#
# For every yes/no and categorical variable: how many patients sit in each
# level, in development and validation, plus a row for missing. A level with
# very few development patients can't be estimated reliably and may need
# merging with a neighbour in Step 5. The `small_in_development` column flags
# levels with fewer than 100 development patients.
#
# The original multi-level Diabetes, Stroke and Smoking fields are included
# too, so the paper can show what the yes/no versions were made from.
#
# Merging is decided on **patient counts and clinical sense only, never on
# outcome rates** (decision 29), which is why this table has no outcome in it.
#
# **What you should see:** the first rows of the table.

# %%
RAW_MULTI = {
    "Diabetes": ("Diabetes (original field)", {0: "None", 1: "Diet", 2: "Oral", 3: "Insulin"}),
    "Stroke": ("Stroke history (original field)", {0: "None", 1: "TIA", 2: "Stroke, recovered",
                                                   3: "Stroke, residual deficit"}),
    "Smoking": ("Smoking (original field)", {0: "Never", 1: "Ex-smoker", 2: "Current"}),
}
cat_rows = []
cat_vars = [(v, nc.VARIABLES[v]["label"], None) for v in nc.variables(types=["binary", "ordinal", "nominal"])]
cat_vars += [(v, lab, lv) for v, (lab, lv) in RAW_MULTI.items()]
for var, label, levels in cat_vars:
    levels = levels or level_map(var, dev[var], val[var])
    per = {}
    for g, d in [("development", dev), ("validation", val)]:
        counts = [int((d[var] == lv).sum()) for lv in levels] + [int(d[var].isna().sum())]
        per[g] = (counts, nc.suppress_counts(counts))
    for i, lab in enumerate(list(levels.values()) + ["Missing"]):
        raw_dev = per["development"][0][i]
        cat_rows.append({
            "variable": label, "level": lab,
            "n development": per["development"][1][i],
            "% development": "" if isinstance(per["development"][1][i], str)
                             else round(100 * raw_dev / len(dev), 1),
            "n validation": per["validation"][1][i],
            "% validation": "" if isinstance(per["validation"][1][i], str)
                            else round(100 * per["validation"][0][i] / len(val), 1),
            "small_in_development": "yes" if (lab != "Missing" and raw_dev < 100) else "",
            "code": var,
        })
category_counts = pd.DataFrame(cat_rows)
category_counts.to_csv(OUT_DIR / "category_counts.csv", index=False)
print(category_counts.head(30).to_string(index=False))
print("\nLevels with fewer than 100 development patients:")
print(category_counts[category_counts["small_in_development"] == "yes"][["variable", "level", "n development"]]
      .to_string(index=False))

# %% [markdown]
# ## 10. How much do the predictors overlap? (SF4)
#
# Two variables "overlap" when knowing one tells you a lot about the other
# (e.g. creatinine and creatinine clearance). Overlap doesn't hurt prediction,
# but it makes the explanation (SHAP) split credit between the two, and in
# regression it makes single coefficients unstable. We **measure** it here and
# **decide** in Step 5 (decision 20); nothing is dropped automatically.
#
# **One number per pair, between 0 (no relation) and 1 (identical).** Which
# measure depends on the two variable types:
#
# | Pair | Measure |
# |---|---|
# | number/ordered x number/ordered | Spearman correlation (absolute value) |
# | yes-no/category x yes-no/category | Cramer's V |
# | number/ordered x yes-no/category | correlation ratio (eta) |
#
# Pairs at 0.7 or above are flagged. Everything uses development patients only
# (validation data never influence modelling choices).
#
# **VIF (variance inflation factor)** asks a different question: how well can
# each predictor be predicted from **all the others together**? A VIF of 1
# means not at all; above 5 means a lot; above 10 means the predictor is
# almost fully explained by the others. It's computed for the three
# cumulative predictor sets (pre-op; pre + intra; pre + intra + post) on
# patients with every value known, leaving out variables missing in more than
# half of development patients (they're listed). VIF is documentation only.
#
# This block takes a minute or two.
#
# **What you should see:** the flagged pairs, then the largest VIFs per set.

# %%
NUMERIC_LIKE = set(nc.variables(types=["continuous", "ordinal"]))


def cramers_v(x, y):
    tab = pd.crosstab(x, y).values.astype(float)
    n = tab.sum()
    if n == 0 or min(tab.shape) < 2:
        return np.nan
    expected = tab.sum(1, keepdims=True) @ tab.sum(0, keepdims=True) / n
    chi2 = ((tab - expected) ** 2 / expected).sum()
    return float(np.sqrt(chi2 / (n * (min(tab.shape) - 1))))


def correlation_ratio(categories, values):
    ok = categories.notna() & values.notna()
    c, v = categories[ok], values[ok].astype(float)
    if len(v) == 0 or v.var() == 0:
        return np.nan
    means = v.groupby(c).agg(["mean", "size"])
    between = (means["size"] * (means["mean"] - v.mean()) ** 2).sum()
    return float(np.sqrt(between / ((v - v.mean()) ** 2).sum()))


num_vars = [v for v in PREDICTORS if v in NUMERIC_LIKE]
spearman = dev[num_vars].astype(float).corr(method="spearman").abs()

assoc = pd.DataFrame(np.nan, index=PREDICTORS, columns=PREDICTORS)
measure = pd.DataFrame("", index=PREDICTORS, columns=PREDICTORS)
for i, a in enumerate(PREDICTORS):
    for b in PREDICTORS[i + 1:]:
        if a in NUMERIC_LIKE and b in NUMERIC_LIKE:
            value, how = spearman.loc[a, b], "Spearman"
        elif a not in NUMERIC_LIKE and b not in NUMERIC_LIKE:
            value, how = cramers_v(dev[a], dev[b]), "Cramer's V"
        else:
            cat, num = (a, b) if b in NUMERIC_LIKE else (b, a)
            value, how = correlation_ratio(dev[cat], dev[num]), "eta"
        assoc.loc[a, b] = assoc.loc[b, a] = value
        measure.loc[a, b] = measure.loc[b, a] = how
for v in PREDICTORS:
    assoc.loc[v, v] = 1.0
short_names = [nc.short(v) for v in PREDICTORS]
assoc.round(2).set_axis(short_names, axis=0).set_axis(short_names, axis=1) \
    .to_csv(OUT_DIR / "associations_matrix.csv")

pairs = [{"variable_1": nc.label(a), "variable_2": nc.label(b), "measure": measure.loc[a, b],
          "strength": round(assoc.loc[a, b], 2), "code_1": a, "code_2": b}
         for i, a in enumerate(PREDICTORS) for b in PREDICTORS[i + 1:] if assoc.loc[a, b] >= 0.4]
top_pairs = pd.DataFrame(pairs).sort_values("strength", ascending=False)
top_pairs.insert(4, "flag_0.7", np.where(top_pairs["strength"] >= 0.7, "yes", ""))
top_pairs.to_csv(OUT_DIR / "associations_top_pairs.csv", index=False)
print("Pairs with strength 0.4 or above (flag = 0.7 or above):")
print(top_pairs.to_string(index=False))

fig, ax = plt.subplots(figsize=(16, 14))
im = ax.imshow(assoc.values, cmap="viridis", vmin=0, vmax=1)
ax.set_xticks(range(len(PREDICTORS)), short_names, rotation=90, fontsize=7)
ax.set_yticks(range(len(PREDICTORS)), short_names, fontsize=7)
ax.set_title("Strength of association between candidate predictors (development, 2012-2016)")
fig.colorbar(im, ax=ax, shrink=0.7, label="0 = none, 1 = identical")
fig.tight_layout()
fig.savefig(OUT_DIR / "associations_matrix.png", dpi=200)
plt.show()


def vif_for(d, var_list):
    share_missing = d[var_list].isna().mean()
    left_out = share_missing[share_missing > 0.5].index.tolist()
    parts = []
    for v in var_list:
        if v in left_out:
            continue
        meta = nc.VARIABLES[v]
        if meta["type"] == "nominal":
            # one 0/1 column per level except the reference; missing stays missing
            for code, label in level_map(v, d[v]).items():
                if code != meta["ref"]:
                    parts.append(((d[v] == code).astype(float).where(d[v].notna()))
                                 .rename(f"{nc.label(v)}: {label}"))
        else:
            parts.append(d[v].astype(float).rename(nc.label(v)))
    X = pd.concat(parts, axis=1).dropna()
    X = X.loc[:, X.std() > 0]
    Z = ((X - X.mean()) / X.std()).values
    # VIF_j = 1 / (1 - R2_j), where R2_j comes from regressing term j on all
    # the other terms. A term fully determined by the others (R2 = 1, e.g.
    # "off-pump" and "on-pump") gets an infinite VIF instead of a misleading number.
    vifs = {}
    for j, name in enumerate(X.columns):
        others = np.delete(Z, j, axis=1)
        coef, *_ = np.linalg.lstsq(others, Z[:, j], rcond=None)
        resid = Z[:, j] - others @ coef
        r2 = 1 - resid.var() / Z[:, j].var()
        vifs[name] = np.inf if r2 > 1 - 1e-9 else 1 / (1 - r2)
    return pd.Series(vifs), len(X), left_out


vif_rows = []
for set_name, stages in [("pre-op", ["pre"]), ("pre + intra-op", ["pre", "intra"]),
                         ("pre + intra + post-op", ["pre", "intra", "post"])]:
    vif, n_used, left_out = vif_for(dev, nc.variables(stages))
    left_out = [nc.label(v) for v in left_out]
    print(f"\n{set_name}: VIF on {n_used:,} complete development patients; "
          f"left out (over 50% missing): {left_out or 'none'}")
    print(vif.sort_values(ascending=False).head(8).round(1).to_string())
    vif_rows += [{"predictor_set": set_name, "term": t,
                  "VIF": "infinite (fully determined by other terms)" if np.isinf(x) else round(x, 2),
                  "n_complete_patients": n_used, "left_out_over_50pct_missing": "; ".join(left_out)}
                 for t, x in vif.items()]
pd.DataFrame(vif_rows).to_csv(OUT_DIR / "vif.csv", index=False)

# %% [markdown]
# ## 11. Outcome rates by year, sex and age band (SF3)
#
# **By year (1996-2019):** the rate of prolonged stay among survivors (primary),
# the two sensitivity versions, and in-hospital mortality. This is the drift
# we discussed: the outcome became less common over time, which is the main
# reason models need recalibrating when used in later years.
#
# **By sex and age band:** the primary outcome rate in each group, development
# and validation. Needed for TRIPOD+AI and to set up the fairness analysis
# (Step 17).
#
# **What you should see:** a table by year, a line chart, and the sex/age table.

# %%
year_rows = []
for year, d in full.groupby("Year"):
    surv, comp, allp = d["outcome_survivors"], d["outcome_composite"], d["outcome_all"]
    mort = d["Mortality"]
    year_rows.append({
        "Year": int(year), "period": d["period"].iloc[0],
        "n analysed (primary)": nc.suppress(surv.notna().sum()),
        "events (primary)": nc.suppress((surv == 1).sum()),
        "rate primary %": nc.rate_text((surv == 1).sum(), surv.notna().sum()),
        "rate composite %": nc.rate_text((comp == 1).sum(), comp.notna().sum()),
        "rate all patients %": nc.rate_text((allp == 1).sum(), allp.notna().sum()),
        "in-hospital mortality %": nc.rate_text((mort == 1).sum(), mort.notna().sum()),
    })
by_year = pd.DataFrame(year_rows)
by_year.to_csv(OUT_DIR / "outcome_by_year.csv", index=False)
print(by_year.to_string(index=False))

fig, ax = plt.subplots(figsize=(9, 4.5))
for col, style in [("rate primary %", "-o"), ("rate composite %", "--"), ("rate all patients %", ":"),
                   ("in-hospital mortality %", "-s")]:
    y = pd.to_numeric(by_year[col], errors="coerce")
    ax.plot(by_year["Year"], y, style, ms=3, label=col.replace(" %", ""))
ax.axvspan(2011.5, 2016.5, color="tab:blue", alpha=0.08, label="development")
ax.axvspan(2016.5, 2019.5, color="tab:orange", alpha=0.08, label="validation")
ax.set_xlabel("Year of operation")
ax.set_ylabel("%")
ax.set_ylim(bottom=0)
ax.legend(fontsize=8, ncol=2)
ax.set_title("Outcome rates by year of operation")
fig.tight_layout()
fig.savefig(OUT_DIR / "outcome_by_year.png", dpi=200)
plt.show()

sa_rows = []
for g, d in [("development", dev), ("validation", val)]:
    groups = [("All", d)] + [(s, d[d["Gender"] == code]) for s, code in [("Male", 0), ("Female", 1)]] + \
             [(f"Age {b}", d[d["age_band"] == b]) for b in nc.AGE_BAND_LABELS]
    for name, sub in groups:
        sa_rows.append({"cohort": g, "group": name, "n": nc.suppress(len(sub)),
                        "events": nc.suppress(sub["y"].sum()), "rate %": nc.rate_text(sub["y"].sum(), len(sub))})
by_sex_age = pd.DataFrame(sa_rows)
by_sex_age.to_csv(OUT_DIR / "outcome_by_sex_age.csv", index=False)
print(by_sex_age.to_string(index=False))

# %% [markdown]
# ## 12. Draft variable dictionary (ST1) and patients per analysis (ST20)
#
# **ST1** is built from the registry in `nacsa_common.py` plus numbers measured
# here: first year the variable is recorded (first year with at least half the
# values present; filled-in zeros make some fields look recorded from 1996, see
# decision 16) and % missing. The columns `models_using` and `handling` are
# filled in after Step 5.
#
# **ST20** starts here: one row per analysis with its patients and events.
# Later steps add their own rows.
#
# **What you should see:** the first rows of the dictionary.

# %%
def levels_text(meta):
    if "levels" not in meta:
        return meta["label"].split(", ")[-1] if ", " in meta["label"] else ""
    return "; ".join(f"{k} = {v}" for k, v in meta["levels"].items())


present_by_year = full.groupby("Year")[ALL_VARS].apply(lambda d: d.notna().mean())
dict_rows = []
for v in ALL_VARS:
    meta = nc.VARIABLES[v]
    recorded = present_by_year.index[present_by_year[v] >= 0.5]
    ref = meta.get("ref", "")
    dict_rows.append({
        "code_name": v,
        "label": meta["label"],
        "source_or_derivation": meta["source"],
        "type": meta["type"],
        "units_or_levels": levels_text(meta),
        "reference_level": meta["levels"].get(ref, ref) if "levels" in meta else "",
        "stage": meta["stage"],
        "models_using": "(Step 5)",
        "first_year_recorded": int(recorded.min()) if len(recorded) else "",
        "pct_missing_development": nc.pct_missing_text(dev[v].isna().sum(), len(dev)),
        "pct_missing_validation": nc.pct_missing_text(val[v].isna().sum(), len(val)),
        "handling": "(Steps 5-6)",
        "statistic_in_tables": STAT.get(v, "n (%)"),
    })
dictionary = pd.DataFrame(dict_rows)
dictionary.to_csv(OUT_DIR / "variable_dictionary_draft.csv", index=False)
print(dictionary.head(12).to_string(index=False))

participants = pd.DataFrame([
    {"step": "4", "analysis": "Cohort description, primary outcome", "cohort": "development 2012-2016",
     "patients": nc.suppress(len(dev)), "events": nc.suppress(dev["y"].sum())},
    {"step": "4", "analysis": "Cohort description, primary outcome", "cohort": "validation 2017-2019",
     "patients": nc.suppress(len(val)), "events": nc.suppress(val["y"].sum())},
])
participants.to_csv(OUT_DIR / "participants_events.csv", index=False)
print("\n", participants.to_string(index=False))

# %% [markdown]
# ## 13. Log and list of files
#
# **What to send back** (from `MyDrive/nacsa/outputs/04_cohort_profile/`):
# every file in the list printed below. They are all aggregate, with small
# counts hidden, so they're safe to share. `nacsa_clean.parquet` stays on your
# Drive.

# %%
log_lines = [
    "Script         : 04_cohort_profile.py",
    f"Run at         : {datetime.now().isoformat(timespec='seconds')}",
    f"Clean file     : {(nc.CLEAN_FILE if nc.CLEAN_FILE.exists() else nc.CLEAN_FILE_CSV).name} "
    f"({len(full)} rows; schema check passed)",
    f"Analysed       : development {len(dev)} (events {int(dev['y'].sum())}), "
    f"validation {len(val)} (events {int(val['y'].sum())})",
    f"Python {sys.version.split()[0]} | pandas {pd.__version__} | numpy {np.__version__} "
    f"| matplotlib {matplotlib.__version__}",
]
(OUT_DIR / "log.txt").write_text("\n".join(log_lines))
print("\n".join(log_lines))
print("\nSaved files:")
for f in sorted(OUT_DIR.iterdir()):
    print("  ", f.name)

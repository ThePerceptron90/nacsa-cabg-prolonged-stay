# =============================================================================
# NACSA PLoS project  |  Step 3: Cleaning
# -----------------------------------------------------------------------------
# HOW TO USE THIS FILE IN COLAB
#   The file is split into blocks. Each block starts with a line "# %%".
#   Copy ONE block at a time into a Colab cell, read the comments, then run it
#   with Shift + Enter. Go top to bottom. (Or use the .ipynb version.)
# =============================================================================

# %% [markdown]
# # Step 3: Cleaning
#
# **What this step does:** applies the cleaning rules we agreed, one at a time,
# and saves a clean copy of the data. This is the first step that *changes*
# data, so it's built to be transparent:
#
# - **The raw file is only read, never overwritten.** The clean file goes into
#   a separate `processed/` folder.
# - **Every rule is logged** with the number of rows it changed. That log
#   becomes a supplementary table in the paper.
# - **All the decisions live in two places at the top** (the drop list in
#   Section 2, the plausible ranges in Section 3). If you change your mind about
#   a column, you edit one line and rerun everything. Nothing is hidden deeper
#   in the code.
#
# **Cleaning is not variable selection.** We remove columns that are broken,
# duplicated or meaningless. Columns that are fine but may or may not become
# predictors are kept; choosing predictors is Phase 2.
#
# **The clean file keeps all years (1996-2019)**, because the training-period
# comparison (Model 3) needs the older patients. The split into older /
# development / validation happens at modelling time, using the `period` column
# this step creates.
#
# **What you'll send back:** five small summary files (Section 12).

# %% [markdown]
# ## 1. Set up and load the raw data
#
# Same as Steps 1 and 2, with two additions:
#
# - `PROCESSED_DIR`: a new folder `MyDrive/nacsa/processed/` for the clean
#   data file. Keeping raw, processed and outputs in separate folders means you
#   can always tell at a glance which file is which.
# - `record_id`: we give every row a permanent number (its row position in the
#   raw file) **before** changing anything. NACSA has no patient ID, so this is
#   our only way to trace a cleaned row back to its raw row if we ever need to
#   check something.
#
# **What you should see:** `Loaded raw data: 380,293 rows x 129 columns`.

# %%
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

pd.set_option("display.max_rows", 300)
pd.set_option("display.max_columns", 60)
pd.set_option("display.width", 250)

from google.colab import drive
drive.mount("/content/drive")

PROJECT_DIR   = Path("/content/drive/MyDrive/nacsa")
DATA_FILE     = "NACSA.csv"            # <-- CHANGE THIS if your file has another name
DATA_PATH     = PROJECT_DIR / "raw" / DATA_FILE
PROCESSED_DIR = PROJECT_DIR / "processed"
OUT_DIR       = PROJECT_DIR / "outputs" / "03_cleaning"
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
OUT_DIR.mkdir(parents=True, exist_ok=True)


def load_data(path):
    """Read the raw data file, whatever its format. Never writes anything."""
    suffix = path.suffix.lower()
    if suffix == ".csv":
        return pd.read_csv(path, low_memory=False)
    if suffix in (".xlsx", ".xls"):
        return pd.read_excel(path)
    if suffix == ".parquet":
        return pd.read_parquet(path)
    if suffix == ".rds":
        import pyreadr
        return pyreadr.read_r(str(path))[None]
    if suffix == ".sav":
        return pd.read_spss(path)
    if suffix == ".dta":
        return pd.read_stata(path)
    raise ValueError(f"Unknown file type: {suffix}")


assert DATA_PATH.exists(), f"File not found: {DATA_PATH}"
raw = load_data(DATA_PATH)
raw = pd.concat([pd.Series(np.arange(len(raw)), name="record_id"), raw.reset_index(drop=True)], axis=1)
print(f"Loaded raw data: {raw.shape[0]:,} rows x {raw.shape[1] - 1} columns")

# %% [markdown]
# ## 2. THE DROP LIST (the main thing to review)
#
# This is a Python **dictionary**: a lookup table of `key: value` pairs. Each
# key is a column name; each value is a pair `(group, reason)`. The script
# drops every column listed here and writes the whole list, with reasons, to
# `dropped_columns.csv` so you can review it.
#
# **To keep a column**, put a `#` at the start of its line (this "comments it
# out", so Python ignores it) and rerun the script from the top.
#
# Groups:
# - **A** duplicates and EuroSCORE helper columns
# - **B** unreliable, or only recorded in some years
# - **C** constant (same value for everyone)
# - **D** replaced by a cleaner derived column built in this script (these are
#   used during cleaning and dropped at the end)
#
# The block finishes with a safety check: every listed column must exist in
# the data, so a typo can't silently do nothing.
#
# **What you should see:** `80 columns will be dropped: A 42 | B 25 | C 2 | D 11`.

# %%
DROP = {
    # --- A: duplicates and EuroSCORE helper columns -------------------------
    "e2":                     ("A", "ES2 divided by 100"),
    "Age_Imputed":            ("A", "Copy of Age prepared for score formulas"),
    "Age_Imputed_ES2":        ("A", "Age floored at 60 for EuroSCORE II"),
    "Age_Imputed_LogES":      ("A", "Age rescaled for logistic EuroSCORE"),
    "Gender_Imputed":         ("A", "Garbled copy of Gender"),
    "Height_Imputed":         ("A", "Height with blanks filled; raw Height kept"),
    "Weight_Imputed":         ("A", "Weight with blanks filled; raw Weight kept"),
    "Cr.Gender":              ("A", "Sex constant for clearance formula (1 or 0.85)"),
    "CrCl":                   ("A", "Set to 100 when creatinine missing; recalculated as crcl"),
    "CrCl.Category":          ("A", "Category of the imputed CrCl"),
    "CrCl.Category.Moderate": ("A", "Dummy of the imputed CrCl category"),
    "CrCl.Category.Severe":   ("A", "Dummy of the imputed CrCl category"),
    "CrCl.Category.Dialysis": ("A", "Dummy of the imputed CrCl category"),
    "LVF1":                   ("A", "Derived from LVEF with missing set to 'good'"),
    "LVF2":                   ("A", "Derived from LVEF with missing set to 'good'"),
    "LVF1.Moderate":          ("A", "LV dummy, labels inconsistent"),
    "LVF1.Poor":              ("A", "LV dummy, labels inconsistent"),
    "LVF2.Moderate":          ("A", "LV dummy, labels inconsistent"),
    "LVF2.Poor":              ("A", "LV dummy, labels inconsistent"),
    "LVF2.VeryPoor":          ("A", "LV dummy, labels inconsistent"),
    ".LVF1.Moderate":         ("A", "LV dummy, mostly empty"),
    ".LVF2.Moderate":         ("A", "LV dummy, mostly empty"),
    ".LVF2.Poor":             ("A", "LV dummy, mostly empty"),
    "Urgency.Urgent":         ("A", "Dummy of Urgency"),
    "Urgency.Emergency":      ("A", "Dummy of Urgency"),
    "Urgency.Salvage":        ("A", "Dummy of Urgency"),
    "Urgency.Log.Emergency":  ("A", "Dummy of Urgency (logistic EuroSCORE definition)"),
    "Diabetes.Diet":          ("A", "Dummy of Diabetes"),
    "Diabetes.Oral":          ("A", "Dummy of Diabetes"),
    "Diabetes.Insulin":       ("A", "Dummy of Diabetes"),
    "CCS4":                   ("A", "Dummy of CCS"),
    "NYHA2":                  ("A", "Dummy of NYHA"),
    "NYHA3":                  ("A", "Dummy of NYHA"),
    "NYHA4":                  ("A", "Dummy of NYHA"),
    "PAP.Moderate":           ("A", "Dummy of PAP"),
    "PAP.Severe":             ("A", "Dummy of PAP"),
    "PreopAF":                ("A", "PreopHR split up, missing set to 0"),
    "PreopVFT":               ("A", "PreopHR split up, missing set to 0"),
    "PreopCHB.pacing":        ("A", "PreopHR split up, missing set to 0"),
    "PreopOtherHR":           ("A", "PreopHR split up, missing set to 0"),
    "Total.Graft.Sites":      ("A", "Identical to DistalGrafts"),
    "PoorMobility":           ("A", "Mobility OR NeuroDys; meaning changes in 2017"),
    # --- B: unreliable, or only recorded in some years ----------------------
    "LMS":                    ("B", "Essentially unused field (473 positives)"),
    "Impeller":               ("B", "Contradicts its own reason field"),
    "Impeller.Reason":        ("B", "Goes with Impeller"),
    "VAD":                    ("B", "25 patients; too rare"),
    "VAD.Reason":             ("B", "Goes with VAD"),
    "Other.Mech.Support":     ("B", "69 patients; too rare"),
    "Other.Mech.Support.Reason": ("B", "Goes with Other.Mech.Support"),
    "IABP.Reason":            ("B", "Describes the device; not a predictor"),
    "PAsys":                  ("B", "0 used for 'not measured'; PAP kept instead"),
    "CircArrest":             ("B", "Not relevant to isolated CABG; 99.6% zero"),
    "Plegia.Timing":          ("B", "Almost no variation"),
    "Median.Sternotomy":      ("B", "Incision fields only exist from 2017"),
    "Partial.Sternotomy":     ("B", "Incision fields only exist from 2017"),
    "Thoracotomy":            ("B", "Incision fields only exist from 2017"),
    "Mini.Thoracotomy":       ("B", "Incision fields only exist from 2017"),
    "Robotic":                ("B", "Incision fields only exist from 2017"),
    "Other.Incision":         ("B", "Incision fields only exist from 2017"),
    "Multiple.Incision":      ("B", "Incision fields only exist from 2017"),
    "Conversion":             ("B", "Only exists from 2017"),
    "Mobility":               ("B", "Only exists from 2017"),
    "Nitrates":               ("B", "Field retired in 2017-2019"),
    "PrevCEA":                ("B", "Field retired from 2012; PVD covers vascular disease"),
    "PrevPVD":                ("B", "Field retired from 2012; PVD covers vascular disease"),
    "NumberGrafts":           ("B", "Second graft count that disagrees; DistalGrafts kept"),
    "PostopDSWI.treatment":   ("B", "Treatment detail of a late complication"),
    # --- C: constant --------------------------------------------------------
    "CABG":                   ("C", "Always 1"),
    "PostInfarctVSD":         ("C", "Always 0"),
    # --- D: replaced by a cleaner derived column (dropped at the end) -------
    "LVEF":                   ("D", "Used to check LVEFC; replaced by lv_impaired"),
    "LVEFC":                  ("D", "Replaced by lv_impaired"),
    "MyoProt":                ("D", "Replaced by myo_protection"),
    "IABP":                   ("D", "Replaced by iabp_preop / iabp_intraop / iabp_postop"),
    "RTT.bleeding.tamponade": ("D", "Replaced by rtt_bleeding"),
    "RTT.valve.other":        ("D", "Merged into rtt_other_cardiac"),
    "RTT.graft":              ("D", "Merged into rtt_other_cardiac"),
    "RTT.cardiac.other":      ("D", "Merged into rtt_other_cardiac"),
    "RTT.rhythmn":            ("D", "Merged into rtt_other_cardiac"),
    "RTT.multiple":           ("D", "Flag for >1 reason; reasons themselves are kept"),
    "RTT.wound":              ("D", "Replaced by rtt_wound"),
}

missing_from_data = [c for c in DROP if c not in raw.columns]
assert not missing_from_data, f"These DROP entries are not in the data (typo?): {missing_from_data}"
groups = pd.Series({c: g for c, (g, _) in DROP.items()}).value_counts().sort_index()
print(f"{len(DROP)} columns will be dropped: " + " | ".join(f"{g} {n}" for g, n in groups.items()))

# %% [markdown]
# ## 3. Plausible ranges
#
# Values outside these ranges are physically impossible or almost certainly
# typing errors (e.g. height typed in metres). They are set to **missing**,
# never deleted along with the patient: the rest of that patient's record is
# still good.
#
# Ranges follow the NICOR data dictionary limits where it gives them, tightened
# to clinically realistic values for adults having CABG.

# %%
RANGES = {              # column: (lowest allowed, highest allowed)
    "Height": (120, 220),   # cm
    "Weight": (30, 250),    # kg (NICOR upper limit 250)
    "BMI":    (12, 70),     # kg/m2, checked after recalculation
    "CPB":    (0, 600),     # minutes
}
LOS_CUTOFF = 10             # prolonged stay = more than 10 whole days

# %% [markdown]
# ## 4. Helpers: the cleaning log
#
# `log(step, rule, n)` records one line: which step, what the rule did, and how
# many rows it changed. At the end, the whole log is saved as a CSV.
#
# `suppress` hides counts 1-9 as before. The log saved for sharing uses it;
# the clean data file itself is not affected.
#
# **What you should see:** `Log ready.`

# %%
SMALL_CELL = 10
LOG = []


def suppress(n):
    n = int(n)
    return f"<{SMALL_CELL}" if 0 < n < SMALL_CELL else n


def log(step, rule, n):
    LOG.append({"step": step, "rule": rule, "rows_changed": int(n)})
    print(f"[{step}] {rule}: {int(n):,} rows")


print("Log ready.")

# %% [markdown]
# ## 5. Two checks before cleaning (nothing changes here)
#
# **5a. Fractional length of stay.** In some years LOS was calculated from
# date *and time*, giving values like 8.04 days. We'll round **down** to whole
# days, the standard way LOS is counted (days since the operation date). This
# check shows how many values that affects, and how many patients sit between
# 10 and 11 days, the only ones whose outcome could change.
#
# **5b. What do the LVEFC codes mean in 2012-2019?** We compare LVEFC with the
# actual ejection fraction within the modern period only. For now the script
# uses the safest reading, which the Step 2 audit supports: code 3 = good
# (EF > 50%), any other code = impaired. If this table shows clean finer grades,
# we'll add them in a later version.
#
# **What you should see:** two small tables.

# %%
los_raw = raw["PostLOS"]
fractional = los_raw.notna() & (los_raw % 1 != 0)
period_raw = pd.cut(raw["Year"], [1995, 2011, 2016, 2019],
                    labels=["older", "development", "validation"]).rename("period")
check_a = pd.DataFrame({
    "n_fractional_LOS": fractional.groupby(period_raw, observed=False).sum(),
    "n_between_10_and_11_days": ((los_raw > 10) & (los_raw < 11)).groupby(period_raw, observed=False).sum(),
}).map(suppress)
print("5a. Fractional LOS values, by period\n", check_a.to_string(), "\n")

modern = raw["Year"] >= 2012
ef_band = pd.cut(raw["LVEF"], [0, 20, 30, 50, 100], labels=["EF 00-20", "EF 21-30", "EF 31-50", "EF 51+"])
ef_band = ef_band.astype(object).where(raw["LVEF"].notna(), "missing")
check_b = pd.crosstab(raw.loc[modern, "LVEFC"].fillna(-1).astype(int).replace(-1, "missing").astype(str),
                      ef_band[modern]).map(suppress)
print("5b. LVEFC (rows) by actual EF band, 2012-2019 only\n", check_b.to_string())
n_ef50 = ((raw["LVEF"] == 50) & (raw["LVEFC"] == 3) & modern).sum()
print(f"\nPatients 2012-2019 with EF exactly 50% coded as LVEFC 3 (good): {suppress(n_ef50)}")

# %% [markdown]
# ## 6. Row-level step: remove exact duplicates
#
# `drop_duplicates(keep="first")` keeps the first copy of each identical row
# and removes the rest. We ignore `record_id` when comparing (it's different
# for every row by construction), otherwise nothing would ever match.
#
# From here on we work on `df`, a copy. `raw` stays untouched in memory in case
# you want to compare.
#
# **What you should see:** about 2,992 rows removed.

# %%
data_cols = [c for c in raw.columns if c != "record_id"]
df = raw.drop_duplicates(subset=data_cols, keep="first").copy()
log("R1", "Exact duplicate rows removed (first copy kept)", len(raw) - len(df))
print(f"Rows now: {len(df):,}")

# %% [markdown]
# ## 7. Outcome: length of stay, death, and the two outcome definitions
#
# **The fixes, in order:**
# 1. Negative LOS is impossible -> missing.
# 2. LOS of 0 days for a patient who went home alive is impossible -> missing.
#    (0 days for a patient who died is genuine and kept.)
# 3. `los_days` = LOS rounded down to whole days (`np.floor`).
#
# **The outcomes:**
# - `outcome_primary` (prolonged stay or death): 1 if died in hospital, or if
#   `los_days` > 10; 0 if alive with `los_days` <= 10; otherwise missing. As
#   agreed, a short stay with unknown survival status is left missing, so
#   those patients drop out at modelling.
# - `outcome_survivors` (the sensitivity analysis): only patients known to
#   have gone home alive; 1 if `los_days` > 10, else 0. Deaths are missing.
#
# `np.select([conditions], [values], default)` checks the conditions in order
# and uses the first one that's true, like a chain of "if ... else if ...".
#
# **What you should see:** four log lines, then the event rates for each
# outcome (expect roughly 16-17% and 15-16%).

# %%
neg = df["PostLOS"] < 0
df.loc[neg, "PostLOS"] = np.nan
log("O1", "Negative PostLOS set to missing", neg.sum())

zero_alive = (df["PostLOS"] == 0) & (df["Mortality"] == 0)
df.loc[zero_alive, "PostLOS"] = np.nan
log("O2", "PostLOS of 0 days in survivors set to missing", zero_alive.sum())

df["los_days"] = np.floor(df["PostLOS"])
changed = df["PostLOS"].notna() & (df["los_days"] != df["PostLOS"])
log("O3", "Fractional PostLOS rounded down to whole days (los_days)", changed.sum())
crossed = (df["PostLOS"] > LOS_CUTOFF) & (df["los_days"] <= LOS_CUTOFF)
log("O3b", "...of which moved from over 10 to 10 days by rounding", crossed.sum())

died, los = df["Mortality"], df["los_days"]
df["outcome_primary"] = np.select(
    [died == 1, los > LOS_CUTOFF, (los <= LOS_CUTOFF) & (died == 0)],
    [1.0, 1.0, 0.0], default=np.nan)
df["outcome_survivors"] = np.where((died == 0) & los.notna(), (los > LOS_CUTOFF).astype(float), np.nan)
log("O4", "outcome_primary could not be determined (missing)", df["outcome_primary"].isna().sum())

for col in ["outcome_primary", "outcome_survivors"]:
    print(f"{col}: {int(df[col].notna().sum()):,} with a value, event rate {100 * df[col].mean():.1f}%")

# %% [markdown]
# ## 8. Body size
#
# Height and weight outside the plausible ranges become missing. Then BMI is
# **recalculated** from the cleaned values (weight in kg / height in m
# squared), so the derivation is ours and consistent. The supplied BMI is
# overwritten, not kept alongside.
#
# `.between(low, high)` is True when a value is inside the range (inclusive).
# We set to missing the values that are present **and** outside the range.

# %%
for col in ["Height", "Weight"]:
    low, high = RANGES[col]
    bad = df[col].notna() & ~df[col].between(low, high)
    df.loc[bad, col] = np.nan
    log("B1", f"{col} outside {low}-{high} set to missing", bad.sum())

df["BMI"] = df["Weight"] / (df["Height"] / 100) ** 2
low, high = RANGES["BMI"]
bad = df["BMI"].notna() & ~df["BMI"].between(low, high)
df.loc[bad, "BMI"] = np.nan
log("B2", f"BMI recalculated from height and weight; outside {low}-{high} set to missing", bad.sum())

# %% [markdown]
# ## 9. Operation details
#
# **Bypass time (`CPB`)** and **cross-clamp time (`XClamp`)**:
# - Off-pump (`PumpCase` = 0): no bypass, so the time is 0 by definition.
#   Blanks become 0. An off-pump patient with a positive clamp time is a
#   contradiction, so that time becomes missing.
# - On-pump (`PumpCase` = 1) with 0 bypass minutes is impossible -> missing.
#   On-pump with 0 clamp minutes is genuine (beating-heart surgery) -> kept.
# - `PumpCase` missing: left exactly as recorded, as you decided (no inference).
#
# **Myocardial protection** becomes one column, `myo_protection`, with three
# values: `off_pump`, `cardioplegia`, `no_cardioplegia` (on-pump without it).
#
# **Grafts:** `DistalGrafts` of 0 is impossible in CABG -> missing. For the
# four target vessels, a blank means "not grafted" (0) when the graft count is
# known; if the count is missing, the vessels are missing too.
#
# Note the order in the code: we find the contradictory off-pump clamp times
# **first**, before filling off-pump blanks with 0, so the two rules can't
# overwrite each other.

# %%
off, on = df["PumpCase"] == 0, df["PumpCase"] == 1

fill = off & df["CPB"].isna()
df.loc[off, "CPB"] = 0
log("P1", "Off-pump: CPB set to 0 (was blank)", fill.sum())
bad = on & (df["CPB"] == 0)
df.loc[bad, "CPB"] = np.nan
log("P2", "On-pump with CPB = 0 set to missing", bad.sum())
bad = df["CPB"] > RANGES["CPB"][1]
df.loc[bad, "CPB"] = np.nan
log("P3", f"CPB over {RANGES['CPB'][1]} min set to missing", bad.sum())

contradiction = off & (df["XClamp"] > 0)
fill = off & df["XClamp"].isna()
df.loc[contradiction, "XClamp"] = np.nan
df.loc[fill, "XClamp"] = 0
log("P4", "Off-pump with positive XClamp set to missing", contradiction.sum())
log("P5", "Off-pump: XClamp set to 0 (was blank)", fill.sum())

df["myo_protection"] = np.select(
    [off, on & (df["MyoProt"] == 1), on & (df["MyoProt"] == 0)],
    ["off_pump", "cardioplegia", "no_cardioplegia"], default="")
df["myo_protection"] = df["myo_protection"].replace("", np.nan)
log("P6", "myo_protection could not be determined (missing)", df["myo_protection"].isna().sum())

bad = df["DistalGrafts"] == 0
df.loc[bad, "DistalGrafts"] = np.nan
log("G1", "DistalGrafts = 0 set to missing", bad.sum())
known = df["DistalGrafts"].notna()
for col in ["RCA", "LAD", "Diag", "Cx"]:
    filled = known & df[col].isna()
    df.loc[filled, col] = 0
    df.loc[~known, col] = np.nan
    log("G2", f"{col}: blank set to 0 (not grafted) where graft count known", filled.sum())

# %% [markdown]
# ## 10. Renal function, LV function, IABP and return to theatre
#
# **`crcl`** (creatinine clearance, mL/min) is recalculated with the
# Cockcroft-Gault formula, the one EuroSCORE II uses, for creatinine in umol/L:
#
# `crcl = (140 - age) x weight x k / creatinine`, with k = 1.23 for men and
# 1.04 for women.
#
# It's only calculated where creatinine, age, weight and sex are all known;
# otherwise it stays missing (no more fake 100s).
#
# **`lv_impaired`**: 0 if `LVEFC` = 3 (good, EF > 50%), 1 for any other code,
# missing if `LVEFC` is missing. See check 5b.
#
# **IABP** splits by timing into three yes/no columns for the three stage
# models. Where `IABP` is missing (mostly before 2011), all three are missing.
#
# **Return to theatre** becomes three columns, each missing where `RTT` itself
# is missing (no more "missing = no"):
# - `rtt_bleeding`: bleeding or tamponade (early event)
# - `rtt_other_cardiac`: graft, valve, other cardiac or rhythm problems
# - `rtt_wound`: sternal resuturing or wound surgery (late event)
#
# **`period`**: older (1996-2011), development (2012-2016), validation (2017-2019).

# %%
k = df["Gender"].map({0: 1.23, 1: 1.04})          # missing sex -> missing k -> missing crcl
df["crcl"] = (140 - df["Age"]) * df["Weight"] * k / df["Creatinine"]
log("K1", "crcl calculated (Cockcroft-Gault)", df["crcl"].notna().sum())

df["lv_impaired"] = np.where(df["LVEFC"].isna(), np.nan, (df["LVEFC"] != 3).astype(float))
log("L1", "lv_impaired could not be determined (missing)", df["lv_impaired"].isna().sum())

iabp_known = df["IABP"].notna()
for name, code in [("iabp_preop", 1), ("iabp_intraop", 2), ("iabp_postop", 3)]:
    df[name] = np.where(iabp_known, (df["IABP"] == code).astype(float), np.nan)
    log("I1", f"{name} = 1", (df[name] == 1).sum())

rtt_known = df["RTT"].notna()
other = df[["RTT.graft", "RTT.valve.other", "RTT.cardiac.other", "RTT.rhythmn"]].max(axis=1)
df["rtt_bleeding"]      = np.where(rtt_known, df["RTT.bleeding.tamponade"], np.nan)
df["rtt_other_cardiac"] = np.where(rtt_known, other, np.nan)
df["rtt_wound"]         = np.where(rtt_known, df["RTT.wound"], np.nan)
for name in ["rtt_bleeding", "rtt_other_cardiac", "rtt_wound"]:
    log("T1", f"{name} = 1", (df[name] == 1).sum())

df["period"] = pd.cut(df["Year"], [1995, 2011, 2016, 2019],
                      labels=["older", "development", "validation"]).astype(str)
print(df["period"].value_counts().to_string())

# %% [markdown]
# ## 11. Drop the listed columns and tidy up
#
# We remove every column in the DROP list, then put the columns in a sensible
# order that mirrors the study design: identifiers and outcomes first, then
# pre-op, intra-op and post-op variables, then the EuroSCORE II benchmark.
# This ordering is only for readability; which variables enter each stage's
# model is decided in Phase 2.
#
# **What you should see:** `Clean data: ~377,300 rows x 63 columns` and the
# list of columns kept.

# %%
df = df.drop(columns=list(DROP))
ORDER = (
    ["record_id", "Year", "period", "Country",
     "outcome_primary", "outcome_survivors", "Mortality", "los_days", "PostLOS"]
    # pre-op
    + ["Age", "Gender", "Height", "Weight", "BMI", "Smoking", "Diabetes", "Hypertension",
       "PulmonaryDisease", "PVD", "Stroke", "NeuroDys", "PrevMI", "RecentMI", "UA", "PCI",
       "PrevSurgery", "PrevCABG", "CCS", "NYHA", "PreopHR", "CADExtent", "lv_impaired", "PAP",
       "PulHTN", "CardiogenicShock", "Inotropes", "VentilatedPreop", "CPS", "Urgency",
       "Cr200", "Creatinine", "crcl", "PreopDialysis", "iabp_preop"]
    # intra-op
    + ["PumpCase", "CPB", "XClamp", "myo_protection", "DistalGrafts", "RCA", "LAD", "Diag", "Cx",
       "iabp_intraop"]
    # post-op
    + ["RTT", "rtt_bleeding", "rtt_other_cardiac", "rtt_wound", "PostopCVA", "PostopDialysis",
       "PostopDSWI", "iabp_postop"]
    # benchmark
    + ["ES2"]
)
leftover = [c for c in df.columns if c not in ORDER]
assert not leftover, f"Columns not placed in ORDER: {leftover}"
df = df[ORDER]
print(f"Clean data: {df.shape[0]:,} rows x {df.shape[1]} columns\n")
print("Columns kept:\n" + ", ".join(df.columns))

# %% [markdown]
# ## 12. Save the clean data and the summaries
#
# **The clean data** is saved as **Parquet**, a compact file format that,
# unlike CSV, remembers each column's type (number, text, category). Later
# steps read it with `pd.read_parquet`. It stays in `processed/` and is
# patient-level, so it is **not** sent to Claude.
#
# **What to send back** (from `MyDrive/nacsa/outputs/03_cleaning/`):
#
# 1. `cleaning_log.csv`: every rule and how many rows it changed
# 2. `dropped_columns.csv`: every dropped column with its group and reason,
#    for your review
# 3. `flow_counts.csv`: patients per period and outcome counts
# 4. `clean_overview.csv`: one summary line per column of the clean data
#    (same format as Step 1)
# 5. `checks.txt` and `log.txt`
#
# **What you should see:** the list of saved files.

# %%
df.to_parquet(PROCESSED_DIR / "nacsa_clean.parquet", index=False)

pd.DataFrame(LOG).assign(rows_changed=lambda t: t["rows_changed"].map(suppress)) \
    .to_csv(OUT_DIR / "cleaning_log.csv", index=False)

pd.DataFrame([{"column": c, "group": g, "reason": r} for c, (g, r) in DROP.items()]) \
    .to_csv(OUT_DIR / "dropped_columns.csv", index=False)

flow = df.groupby("period").agg(
    n_patients=("record_id", "size"),
    n_outcome_primary=("outcome_primary", "count"),
    events_primary=("outcome_primary", "sum"),
    n_outcome_survivors=("outcome_survivors", "count"),
    events_survivors=("outcome_survivors", "sum"),
    n_died=("Mortality", lambda s: (s == 1).sum()),
)
flow["rate_primary_%"] = (100 * flow["events_primary"] / flow["n_outcome_primary"]).round(2)
flow["rate_survivors_%"] = (100 * flow["events_survivors"] / flow["n_outcome_survivors"]).round(2)
count_cols = [c for c in flow.columns if not c.startswith("rate")]
flow[count_cols] = flow[count_cols].map(suppress)
flow.to_csv(OUT_DIR / "flow_counts.csv")
print(flow.to_string(), "\n")


def describe_column(s):
    """One summary line per column, same rules as Step 1 (no min/max, small counts hidden)."""
    present = s.dropna()
    row = {"column": s.name, "dtype": str(s.dtype), "pct_missing": round(100 * s.isna().mean(), 1),
           "n_unique": int(present.nunique())}
    if row["n_unique"] == 0:
        return {**row, "summary": "ALL MISSING"}
    if pd.api.types.is_numeric_dtype(s) and row["n_unique"] > 10:
        q = present.quantile([0.01, 0.5, 0.99])
        row["summary"] = f"p1={q.iloc[0]:.4g} | median={q.iloc[1]:.4g} | p99={q.iloc[2]:.4g}"
    else:
        counts = present.value_counts().head(8)
        row["summary"] = " | ".join(f"{k}: {suppress(v)}" for k, v in counts.items())
    return row


pd.DataFrame([describe_column(df[c]) for c in df.columns if c != "record_id"]) \
    .to_csv(OUT_DIR / "clean_overview.csv", index=False)

(OUT_DIR / "checks.txt").write_text(
    "5a. Fractional LOS values, by period\n" + check_a.to_string()
    + "\n\n5b. LVEFC (rows) by actual EF band, 2012-2019 only\n" + check_b.to_string()
    + f"\n\nEF exactly 50% coded LVEFC 3, 2012-2019: {suppress(n_ef50)}\n")

log_lines = [
    "Script         : 03_clean.py",
    f"Run at         : {datetime.now().isoformat(timespec='seconds')}",
    f"Raw file       : {DATA_PATH.name} ({len(raw)} rows)",
    f"Clean file     : nacsa_clean.parquet ({df.shape[0]} rows x {df.shape[1]} columns)",
    f"Python {sys.version.split()[0]} | pandas {pd.__version__} | numpy {np.__version__}",
]
(OUT_DIR / "log.txt").write_text("\n".join(log_lines))
print("\n".join(log_lines))
print("\nSaved files:")
for f in sorted(OUT_DIR.iterdir()):
    print("  ", f.name)
print("  ", PROCESSED_DIR / "nacsa_clean.parquet", "(stays on your Drive; do not share)")

# =============================================================================
# NACSA PLoS project  |  Step 2: Deep audit (still look-only, nothing changes)
# -----------------------------------------------------------------------------
# HOW TO USE THIS FILE IN COLAB
#   The file is split into blocks. Each block starts with a line "# %%".
#   1. Open a new, empty Colab notebook (or continue in the Step 1 notebook).
#   2. Copy ONE block (from one "# %%" line down to the next), paste it into a
#      Colab cell, read the comments, then run it with Shift + Enter.
#   3. Repeat block by block, top to bottom.
#   Lines starting with "#" are comments: explanations for you. Python ignores them.
# =============================================================================

# %% [markdown]
# # Step 2: Deep audit
#
# **What this step does:** answers the open questions from Step 1 and the data
# dictionary, by cross-tabulating columns against each other and against year.
#
# **What it does not do:** it changes nothing. The raw file is only read.
#
# **The two key tools in this step:**
#
# 1. **Cross-tabulation (cross-tab).** A table that counts patients for every
#    combination of two columns. Example: rows = `PreopHR` codes, columns =
#    `PreopAF` (0/1). If `PreopAF` really is "PreopHR == 1", then all patients
#    with `PreopHR = 1` should sit in the `PreopAF = 1` column and nobody else.
#    Any patient off that pattern tells us something. Cross-tabs are the
#    simplest and most powerful way to check how two columns relate.
#    We always show "missing" as its own row/column, so blanks are never hidden.
#
# 2. **Breaking things down by year.** NACSA changed its dataset several times
#    between 1996 and 2019. A field that only started being collected in 2012
#    looks "70% missing" overall, but it's really "100% missing before 2012 and
#    complete after". That difference decides how we handle it, so we look at
#    every column year by year.
#
# **What you'll send back at the end:** a text report, three small CSV tables,
# two heatmap images and the log. Section 9 lists them.
#
# **Privacy rules are the same as Step 1:** only counts and percentages leave
# Colab, counts from 1 to 9 are shown as "<10", and no minimum/maximum values.

# %% [markdown]
# ## 1. Set up and load the data
#
# This block repeats what Step 1 did: import the libraries, connect Drive, set
# the file locations, and read the file. One new library appears:
#
# - **matplotlib** (`plt`): Python's standard plotting library. We use it for
#   two heatmaps in Section 3.
#
# We also put the file-reading code inside a **function** called `load_data`.
# A function is a named, reusable recipe: you write the steps once, then "call"
# it by name. We'll reuse this same function in every later step, so every step
# reads the data in exactly the same way.
#
# Change `DATA_FILE` to your file name (same as in Step 1).
#
# **What you should see:** `Mounted at /content/drive` (or a note it's already
# mounted), then `Loaded: 380,293 rows x 129 columns`.

# %%
import sys
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

pd.set_option("display.max_rows", 300)
pd.set_option("display.max_columns", 60)
pd.set_option("display.width", 250)

from google.colab import drive
drive.mount("/content/drive")

PROJECT_DIR = Path("/content/drive/MyDrive/nacsa")
DATA_FILE   = "NACSA.csv"            # <-- CHANGE THIS if your file has another name
DATA_PATH   = PROJECT_DIR / "raw" / DATA_FILE
OUT_DIR     = PROJECT_DIR / "outputs" / "02_audit"
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
df = load_data(DATA_PATH)
print(f"Loaded: {df.shape[0]:,} rows x {df.shape[1]} columns")

# %% [markdown]
# ## 2. Small helper functions
#
# We'll build ~30 tables in this step. Rather than repeat the same code 30
# times, we write a few small helper functions once:
#
# - `suppress(n)`: same as Step 1, turns counts 1-9 into "<10".
# - `as_label(column)`: turns a column into text labels and writes **"missing"**
#   for blanks (and shows codes like `1.0` as `1`). Without this, `pd.crosstab` silently drops missing values,
#   which is exactly what we must not do in an audit.
# - `xtab(a, b)`: builds the cross-tab of column `a` (rows) against column `b`
#   (columns), with missing shown and small counts suppressed.
# - `show(title, table, explain)`: prints a table with a title and a one-line
#   explanation, and also saves it into a list called `REPORT`, which we write
#   to a text file at the end. So everything you see on screen also ends up in
#   the file you send me.
# - `ERA`: groups the 24 years into five eras (1996-2000, 2001-05, 2006-10,
#   2011-14, 2015-19). Year-by-year tables for 30 cross-tabs would be
#   unreadable; eras are enough to see when a field changes. The last era is
#   exactly our temporal validation cohort.
#
# **What you should see:** `Helpers ready.` and the number of patients per era.

# %%
SMALL_CELL = 10
REPORT = []


def suppress(n):
    """Hide counts from 1 to 9 (privacy)."""
    n = int(n)
    return f"<{SMALL_CELL}" if 0 < n < SMALL_CELL else n


def as_label(s):
    """Column -> text labels, with blanks written as 'missing'.
    Whole-number codes stored as decimals (1.0) are shown as 1."""
    if pd.api.types.is_float_dtype(s) and (s.dropna() % 1 == 0).all():
        s = s.astype("Int64")
    return s.astype("object").where(s.notna(), "missing").astype(str)


def xtab(a, b, data=None):
    """Cross-tab of column a (rows) by column b (columns), missing shown, small cells hidden."""
    data = df if data is None else data
    ra = a if isinstance(a, pd.Series) else as_label(data[a])
    rb = b if isinstance(b, pd.Series) else as_label(data[b])
    return pd.crosstab(ra, rb).map(suppress)


def show(title, table, explain=""):
    """Print a table and keep a copy for the report file."""
    text = f"\n{'=' * 78}\n{title}\n"
    if explain:
        text += f"  ({explain})\n"
    text += "-" * 78 + "\n" + (table.to_string() if hasattr(table, "to_string") else str(table))
    print(text)
    REPORT.append(text)


ERA = pd.cut(df["Year"], bins=[1995, 2000, 2005, 2010, 2014, 2019],
             labels=["1996-2000", "2001-2005", "2006-2010", "2011-2014", "2015-2019"]).astype(str).rename("Era")

print("Helpers ready.")
print(ERA.value_counts().sort_index().to_string())

# %% [markdown]
# ## 3. The outcome: length of stay and in-hospital death
#
# This is the most important check in the whole audit.
#
# **The question.** `PostLOS` is missing for 6,816 patients and 6,842 patients
# died. If most deaths have no LOS, the old analysis (which dropped missing
# LOS) silently removed nearly every death.
#
# **3a** cross-tabs "is LOS missing?" against `Mortality`.
#
# **3b** describes LOS separately for survivors and deaths, using percentiles
# (no min/max).
#
# **3c** counts suspicious LOS values: 0 days (discharged or died on the day of
# surgery) and over 100 days.
#
# **3d** builds, per year: number of patients, % with missing LOS, % died, %
# with LOS > 10 days, and % with our proposed primary outcome (LOS > 10 days
# **or** died in hospital). We're allowed to look at outcome rates by year:
# it's description, not model building, and the masterplan requires it.
#
# The composite outcome is defined like this, in order:
#
# - died in hospital -> 1 (whatever the LOS)
# - otherwise LOS > 10 -> 1
# - otherwise LOS <= 10 and known to be alive -> 0
# - anything else (both unknown) -> missing
#
# `np.select` is a tidy way to write "if this, then that; else if ..." for a
# whole column at once.
#
# **What you should see:** four tables. In 3a, look at the "1" (died) column:
# how many deaths have LOS present vs missing?

# %%
los = df["PostLOS"]
died = df["Mortality"]

los_status = np.where(los.isna(), "LOS missing", "LOS present")
show("3a. Is PostLOS missing? (rows) by Mortality (columns: 0 alive, 1 died)",
     xtab(pd.Series(los_status, name="PostLOS", index=df.index), "Mortality"),
     "the key check: do deaths have a length of stay?")

pcts = [0.05, 0.25, 0.5, 0.75, 0.95]
by_status = los.groupby(as_label(died)).quantile(pcts).unstack()
by_status.columns = [f"p{int(p * 100)}" for p in pcts]
by_status.insert(0, "n_with_LOS", los.notna().groupby(as_label(died)).sum().map(suppress))
show("3b. PostLOS (days) for survivors (0), deaths (1) and unknown status", by_status,
     "percentiles; deaths' LOS = days from surgery to death if recorded")

suspicious = pd.DataFrame({
    "LOS = 0 days":    (los == 0).groupby(as_label(died)).sum(),
    "LOS > 100 days":  (los > 100).groupby(as_label(died)).sum(),
    "LOS > 365 days":  (los > 365).groupby(as_label(died)).sum(),
}).map(suppress)
show("3c. Suspicious LOS values, by Mortality", suspicious)

composite = pd.Series(
    np.select([died == 1, los > 10, (los <= 10) & (died == 0)], [1.0, 1.0, 0.0], default=np.nan),
    index=df.index)
outcome_by_year = pd.DataFrame({
    "n_patients":        df.groupby("Year").size(),
    "pct_LOS_missing":   los.isna().groupby(df["Year"]).mean() * 100,
    "pct_died":          (died == 1).groupby(df["Year"]).mean() * 100,
    "pct_LOS_over_10":   (los > 10)[los.notna()].groupby(df["Year"]).mean() * 100,
    "pct_composite":     composite.groupby(df["Year"]).mean() * 100,
    "pct_composite_missing": composite.isna().groupby(df["Year"]).mean() * 100,
}).round(2)
outcome_by_year["n_patients"] = outcome_by_year["n_patients"].map(suppress)
show("3d. Outcome rates by year", outcome_by_year,
     "pct_LOS_over_10 is among patients with a known LOS")

# %% [markdown]
# ## 4. Missingness by year for every column (+ heatmap)
#
# **The idea.** For every column and every year, the % of patients with a
# blank. The result is a table with one row per column and one column per
# year (129 x 24). We save it as a CSV and draw it as a **heatmap**: a grid
# where each cell's colour shows its value (white = complete, dark blue =
# entirely missing). Patterns jump out: a field introduced in 2012 shows as a
# dark block on the left and white on the right.
#
# A dashed line marks 2015, where our temporal validation starts. A field that
# is dark on the left and white on the right of that line is a problem for
# temporal validation: the model would learn from a period when it barely
# existed.
#
# The code:
# - `df.isna()` makes a True/False table (True = missing).
# - `.groupby(df["Year"]).mean()` gives, per year, the share of True = the %
#   missing (True counts as 1, False as 0).
# - `.T` flips the table so columns become rows.
#
# Only columns with at least some missing values are drawn (the rest would be
# all-white rows).
#
# **What you should see:** a tall heatmap. Look for dark blocks that start or
# stop at a particular year.

# %%
def heatmap(mat, title, colour_label, path, vmax=100):
    """Draw a column-by-year grid: light = low, dark = high. One hue only."""
    years = list(mat.columns)
    fig, ax = plt.subplots(figsize=(11, max(4, 0.17 * len(mat) + 1.6)))
    im = ax.imshow(mat.values.astype(float), aspect="auto", cmap="Blues",
                   vmin=0, vmax=vmax, interpolation="nearest")
    ax.set_xticks(range(len(years)))
    ax.set_xticklabels(years, rotation=90, fontsize=8)
    ax.set_yticks(range(len(mat)))
    ax.set_yticklabels(mat.index, fontsize=7)
    if 2015 in years:
        ax.axvline(years.index(2015) - 0.5, color="#333333", lw=1.2, ls="--")
        ax.text(years.index(2015) - 0.3, -0.8, "validation starts", fontsize=8, color="#333333")
    for side in ax.spines.values():
        side.set_visible(False)
    bar = fig.colorbar(im, ax=ax, fraction=0.025, pad=0.01)
    bar.set_label(colour_label, fontsize=9)
    ax.set_title(title, loc="left", fontsize=11, pad=14)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.show()


missing_by_year = (df.isna().groupby(df["Year"]).mean().T * 100).round(1)
missing_by_year.index.name = "column"
missing_by_year.to_csv(OUT_DIR / "missing_by_year.csv")

has_missing = missing_by_year[missing_by_year.max(axis=1) > 0]
has_missing = has_missing.loc[df[has_missing.index].isna().mean().sort_values(ascending=False).index]
heatmap(has_missing, "Missing values by year: dark = mostly missing",
        "% missing", OUT_DIR / "missing_by_year.png")
print(f"{len(has_missing)} of {df.shape[1]} columns have at least some missing values.")

# %% [markdown]
# ## 5. Hidden missingness: % of patients with a non-zero value, by year
#
# **The idea.** Section 4 can't see blanks that were already replaced by 0.
# But it leaves a trace: if a field didn't exist before 2012 and blanks became
# 0, the % of patients with a "yes" (non-zero) value will be 0% in early years
# and then suddenly jump. Real clinical prevalence changes slowly; a jump from
# 0% to 5% in one year is a recording change, not a medical one.
#
# We do this for every **categorical** column (10 or fewer distinct values).
# Prevalences differ hugely between columns (hypertension ~70%, salvage ~0.2%),
# so on the heatmap each row is scaled to its own highest year (= 100). That
# way a rare field's pattern is as visible as a common one's. The CSV keeps the
# real percentages.
#
# Privacy: a percentage based on 1-9 patients is replaced by "<10 pts" in the
# CSV.
#
# **What you should see:** a second heatmap. Look for rows that are white on
# the left and suddenly blue from some year on.

# %%
categorical = [c for c in df.columns
               if c not in ("Year", "Country")
               and pd.api.types.is_numeric_dtype(df[c])
               and df[c].nunique(dropna=True) <= 10]

nonzero = df[categorical].ne(0) & df[categorical].notna()
nz_count = nonzero.groupby(df["Year"]).sum().T
nz_pct = (nonzero.groupby(df["Year"]).mean().T * 100).round(2)

nz_csv = nz_pct.astype(object).where(~((nz_count > 0) & (nz_count < SMALL_CELL)), "<10 pts")
nz_csv.index.name = "column"
nz_csv.to_csv(OUT_DIR / "nonzero_by_year.csv")

row_peak = nz_pct.max(axis=1)
relative = nz_pct[row_peak > 0].div(row_peak[row_peak > 0], axis=0) * 100
heatmap(relative, "Share of patients with a non-zero value, by year (each row scaled to its peak year)",
        "% of the row's peak", OUT_DIR / "nonzero_by_year.png")

# %% [markdown]
# ## 6. Resolving the contradictions, one group at a time
#
# Each block below cross-tabs columns that should agree with each other. For
# each, the comment says **what we expect** if the data are clean. Anything
# else is a finding.
#
# ### 6a. LV function (four sources disagree)
#
# We have: `LVEF` (actual %), `LVEFC` (category, recoded), `LVF1` (3 levels),
# `LVF2` (4 levels), and the dummy columns. We turn `LVEF` into the standard
# EuroSCORE II bands (0-20, 21-30, 31-50, 51+ %) with `pd.cut`, then compare
# every categorical version against those bands and against each other.
#
# **Expect:** each category code lines up with one EF band. That tells us
# which code means "good", "moderate", "poor", "very poor". Then the era tables
# show which source exists in which years.

# %%
lvef_band = pd.cut(df["LVEF"], bins=[0, 20, 30, 50, 100],
                   labels=["EF 00-20", "EF 21-30", "EF 31-50", "EF 51+"]).astype(object)
lvef_band = pd.Series(lvef_band, index=df.index).where(df["LVEF"].notna(), "missing").astype(str).rename("LVEF band")

show("6a-1. LVEFC (rows) by actual LVEF band", xtab("LVEFC", lvef_band), "which LVEFC code = which EF range?")
show("6a-2. LVF1 (rows) by actual LVEF band", xtab("LVF1", lvef_band))
show("6a-3. LVF2 (rows) by actual LVEF band", xtab("LVF2", lvef_band))
show("6a-4. LVF1 (rows) by LVF2", xtab("LVF1", "LVF2"))
show("6a-5. LVF1 (rows) by LVEFC", xtab("LVF1", "LVEFC"), "do the two systems agree?")
show("6a-6. LVEFC (rows) by era", xtab("LVEFC", ERA))
show("6a-7. LVF1 (rows) by era", xtab("LVF1", ERA))
show("6a-8. Dummy columns: LVF1.Poor and LVF1.Moderate (rows) by LVEFC",
     xtab((as_label(df["LVF1.Poor"]) + " / " + as_label(df["LVF1.Moderate"])).rename("Poor / Moderate"), "LVEFC"),
     "rows read 'LVF1.Poor / LVF1.Moderate'")

# %% [markdown]
# ### 6b. Rhythm and neurology
#
# **Expect:** `PreopAF = 1` exactly when `PreopHR = 1`, and `PreopHR` missing
# rows sit in `PreopAF = 0` (blanks filled as 0). For neurology: how `Stroke`
# (history of TIA/CVA), `NeuroDys` and the mobility columns relate, and when
# `Mobility` starts being recorded.

# %%
show("6b-1. PreopHR (rows) by PreopAF", xtab("PreopHR", "PreopAF"), "expect PreopAF=1 only when PreopHR=1")
show("6b-2. PreopHR (rows) by PreopCHB.pacing", xtab("PreopHR", "PreopCHB.pacing"))
show("6b-3. Stroke (rows) by NeuroDys", xtab("Stroke", "NeuroDys"))
show("6b-4. PoorMobility (rows) by 'Mobility / NeuroDys'",
     xtab("PoorMobility", (as_label(df["Mobility"]) + " / " + as_label(df["NeuroDys"])).rename("Mobility / NeuroDys")),
     "is PoorMobility = Mobility OR NeuroDys?")
show("6b-5. Mobility (rows) by era", xtab("Mobility", ERA))

# %% [markdown]
# ### 6c. Bypass, cross-clamp and myocardial protection
#
# **The idea.** Off-pump patients (`PumpCase = 0`) should have a bypass time of
# 0 (or "not applicable"), and on-pump patients a positive time. We sort each
# time into three states: missing / zero / positive, and cross-tab with
# `PumpCase`.
#
# **Expect:** on-pump -> positive; off-pump -> zero or missing. On-pump
# patients with zero bypass time are errors.

# %%
def time_state(s):
    return pd.Series(np.select([s.isna(), s == 0], ["missing", "zero"], default="positive"),
                     index=s.index, name=s.name)


show("6c-1. PumpCase (rows: 1 on-pump, 0 off-pump) by CPB time state", xtab("PumpCase", time_state(df["CPB"])))
show("6c-2. PumpCase (rows) by cross-clamp time state", xtab("PumpCase", time_state(df["XClamp"])))
show("6c-3. PumpCase (rows) by MyoProt (0 non-cardioplegic, 1 cardioplegia)", xtab("PumpCase", "MyoProt"))
show("6c-4. MyoProt (rows) by Plegia.Timing", xtab("MyoProt", "Plegia.Timing"))
show("6c-5. Conversion (rows) by PumpCase", xtab("Conversion", "PumpCase"), "is Conversion = off-pump converted to on-pump?")

# %% [markdown]
# ### 6d. Renal function
#
# **Expect:** `CrCl` = 100 exactly where `Creatinine` is missing (confirming
# the fill-in). `Cr200 = 1` where creatinine > 200. Also: in which eras actual
# creatinine exists, and how the dialysis codes relate to the renal category.

# %%
cr_state = pd.Series(np.select([df["Creatinine"].isna(), df["Creatinine"] > 200], ["missing", ">200"], default="<=200"),
                     index=df.index, name="Creatinine")
show("6d-1. Creatinine state (rows) by 'CrCl is exactly 100?'",
     xtab(cr_state, (df["CrCl"] == 100).map({True: "CrCl = 100", False: "CrCl other"}).rename("CrCl")))
show("6d-2. Creatinine state (rows) by Cr200", xtab(cr_state, "Cr200"))
show("6d-3. Creatinine state (rows) by era", xtab(cr_state, ERA))
show("6d-4. Cr200 (rows) by era", xtab("Cr200", ERA))
show("6d-5. PreopDialysis (rows) by CrCl.Category", xtab("PreopDialysis", "CrCl.Category"))

# %% [markdown]
# ### 6e. Grafts, target vessels and left main disease
#
# **Expect:** `DistalGrafts` and `Total.Graft.Sites` identical; `NumberGrafts`
# close to `DistalGrafts`. For target vessels: the number of territories
# grafted (RCA + LAD + Diag + Cx, blanks counted as 0) should never exceed the
# number of distal grafts. For left main stem: in CABG we'd expect 20-30% LMS
# disease; we check it against `CADExtent` and by era.

# %%
show("6e-1. NumberGrafts (rows) by DistalGrafts", xtab("NumberGrafts", "DistalGrafts"))
same = (df["DistalGrafts"] == df["Total.Graft.Sites"]) | (df["DistalGrafts"].isna() & df["Total.Graft.Sites"].isna())
show("6e-2. Rows where DistalGrafts equals Total.Graft.Sites", same.value_counts().map(suppress))
territories = df[["RCA", "LAD", "Diag", "Cx"]].fillna(0).sum(axis=1).astype(int).rename("n_territories")
show("6e-3. Territories grafted (rows) by DistalGrafts", xtab(territories, "DistalGrafts"))
show("6e-4. RCA (rows) by era", xtab("RCA", ERA), "is a blank RCA 'not grafted' or 'not recorded'?")
show("6e-5. LMS (rows) by CADExtent", xtab("LMS", "CADExtent"))
show("6e-6. LMS (rows) by era", xtab("LMS", ERA))

# %% [markdown]
# ### 6f. Mechanical support and return to theatre
#
# **Expect:** a device's "reason" is 0 exactly when the device wasn't used.
# Impeller use rising sharply in one era would suggest a coding problem. For
# return to theatre: patients with `RTT = 1` should have at least one subtype
# flagged, and `RTT = 0` none.

# %%
show("6f-1. IABP timing (rows) by IABP.Reason", xtab("IABP", "IABP.Reason"))
show("6f-2. Impeller timing (rows) by Impeller.Reason", xtab("Impeller", "Impeller.Reason"))
show("6f-3. IABP (rows) by era", xtab("IABP", ERA))
show("6f-4. Impeller (rows) by era", xtab("Impeller", ERA))
rtt_cols = ["RTT.bleeding.tamponade", "RTT.valve.other", "RTT.graft", "RTT.cardiac.other",
            "RTT.wound", "RTT.rhythmn", "RTT.multiple"]
n_subtypes = df[rtt_cols].sum(axis=1).astype(int).rename("n_RTT_subtypes")
show("6f-5. RTT (rows) by number of RTT subtypes flagged", xtab("RTT", n_subtypes))

# %% [markdown]
# ## 7. Duplicates and country
#
# **7a.** Step 1 found 2,992 exact duplicate rows. Here we see which years
# they're in. Duplicates concentrated in one or two years suggest a data
# submission error in those years.
#
# **7b.** Country codes by era: are codes 4-6 present throughout, or only in
# some years (which would hint at what they are)?

# %%
dup = df.duplicated(keep="first")
show("7a. Exact duplicate rows by year", dup.groupby(df["Year"]).sum().map(suppress).rename("n_duplicates").to_frame())
show("7b. Country code (rows) by era", xtab("Country", ERA))

# %% [markdown]
# ## 8. Plausibility of continuous values
#
# **The idea.** For each continuous column we set a range of values that are
# physically possible (mostly from the NICOR dictionary's own limits) and count
# patients outside it. These counts tell us how many values we'll have to set
# to missing during cleaning.
#
# A value outside the range is almost always a typing or unit error (e.g.
# height entered in metres, 1.75, instead of cm, 175).
#
# **What you should see:** one table. Most counts should be small.

# %%
limits = {                      # column: (lowest plausible, highest plausible)
    "Age":        (18, 100),
    "Height":     (120, 220),   # cm
    "Weight":     (30, 250),    # kg (NICOR limit 250)
    "BMI":        (12, 70),
    "Creatinine": (20, 1500),   # umol/L
    "LVEF":       (5, 90),      # %
    "PAsys":      (10, 130),    # mmHg; 0 is the 'not measured' code
    "CPB":        (0, 600),     # min
    "XClamp":     (0, 360),     # min (NICOR limit 360)
    "PostLOS":    (0, 365),     # days
}
rows = []
for col, (low, high) in limits.items():
    s = df[col]
    rows.append({"column": col, "plausible range": f"{low} to {high}",
                 "n_present": suppress(s.notna().sum()),
                 "n_below": suppress((s < low).sum()),
                 "n_above": suppress((s > high).sum())})
show("8. Values outside plausible ranges", pd.DataFrame(rows).set_index("column"))

# %% [markdown]
# ## 9. Save everything and send it to Claude
#
# **What to send back** (from `MyDrive/nacsa/outputs/02_audit/`):
#
# 1. `audit_report.txt`: every table printed above, in order
# 2. `missing_by_year.csv` and `missing_by_year.png`
# 3. `nonzero_by_year.csv` and `nonzero_by_year.png`
# 4. `outcome_by_year.csv`
# 5. `log.txt`
#
# As before, glance through them before sending. They contain only counts and
# percentages.

# %%
(OUT_DIR / "audit_report.txt").write_text("\n".join(REPORT))
outcome_by_year.to_csv(OUT_DIR / "outcome_by_year.csv")

log_lines = [
    "Script         : 02_deep_audit.py",
    f"Run at         : {datetime.now().isoformat(timespec='seconds')}",
    f"Data file      : {DATA_PATH.name}",
    f"Rows x columns : {df.shape[0]} x {df.shape[1]}",
    f"Tables in report: {len(REPORT)}",
    f"Python {sys.version.split()[0]} | pandas {pd.__version__} | numpy {np.__version__}",
]
(OUT_DIR / "log.txt").write_text("\n".join(log_lines))
print("\n".join(log_lines))
print("\nSaved files:")
for f in sorted(OUT_DIR.iterdir()):
    print("  ", f.name)

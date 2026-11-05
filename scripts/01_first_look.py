# =============================================================================
# NACSA PLoS project  |  Step 1: First look at the data
# -----------------------------------------------------------------------------
# HOW TO USE THIS FILE IN COLAB
#   The file is split into blocks. Each block starts with a line "# %%".
#   1. Open a new, empty Colab notebook.
#   2. Copy ONE block (from one "# %%" line down to the next), paste it into a
#      Colab cell, read the comments, then run it with Shift + Enter.
#   3. Repeat block by block, top to bottom.
#   Lines starting with "#" are comments: explanations for you. Python ignores them.
# =============================================================================

# %% [markdown]
# # Step 1 — First look at the NACSA data
#
# **What this step does:** opens the dataset and describes it. That's all.
#
# **What it does *not* do:** it changes nothing. No cleaning, no deleting, no
# imputing. The raw file is never overwritten.
#
# **Why start here?** Before you can make good decisions about a dataset (what
# to clean, what to drop, what to merge), you need to *see* it: how big it is,
# what type each column is, how much is missing, and what values actually
# appear. Analysts who skip this step end up discovering problems halfway
# through modelling, when they are expensive to fix.
#
# **What you'll send back to Claude at the end:** three small summary files (no
# patient-level rows). Section 8 tells you exactly which ones.
#
# Read the explanation above each block *before* running it, including the part
# called **"What you should see"**.

# %% [markdown]
# ## 1. Load the tools (libraries)
#
# **The idea.** Python on its own is a general language. For data work we
# "import" ready-made toolkits, called *libraries*:
#
# - **pandas** (`pd`): the main tool for tables. A table in pandas is called a
#   **DataFrame**. Think of it as an Excel sheet that you control with code:
#   rows are patients, columns are variables.
# - **numpy** (`np`): fast maths on numbers. pandas is built on top of it.
# - **pathlib** (`Path`): handles file and folder locations cleanly.
#
# `import pandas as pd` means "load pandas, and let me call it `pd` for short".
# That's a universal convention, so you'll see `pd.` everywhere.
#
# The two `pd.set_option` lines only change how tables are *displayed* (show up
# to 200 rows and 50 columns instead of hiding some with `...`).
#
# The last line prints the library versions. We record these because results can
# differ slightly between versions, and a good paper reports them.
#
# **What you should see:** three version numbers printed. No error.

# %%
import sys
from pathlib import Path

import numpy as np
import pandas as pd

pd.set_option("display.max_rows", 200)
pd.set_option("display.max_columns", 50)
pd.set_option("display.width", 200)

print("Python :", sys.version.split()[0])
print("pandas :", pd.__version__)
print("numpy  :", np.__version__)

# %% [markdown]
# ## 2. Connect Colab to your Google Drive
#
# **The idea.** Colab runs on a temporary Google computer that is wiped when you
# close it. Your data lives in your Google Drive. "Mounting" Drive makes your
# Drive folders appear inside that temporary computer at the location
# `/content/drive/MyDrive/`, so the code can read files from it.
#
# When you run this cell, Google will ask you to sign in and give permission.
# That's expected.
#
# > **Governance reminder:** only do this if your NICOR data sharing agreement
#   allows the data to be stored in and processed on Google's cloud. If not, the
#   same code runs unchanged in Jupyter on an approved computer (just skip
#   this cell).
#
# **What you should see:** `Mounted at /content/drive`.

# %%
from google.colab import drive
drive.mount("/content/drive")

# %% [markdown]
# ## 3. Tell the code where things are
#
# **The idea.** We set file locations **once, at the top**, instead of typing
# them throughout the code. If the file moves, you change one line, not
# twenty.
#
# We use this folder layout inside your Drive (create the folders if they don't
# exist):
#
# ```
# MyDrive/
#   nacsa/
#     raw/        <- the original data file lives here and is NEVER modified
#     outputs/    <- everything the scripts produce goes here
# ```
#
# Keeping the raw file untouched and separate is a basic rule of reproducible
# research: you can always go back to the original and re-run every step.
#
# **What you need to do:** change `DATA_FILE` to the real name of your file
# (keep the quotes). The line `assert DATA_PATH.exists()` is a safety check: if
# the file isn't where we said, the code stops with a clear message instead
# of failing confusingly later.
#
# **What you should see:** `Found data file: ...` and the file size in MB.

# %%
PROJECT_DIR = Path("/content/drive/MyDrive/nacsa")
DATA_FILE   = "NACSA.csv"            # <-- CHANGE THIS to your file's real name

DATA_PATH   = PROJECT_DIR / "raw" / DATA_FILE
OUT_DIR     = PROJECT_DIR / "outputs" / "01_first_look"
OUT_DIR.mkdir(parents=True, exist_ok=True)   # creates the output folder if needed

assert DATA_PATH.exists(), f"File not found: {DATA_PATH}. Check the folder and the file name."
print("Found data file:", DATA_PATH)
print(f"Size: {DATA_PATH.stat().st_size / 1e6:.1f} MB")
print("Outputs will be saved in:", OUT_DIR)

# %% [markdown]
# ## 4. Read the file into a DataFrame
#
# **The idea.** Data files come in different formats (`.csv`, `.xlsx`, R's
# `.rds`, SPSS `.sav`, Stata `.dta`, `.parquet`). Each needs a different reading
# function. This block looks at the file's ending (its *suffix*) and picks the
# right one, so it works whatever format you have.
#
# A few details worth knowing:
#
# - `low_memory=False` (for CSV) makes pandas read the whole column before
#   deciding its type. Without it, a big file can end up with columns of mixed
#   types.
# - When pandas reads a CSV, it automatically treats empty cells and text like
#   `NA` or `NaN` as **missing** (shown as `NaN`). That's what we want. But
#   registries sometimes use **special codes** for missing, like `-9`, `99` or
#   `"Unknown"`. pandas can't know those mean "missing", so it will treat them
#   as real values. We will look for such codes in Section 6.
# - If your file is `.rds` (from R), Colab needs the extra library `pyreadr`;
#   the block installs it automatically.
#
# The result is stored in a variable called `df` (short for DataFrame). From now
# on, `df` *is* the dataset in memory. **Nothing we do to `df` changes the file
# on disk.**
#
# **What you should see:** `Loaded: N rows x M columns`. You told me the column
# list has about 150 columns; the rows should be in the region of 380,000 if
# this is the full isolated-CABG extract.

# %%
suffix = DATA_PATH.suffix.lower()

if suffix == ".csv":
    df = pd.read_csv(DATA_PATH, low_memory=False)
elif suffix in (".xlsx", ".xls"):
    df = pd.read_excel(DATA_PATH)
elif suffix == ".parquet":
    df = pd.read_parquet(DATA_PATH)
elif suffix == ".rds":
    try:
        import pyreadr
    except ImportError:
        import subprocess
        subprocess.run([sys.executable, "-m", "pip", "install", "-q", "pyreadr"], check=True)
        import pyreadr
    df = pyreadr.read_r(str(DATA_PATH))[None]
elif suffix == ".sav":
    try:
        import pyreadstat  # noqa: F401
    except ImportError:
        import subprocess
        subprocess.run([sys.executable, "-m", "pip", "install", "-q", "pyreadstat"], check=True)
    df = pd.read_spss(DATA_PATH)
elif suffix == ".dta":
    df = pd.read_stata(DATA_PATH)
else:
    raise ValueError(f"Don't know how to read '{suffix}' files yet - tell Claude the format.")

print(f"Loaded: {df.shape[0]:,} rows x {df.shape[1]} columns")
print(f"Memory used: {df.memory_usage(deep=True).sum() / 1e6:.0f} MB")

# %% [markdown]
# ## 5. Basic structure: column types and duplicate rows
#
# ### 5a. Column types
#
# **The idea.** Every column has a *data type* (`dtype`) that tells pandas what
# kind of values it holds:
#
# | dtype | Meaning | Example |
# |---|---|---|
# | `int64` | whole numbers, no missing values | number of grafts |
# | `float64` | numbers with decimals, **or** whole numbers that have missing values (pandas needs a decimal type to store `NaN`) | creatinine, BMI |
# | `object` / `str` | text | `"Elective"`, `"Male"` |
# | `bool` | True/False | |
# | `category` | text with a fixed set of levels (common in files that came from R) | |
#
# Why this matters: a column we expect to be a number (like `Creatinine`)
# showing up as text usually means something odd is hiding inside it, such as
# `"<50"` or `"unknown"`. We'd need to fix that before any analysis.
#
# **What you should see:** a short count of how many columns have each type.

# %%
print("Number of columns of each type:")
print(df.dtypes.astype(str).value_counts().to_string())

# %% [markdown]
# ### 5b. Exact duplicate rows
#
# **The idea.** We have no patient ID, so we can't find repeat patients. But we
# *can* count rows that are **identical in every single column**. With ~150
# columns, two genuinely different patients being identical everywhere is very
# unlikely, so exact duplicates often point to a data extraction error.
#
# We only **count** them here. We don't delete anything; that's a decision for
# the cleaning step.
#
# **What you should see:** a number. Zero or close to zero is reassuring. A
# large number (hundreds or more) is something to discuss.

# %%
n_dup = int(df.duplicated().sum())
print(f"Exact duplicate rows: {n_dup:,} ({100 * n_dup / len(df):.2f}% of rows)")

# %% [markdown]
# ## 6. One summary line per column (the core of this step)
#
# **The idea.** We build a table with **one row per column of the dataset**,
# describing it. This is the table we'll go through together, column by column,
# to decide what to keep, recode, merge or drop.
#
# For each column we record:
#
# | Field | What it tells us |
# |---|---|
# | `dtype` | data type (see 5a) |
# | `n_missing`, `pct_missing` | how many values are missing |
# | `n_unique` | number of distinct values. 1 = constant column. 2 = probably yes/no. Hundreds = continuous |
# | `top_value`, `top_pct` | the most common value and what % of non-missing rows have it. A `top_pct` of 99.5 means the column is *nearly* constant (the "near-zero variance" issue we discussed) |
# | `n_not_top` | how many patients have a value **other than** the most common one. This is the *absolute count* that tells us whether a rare feature still has enough patients to be useful |
# | `summary` | for continuous columns: percentiles. For categorical columns: the most frequent values and their counts |
#
# ### Two privacy safeguards built into this table
#
# 1. **No minimum or maximum.** The min and max of a column are each a single
#   patient's value, which counts as patient-level data. Instead we report
#   **percentiles**: `p0.1` is the value below which 0.1% of patients fall
#   (about 380 people), `p99.9` the value above which 0.1% fall. These still
#   reveal impossible values (e.g. if `p0.1` of age is 2, something is wrong)
#   without exposing any one person. For columns with fewer filled-in values,
#   the code automatically uses less extreme percentiles (p1/p99, or p10/p90
#   under 1,000 values), so no percentile ever rests on a handful of patients.
# 2. **Small-cell suppression.** Any count between 1 and 9 is shown as `<10`.
#   NICOR and most data agreements require this for anything leaving the secure
#   environment.
#
# ### How the code works, in plain words
#
# - `def ...:` defines a small reusable function. `suppress(n)` replaces small
#   counts with `"<10"`.
# - `describe_column(s)` takes one column (`s`, a pandas *Series*) and returns a
#   dictionary of the fields above.
# - The line starting `overview = pd.DataFrame(...)` runs that function on every
#   column and stacks the results into one table.
# - We then sort it so the most-missing columns are at the top.
#
# **What you should see:** a long table with ~150 rows. Don't try to interpret
# it all now. Skim it and notice: which columns are mostly missing? Which have
# `n_unique` of 1? Which numeric columns look like they might contain special
# codes (e.g. `p99.9` of 999)?

# %%
SMALL_CELL = 10   # counts from 1 to 9 are hidden as "<10"


def suppress(n):
    # Hide small counts to protect patient privacy.
    n = int(n)
    return f"<{SMALL_CELL}" if 0 < n < SMALL_CELL else n


def describe_column(s):
    n_total   = len(s)
    n_missing = int(s.isna().sum())
    present   = s.dropna()
    n_unique  = int(present.nunique())

    row = {
        "column":      s.name,
        "dtype":       str(s.dtype),
        "n_missing":   suppress(n_missing),
        "pct_missing": round(100 * n_missing / n_total, 1),
        "n_unique":    n_unique,
    }

    if n_unique == 0:                       # column is completely empty
        row.update(top_value=None, top_pct=None, n_not_top=0, summary="ALL MISSING")
        return row

    counts = present.value_counts()
    row["top_value"] = counts.index[0]
    row["top_pct"]   = round(100 * counts.iloc[0] / len(present), 2)
    row["n_not_top"] = suppress(len(present) - counts.iloc[0])

    is_number = pd.api.types.is_numeric_dtype(s) and not pd.api.types.is_bool_dtype(s)
    if is_number and n_unique > 10:
        # Extreme percentiles are only safe when many values exist (p0.1 of 500 values
        # would be a single patient), so the percentiles shown depend on the count.
        if len(present) >= 10_000:
            probs, labels = [0.001, 0.01, 0.5, 0.99, 0.999], ["p0.1", "p1", "median", "p99", "p99.9"]
        elif len(present) >= 1_000:
            probs, labels = [0.01, 0.05, 0.5, 0.95, 0.99], ["p1", "p5", "median", "p95", "p99"]
        else:
            probs, labels = [0.1, 0.5, 0.9], ["p10", "median", "p90"]
        q = present.quantile(probs)
        row["summary"] = " | ".join(f"{lab}={val:.4g}" for lab, val in zip(labels, q.values))
    else:
        top = counts.head(8)
        row["summary"] = " | ".join(f"{k}: {suppress(v)}" for k, v in top.items())
        if n_unique > 8:
            row["summary"] += f" | ... (+{n_unique - 8} more)"
    return row


overview = pd.DataFrame([describe_column(df[c]) for c in df.columns])
overview = overview.sort_values("pct_missing", ascending=False).reset_index(drop=True)
print(overview.to_string())

# %% [markdown]
# ### 6b. Quick flags
#
# **The idea.** Rather than eyeballing 150 rows, we let the code list the
# columns that fall into each "needs a decision" group. These are **flags for
# discussion, not decisions**. As we discussed, a rare column can still be very
# important (e.g. cardiogenic shock).
#
# - **Completely empty**: 100% missing.
# - **Constant**: only one distinct value.
# - **Mostly missing**: more than 50% missing.
# - **Nearly constant**: the most common value covers 99% or more of patients.
#
# **What you should see:** four short lists of column names.

# %%
flags = {
    "Completely empty (100% missing)": overview.loc[overview["pct_missing"] == 100, "column"],
    "Constant (1 distinct value)":     overview.loc[overview["n_unique"] == 1, "column"],
    "Mostly missing (>50%)":           overview.loc[(overview["pct_missing"] > 50) & (overview["pct_missing"] < 100), "column"],
    "Nearly constant (top value >=99%)": overview.loc[(overview["top_pct"] >= 99) & (overview["n_unique"] > 1), "column"],
}
for label, cols in flags.items():
    print(f"\n{label}: {len(cols)} columns")
    print("   " + (", ".join(cols) if len(cols) else "(none)"))

# %% [markdown]
# ## 7. How many patients per year and per country
#
# **The idea.** Our whole validation strategy depends on `Year` (training on
# 1996–2014, testing on 2015–2019) and we'll use `Country` as a minor check. So
# we need to know exactly what values these columns contain:
#
# - Is `Year` a calendar year (2010) or a financial year (2010/11)? Are there
#   odd values outside 1996–2019?
# - How many operations per year? A sudden drop or jump can mean a change in
#   data collection, not in surgery.
# - Which country labels appear, and how many patients each?
#
# **What you should see:** two small tables of counts.

# %%
year_counts = df["Year"].value_counts(dropna=False).sort_index().rename("n_patients").to_frame()
year_counts["n_patients"] = year_counts["n_patients"].map(suppress)
print(year_counts.to_string())

# %%
country_counts = df["Country"].value_counts(dropna=False).rename("n_patients").to_frame()
country_counts["n_patients"] = country_counts["n_patients"].map(suppress)
print(country_counts.to_string())

# %% [markdown]
# ## 8. Save the summaries and send them to Claude
#
# **The idea.** We save the summary tables as CSV files in the outputs folder.
# They contain **only aggregate information**: no rows of patient data, small
# counts suppressed, no minimums or maximums.
#
# We also save a small text "log" recording when this ran, on how many rows, and
# with which library versions. Keeping a log for every step is how we'll make
# the whole analysis reproducible for the paper.
#
# **What to send back:** download these three files from
# `MyDrive/nacsa/outputs/01_first_look/` and attach them in the chat:
#
# 1. `column_overview.csv`
# 2. `year_country_counts.csv`
# 3. `log.txt`
#
# Before you send them, open `column_overview.csv` and glance at the `summary`
# column yourself. If anything there looks like it could identify a person (it
# shouldn't), tell me before sharing.

# %%
from datetime import datetime

overview.to_csv(OUT_DIR / "column_overview.csv", index=False)

yc = year_counts.reset_index().rename(columns={"index": "value", "Year": "value"})
yc.insert(0, "variable", "Year")
cc = country_counts.reset_index().rename(columns={"index": "value", "Country": "value"})
cc.insert(0, "variable", "Country")
pd.concat([yc, cc], ignore_index=True).to_csv(OUT_DIR / "year_country_counts.csv", index=False)

log_lines = [
    f"Script         : 01_first_look.py",
    f"Run at         : {datetime.now().isoformat(timespec='seconds')}",
    f"Data file      : {DATA_PATH.name}",
    f"Rows x columns : {df.shape[0]} x {df.shape[1]}",
    f"Exact duplicate rows : {n_dup}",
    f"Python {sys.version.split()[0]} | pandas {pd.__version__} | numpy {np.__version__}",
]
(OUT_DIR / "log.txt").write_text("\n".join(log_lines))

print("\n".join(log_lines))
print("\nSaved files:")
for f in sorted(OUT_DIR.iterdir()):
    print("  ", f.name)

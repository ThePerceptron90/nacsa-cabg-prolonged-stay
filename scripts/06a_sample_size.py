# =============================================================================
# NACSA PLoS project  |  Step 6a: Sample size
# -----------------------------------------------------------------------------
# HOW TO USE THIS FILE
#   Claude runs this file in its own workspace (decision 56). To run it
#   yourself in Colab, copy ONE block at a time (each starts with "# %%") into
#   a cell, or open the .ipynb version, and run top to bottom.
# =============================================================================

# %% [markdown]
# # Step 6a: Is the sample big enough?
#
# **What this step does:** it checks formally that the development cohort is
# large enough to build each model without serious overfitting, and that the
# validation cohort is large enough to measure performance precisely. It
# answers TRIPOD+AI item 10 and produces supplementary table ST7.
#
# **Methods** (agreed 6 Oct 2026, decision 73):
#
# - Development: Riley et al., BMJ 2020 (three criteria for a binary outcome:
#   shrinkage >= 0.90; optimism in Nagelkerke R2 <= 0.05; overall risk within
#   +/- 0.05).
# - Validation: Riley et al., Statistics in Medicine 2021 (95% CI widths:
#   observed/expected ratio 0.2, calibration slope 0.2, C-statistic 0.1).
# - Expected model performance, needed as an input: the more demanding of
#   (A) the C-statistic of EuroSCORE II for our outcome in development
#   patients (an existing model in the same population; our models contain
#   everything it does) and (B) Riley's default when nothing is known
#   (Cox-Snell R2 = 15% of its maximum). Sensitivity: C = 0.65 and C = 0.80.
# - Number of parameters: regression terms with splines from Step 5 (the
#   honest, larger count).
#
# The validation calculation uses the development outcome rate as the
# anticipated rate, so no validation outcome is used here (decision 25).
#
# **Outputs** (in `outputs/06a_sample_size/`):
#
# | File | What it shows | Paper |
# |---|---|---|
# | `ST7_development.csv` | required vs available patients and events, per model | ST7 |
# | `ST7_validation.csv` | required vs available validation patients | ST7 |
# | `ST7_assumptions.csv` | every input used, with its source | ST7 |
# | `log.txt` | run details | |

# %% [markdown]
# ## 1. Set up and load
#
# Only development patients are loaded with outcomes. For the validation
# cohort we only need its size and number of events, which Step 4 already
# reported (decision 25: no validation outcome enters any modelling choice).
#
# **What you should see:** one `load_clean:` line.

# %%
import importlib
import os
import sys
from datetime import datetime

import numpy as np
import pandas as pd
from scipy.optimize import brentq
from scipy.stats import norm

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

OUT_DIR = nc.OUTPUTS_DIR / "06a_sample_size"
OUT_DIR.mkdir(parents=True, exist_ok=True)

dev = nc.prepare_for_model(nc.load_clean(period="development", outcome="survivors"))
PHI = dev["y"].mean()                      # anticipated outcome rate
print(f"\nDevelopment outcome rate: {100 * PHI:.1f}%")

# Validation cohort size and events (reported in Step 4, outputs/04_cohort_profile/flow_counts.csv)
VAL_N, VAL_EVENTS = 32_104, 4_571

# %% [markdown]
# ## 2. Number of parameters per model
#
# Same rule as Step 5: yes/no = 1 term; a category with k levels = k - 1
# terms; a continuous predictor = 3 terms (restricted cubic spline, 4 knots).
#
# **What you should see:** 48, 61, 63, 68 for the main sets; 43, 55, 56 for
# the all-era sets.

# %%
SPLINE_TERMS = 3


def n_terms(var):
    t = nc.VARIABLES[var]["type"]
    if t == "continuous":
        return SPLINE_TERMS
    if t == "binary":
        return 1
    levels = nc.MODEL_LEVELS.get(var, nc.VARIABLES[var].get("levels", {}))
    return len(levels) - 1


MODEL_SETS = [(nc.MODEL_LABELS[m], nc.PREDICTOR_SETS[m]) for m in nc.PREDICTOR_SETS] + \
             [(nc.MODEL_LABELS[m] + " [all-era set]", nc.ALL_ERA_SETS[m]) for m in nc.ALL_ERA_SETS]
PARAMS = {name: sum(n_terms(v) for v in vs) for name, vs in MODEL_SETS}
print(pd.Series(PARAMS).to_string())

# %% [markdown]
# ## 3. From a C-statistic to the Cox-Snell R2
#
# Riley's formulas need the expected **Cox-Snell R2**, but performance is
# usually expressed as a **C-statistic**. Riley, Van Calster and Collins
# (Stat Med 2021) convert one to the other by imagining a very large
# population in which the model's risk scores follow a normal distribution:
#
# 1. choose the spread of the risk scores so that the C-statistic equals the
#    target, and their average so that the outcome rate equals ours;
# 2. compute how much better that model fits than "everyone has the average
#    risk": this gives the Cox-Snell R2.
#
# We do this exactly on a fine grid of 200,000 points instead of by random
# simulation, so the result is the same every time.
#
# **What you should see:** the maximum possible Cox-Snell R2 for our outcome
# rate, and a check that the conversion reproduces C = 0.70.

# %%
GRID = norm.ppf((np.arange(200_000) + 0.5) / 200_000)      # standard normal quantiles


def _expit(x):
    return 1 / (1 + np.exp(-x))


def lp_distribution(sigma, phi):
    """Risk-score grid with spread sigma, shifted so the average risk equals phi."""
    mu = brentq(lambda m: _expit(m + sigma * GRID).mean() - phi, -20, 20)
    return _expit(mu + sigma * GRID), mu


def c_statistic(p):
    """C-statistic of risk scores p (sorted ascending) for an outcome drawn with those risks."""
    nonevent_below = np.concatenate([[0.0], np.cumsum(1 - p)[:-1]])
    return float((p * nonevent_below).sum() / (p.sum() * (1 - p).sum()))


def cox_snell_from_c(c_target, phi):
    sigma = brentq(lambda s: c_statistic(lp_distribution(s, phi)[0]) - c_target, 0.01, 10)
    p, _ = lp_distribution(sigma, phi)
    ll_model = (p * np.log(p) + (1 - p) * np.log(1 - p)).mean()      # per patient
    ll_null = phi * np.log(phi) + (1 - phi) * np.log(1 - phi)
    return 1 - np.exp(-2 * (ll_model - ll_null)), sigma


LL_NULL = PHI * np.log(PHI) + (1 - PHI) * np.log(1 - PHI)
MAX_R2 = 1 - np.exp(2 * LL_NULL)
r2_check, _ = cox_snell_from_c(0.70, PHI)
print(f"Maximum possible Cox-Snell R2 for a {100 * PHI:.1f}% outcome: {MAX_R2:.3f}")
print(f"Check: C = 0.70 corresponds to Cox-Snell R2 = {r2_check:.3f}")

# %% [markdown]
# ## 4. The anticipated performance
#
# **(A) EuroSCORE II's C-statistic** for our outcome in development patients
# (patients with a missing EuroSCORE II, fewer than 10, are left out of this
# one number).
#
# **(B) Riley's default:** Cox-Snell R2 = 15% of the maximum.
#
# The main calculation uses whichever gives the **smaller** R2, because a
# smaller R2 demands more patients (the cautious choice).
#
# **What you should see:** both anchors and which one is used.

# %%
es2 = dev[["ES2", "y"]].dropna()
order = np.argsort(es2["ES2"].values, kind="mergesort")
ranks = pd.Series(es2["ES2"].values).rank().values           # average ranks handle ties
n1 = int(es2["y"].sum()); n0 = len(es2) - n1
C_ES2 = (ranks[es2["y"].values == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)
R2_ES2, _ = cox_snell_from_c(C_ES2, PHI)
R2_DEFAULT = 0.15 * MAX_R2

anchors = {"A: EuroSCORE II in development data": R2_ES2, "B: Riley default (15% of maximum)": R2_DEFAULT}
MAIN_ANCHOR = min(anchors, key=anchors.get)
R2_MAIN = anchors[MAIN_ANCHOR]
print(f"(A) EuroSCORE II C-statistic {C_ES2:.3f} -> Cox-Snell R2 {R2_ES2:.3f}")
print(f"(B) Riley default -> Cox-Snell R2 {R2_DEFAULT:.3f}")
print(f"Main calculation uses: {MAIN_ANCHOR} (R2 = {R2_MAIN:.3f})")

# %% [markdown]
# ## 5. Development cohort: Riley 2020
#
# For each model, three minimum sample sizes; the largest is the answer:
#
# 1. **Shrinkage >= 0.90**: n = p / ((S - 1) ln(1 - R2/S)), S = 0.90.
# 2. **Optimism in Nagelkerke R2 <= 0.05**: the same formula with
#    S = R2 / (R2 + 0.05 x maximum R2).
# 3. **Overall risk within +/- 0.05**: n = (1.96 / 0.05)^2 x rate x (1 - rate).
#
# p = number of regression terms. We then compare with what we have, and
# repeat the calculation for the sensitivity C-statistics 0.65 and 0.80.
#
# **What you should see:** a table where every "available / required" ratio
# is well above 1.

# %%
S_TARGET, DELTA, MARGIN = 0.90, 0.05, 0.05
AVAILABLE = {"main": (len(dev), int(dev["y"].sum()))}


def riley_development(p, r2, phi, max_r2):
    n_shrink = p / ((S_TARGET - 1) * np.log(1 - r2 / S_TARGET))
    s_opt = r2 / (r2 + DELTA * max_r2)
    n_opt = p / ((s_opt - 1) * np.log(1 - r2 / s_opt))
    n_risk = (1.96 / MARGIN) ** 2 * phi * (1 - phi)
    return int(np.ceil(n_shrink)), int(np.ceil(n_opt)), int(np.ceil(n_risk))


scenarios = [("Main (" + MAIN_ANCHOR.split(":")[0] + ")", R2_MAIN)]
for c in (0.65, 0.80):
    scenarios.append((f"Sensitivity: C = {c:.2f}", cox_snell_from_c(c, PHI)[0]))

dev_rows = []
for scen, r2 in scenarios:
    for name, _ in MODEL_SETS:
        p = PARAMS[name]
        n_a, n_b, n_c = riley_development(p, r2, PHI, MAX_R2)
        need = max(n_a, n_b, n_c)
        n_have, e_have = AVAILABLE["main"]
        dev_rows.append({
            "scenario": scen, "model": name, "regression terms": p, "Cox-Snell R2 assumed": round(r2, 3),
            "n for shrinkage >= 0.90": n_a, "n for optimism <= 0.05": n_b, "n for overall risk +/-0.05": n_c,
            "minimum patients required": need, "minimum events required": int(np.ceil(need * PHI)),
            "events per term required": round(need * PHI / p, 1),
            "patients available (development)": n_have, "events available": e_have,
            "available / required": round(n_have / need, 1),
        })
st7_dev = pd.DataFrame(dev_rows)
st7_dev.to_csv(OUT_DIR / "ST7_development.csv", index=False)
print(st7_dev[st7_dev["scenario"].str.startswith("Main")]
      [["model", "regression terms", "minimum patients required", "minimum events required",
        "available / required"]].to_string(index=False))
print("\nSmallest ratio across all scenarios:", st7_dev["available / required"].min())

# %% [markdown]
# ## 6. Validation cohort: Riley 2021
#
# Minimum size to estimate, with the target 95% CI width:
#
# - **observed/expected ratio** (width 0.2): from the standard error of
#   log(O/E), which depends only on the outcome rate;
# - **C-statistic** (width 0.1): Newcombe's standard error formula;
# - **calibration slope** (width 0.2): from the information a perfectly
#   calibrated model gives about the slope, using the same risk-score
#   distribution as above.
#
# The C-statistic and slope criteria depend on the assumed C, so they are
# shown for the main anchor and both sensitivity values.
#
# **What you should see:** every required size far below 32,104.

# %%
Z = norm.ppf(0.975)


def n_oe(phi, width=0.2):
    se = np.arcsinh(width / 2) / Z            # CI exp(+/- Z*se) around 1 has this width
    return int(np.ceil((1 - phi) / (phi * se ** 2)))


def n_cstat(c, phi, width=0.1):
    target_se = width / (2 * Z)

    def se(n):
        n1, n0 = n * phi, n * (1 - phi)
        q1, q2 = c / (2 - c), 2 * c ** 2 / (1 + c)
        return np.sqrt((c * (1 - c) + (n1 - 1) * (q1 - c ** 2) + (n0 - 1) * (q2 - c ** 2)) / (n1 * n0))
    return int(np.ceil(brentq(lambda n: se(n) - target_se, 20, 1e7)))


def n_slope(c, phi, width=0.2):
    sigma = brentq(lambda s: c_statistic(lp_distribution(s, phi)[0]) - c, 0.01, 10)
    p, mu = lp_distribution(sigma, phi)
    lp = mu + sigma * GRID
    w = p * (1 - p)
    info = np.array([[w.mean(), (w * lp).mean()], [(w * lp).mean(), (w * lp ** 2).mean()]])
    var_slope_per_patient = np.linalg.inv(info)[1, 1]
    return int(np.ceil(var_slope_per_patient / (width / (2 * Z)) ** 2))


C_MAIN = C_ES2 if MAIN_ANCHOR.startswith("A") else brentq(
    lambda c: cox_snell_from_c(c, PHI)[0] - R2_MAIN, 0.55, 0.95)
val_rows = []
for scen, c in [("Main", C_MAIN), ("Sensitivity: C = 0.65", 0.65), ("Sensitivity: C = 0.80", 0.80)]:
    req = {"O/E ratio (width 0.2)": n_oe(PHI), "C-statistic (width 0.1)": n_cstat(c, PHI),
           "calibration slope (width 0.2)": n_slope(c, PHI)}
    need = max(req.values())
    val_rows.append({"scenario": scen, "C-statistic assumed": round(c, 3),
                     **{f"n for {k}": v for k, v in req.items()},
                     "minimum patients required": need, "minimum events required": int(np.ceil(need * PHI)),
                     "patients available (validation)": VAL_N, "events available": VAL_EVENTS,
                     "available / required": round(VAL_N / need, 1)})
st7_val = pd.DataFrame(val_rows)
st7_val.to_csv(OUT_DIR / "ST7_validation.csv", index=False)
print(st7_val.drop(columns=["patients available (validation)", "events available"]).to_string(index=False))

# %% [markdown]
# ## 7. Assumptions table and log
#
# Every input used, with where it came from, so the calculation can be
# reproduced from the paper alone.

# %%
assumptions = pd.DataFrame([
    ("Outcome rate (anticipated, development)", f"{100 * PHI:.1f}%", "Development cohort, Step 4"),
    ("Maximum Cox-Snell R2 at this rate", f"{MAX_R2:.3f}", "Riley et al. BMJ 2020"),
    ("EuroSCORE II C-statistic for the outcome (development)", f"{C_ES2:.3f}", "Anchor A"),
    ("Cox-Snell R2 from anchor A", f"{R2_ES2:.3f}", "Riley, Van Calster, Collins, Stat Med 2021 conversion"),
    ("Cox-Snell R2 from anchor B", f"{R2_DEFAULT:.3f}", "15% of maximum, Riley et al. BMJ 2020"),
    ("Anchor used in main calculation", MAIN_ANCHOR, "The more demanding of A and B"),
    ("Shrinkage target", "0.90", "Riley et al. BMJ 2020"),
    ("Maximum optimism in Nagelkerke R2", "0.05", "Riley et al. BMJ 2020"),
    ("Margin of error for overall risk", "0.05", "Riley et al. BMJ 2020"),
    ("Validation CI widths: O/E, slope, C", "0.2, 0.2, 0.1", "Riley et al. Stat Med 2021"),
    ("Regression terms", "Splines (3 terms) for continuous predictors", "Step 5, model_summary.csv"),
    ("Sensitivity C-statistics", "0.65 and 0.80", "Decision 73"),
], columns=["input", "value", "source"])
assumptions.to_csv(OUT_DIR / "ST7_assumptions.csv", index=False)
print(assumptions.to_string(index=False))

clean_name = (nc.CLEAN_FILE if nc.CLEAN_FILE.exists() else nc.CLEAN_FILE_CSV).name
log_lines = [
    "Script         : 06a_sample_size.py",
    f"Run at         : {datetime.now().isoformat(timespec='seconds')}",
    f"Clean file     : {clean_name} (schema check passed)",
    f"Development    : {len(dev)} patients, {int(dev['y'].sum())} events",
    f"Main anchor    : {MAIN_ANCHOR}; smallest available/required ratio "
    f"{st7_dev['available / required'].min()} (development), {st7_val['available / required'].min()} (validation)",
    f"Python {sys.version.split()[0]} | pandas {pd.__version__} | numpy {np.__version__} "
    f"| scipy {__import__('scipy').__version__}",
]
(OUT_DIR / "log.txt").write_text("\n".join(log_lines))
print("\n" + "\n".join(log_lines))

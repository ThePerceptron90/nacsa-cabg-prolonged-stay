# %% [markdown]
# # Step 18: final exhibits and the supplementary file (decisions 151-153)
#
# Collects the aggregate outputs of Steps 3-17, gives every exhibit its final number,
# draws the exhibits not yet drawn (Figure 1; SF1-SF3 from the clean file, all years;
# SF9 AUC by year; SF10 training periods), reshapes long results into compact
# journal-style tables (estimate (95% CI), one column per measure), and writes:
#   - main_text_exhibits.docx / .pdf  (Tables 1-3, Figures 1-6, graphical abstract; decisions 151, 159, 164)
#   - supplementary_material.docx / .pdf  (every ST and SF; decision 152)
#   - figures/ (final names, PDF and PNG) and tables/ (final CSVs)
# ST18 (TRIPOD+AI), ST22 (RECORD) and ST23 (CODE-EHR) give manuscript locations; ST19 (PROBAST+AI) is a self-assessment;
# page numbers are added at Step 19 and this script is simply run again (decision 153).
# Aggregate data only; counts 1-9 already suppressed by the steps that made them, and
# the new by-year figures hide any cell based on 1-9 patients.

# %%
import importlib
import json
import os
import re
import shutil
import subprocess
import sys
import textwrap
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from docx import Document
from docx.enum.section import WD_ORIENT
from docx.enum.style import WD_STYLE_TYPE
from docx.enum.table import WD_TABLE_ALIGNMENT
from docx.enum.text import WD_BREAK
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Cm, Pt, RGBColor

CODE_DIR = os.environ.get("NACSA_CODE_DIR", "/content/drive/MyDrive/nacsa/code")
sys.path.insert(0, CODE_DIR)
import nacsa_common as nc
import nacsa_modelling as nm
import nacsa_plots as npl
for mod in (nc, nm, npl):
    importlib.reload(mod)

REPO = Path(CODE_DIR).parent
O = nc.OUTPUTS_DIR
OUT_DIR = O / "18_final_exhibits"
FIG, TAB = OUT_DIR / "figures", OUT_DIR / "tables"
for d in (FIG, TAB):
    d.mkdir(parents=True, exist_ok=True)
STAGES = ["preop", "end_of_surgery", "early_postop"]
STAGE_NAME = {"preop": "Pre-operative", "end_of_surgery": "Intra-operative", "early_postop": "Early post-operative",
              "complication_aware": "Complication-aware (ceiling)"}
MODEL_NAME = {"M0": "EuroSCORE II (recalibrated)", "M1": "Logistic regression", "M2": "Ridge regression (splines)",
              "M3": "XGBoost", "M4": "Random forest", "M5": "Neural network (MLP)"}
LOG, CHECKS = [], []


def log(msg):
    print(msg, flush=True)
    LOG.append(msg)


def check(name, ok, detail=""):
    CHECKS.append((name, bool(ok)))
    log(f"[{'PASS' if ok else 'FAIL'}] {name}" + (f": {detail}" if detail else ""))
    if not ok:
        raise AssertionError(f"Check failed: {name}")


def read_text(path):
    """Descriptive tables: keep labels such as 'None' as text (pandas reads 'None' as missing
    by default, which blanked three Table 1 level labels)."""
    return pd.read_csv(path, keep_default_na=False, na_values=[""])


def ci(est, lo, hi, d=3):
    if pd.isna(est):
        return ""
    return f"{est:.{d}f} ({lo:.{d}f} to {hi:.{d}f})"


def wide(df, index, measures, labels=None, d=3, col="measure"):
    """Long table (index..., measure, estimate, ci_low, ci_high) -> one row per index,
    one column per measure as 'estimate (low to high)'."""
    df = df[df[col].isin(measures)].copy()
    df["value"] = [ci(e, lo, hi, d) for e, lo, hi in zip(df["estimate"], df["ci_low"], df["ci_high"])]
    w = df.pivot_table(index=index, columns=col, values="value", aggfunc="first", sort=False)
    w = w[[m for m in measures if m in w.columns]].reset_index()
    return w.rename(columns=labels or {})


MEASURE_LABEL = {"AUC": "AUC", "PR_AUC": "PR-AUC", "event_rate": "Event rate", "CITL": "Calibration-in-the-large",
                 "slope": "Calibration slope", "ICI": "ICI", "E50": "E50", "E90": "E90", "Brier": "Brier score",
                 "scaled_Brier": "Scaled Brier", "NB_10": "NB 10%", "NB_15": "NB 15%", "NB_20": "NB 20%",
                 "NB_30": "NB 30%", "NB_all_10": "Flag-all NB 10%", "NB_all_15": "Flag-all NB 15%",
                 "NB_all_20": "Flag-all NB 20%", "NB_all_30": "Flag-all NB 30%"}
DISC_CAL = ["AUC", "PR_AUC", "CITL", "slope", "ICI", "E50", "E90", "Brier", "scaled_Brier"]
NB = ["NB_10", "NB_15", "NB_20", "NB_30", "NB_all_10", "NB_all_15", "NB_all_20", "NB_all_30"]
CLS = [f"top{s}_{k}" for s in (10, 20) for k in ("sensitivity", "specificity", "PPV", "NPV")]
for c in CLS:
    MEASURE_LABEL[c] = c.replace("top10_", "Top 10%: ").replace("top20_", "Top 20%: ")


def nice_models(df, col="model"):
    df[col] = df[col].map(lambda m: MODEL_NAME.get(m, m))
    return df


def nice_stages(df, col="stage"):
    df[col] = df[col].map(lambda s: STAGE_NAME.get(s, s))
    return df


tables, figures = {}, {}          # id -> dict(title, note, df / path, ...)


def relabel(x):
    """Moment names agreed 28 Oct 2026 (decision 174): the end-of-surgery model is called intra-operative. Applied to
    every table cell, title and note, so labels written by earlier steps match the manuscript."""
    if isinstance(x, pd.DataFrame):
        x = x.copy()
        x.columns = [relabel(c) for c in x.columns]
        return x.map(relabel)
    if not isinstance(x, str):
        return x
    for old, new in (("End of surgery", "Intra-operative"), ("end of surgery", "intra-operative"),
                     ("End-of-surgery", "Intra-operative"), ("end-of-surgery", "intra-operative"),
                     ("intra-operative (intra-operative)", "intra-operative (end of surgery)")):
        x = x.replace(old, new)
    return x


def add_table(tid, title, df, note="", landscape=False, font=7, parts=None, main=False):
    title, note, df = relabel(title), relabel(note), relabel(df)
    parts = [(relabel(t), relabel(d)) for t, d in parts] if parts else parts
    tables[tid] = {"title": title, "note": note, "df": df, "landscape": landscape, "font": font, "parts": parts, "main": main}
    if parts:
        for k, (sub_title, sub_df) in enumerate(parts):
            sub_df.to_csv(TAB / f"{tid}_{k + 1}.csv", index=False)
    else:
        df.to_csv(TAB / f"{tid}.csv", index=False)


def add_figure(fid, title, note, src_stem=None, draw=None, alt=""):
    """Copy an existing figure (src_stem without extension) or save a drawn one."""
    if src_stem is not None:
        for ext in ("pdf", "png"):
            src = Path(f"{src_stem}.{ext}")
            if src.exists():
                shutil.copy(src, FIG / f"{fid}.{ext}")
    title, note, alt = relabel(title), relabel(note), relabel(alt)
    figures[fid] = {"title": title, "note": note, "png": FIG / f"{fid}.png", "alt": alt}


def save(fig, fid):
    for ext in ("pdf", "png"):
        fig.savefig(FIG / f"{fid}.{ext}", dpi=300)
    plt.close(fig)


# %% [markdown]
# ## 1. Main-text tables (decision 151)

# %%
t1 = read_text(O / "04_cohort_profile" / "table1_main.csv").fillna("")
cad = t1.index[t1["variable"] == "Diseased coronary vessels"][0]
nxt = cad + 1                                               # level 'None' (code 0) of diseased vessels
check("Table 1: diseased-vessels level 'None' found under its header", t1.loc[nxt, "level"] == "None")
t1.loc[nxt, "level"] = "None recorded (treated as missing in the models)"
check("Table 1: no blank level label under a categorical variable",
      not ((t1["level"].isin(["", "nan"])) & (t1["variable"] == "")).any())
add_table("Table1", "Baseline characteristics of patients discharged alive: development (2012-16) and temporal "
          "evaluation (2017-19) cohorts", t1, font=6.5, main=True,
          note="SMD = standardised mean difference. Counts of 1-9 shown as '<10', with a second count hidden ('suppressed') "
               "where needed so the first cannot be recovered. Percentages missing are before imputation.")

perf = pd.read_csv(O / "12_temporal_validation" / "performance_2017_19.csv")
p_models = perf[~perf["what"].str.contains(" - ")].rename(columns={"what": "model"})
t2a = wide(p_models[p_models["stage"].isin(STAGES)], ["stage", "model"], ["AUC", "CITL", "slope", "scaled_Brier"], MEASURE_LABEL)
t2a = nice_models(nice_stages(t2a))
t2a.loc[t2a["model"] == MODEL_NAME["M3"], "model"] = MODEL_NAME["M3"] + " (headline)"
t2a = t2a.rename(columns={"stage": "Prediction moment", "model": "Model"})
ver = pd.read_csv(O / "12_temporal_validation" / "stage_verdicts.csv")
# Table 3 (decision 170): the verdict with its evidence. AUC differences as in Step 12; net-benefit differences
# (per 100 patients) at the four key thresholds with paired 95% CIs from the SAME 2,000 bootstrap resamples Step 12
# saved (patients and resamples are identical at every moment). The verdict itself used point estimates (protocol
# 9.7); the net-benefit CIs are descriptive.
REPS = {s: dict(np.load(nc.PROJECT_DIR / "work" / "validation" / f"scores_all_{s}_replicates.npz")) for s in STAGES}
pt = perf.set_index(["stage", "what", "measure"])["estimate"]
rows3 = []
for _, r in ver[ver["comparison"].isin(["best vs best", "vs EuroSCORE II"])].iterrows():
    a, b, ma, mb = r["from"], r["to"], r["model_from"], r["model_to"]
    label = (f"{STAGE_NAME[a]} to {STAGE_NAME[b].lower()}" if r["comparison"] == "best vs best"
             else f"{STAGE_NAME[a]}: headline model vs EuroSCORE II")
    row = {"Comparison": label,
           "AUC difference (95% CI)": ci(r["AUC_gain"], r["ci_low"], r["ci_high"], 4)}
    for t in (10, 15, 20, 30):
        d = 100 * (REPS[b][f"{mb}|NB_{t}"] - REPS[a][f"{ma}|NB_{t}"])
        e = 100 * (pt[(b, mb, f"NB_{t}")] - pt[(a, ma, f"NB_{t}")])
        lo, hi = np.percentile(d, [2.5, 97.5])
        row[f"Net benefit difference at {t}%"] = ci(e, lo, hi, 2)
    row["Adds value"] = "Yes" if bool(r["adds_value"]) else "No"
    rows3.append(row)
t3 = pd.DataFrame(rows3)
check("Table 3: three stage steps and three comparisons with EuroSCORE II", len(t3) == 6)
check("Table 3: every point estimate inside its 95% CI",
      all(float(v.split(" (")[0]) >= float(v.split("(")[1].split(" to ")[0]) - 1e-9 and
          float(v.split(" (")[0]) <= float(v.split(" to ")[1].rstrip(")")) + 1e-9
          for c in t3.columns if c.startswith("Net") for v in t3[c]))
add_table("Table2", "Performance of every model at each prediction moment on 2017-19 patients (temporal evaluation)",
          t2a, font=7, main=True,
          note="95% confidence intervals from 2,000 paired bootstrap resamples. Calibration-in-the-large: 0 is ideal (negative = "
               "risks too high on average); calibration slope: 1 is ideal. The headline model at each moment was chosen on "
               "2012-16 data before this evaluation.")
add_table("Table3", "Does each later prediction moment add value? Headline model (XGBoost) on 2017-19 patients", t3,
          font=7, main=True,
          note="Differences are later moment minus earlier moment, or headline model minus recalibrated EuroSCORE II. Net "
               "benefit differences are per 100 patients (net extra prolonged stays correctly flagged, after penalising false "
               "flags) at threshold risks of 10%, 15%, 20% and 30%. Pre-specified rule: a later moment adds value when the 95% "
               "CI of the AUC difference lies above zero and net benefit is higher at all four thresholds than the earlier "
               "model's and than flagging everyone or no one; net benefit was judged on point estimates, and its CIs are "
               "descriptive. 95% CIs from the same 2,000 paired bootstrap resamples. Same-algorithm comparisons are in ST11.")

# %% [markdown]
# ## 2. Main-text figures (decision 151)

# %%
flow = pd.read_csv(O / "04_cohort_profile" / "flow_counts.csv", index_col=0)
fig, ax = plt.subplots(figsize=(npl.DOUBLE_COLUMN_WIDTH, 4.6), layout="constrained")
ax.set_xlim(0, 100)
ax.set_ylim(-3, 101)
ax.axis("off")
box = dict(boxstyle="round,pad=0.4", fc="white", ec="0.3", lw=0.6)
cols = {"older": 16, "development": 50, "validation": 84}
heads = {"older": "1996-2011\n(training set C only)", "development": "2012-2016\nDevelopment", "validation": "2017-2019\nTemporal evaluation"}
fmt = lambda v: f"{int(float(v)):,}" if str(v).replace(".", "").isdigit() else str(v)
for p, x in cols.items():
    c = flow[p]
    ax.text(x, 95, heads[p], ha="center", va="center", fontsize=7.5, fontweight="bold")
    ax.text(x, 84, f"Isolated CABG records\nn = {fmt(c['All records'])}", ha="center", va="center", fontsize=6.5, bbox=box)
    ax.text(x, 66, "Excluded:\n"
                   f"died in hospital {fmt(c['Excluded: died in hospital'])}\n"
                   f"survival status unknown {fmt(c['Excluded: survival status unknown'])}\n"
                   f"length of stay unknown {fmt(c['Excluded: discharged alive, length of stay unknown'])}",
            ha="center", va="center", fontsize=5.8, bbox=box)
    ax.text(x, 46, f"Analysed (discharged alive)\nn = {fmt(c['Analysed (primary, discharged alive)'])}\n"
                   f"stay over 10 days: {fmt(c['Events: stay over 10 days (primary)'])} "
                   f"({c['Rate, primary (%)']}%)", ha="center", va="center", fontsize=6.5, bbox=box)
    for y0, y1 in ((80.5, 72.5), (59.5, 51)):
        ax.annotate("", xy=(x, y1), xytext=(x, y0), arrowprops=dict(arrowstyle="->", lw=0.6, color="0.3"))
ax.text(50, 33, "Development: 5-fold nested cross-validation (internal) and final models fitted on all 2012-16 patients, "
               "locked before evaluation", ha="center", fontsize=6.3, style="italic")
moments = [("Pre-operative", "preop"), ("Intra-operative", "end_of_surgery"), ("Early post-operative", "early_postop")]
for k, (lab, st) in enumerate(moments):
    x = 18 + k * 32
    ax.text(x, 16, f"{lab}\n{len(nc.predictors(st))} predictors", ha="center", va="center", fontsize=7, bbox=dict(
        boxstyle="round,pad=0.5", fc="#eef3f8", ec="#2b5d8a", lw=0.8))
    if k < 2:
        ax.annotate("", xy=(x + 22, 16), xytext=(x + 10, 16), arrowprops=dict(arrowstyle="->", lw=0.9, color="#2b5d8a"))
ax.text(50, 3, "Three prediction moments, each adding the information known by then.\nModels at every moment: EuroSCORE II, "
               "logistic and ridge regression, XGBoost, random forest, neural network.", ha="center", va="center", fontsize=6.3)
save(fig, "Figure1")
add_figure("Figure1", "Study design and patient flow",
           "Patients who died in hospital, or whose survival status or length of stay was not recorded, are excluded from the "
           "primary analysis (they are included in sensitivity analyses S1-S2). Counts of 1-9 would be shown as '<10'.")
VAL = O / "12_temporal_validation" / "figures"
# Figures 2-4 redrawn from the saved 2017-19 risks of the locked models (Step 12), with panels lettered A-C
# and no figure title (the legend carries it); identical curves, nothing refitted (decision 159).
val = nc.load_clean(period="validation", allow_validation=True, verbose=False)
_, y_val, ex_val = nm.get_stage_data("preop", period="validation", allow_validation=True, df=val, verbose=False)
y_val = y_val.to_numpy()
by_stage = {}
for st in STAGES:
    r = pd.read_csv(nc.PROJECT_DIR / "work" / "validation" / f"risks_{st}.csv.gz").set_index("record_id").loc[ex_val["record_id"]]
    by_stage[st] = (y_val, {m: r[m].to_numpy() for m in ("M0", "M2", "M3")})
auc_ci = {st: {m: tuple(p_models[(p_models["stage"] == st) & (p_models["model"] == m) & (p_models["measure"] == "AUC")]
                        [["estimate", "ci_low", "ci_high"]].iloc[0]) for m in ("M0", "M2", "M3")} for st in STAGES}
npl.stage_figure("roc", {st: (y, p, {"ci": auc_ci[st]}) for st, (y, p) in by_stage.items()}, FIG / "Figure2", letters=True)
npl.stage_figure("calibration", by_stage, FIG / "Figure3", letters=True)
npl.stage_figure("decision", by_stage, FIG / "Figure4", letters=True)
add_figure("Figure2", "ROC curves on 2017-19 patients at each prediction moment",
           "(A) pre-operative, (B) end of surgery, (C) early post-operative. Ridge regression with splines, XGBoost (headline "
           "model) and recalibrated EuroSCORE II; AUC with 95% CI in each legend. All six models: Supplementary Figure SF5.",
           alt="Three panels of receiver operating characteristic curves, one per prediction moment, each showing three models "
               "as lines rising from the bottom left to the top right above the diagonal of no discrimination.")
add_figure("Figure3", "Calibration curves on 2017-19 patients at each prediction moment",
           "(A) pre-operative, (B) end of surgery, (C) early post-operative. Lowess-smoothed observed proportion against "
           "predicted risk; the diagonal is perfect calibration; histograms of predicted risks below. All models: SF7.",
           alt="Three panels plotting observed against predicted risk of prolonged stay for three models, with a diagonal "
               "reference line and a histogram of predicted risks under each panel.")
add_figure("Figure4", "Decision curves on 2017-19 patients at each prediction moment",
           "(A) pre-operative, (B) end of surgery, (C) early post-operative. Net benefit across threshold risks; flagging "
           "everyone and no one shown for reference; key thresholds 10-30%. All models: SF8.",
           alt="Three panels of net benefit against threshold risk for three models and for the flag-everyone and flag-no-one "
               "strategies.")
add_figure("Figure5", "What the headline model (XGBoost) uses at each prediction moment",
           "(A-C) Mean absolute SHAP value of the 12 most important predictors at the pre-operative, end-of-surgery and early "
           "post-operative moments (log-odds scale). (D-F) SHAP value of each of 5,000 sampled 2017-19 patients for the same "
           "predictors, coloured by the predictor's value (blue low, red high). Contributions to prediction, not causal effects.",
           O / "15_explainability" / "figures" / "F5_shap_importance_beeswarm",
           alt="Six panels: three horizontal bar charts of predictor importance, one per prediction moment, above three "
               "dot plots in which each dot is one patient's contribution for each predictor, coloured from blue to red.")
add_figure("Figure6", "Two methods, one shape: ridge-regression spline curves and XGBoost SHAP dependence for age, creatinine "
           "and bypass time (end-of-surgery models)",
           "(A-C) Ridge regression with restricted cubic splines: log odds ratio relative to the development median. (D-F) "
           "XGBoost: SHAP value of each of 5,000 sampled 2017-19 patients. Both on the log-odds scale; each column covers the "
           "1st-99th percentile of the predictor. Contributions to prediction, not causal effects.",
           O / "15_explainability" / "figures" / "F6_ridge_vs_shap",
           alt="Six panels in two rows: smooth rising curves from ridge regression for age, creatinine and bypass time above "
               "scatter plots of XGBoost contributions for the same three predictors, which follow similar rising patterns.")

# %% [markdown]
# ## 2b. Graphical abstract (decisions 156 and 164): drawn by script, numbers read from the outputs

# %%
auc = {st: float(p_models[(p_models["stage"] == st) & (p_models["model"] == "M3") & (p_models["measure"] == "AUC")]["estimate"].iloc[0])
       for st in STAGES + ["complication_aware"]}
auc_m2 = {st: float(p_models[(p_models["stage"] == st) & (p_models["model"] == "M2") & (p_models["measure"] == "AUC")]["estimate"].iloc[0])
          for st in STAGES}
auc_es = float(p_models[(p_models["stage"] == "preop") & (p_models["model"] == "M0") & (p_models["measure"] == "AUC")]["estimate"].iloc[0])
citl = pd.read_csv(O / "12_temporal_validation" / "performance_by_year.csv")
citl = citl[(citl["what"] == "M3") & (citl["measure"] == "CITL") & citl["stage"].isin(STAGES)]
bb = ver[ver["comparison"].isin(["best vs best", "vs EuroSCORE II"])]
# the four take-home messages (Daniel Fudulu's main findings, decision 175) are printed only if the outputs support them
shap_imp = pd.read_csv(O / "15_explainability" / "shap_importance_by_variable.csv")
top3 = {st: list(shap_imp[shap_imp["stage"] == st].sort_values("mean_abs_shap", ascending=False)["variable"].head(3))
        for st in STAGES}
fdiff_ga = pd.read_csv(O / "17_fairness_subgroups" / "ST17_differences_from_reference.csv")
fdiff_ga = fdiff_ga[(fdiff_ga["stage"] == "early_postop") & (fdiff_ga["model"] == "XGBoost") & (fdiff_ga["measure"] == "AUC")]
check("GA message 1: later moments add less than 0.02 AUC over the pre-operative model, and every moment beats EuroSCORE II",
      auc["early_postop"] - auc["preop"] < 0.02 and bb[bb["comparison"] == "vs EuroSCORE II"]["adds_value"].all())
check("GA message 2: ridge regression within 0.01 AUC of XGBoost at every moment",
      all(abs(auc[st] - auc_m2[st]) < 0.01 for st in STAGES))
check("GA message 3: age is the top predictor and creatinine in the top three at every moment, urgency in the pre-operative top three",
      all(top3[st][0] == "Age" and "Creatinine" in top3[st] for st in STAGES) and "Urgency" in top3["preop"])
check("GA message 4: AUC lower in women and in patients aged 80+ (CI below zero), and risks too high in every 2017-19 year, "
      "increasingly so",
      (fdiff_ga[fdiff_ga["group"].isin(["Female", "80+"])]["ci_high"] < 0).all() and len(fdiff_ga[fdiff_ga["group"].isin(["Female", "80+"])]) == 2
      and (citl["estimate"] < 0).all() and all(citl[(citl["stage"] == st) & (citl["year"] == 2019)]["estimate"].iloc[0]
                                              < citl[(citl["stage"] == st) & (citl["year"] == 2017)]["estimate"].iloc[0] for st in STAGES))
n_dev = fmt(flow.loc["Analysed (primary, discharged alive)", "development"])
n_val = fmt(flow.loc["Analysed (primary, discharged alive)", "validation"])
with plt.rc_context({"font.family": "sans-serif", "font.sans-serif": ["Liberation Sans", "Arial", "DejaVu Sans"]}):
    fig = plt.figure(figsize=(18 / 2.54, 12.5 / 2.54))
    navy, gold, grey = "#1f3b5c", npl.MODEL_STYLE["M3"]["color"], "0.35"
    fig.text(0.5, 0.955, "Staged prediction of prolonged hospital stay after coronary bypass surgery", ha="center",
             va="center", fontsize=12, fontweight="bold", color=navy)
    left = fig.add_axes([0.015, 0.15, 0.245, 0.74])
    left.axis("off")
    left.add_patch(plt.Rectangle((0, 0), 1, 1, transform=left.transAxes, fc="#eef3f8", ec=navy, lw=0.8))
    left.text(0.07, 0.95, "Data and design", fontsize=10, fontweight="bold", color=navy, va="top")
    left.text(0.07, 0.83, "UK National Adult Cardiac\nSurgery Audit\n\n"
                          f"Developed: 2012-16\n{n_dev} patients\n\n"
                          f"Tested once: 2017-19\n{n_val} patients\n\n"
                          "Outcome: hospital stay\nover 10 days\n\n"
                          "6 models, including\nEuroSCORE II", fontsize=9, va="top", linespacing=1.15)
    ax = fig.add_axes([0.36, 0.26, 0.40, 0.6])
    xs = [0, 1, 2]
    ax.axhline(auc_es, color=grey, lw=1.2)
    ax.text(1.4, auc_es + 0.002, f"EuroSCORE II {auc_es:.3f}", fontsize=8.5, va="bottom", color=grey)
    ax.axhline(auc["complication_aware"], color=navy, lw=1.2, ls="--")
    ax.text(-0.35, auc["complication_aware"] + 0.0025, f"Ceiling {auc['complication_aware']:.3f}: knowing every later complication",
            fontsize=8, va="bottom", color=navy)
    ys = [auc[st] for st in STAGES]
    ax.plot(xs, ys, "-o", color=gold, lw=2.2, ms=7, mec="white", mew=1)
    for x, yv in zip(xs, ys):
        ax.text(x, yv + 0.0045, f"{yv:.3f}", ha="center", va="bottom", fontsize=10, fontweight="bold", color="#7a5300")
    ax.set_xlim(-0.4, 3.2)
    ax.set_ylim(0.695, 0.80)
    ax.set_yticks([0.70, 0.75, 0.80])
    ax.set_xticks([0, 1, 2, 3], ["Pre-\noperative", "Intra-\noperative", "Early post-\noperative\n(day 1-2)", "Discharge"], fontsize=8.5)
    ax.tick_params(axis="y", labelsize=8.5)
    ax.set_ylabel("Discrimination (AUC),\n2017-19 patients", fontsize=9)
    ax.grid(False)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    kw = dict(transform=ax.transAxes, color="k", clip_on=False, lw=0.8)
    for dy in (0.0, 0.025):                                       # axis break: the scale starts at 0.70, not 0
        ax.plot([-0.025, 0.025], [0.03 + dy, 0.06 + dy], **kw)
    ax.text(-0.35, 0.797, "Headline model (XGBoost) at each moment", fontsize=8.5, va="top", color="#7a5300")
    right = fig.add_axes([0.80, 0.15, 0.19, 0.74])
    right.axis("off")
    right.text(0.0, 0.95, "Take-home", fontsize=10, fontweight="bold", color=navy, va="top")
    msgs = ["Most prediction is available before surgery; later stages add little",
            "Penalised regression predicts as well as machine learning",
            "Age, kidney function and urgency drive the risk",
            "Less accurate in women and over-80s; risk levels drift"]
    for k, m_ in enumerate(msgs):
        right.text(0.0, 0.83 - k * 0.215, f"{k + 1}", fontsize=11, fontweight="bold", color=gold, va="top")
        right.text(0.13, 0.83 - k * 0.215, textwrap.fill(m_, 19), fontsize=8.5, va="top", linespacing=1.1)
    fig.text(0.5, 0.045, "For bed and discharge planning, not for treatment decisions.", ha="center", fontsize=9,
             style="italic", color=navy)
    for ext in ("pdf", "png"):
        fig.savefig(FIG / f"GraphicalAbstract.{ext}", dpi=300)
    plt.close(fig)
add_figure("GraphicalAbstract", "Graphical abstract",
           "Drawn by script from the output files (18.0 x 12.5 cm, 300 dpi, vector PDF); every number is read from the 2017-19 "
           "results. The vertical axis starts at 0.70 (axis break).",
           alt="Left: data and design. Centre: the headline model's AUC rising across three prediction moments, above a "
               "EuroSCORE II line and below a dashed ceiling line. Right: four take-home messages.")

# %% [markdown]
# ## 3. Supplementary figures

# %%
full = nc.load_clean(period="all", outcome=None, allow_validation=True, verbose=False)
check("all 1996-2019 records loaded for the by-year description", full["Year"].between(1996, 2019).all(), f"{len(full):,}")
preds = list(dict.fromkeys(nc.predictors("complication_aware")))
years = sorted(full["Year"].unique())
miss = pd.DataFrame({y: full.loc[full["Year"] == y, preds].isna().mean() * 100 for y in years})
nmiss = pd.DataFrame({y: full.loc[full["Year"] == y, preds].isna().sum() for y in years})
nyear = full.groupby("Year").size()
nobs = pd.DataFrame(nyear.loc[years].to_numpy()[None, :] - nmiss.to_numpy(), index=nmiss.index, columns=nmiss.columns)
hide = ((nmiss >= 1) & (nmiss <= 9)) | ((nobs >= 1) & (nobs <= 9))
lab = lambda v: nc.VARIABLES.get(v, {}).get("label", v)
fig, ax = plt.subplots(figsize=(npl.DOUBLE_COLUMN_WIDTH, 8.5), layout="constrained")
cmap = plt.get_cmap("Greys").copy()
cmap.set_bad("#e8a87c")                                     # suppressed cells get their own colour (never white = 0%)
im = ax.imshow(np.ma.masked_invalid(miss.where(~hide).to_numpy()), aspect="auto", cmap=cmap, vmin=0, vmax=100)
ax.grid(False)
ax.set_yticks(range(len(preds)), [lab(v) for v in preds], fontsize=5.5)
ax.set_xticks(range(len(years)), years, fontsize=5.5, rotation=90)
fig.colorbar(im, ax=ax, shrink=0.5, label="% missing")
save(fig, "SF1")
add_figure("SF1", "Missing values by year of operation, 1996-2019, for every predictor",
           "Computed from the cleaned registry file (all operations). Darker = more missing. Orange cells: suppressed because "
           "the number of patients with a recorded (or missing) value was 1-9; in practice these are years where the predictor "
           "was almost never recorded. The raw-file audit version (Step 2) informed the choice of 2012-16 for development.")

keyv = ["Creatinine", "PreopDialysis", "iabp_intraop"]
fig, ax = plt.subplots(figsize=(npl.SINGLE_COLUMN_WIDTH if hasattr(npl, "SINGLE_COLUMN_WIDTH") else 3.5, 2.6),
                       layout="constrained")
for v in keyv:
    r = 100 - miss.loc[v].where(~hide.loc[v])
    ax.plot(years, r, marker="o", ms=2, lw=1, label=lab(v))
ax.set_ylabel("% recorded", fontsize=7)
ax.set_ylim(0, 102)
ax.tick_params(labelsize=6)
ax.legend(fontsize=6, frameon=False)
save(fig, "SF2")
add_figure("SF2", "Key predictors recorded by year of operation, 1996-2019",
           "Creatinine, pre-operative dialysis and intra-operative balloon pump; poorly recorded early years motivated the "
           "development period and the all-era predictor sets of training sets B and C (decisions 6, 18 and 69).")

alive = full[full["Mortality"] != 1]
rows = []
for y in years:
    a = alive[alive["Year"] == y]["outcome_survivors"].dropna()
    m = full[full["Year"] == y]["Mortality"].dropna()
    rows.append({"year": y, "long": a.sum(), "n_alive": len(a), "deaths": m.sum(), "n_all": len(m)})
oy = pd.DataFrame(rows)
rate = lambda e, n: np.where((e.between(1, 9)) | ((n - e).between(1, 9)), np.nan, 100 * e / n)
fig, ax = plt.subplots(figsize=(3.5, 2.6), layout="constrained")
ax.plot(oy["year"], rate(oy["long"], oy["n_alive"]), marker="o", ms=2, lw=1, label="Stay over 10 days (discharged alive)")
ax.plot(oy["year"], rate(oy["deaths"], oy["n_all"]), marker="s", ms=2, lw=1, label="In-hospital death")
ax.axvspan(2011.5, 2016.5, color="0.9", lw=0)
ax.axvspan(2016.5, 2019.5, color="0.8", lw=0)
ax.set_ylabel("%", fontsize=7)
ax.tick_params(labelsize=6)
ax.legend(fontsize=6, frameon=False)
save(fig, "SF3")
add_figure("SF3", "Prolonged stay and in-hospital mortality by year of operation, 1996-2019",
           "Light shading: development years (2012-16); darker shading: temporal evaluation (2017-19).")
add_figure("SF4", "Associations between candidate predictors in the development cohort",
           "Correlation (or association) matrix used to check overlaps before modelling (Steps 4-5; decisions 20 and 64).",
           O / "04_cohort_profile" / "associations_matrix")
# SF5-SF8 redrawn here (as Figures 2-4) from the saved Step 12 risks, so panel titles carry the agreed moment names
# (decision 174); identical curves to Step 12's val_*_A/B figures, nothing refitted.
ALL_M = ["M0", "M1", "M2", "M3", "M4", "M5"]
by_all, by_A = {}, {}
for st in STAGES:
    r = pd.read_csv(nc.PROJECT_DIR / "work" / "validation" / f"risks_{st}.csv.gz").set_index("record_id").loc[ex_val["record_id"]]
    by_all[st] = (y_val, {m: r[m].to_numpy() for m in ALL_M})
    by_A[st] = (y_val, {m: r[m].to_numpy() for m in ("M0", "M2", "M3")})
ci_all = {st: {m: tuple(p_models[(p_models["stage"] == st) & (p_models["model"] == m) & (p_models["measure"] == "AUC")]
                        [["estimate", "ci_low", "ci_high"]].iloc[0]) for m in ALL_M} for st in STAGES}
npl.stage_figure("roc", {st: (y, p, {"ci": ci_all[st]}) for st, (y, p) in by_all.items()}, FIG / "SF5", letters=True)
npl.stage_figure("pr", by_A, FIG / "SF6a", letters=True)
npl.stage_figure("pr", by_all, FIG / "SF6b", letters=True)
npl.stage_figure("calibration", by_all, FIG / "SF7", letters=True)
npl.stage_figure("decision", by_all, FIG / "SF8", letters=True)
plt.close("all")
add_figure("SF5", "ROC curves on 2017-19 patients, all models", "Version B of Figure 2 (decision 119).")
add_figure("SF6a", "Precision-recall curves on 2017-19 patients (three models)",
           "Horizontal line: event rate, the value of an uninformative model.")
add_figure("SF6b", "Precision-recall curves on 2017-19 patients, all models", "")
add_figure("SF7", "Calibration curves on 2017-19 patients, all models", "As Figure 3, all six models.")
add_figure("SF8", "Decision curves on 2017-19 patients, all models", "As Figure 4, all six models.")

by_year = pd.read_csv(O / "12_temporal_validation" / "performance_by_year.csv")
fig, axes = plt.subplots(1, 3, figsize=(npl.DOUBLE_COLUMN_WIDTH, 2.6), layout="constrained", sharey=True)
for ax, st in zip(axes, STAGES):
    for k, m in enumerate(["M3", "M2", "M0"]):
        d = by_year[(by_year["stage"] == st) & (by_year["what"] == m) & (by_year["measure"] == "AUC")].sort_values("year")
        ax.errorbar(d["year"] + (k - 1) * 0.1, d["estimate"], yerr=[d["estimate"] - d["ci_low"], d["ci_high"] - d["estimate"]],
                    fmt="o-", ms=2.5, lw=0.8, color=npl.MODEL_STYLE[m]["color"], label=MODEL_NAME[m])
    ax.set_title(STAGE_NAME[st], loc="left", fontweight="bold", fontsize=8)
    ax.set_xticks([2017, 2018, 2019])
    ax.tick_params(labelsize=6)
axes[0].set_ylabel("AUC (95% CI)", fontsize=7)
fig.legend(*axes[0].get_legend_handles_labels(), loc="outside lower center", ncol=3, fontsize=6.5, frameon=False)
save(fig, "SF9")
add_figure("SF9", "Discrimination by year of operation within 2017-19", "Frozen models (decision 104); values in ST15.")

tp = pd.read_csv(O / "14_training_periods" / "training_period_comparison.csv")
tp = tp[~tp["what"].str.contains(r"\|| - ", regex=True)]
order = ["A_M2", "B_M2", "C_M2", "A_M3", "B_M3", "C_M3"]
names = {"A_M2": "A 2012-16, full", "B_M2": "B 2012-16, all-era", "C_M2": "C 1996-2016, all-era"}
fig, axes = plt.subplots(2, 3, figsize=(npl.DOUBLE_COLUMN_WIDTH, 3.6), layout="constrained", sharey=True)
for c, st in enumerate(STAGES):
    for r, meas in enumerate(["AUC", "CITL"]):
        ax = axes[r, c]
        for i, w in enumerate(order[::-1]):
            d = tp[(tp["stage"] == st) & (tp["what"] == w) & (tp["measure"] == meas)]
            if len(d):
                d = d.iloc[0]
                ax.errorbar(d["estimate"], i, xerr=[[d["estimate"] - d["ci_low"]], [d["ci_high"] - d["estimate"]]], fmt="o",
                            ms=2.5, lw=0.8, color=npl.MODEL_STYLE[w[-2:]]["color"])
        if meas == "CITL":
            ax.axvline(0, color="grey", lw=0.5)
        ax.set_xlabel("AUC" if meas == "AUC" else "Calibration-in-the-large", fontsize=6.5)
        ax.tick_params(labelsize=6)
        if r == 0:
            ax.set_title(STAGE_NAME[st], loc="left", fontweight="bold", fontsize=8)
for r in range(2):
    axes[r, 0].set_yticks(range(len(order)), [f"{names[w[:2] + 'M2']} - {'XGBoost' if w.endswith('M3') else 'ridge'}"
                                              for w in order[::-1]], fontsize=5.5)
save(fig, "SF10")
add_figure("SF10", "Training-period comparison on 2017-19 patients",
           "Training sets A (2012-16, full predictors; the main models), B (2012-16, all-era predictors) and C (1996-2016, "
           "all-era predictors); ridge regression and XGBoost (decision 18). Values in ST14.")
F15 = O / "15_explainability" / "figures"
add_figure("SF11", "Ridge regression (early post-operative model): continuous predictors as odds ratio versus the median",
           "5th-95th percentile of each predictor; all stages in ST13 companion CSV.", F15 / "SF11_ridge_spline_curves")
add_figure("SF12", "SHAP dependence plots for age, creatinine and bypass time (XGBoost, end-of-surgery model)",
           "1st-99th percentile of each predictor (decision 143).", F15 / "SF12_shap_dependence")
for k, st in enumerate(STAGES):
    add_figure(f"SF12b_{k + 1}", f"SHAP dependence plots for the 10 most important predictors: {STAGE_NAME[st].lower()} model",
               "Continuous predictors over their 1st-99th percentile; categorical predictors as one strip per level "
               "(decision 144).", F15 / f"SF12b_shap_dependence_top10_{st}")
F17 = O / "17_fairness_subgroups" / "figures"
add_figure("SF13a", "Performance of XGBoost by sex, age band and urgency on 2017-19 patients",
           "Error rates use one cut-off for everyone (top 20% of predicted risk over all 2017-19 patients; decision 149). "
           "Values for all models in ST17.", F17 / "SF13a_subgroups")
add_figure("SF13b", "Calibration-in-the-large and slope by year of operation within 2017-19", "", F17 / "SF13b_calibration_by_year")
add_figure("SF14", "SHAP importance by pre-specified clinical group (XGBoost)",
           "Related predictors summed into clinical groups (decision 105), because standard SHAP splits credit between "
           "dependent predictors.", F15 / "F4_shap_importance_grouped")
F10 = O / "10b_internal_scoring" / "figures"
add_figure("SF15a", "ROC curves, 2012-16 out-of-fold predictions (three models)", "Internal validation (decision 119).", F10 / "SF15_roc_dev_A")
add_figure("SF15b", "ROC curves, 2012-16 out-of-fold predictions, all models", "", F10 / "SF15_roc_dev_B")
add_figure("SF16", "Precision-recall curves, 2012-16 out-of-fold predictions", "", F10 / "SF16_pr_dev")
add_figure("SF17", "Calibration curves, 2012-16 out-of-fold predictions", "", F10 / "SF17_calibration_dev")
add_figure("SF18", "Decision curves, 2012-16 out-of-fold predictions", "", F10 / "SF18_decision_dev")
add_figure("SF19", "Sensitivity analyses: AUC on 2017-19 patients by analysis",
           "S1-S4 use other outcome definitions and S6 other patients, so their AUCs are descriptive, not a test against the "
           "main analysis (decision 148). Values in ST16.", O / "16d_sensitivity_summary" / "figures" / "SF19_sensitivity_auc")
check("every figure file exists", all(f["png"].exists() for f in figures.values()),
      ", ".join(k for k, f in figures.items() if not f["png"].exists()))

# %% [markdown]
# ## 4. Supplementary tables

# %%
vd = read_text(O / "05_predictor_spec" / "variable_dictionary.csv").fillna("")
vd["models_using"] = (vd["models_using"].str.replace("B/C M", "B/C stage ", regex=False)
                      .str.replace(r"\bM1\b", "pre-op", regex=True).str.replace(r"\bM2\b", "end of surgery", regex=True)
                      .str.replace(r"\bM3\b", "early post-op", regex=True).str.replace(r"\bS\b", "complication-aware", regex=True)
                      .str.replace("B/C stage 1", "B/C pre-op").str.replace("B/C stage 2", "B/C end of surgery")
                      .str.replace("B/C stage 3", "B/C early post-op"))
vd = vd.rename(columns={"code_name": "Code", "label": "Variable", "source_or_derivation": "Source or derivation",
                        "type": "Type", "units_or_levels": "Units or levels", "reference_level": "Reference",
                        "stage": "Known at", "models_using": "Stage models using it", "first_year_recorded": "First year recorded",
                        "pct_missing_development": "% missing 2012-16", "pct_missing_validation": "% missing 2017-19",
                        "handling": "Handling"})
add_table("ST1", "Variable dictionary", vd, landscape=True, font=6,
          note="'Stage models using it': the stage models (pre-operative, end of surgery, early post-operative, "
               "complication-aware ceiling) whose predictor set contains the variable; B/C = all-era version used by "
               "training sets B and C (decision 18).")
add_table("ST2", "Patients per level of every categorical variable", read_text(O / "04_cohort_profile" / "category_counts.csv").fillna(""),
          note="Counts of 1-9 shown as '<10'; 'suppressed' = secondary suppression so that a hidden count cannot be "
               "recovered from totals.")
add_table("ST3", "Cleaning rules and number of records changed", read_text(O / "03_clean" / "cleaning_log.csv"),
          note="Step 3 cleaning of the 1996-2019 extract (all years).")
add_table("ST4", "Columns dropped during cleaning, with reasons", read_text(O / "03_clean" / "dropped_columns.csv"),
          note="Group A: derived copies or score components; B: unusable for prediction (too rare, retired, introduced "
               "after 2017 or inconsistent); C: constant; D: replaced by a cleaned derived variable.")
mo = read_text(O / "04_cohort_profile" / "missing_overall.csv").fillna("")
ms = read_text(O / "04_cohort_profile" / "missing_by_sex_age.csv").fillna("")
add_table("ST5", "Missing values by variable: development vs evaluation, and by sex and age band", mo,
          parts=[("A. Development (2012-16) and evaluation (2017-19)", mo), ("B. Development, by sex and age band", ms)])
B6 = O / "06b_missing_data"
s5b = [("A. Missing values by year, 2012-19", "missing_by_year"), ("B. Long-stay rate when each predictor is missing or recorded", "missing_outcome_by_status"),
       ("C. Balloon pump field blank vs recorded", "missing_iabp_comparison"), ("D. Balloon pump imputation check", "missing_iabp_imputation_check"),
       ("E. Post-operative fields: blank vs no vs yes", "missing_postop_comparison"), ("F. Co-missingness between predictors", "missing_comissingness"),
       ("G. Complete cases per model", "missing_complete_cases")]
parts = [(t, read_text(B6 / f"{f}.csv").fillna("")) for t, f in s5b]
add_table("ST5b", "Missing-data evidence supporting the imputation plan (decisions 74-85)", parts[0][1], landscape=True,
          font=6, parts=parts)
tbo = read_text(O / "04_cohort_profile" / "table_by_outcome.csv").fillna("")
osa = read_text(O / "04_cohort_profile" / "outcome_by_sex_age.csv").fillna("")
add_table("ST6", "Baseline characteristics by outcome (development) and outcome rates by sex and age band", tbo, landscape=True,
          parts=[("A. Stay 10 days or less vs over 10 days, development cohort", tbo), ("B. Outcome rates by sex and age band", osa)])
S7 = O / "06a_sample_size"
parts = [("A. Development (Riley et al. 2020)", pd.read_csv(S7 / "ST7_development.csv")),
         ("B. Evaluation (Riley et al. 2021)", pd.read_csv(S7 / "ST7_validation.csv")),
         ("C. Assumptions", pd.read_csv(S7 / "ST7_assumptions.csv"))]
add_table("ST7", "Sample-size calculations", parts[0][1], landscape=True, parts=parts)

space = json.loads((REPO / "config" / "search_spaces.json").read_text())
PLAIN = {
    "learning_rate": "How much each new tree corrects the previous ones (smaller = slower, steadier learning)",
    "max_depth": "Largest number of questions a single tree may ask (depth)",
    "min_child_weight": "Smallest amount of patient information allowed in a tree's final group",
    "subsample": "Share of patients each tree is built on",
    "colsample_bytree": "Share of predictors each tree may use",
    "reg_lambda": "Penalty that shrinks each tree's predictions",
    "max_features": "Share of predictors considered at each split",
    "min_samples_leaf": "Smallest number of patients in a final group (leaf)",
    "max_samples": "Share of patients drawn for each tree",
    "n_layers": "Number of hidden layers",
    "n_neurons": "Neurons per hidden layer",
    "batch_size": "Patients per training step",
    "alpha": "Weight decay: penalty on large network weights",
    "learning_rate_init": "Step size of the network's training",
}
rows = []
for key, spec in space.items():
    if not isinstance(spec, dict) or "tuned" not in spec:
        continue
    for h, r in spec["tuned"].items():
        rng = (f"{r.get('low')} to {r.get('high')}" + (" (log scale)" if r.get("log") else "")) if "low" in r else \
            ", ".join(map(str, r.get("choices", [])))
        rows.append({"Model": key.split("_", 1)[1].replace("_", " "), "Setting": h, "Plain meaning": PLAIN.get(h, ""), "Searched over": rng})
rows.append({"Model": "ridge", "Setting": "lambda", "Plain meaning": "Strength of the ridge penalty",
             "Searched over": "30 values from 10 to 0.00001 (log scale)"})
st8a = pd.DataFrame(rows)
fs = pd.read_csv(O / "11_final_models" / "final_settings_ml.csv")
fs = nice_models(nice_stages(fs)).round(4).fillna("")
ridge_final = []
for st in ["preop", "end_of_surgery", "early_postop", "complication_aware"]:
    r = json.loads((O / "tuning" / "primary" / f"M2_{st}_final.json").read_text())
    ridge_final.append({"stage": STAGE_NAME[st], "lambda": r["best"]["settings"]["lambda"]})
ss = pd.read_csv(O / "10_ml_models" / "search_summary.csv").fillna("")
ss = nice_models(nice_stages(ss))
add_table("ST8", "Hyperparameters: meaning, search ranges and selected values", st8a, landscape=True,
          parts=[("A. Settings searched (30 Optuna trials per search; decision 95)", st8a),
                 ("B. Values selected for the final models (all 2012-16 patients)", fs),
                 ("C. Ridge penalty selected for the final models", pd.DataFrame(ridge_final)),
                 ("D. Searches in the 5 outer rounds of cross-validation", ss)])
add_table("ST8b", "Algorithm inclusion rule applied to every candidate algorithm",
          pd.read_json(REPO / "docs" / "supplement" / "ST8b_algorithm_inclusion.json", orient="records"), landscape=True, font=6,
          note="All five criteria required (decisions 91 and 102).")

trows = []
for folder in sorted((O / "tuning").iterdir()):
    for f in sorted(folder.glob("*.json")):
        r = json.loads(f.read_text())
        for t in r.get("trials", []):
            prm = ", ".join(f"{k}={v:.3g}" if isinstance(v, float) else f"{k}={v}" for k, v in (t.get("params") or {}).items())
            trows.append({"set": folder.name, "model": MODEL_NAME[r["model"]], "stage": STAGE_NAME.get(r["stage"], r["stage"]),
                          "round": r["round"], "trial": t.get("number"), "settings": prm, "trees/epochs": t.get("steps"),
                          "calibration": t.get("calibration"), "inner log loss":
                              round(t["log_loss"], 5) if t.get("log_loss") is not None else "", "status": t.get("state")})
st8c = pd.DataFrame(trows)
log(f"ST8c: {len(st8c):,} tuning records from {st8c.groupby(['set', 'model', 'stage', 'round']).ngroups} searches")

st9 = pd.read_csv(O / "10b_internal_scoring" / "ST9_internal_performance.csv")
st9m = st9[~st9["what"].str.contains(" - ")].rename(columns={"what": "model"})
st9a = nice_models(nice_stages(wide(st9m, ["stage", "model"], DISC_CAL, MEASURE_LABEL)))
sel = pd.read_csv(O / "10b_internal_scoring" / "model_selection.csv")
sel = nice_models(nice_stages(sel))
add_table("ST9", "Internal (cross-validated) performance on 2012-16, every model and stage, and the model-selection rule",
          st9a, landscape=True, font=6,
          parts=[("A. Pooled out-of-fold performance (95% CI)", st9a), ("B. Headline-model rule (decision 88)", sel)])

p10 = nice_models(nice_stages(p_models.copy()))
add_table("ST10", "Full evaluation measures on 2017-19 patients", None, landscape=True, font=6,
          parts=[("A. Discrimination, calibration and overall accuracy", wide(p10, ["stage", "model"], DISC_CAL, MEASURE_LABEL)),
                 ("B. Net benefit at the key thresholds", wide(p10, ["stage", "model"], NB, MEASURE_LABEL, d=4)),
                 ("C. Classification when the top 10% / 20% of predicted risk are flagged", wide(p10, ["stage", "model"], CLS, MEASURE_LABEL))],
          note="ICI, E50, E90: mean, median and 90th percentile of the absolute difference between the smoothed calibration "
               "curve and the predicted risk. NB = net benefit.")
pdif = perf[perf["what"].str.contains(" - ")].copy()
pdif["comparison"] = pdif["what"].map(lambda w: " vs ".join(MODEL_NAME.get(x.strip(), x) for x in w.split(" - ")))
pdif = nice_stages(pdif)
v_all = ver.copy()
v_all["AUC difference (95% CI)"] = [ci(e, lo, hi, 4) for e, lo, hi in zip(v_all["AUC_gain"], v_all["ci_low"], v_all["ci_high"])]
v_all = nice_stages(nice_stages(v_all, "from"), "to")
v_all["model_from"], v_all["model_to"] = v_all["model_from"].map(MODEL_NAME), v_all["model_to"].map(MODEL_NAME)
add_table("ST11", "Paired comparisons on 2017-19 patients", None, landscape=True, font=6,
          parts=[("A. Each model vs ridge regression (difference, 95% CI)",
                  wide(pdif, ["stage", "comparison"], ["AUC", "Brier", "scaled_Brier", "NB_10", "NB_15", "NB_20", "NB_30"], MEASURE_LABEL, d=4)),
                 ("B. Stage-to-stage comparisons (all)", v_all[["from", "to", "comparison", "model_from", "model_to",
                                                                "AUC difference (95% CI)", "rerun_10000", "NB_improves_all_4", "adds_value"]])])
up = pd.read_csv(O / "12_temporal_validation" / "updating_2018_19.csv")
up["model"] = up["what"].str.split("_").str[0].map(MODEL_NAME)
up["version"] = up["what"].str.split("_").str[1].map({"frozen": "Frozen", "citl": "Intercept updated", "logistic": "Logistic recalibration"})
add_table("ST12", "Recalibration scenario: models updated on 2017 patients, evaluated on 2018-19",
          nice_stages(wide(up, ["stage", "model", "version"], ["AUC", "CITL", "slope", "ICI", "scaled_Brier", "NB_15"], MEASURE_LABEL)),
          landscape=True, font=6, note="Decision 97.")
orr = pd.read_csv(O / "15_explainability" / "SF11_ridge_odds_ratios.csv")
orr["OR (95% CI)"] = [ci(e, lo, hi, 2) for e, lo, hi in zip(orr["odds_ratio"], orr["ci_low"], orr["ci_high"])]
orw = orr.pivot_table(index="term", columns="stage", values="OR (95% CI)", aggfunc="first", sort=False).reset_index()
orw = orw[["term"] + [s for s in STAGES if s in orw.columns]].rename(columns=STAGE_NAME).fillna("")
add_table("ST13", "Ridge regression: odds ratios for categorical and binary predictors", orw, landscape=True,
          note="Per unit of each predictor's coding, from an unpenalised logistic regression on the same design (the locked "
               "ridge penalty is effectively zero; decision 143). Continuous predictors are modelled with splines and shown "
               "as curves (SF11).")
tpw = tp.copy()
tpw["training set"] = tpw["what"].str[0]
tpw["model"] = tpw["what"].str[2:].map(MODEL_NAME)
add_table("ST14", "Training-period comparison on 2017-19 patients",
          nice_stages(wide(tpw, ["stage", "model", "training set"], ["AUC", "CITL", "slope", "ICI", "scaled_Brier", "NB_15"], MEASURE_LABEL)),
          landscape=True, font=6, note="Training sets A, B, C as in SF10 (decision 18). Paired differences in the Step 14 outputs.")
by = by_year[~by_year["what"].str.contains(" - ")].rename(columns={"what": "model"})
add_table("ST15", "Performance by year of operation, 2017, 2018 and 2019 (frozen models)",
          nice_models(nice_stages(wide(by, ["stage", "model", "year"], ["AUC", "CITL", "slope", "ICI", "scaled_Brier"], MEASURE_LABEL))),
          landscape=True, font=6, note="Decision 104.")
s16 = pd.read_csv(O / "16d_sensitivity_summary" / "ST16_sensitivity.csv")
s16 = nice_models(nice_stages(wide(s16, ["analysis", "stage", "model"], ["AUC", "CITL", "slope", "ICI", "scaled_Brier", "NB_15"], MEASURE_LABEL)))
sgain = pd.read_csv(O / "16a_outcome_sensitivity" / "ST16_S1_S4_stage_comparisons.csv")
sgain["AUC difference (95% CI)"] = [ci(e, lo, hi, 4) for e, lo, hi in zip(sgain["AUC_gain"], sgain["ci_low"], sgain["ci_high"])]
sgain = nice_stages(nice_stages(sgain, "from"), "to")[["analysis", "from", "to", "AUC difference (95% CI)", "CI_above_zero"]]
pair_rows = []
for f, tag in ((O / "16b_mice_sensitivity" / "ST16_S5_performance.csv", "S5"), (O / "16c_complete_case_sensitivity" / "ST16_S6_performance.csv", "S6")):
    d = pd.read_csv(f)
    d = d[d["what"].str.contains(" - ")].copy()
    d["model"] = d["what"].str.split("_").str[1].str.split(" ").str[0]
    d["analysis"] = tag
    pair_rows.append(d)
pdiff = nice_models(nice_stages(wide(pd.concat(pair_rows), ["analysis", "stage", "model"], ["AUC", "CITL", "slope", "scaled_Brier"], MEASURE_LABEL, d=4)))
s5c = pd.read_csv(O / "16b_mice_sensitivity" / "ST16_S5_pooled_M1_coefficients.csv")
s5c["OR (95% CI)"] = [ci(e, lo, hi, 2) for e, lo, hi in zip(s5c["odds_ratio"], s5c["ci_low"], s5c["ci_high"])]
s5c = nice_stages(s5c)[["stage", "imputations", "term", "OR (95% CI)", "fmi"]].rename(columns={"fmi": "Fraction of missing information"})
cnt = pd.concat([pd.read_csv(O / "16a_outcome_sensitivity" / "ST16_S1_S4_patients_events.csv"),
                 nice_stages(pd.read_csv(O / "16c_complete_case_sensitivity" / "ST16_S6_patients_events.csv"))], ignore_index=True).fillna("")
add_table("ST16", "Sensitivity analyses S1-S6", None, landscape=True, font=6,
          parts=[("A. Performance on 2017-19 by analysis (95% CI)", s16),
                 ("B. S1-S4: stage-to-stage AUC differences (XGBoost)", sgain),
                 ("C. S5 and S6: difference from the main models on the same patients", pdiff),
                 ("D. S5: logistic regression odds ratios pooled with Rubin's rules", s5c),
                 ("E. Patients and events in S1-S4 (2017-19) and S6", cnt)],
          note="S1 composite outcome (stay over 10 days or in-hospital death); S2 all patients; S3 cut-off over 7 days; S4 over "
               "14 days; S5 multiple imputation (10 imputations; 48 for the complication-aware stage by the decision 78 rule); "
               "S6 complete cases (decisions 99 and 145-148).")
fair = pd.read_csv(O / "17_fairness_subgroups" / "ST17_fairness_subgroups.csv")
fw = wide(fair, ["stage", "model", "factor", "group", "patients", "events"],
          ["AUC", "O/E", "CITL", "slope", "FNR_top10", "FPR_top10", "FNR_top20", "FPR_top20"],
          {"O/E": "Observed/expected", "CITL": "Calibration-in-the-large", "slope": "Calibration slope",
           "FNR_top10": "Missed long stays, top 10%", "FPR_top10": "Flagged short stays, top 10%",
           "FNR_top20": "Missed long stays, top 20%", "FPR_top20": "Flagged short stays, top 20%"})
fdiff = pd.read_csv(O / "17_fairness_subgroups" / "ST17_differences_from_reference.csv")
fdiff = fdiff.rename(columns={"difference": "estimate"})
fd = wide(fdiff, ["stage", "model", "factor", "group", "reference"], ["AUC", "O/E", "FNR_top20", "FPR_top20"],
          {"O/E": "Observed/expected", "FNR_top20": "Missed long stays, top 20%", "FPR_top20": "Flagged short stays, top 20%"})
add_table("ST17", "Fairness and subgroups on 2017-19 patients", None, landscape=True, font=6,
          parts=[("A. By sex, age band and urgency (95% CI)", nice_stages(fw)),
                 ("B. Differences from the reference group (95% CI; descriptive)", nice_stages(fd))],
          note="One cut-off for everyone: the top 10% / 20% of predicted risk over all 2017-19 patients (decision 149). "
               "Reference groups: men; age 60-69; elective (decision 150).")

# Reporting checklists with manuscript locations (decision 177): TRIPOD+AI (+ for Abstracts), RECORD and CODE-EHR
# (decision 160). Item texts abridged from the sources opened on Europe PMC (RECORD PMC4595218, CODE-EHR PMC9403753).
CK = REPO / "docs" / "checklists"
tri = json.loads((CK / "tripod_ai_checklist.json").read_text())
tri = pd.DataFrame(tri[1:], columns=tri[0])
add_table("ST18", "TRIPOD+AI checklist, with TRIPOD+AI for Abstracts", None, landscape=True, font=6,
          parts=[("A. TRIPOD+AI (Collins et al. BMJ 2024;385:e078378)", tri[tri["Part"].str.startswith("A")].drop(columns="Part")),
                 ("B. TRIPOD+AI for Abstracts", tri[tri["Part"].str.startswith("B")].drop(columns=["Part", "Section"]))],
          note="Locations refer to the manuscript sections and supplementary items; requirement wording abridged.")
rec = json.loads((CK / "record_checklist.json").read_text())
add_table("ST22", "RECORD checklist (STROBE items and RECORD extensions)", pd.DataFrame(rec[1:], columns=rec[0]),
          landscape=True, font=6, note="Benchimol et al. PLoS Med 2015;12:e1001885. Wording abridged; not applicable items "
                                        "relate to data linkage, which this study did not use.")
ce = json.loads((CK / "code_ehr_checklist.json").read_text())
add_table("ST23", "CODE-EHR minimum standards checklist", pd.DataFrame(ce[1:], columns=ce[0]), landscape=True, font=6,
          note="Kotecha et al. BMJ 2022;378:e069048. Self-rating per area: minimum standard not met, minimum met, or "
               "preferred met. Summary: all five minimum standards met; four of five preferred standards met.")
add_table("ST19", "PROBAST+AI self-assessment", pd.read_json(REPO / "docs" / "supplement" / "ST19_probast_ai_self_assessment.json", orient="records"),
          landscape=True, font=6,
          note="Moons et al. BMJ 2025;388:e082505 (PROBAST+AI, step 3 signalling questions). Answers: yes, probably yes, "
               "probably no, no, no information or not applicable. Completed by the analysis team.")

pe = []
pe.append({"analysis": "Cohort description (primary outcome)", "set": "Development 2012-16", "patients": 75814, "events": 12023})
pe.append({"analysis": "Cohort description (primary outcome)", "set": "Evaluation 2017-19", "patients": 32104, "events": 4571})
fc = pd.read_csv(O / "08_infrastructure" / "folds_check.csv")
for _, r in fc.iterrows():
    pe.append({"analysis": "Internal validation, primary outcome", "set": f"Outer fold {int(r['outer_fold'])} (held out)",
               "patients": int(r["patients_primary"]), "events": int(r["events_primary"])})
older = flow["older"]
pe.append({"analysis": "Training set C (1996-2016)", "set": "Development", "patients": int(older["Analysed (primary, discharged alive)"]) + 75814,
           "events": int(older["Events: stay over 10 days (primary)"]) + 12023})
for _, r in pd.read_csv(O / "16a_outcome_sensitivity" / "ST16_S1_S4_patients_events.csv").iterrows():
    pe.append({"analysis": f"Sensitivity {r['analysis']} ({r['outcome']}, over {r['cutoff_days']} days)", "set": "Evaluation 2017-19",
               "patients": r["patients_2017_19"], "events": r["events_2017_19"]})
for _, r in pd.read_csv(O / "16c_complete_case_sensitivity" / "ST16_S6_patients_events.csv").iterrows():
    pe.append({"analysis": f"Sensitivity S6 complete cases, {STAGE_NAME[r['stage']].lower()}", "set": r["set"],
               "patients": r["patients"], "events": r["events"]})
sub = fair[(fair["stage"] == "preop") & (fair["model"] == "XGBoost") & (fair["measure"] == "AUC")]
for _, r in sub.iterrows():
    pe.append({"analysis": f"Subgroup: {r['factor']}", "set": f"Evaluation 2017-19, {r['group']}", "patients": r["patients"], "events": r["events"]})
st20 = pd.DataFrame(pe)
check("ST20: no count of 1-9 patients or events", not (st20[["patients", "events"]].apply(lambda c: c.between(1, 9)).any().any()))
add_table("ST20", "Participants and outcome events in every analysis", st20,
          note="Development (all 2012-16) and evaluation (2017-19) for the primary outcome; training set C adds 1996-2011.")
add_table("ST21", "Glossary of methods terms in plain language", pd.read_json(REPO / "docs" / "supplement" / "ST21_glossary.json",
          orient="records"), note="For clinical readers (decision 48).")
add_table("ST8c", "Every tuning trial of every hyperparameter search", st8c, landscape=True, font=5,
          note="One row per trial (machine-learning models) or per penalty value (ridge regression); inner log loss on the 25% "
               "inner scoring part (decision 92); calibration = the post-hoc step chosen within the trial (decision 134). "
               "Sets: primary = main analysis; set_B / set_C = training-period comparison; sensitivity_S1-S6 = sensitivity analyses.")

# %% [markdown]
# ## 5. Word documents and PDFs

# %%
def set_landscape(section, on):
    section.orientation = WD_ORIENT.LANDSCAPE if on else WD_ORIENT.PORTRAIT
    w, h = Cm(29.7), Cm(21.0)
    section.page_width, section.page_height = (w, h) if on else (h, w)
    for side in ("left_margin", "right_margin", "top_margin", "bottom_margin"):
        setattr(section, side, Cm(1.5))


def shade(cell, hexcolour):
    tcPr = cell._tc.get_or_add_tcPr()
    sh = OxmlElement("w:shd")
    sh.set(qn("w:val"), "clear")
    sh.set(qn("w:color"), "auto")
    sh.set(qn("w:fill"), hexcolour)
    tcPr.append(sh)


def repeat_header(row):
    trPr = row._tr.get_or_add_trPr()
    el = OxmlElement("w:tblHeader")
    el.set(qn("w:val"), "true")
    trPr.append(el)


def write_df(doc, df, font, width_cm, shaded=True):
    """Table with column widths proportional to content (capped), compact rows, header
    repeated on every page."""
    df = df.astype(object).where(df.notna(), "").map(str)   # real gaps become blank; the label "None" stays
    t = doc.add_table(rows=1, cols=len(df.columns))
    t.style = "Table Grid"
    t.alignment = WD_TABLE_ALIGNMENT.CENTER
    t.autofit = False
    need = [max(min(max([len(str(c))] + [len(v) for v in df[c]]), 40) + 3, 7) for c in df.columns]
    widths = [Cm(width_cm * n / sum(need)) for n in need]
    sname = f"Table text {font}"                           # paragraph mark in the table font, so rows stay low
    if sname not in [x.name for x in doc.styles]:
        sty = doc.styles.add_style(sname, WD_STYLE_TYPE.PARAGRAPH)
        sty.base_style = doc.styles["Normal"]
        sty.font.size = Pt(font)
        pf = sty.paragraph_format
        pf.space_after, pf.space_before, pf.line_spacing = Pt(0), Pt(0), 1.0

    def fill(cell, text, bold=False):
        cell.text = text
        for p in cell.paragraphs:
            p.style = doc.styles[sname]
            for r in p.runs:
                r.font.bold = bold
    for j, c in enumerate(df.columns):
        fill(t.rows[0].cells[j], str(c), bold=True)
        if shaded:                                            # main-text tables: no colour (decision 159)
            shade(t.rows[0].cells[j], "E7E7E7")
    repeat_header(t.rows[0])
    for row in df.itertuples(index=False):
        cells = t.add_row().cells
        for j, v in enumerate(row):
            fill(cells[j], v)
    for j, col in enumerate(t.columns):                    # grid widths (read by LibreOffice) and cell widths (Word)
        col.width = widths[j]
    for row in t.rows:
        for j, cell in enumerate(row.cells):
            cell.width = widths[j]
    tblPr = t._tbl.tblPr                                    # tight cell margins (0.05 cm left/right)
    mar = OxmlElement("w:tblCellMar")
    for side in ("left", "right"):
        el = OxmlElement(f"w:{side}")
        el.set(qn("w:w"), "28")
        el.set(qn("w:type"), "dxa")
        mar.append(el)
    tblPr.append(mar)


def light_png(src):
    """A lighter copy of a figure for the Word/PDF documents (JPEG, 1,600 px wide); the
    full-resolution PDF/PNG files stay in figures/ for the journal."""
    from PIL import Image
    dst = OUT_DIR / "doc_images" / (Path(src).stem + ".jpg")
    dst.parent.mkdir(exist_ok=True)
    im = Image.open(src).convert("RGB")
    if im.width > 1600:
        im = im.resize((1600, round(im.height * 1600 / im.width)), Image.LANCZOS)
    im.save(dst, "JPEG", quality=90, optimize=True)
    return dst


def new_section(doc, landscape):
    sec = doc.add_section()
    set_landscape(sec, landscape)
    return sec


REFS = {r[0]: r[1] for r in json.loads((REPO / "docs" / "supplement" / "key_references.json").read_text())[1:]}


def expand(token):
    """'R61-R63' -> ['R61', 'R62', 'R63']; 'R4' -> ['R4']."""
    a, _, b = token.strip().partition("-")
    return [f"R{i}" for i in range(int(a[1:]), int(b[1:]) + 1)] if b else [a]


def compress(nums):
    nums, out, i = sorted(set(nums)), [], 0
    while i < len(nums):
        j = i
        while j + 1 < len(nums) and nums[j + 1] == nums[j] + 1:
            j += 1
        out.append(str(nums[i]) if i == j else f"{nums[i]}-{nums[j]}")
        i = j + 1
    return ", ".join(out)


def add_rich(paragraph, text, size=None):
    """Write text with **bold** spans."""
    for k, part in enumerate(re.split(r"\*\*", text)):
        if part:
            r = paragraph.add_run(part)
            r.bold = k % 2 == 1
            if size:
                r.font.size = Pt(size)


def write_methods(doc, md_path):
    """Supplementary Methods from Markdown (# title, ## sections, - bullets, **bold**);
    citations [R..] become numbers in order of first use, listed at the end."""
    order = []

    def cite(m):
        ids = [x for tok in m.group(1).split(",") for x in expand(tok)]
        for x in ids:
            if x not in order:
                order.append(x)
        return "[" + compress([order.index(x) + 1 for x in ids]) + "]"

    for block in md_path.read_text().split("\n\n"):
        block = re.sub(r"\[(R\d+(?:-R\d+)?(?:,\s*R\d+(?:-R\d+)?)*)\]", cite, block.strip())
        if not block:
            continue
        if block.startswith("# "):
            doc.add_heading(block[2:], level=1)
        elif block.startswith("## "):
            doc.add_heading(block[3:], level=2)
        elif block.startswith("- "):
            for item in block.split("\n"):
                add_rich(doc.add_paragraph(style="List Bullet"), item[2:].strip())
        else:
            add_rich(doc.add_paragraph(), block.replace("\n", " "))
    missing = [x for x in order if x not in REFS]
    check("every citation in the Supplementary Methods is in the reference list", not missing, ", ".join(missing))
    doc.add_heading("References for the Supplementary Methods", level=2)
    for k, x in enumerate(order, 1):
        add_rich(doc.add_paragraph(), f"{k}. {REFS[x]}", size=8)
    return len(order)


def build(doc_path, title, ids_tables, ids_figures, label, ids_last=(), methods=None):
    doc = Document()
    st = doc.styles["Normal"]
    st.font.name, st.font.size = "Arial", Pt(9)
    set_landscape(doc.sections[0], False)
    doc.add_heading(title, level=0)
    doc.add_paragraph("Prolonged length of stay after coronary artery bypass grafting: staged, interpretable prediction "
                      "model development and temporal evaluation in a national registry (UK National Adult Cardiac Surgery "
                      "Audit, 1996 to March 2019).")
    doc.add_paragraph(f"Generated {datetime.now():%d %B %Y} by scripts/18_final_exhibits.py from the aggregate outputs of "
                      "Steps 3-17. Draft: checklist page numbers are added at Step 19.")
    doc.add_heading("Contents", level=1)
    if methods:
        for line in methods.read_text().splitlines():
            if line.startswith("## "):
                doc.add_paragraph(f"Supplementary Methods {line[3:]}", style="List Bullet")
    for tid in ids_tables:
        doc.add_paragraph(f"{label(tid)}. {tables[tid]['title']}", style="List Bullet")
    for fid in ids_figures:
        doc.add_paragraph(f"{label(fid)}. {figures[fid]['title']}", style="List Bullet")
    for tid in ids_last:
        doc.add_paragraph(f"{label(tid)}. {tables[tid]['title']}", style="List Bullet")
    def put_tables(ids):
        for tid in ids:
            put_table(tid)

    def put_table(tid):
        t = tables[tid]
        new_section(doc, t["landscape"])
        doc.add_heading(f"{label(tid)}. {t['title']}", level=1)
        for sub_title, sub_df in (t["parts"] or [("", t["df"])]):
            if sub_title:
                doc.add_heading(sub_title, level=2)
            write_df(doc, sub_df, t["font"], 26.7 if t["landscape"] else 18.0, shaded=not t.get("main"))
        if t["note"]:
            p = doc.add_paragraph(t["note"])
            p.runs[0].font.size, p.runs[0].italic = Pt(7), True
    if methods:
        new_section(doc, False)
        n_refs = write_methods(doc, methods)
        log(f"Supplementary Methods: {n_refs} references cited")
    put_tables(ids_tables)
    for fid in ids_figures:
        f = figures[fid]
        new_section(doc, False)
        doc.add_heading(f"{label(fid)}. {f['title']}", level=1)
        doc.add_picture(str(light_png(f["png"])), width=Cm(17.5))
        if f["note"]:
            p = doc.add_paragraph(f["note"])
            p.runs[0].font.size, p.runs[0].italic = Pt(7.5), True
        if f.get("alt"):
            p = doc.add_paragraph(f"Alt text: {f['alt']}")
            p.runs[0].font.size = Pt(7.5)
    put_tables(ids_last)
    doc.save(doc_path)
    pdf_filter = ('pdf:writer_pdf_Export:{"UseTaggedPDF":{"type":"boolean","value":"false"},'   # untagged: 3.5 MB instead of 23 MB
                  '"ExportBookmarks":{"type":"boolean","value":"false"}}')
    r = subprocess.run(["soffice", "--headless", "--convert-to", pdf_filter, "--outdir", str(OUT_DIR), str(doc_path)],
                       capture_output=True, text=True, timeout=3600)
    pdf = OUT_DIR / (Path(doc_path).stem + ".pdf")
    check(f"{Path(doc_path).name} converted to PDF", pdf.exists() and pdf.stat().st_size > 0, r.stderr[-300:])
    return pdf


main_label = lambda x: "Graphical abstract" if x == "GraphicalAbstract" else x.replace("Table", "Table ").replace("Figure", "Figure ")
supp_label = lambda x: ("Supplementary Table " if x.startswith("ST") else "Supplementary Figure ") + x.replace("_", ".")
build(OUT_DIR / "main_text_exhibits.docx", "Main-text tables and figures", ["Table1", "Table2", "Table3"],
      [f"Figure{i}" for i in range(1, 7)] + ["GraphicalAbstract"], main_label)
st_order = ["ST1", "ST2", "ST3", "ST4", "ST5", "ST5b", "ST6", "ST7", "ST8", "ST8b", "ST9", "ST10", "ST11", "ST12", "ST13",
            "ST14", "ST15", "ST16", "ST17", "ST18", "ST19", "ST20", "ST21", "ST22", "ST23"]
sf_order = [k for k in figures if k.startswith("SF")]
pdf = build(OUT_DIR / "supplementary_material.docx", "Supplementary material", st_order, sf_order, supp_label, ["ST8c"],
            methods=REPO / "docs" / "supplement" / "extended_methods.md")
check("every supplementary table and figure is in the supplement", set(st_order) | set(sf_order) | {"ST8c"} ==
      {k for k in tables if k.startswith("ST")} | {k for k in figures if k.startswith("SF")})
pages = subprocess.run(["pdfinfo", str(pdf)], capture_output=True, text=True).stdout
log(f"Supplement: {[l for l in pages.splitlines() if l.startswith('Pages')]}; "
    f"{len(st_order) + 1} tables, {len(sf_order)} figures; size {pdf.stat().st_size / 1e6:.1f} MB")
log(f"Checks passed: {sum(ok for _, ok in CHECKS)} of {len(CHECKS)}")
log(f"Run finished {datetime.now():%Y-%m-%d %H:%M}")
(OUT_DIR / "log.txt").write_text("\n".join(LOG) + "\n")

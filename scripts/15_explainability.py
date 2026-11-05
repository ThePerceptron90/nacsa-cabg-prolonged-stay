# %% [markdown]
# # Step 15: explainability (decision on SHAP: headline model per stage, 5,000 patients
# from 2017-19, predictors only; per-variable and pre-specified clinical-group importance)
#
# The headline model is XGBoost at all three stages (decision 88, Step 10b); its locked
# final models have no calibration step, so SHAP values are on the log-odds scale.
# Exact TreeSHAP from XGBoost itself (pred_contribs; Lundberg et al. 2020), checked to
# add up to the model's log-odds for every patient. Ridge (M2) shown as odds ratios
# (categories) and spline curves (continuous predictors), SF11. Privacy: dependence plots
# show only values between the 1st and 99th percentiles (no minimum or maximum shown).

# %%
import importlib
import os
import pickle
import textwrap
import sys
from datetime import datetime

import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import numpy as np
import pandas as pd
import statsmodels.api as sm
import xgboost as xgb

CODE_DIR = os.environ.get("NACSA_CODE_DIR", "/content/drive/MyDrive/nacsa/code")
sys.path.insert(0, CODE_DIR)
import nacsa_common as nc
import nacsa_modelling as nm
import nacsa_plots as npl
for mod in (nc, nm, npl):
    importlib.reload(mod)

OUT_DIR = nc.OUTPUTS_DIR / "15_explainability"
FIG_DIR = OUT_DIR / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)
LOCK = nc.PROJECT_DIR / "work" / "final_models"
STAGES = ["preop", "end_of_surgery", "early_postop"]
GROUPS = {"MI history": ["PrevMI", "RecentMI"], "Angina": ["UA", "CCS"], "Neurological": ["stroke_any", "NeuroDys"],
          "Coronary anatomy and grafting": ["CADExtent", "DistalGrafts", "RCA", "LAD", "Diag", "Cx"],
          "Kidney": ["Creatinine", "PreopDialysis"], "Critical illness": ["CPS", "CardiogenicShock"],
          "Balloon pump": ["iabp_intraop", "iabp_postop"]}
LOG = []


def log(msg):
    print(msg, flush=True)
    LOG.append(msg)


def var_of(feature):
    return feature.split("=")[0].split("_rcs")[0]


val = nc.load_clean(period="validation", allow_validation=True, verbose=False)
dev = nc.load_clean(period="development", verbose=False)

# %% [markdown]
# ## 1. SHAP for the headline model (XGBoost) on 5,000 patients from 2017-19

# %%
imp_rows, grp_rows, shap_store = [], [], {}
for stage in STAGES:
    X, _, _ = nm.get_stage_data(stage, period="validation", allow_validation=True, df=val, verbose=False)
    sample = np.sort(np.random.default_rng(nm.MODEL_SEED).choice(len(X), 5000, replace=False))
    b = pickle.load(open(LOCK / f"{stage}_M3.pkl", "rb"))
    assert b["fitted"].calibrator.method == "none", "headline model has a calibration step: SHAP would not be on its output scale"
    F = b["imputer"].transform(X.iloc[sample])
    Z = np.asarray(b["inputs"].transform(F), float)
    names = b["inputs"].get_feature_names_out()
    booster = b["fitted"].fitted.get_booster()
    contrib = booster.predict(xgb.DMatrix(Z), pred_contribs=True)
    margin = booster.predict(xgb.DMatrix(Z), output_margin=True)
    gap = np.abs(contrib.sum(axis=1) - margin).max()
    log(f"[{'PASS' if gap < 1e-4 else 'FAIL'}] {stage}: SHAP values add up to the model's log-odds (largest gap {gap:.1e})")
    S = pd.DataFrame(contrib[:, :-1], columns=names).T.groupby(var_of).sum().T      # per original variable
    shap_store[stage] = (S, F)
    for v in S.columns:
        imp_rows.append({"stage": stage, "variable": v, "mean_abs_shap": float(S[v].abs().mean())})
    for g, members in GROUPS.items():
        present = [m for m in members if m in S.columns]
        if present:
            grp_rows.append({"stage": stage, "group": g, "members": ", ".join(present),
                             "mean_abs_shap": float(S[present].sum(axis=1).abs().mean())})
imp = pd.DataFrame(imp_rows).sort_values(["stage", "mean_abs_shap"], ascending=[True, False])
grp = pd.DataFrame(grp_rows)
# grouped ranking: clinical groups replace their members, other variables stay as they are
rank_rows = []
for stage in STAGES:
    S, _ = shap_store[stage]
    grouped = {g: S[[m for m in ms if m in S.columns]].sum(axis=1) for g, ms in GROUPS.items() if any(m in S.columns for m in ms)}
    in_group = {m for ms in GROUPS.values() for m in ms}
    for v in S.columns:
        if v not in in_group:
            grouped[v] = S[v]
    for k, s in grouped.items():
        rank_rows.append({"stage": stage, "variable_or_group": k, "mean_abs_shap": float(s.abs().mean())})
ranked_groups = pd.DataFrame(rank_rows).sort_values(["stage", "mean_abs_shap"], ascending=[True, False])

# %% [markdown]
# ## 2. Figures: importance (Figure 4), beeswarm (SF14), dependence (SF12)

# %%
def label(v):
    return nc.VARIABLES.get(v, {}).get("label", v) if isinstance(nc.VARIABLES.get(v), dict) else v


def wrapped(v):
    return textwrap.fill(label(v), 24)                      # long labels on two lines, so the bars keep their width


def stage_panels(height):
    """One panel per stage, each in its own sub-figure so its title sits above its own
    labels and never runs past the figure edge."""
    fig = plt.figure(figsize=(npl.DOUBLE_COLUMN_WIDTH, height), layout="constrained")
    subs = fig.subfigures(1, 3, wspace=0.02)
    axes = [sf.subplots() for sf in subs]
    for sf, stage in zip(subs, STAGES):
        sf.suptitle(npl.STAGE_TITLES[stage], x=0.02, ha="left", fontweight="bold", fontsize=8.5)
    return fig, axes


for name, table, key in (("F4_shap_importance", imp, "variable"), ("F4_shap_importance_grouped", ranked_groups, "variable_or_group")):
    fig, axes = stage_panels(4.2)
    xmax = table["mean_abs_shap"].max() * 1.05                # same scale in every panel
    for ax, stage in zip(axes, STAGES):
        t = table[table["stage"] == stage].head(15)[::-1]
        ax.barh([wrapped(v) for v in t[key]], t["mean_abs_shap"], color=npl.MODEL_STYLE["M3"]["color"])
        ax.set_xlim(0, xmax)
        ax.xaxis.set_major_locator(MaxNLocator(3))
        ax.tick_params(axis="y", labelsize=5.5)
    fig.supxlabel("Mean |SHAP| (log-odds)", fontsize=7.5)
    for ext in ("pdf", "png"):
        fig.savefig(FIG_DIR / f"{name}.{ext}", dpi=300)
    plt.close(fig)

fig, axes = stage_panels(4.2)
for ax, stage in zip(axes, STAGES):
    S, F = shap_store[stage]
    top = imp[imp["stage"] == stage].head(12)["variable"].tolist()[::-1]
    for i, v in enumerate(top):
        x = pd.to_numeric(F[v], errors="coerce").to_numpy(float)
        col = (pd.Series(x).rank(pct=True)).to_numpy()
        jitter = np.random.default_rng(i).uniform(-0.3, 0.3, len(x))
        ax.scatter(S[v], i + jitter, c=col, cmap="coolwarm", s=1, alpha=0.4, linewidths=0, rasterized=True)
    ax.set_yticks(range(len(top)), [wrapped(v) for v in top], fontsize=5.5)
    ax.axvline(0, color="grey", lw=0.5)
fig.supxlabel("SHAP value (log-odds). Colour: predictor value, blue = low, red = high", fontsize=7.5)
for ext in ("pdf", "png"):
    fig.savefig(FIG_DIR / f"SF14_shap_beeswarm.{ext}", dpi=300)
plt.close(fig)

# Figure 5 (decision 151, Hamid): importance and beeswarm in one figure; one column per
# stage, mean |SHAP| bars on top and the beeswarm below, same 12 predictors in the same order.
fig = plt.figure(figsize=(npl.DOUBLE_COLUMN_WIDTH, 7.6), layout="constrained")
subs = fig.subfigures(1, 3, wspace=0.02)
xmax = imp["mean_abs_shap"].max() * 1.05
for k, (sf, stage) in enumerate(zip(subs, STAGES)):
    sf.suptitle(npl.STAGE_TITLES[stage], x=0.02, ha="left", fontweight="bold", fontsize=8.5)
    ax_bar, ax_bee = sf.subplots(2, 1, sharey=True)
    ax_bar.set_title("ABC"[k], loc="left", fontweight="bold", fontsize=10)      # panels A-F (decision 159)
    ax_bee.set_title("DEF"[k], loc="left", fontweight="bold", fontsize=10)
    S, F = shap_store[stage]
    t = imp[imp["stage"] == stage].head(12)[::-1]
    top = t["variable"].tolist()
    ax_bar.barh(range(len(top)), t["mean_abs_shap"], color=npl.MODEL_STYLE["M3"]["color"])
    ax_bar.set_xlim(0, xmax)
    ax_bar.xaxis.set_major_locator(MaxNLocator(3))
    ax_bar.set_xlabel("Mean |SHAP| (log-odds)", fontsize=6.5)
    for i, v in enumerate(top):
        x = pd.to_numeric(F[v], errors="coerce").to_numpy(float)
        col = (pd.Series(x).rank(pct=True)).to_numpy()
        jitter = np.random.default_rng(i).uniform(-0.3, 0.3, len(x))
        ax_bee.scatter(S[v], i + jitter, c=col, cmap="coolwarm", s=1, alpha=0.4, linewidths=0, rasterized=True)
    ax_bee.axvline(0, color="grey", lw=0.5)
    ax_bee.set_xlabel("SHAP value (log-odds)", fontsize=6.5)
    ax_bar.set_yticks(range(len(top)), [wrapped(v) for v in top], fontsize=5.5)
    ax_bee.tick_params(axis="y", labelleft=True, labelsize=5.5)
    for ax in (ax_bar, ax_bee):
        ax.tick_params(axis="x", labelsize=6)
fig.supxlabel("Top: average size of each predictor's contribution. Bottom: one dot per patient; "
              "colour = predictor value (blue low, red high).", fontsize=6.5)
for ext in ("pdf", "png"):
    fig.savefig(FIG_DIR / f"F5_shap_importance_beeswarm.{ext}", dpi=300)
plt.close(fig)

S, F = shap_store["end_of_surgery"]
fig, axes = plt.subplots(1, 3, figsize=(npl.DOUBLE_COLUMN_WIDTH, 2.4), layout="constrained")
for ax, v in zip(axes, ["Age", "Creatinine", "CPB"]):
    x = F[v].to_numpy(float)
    lo, hi = np.percentile(x, [1, 99])
    k = (x >= lo) & (x <= hi)                               # no minimum or maximum shown
    ax.scatter(x[k], S[v][k], s=1, alpha=0.3, color=npl.MODEL_STYLE["M3"]["color"], linewidths=0, rasterized=True)
    ax.axhline(0, color="grey", lw=0.5)
    ax.set_xlabel(label(v))
    ax.set_ylabel("SHAP value (log-odds)")
fig.suptitle("XGBoost, intra-operative model, 2017-19 sample (1st-99th percentile of each predictor)", x=0.01, ha="left", fontsize=8)
for ext in ("pdf", "png"):
    fig.savefig(FIG_DIR / f"SF12_shap_dependence.{ext}", dpi=300)
plt.close(fig)

# SF12b (Hamid, 8 Oct 2026; decision 144): dependence plots for the 10 most important
# predictors of each stage. Continuous: 1st-99th percentile only. Binary and categorical:
# one jittered strip per level; levels with fewer than 10 sampled patients left out.
MIN_LEVEL = 10
dropped_levels = []
for stage in STAGES:
    S, F = shap_store[stage]
    top10 = imp[imp["stage"] == stage].head(10)["variable"].tolist()
    fig, axes = plt.subplots(2, 5, figsize=(npl.DOUBLE_COLUMN_WIDTH, 4.4), layout="constrained")
    for ax, v in zip(axes.flat, top10):
        if nc.VARIABLES[v]["type"] == "continuous":
            x = F[v].to_numpy(float)
            lo, hi = np.percentile(x, [1, 99])
            k = (x >= lo) & (x <= hi)
            ax.scatter(x[k], S[v][k], s=1, alpha=0.3, color=npl.MODEL_STYLE["M3"]["color"], linewidths=0, rasterized=True)
        else:
            counts = F[v].value_counts()
            levels = sorted(counts.index, key=lambda z: (pd.to_numeric(pd.Series([z]), errors="coerce").iloc[0], str(z)))
            shown = [lv for lv in levels if counts[lv] >= MIN_LEVEL]
            dropped_levels += [{"stage": stage, "variable": v, "level": str(lv)} for lv in levels if counts[lv] < MIN_LEVEL]
            for i, lv in enumerate(shown):
                y = S[v].to_numpy()[(F[v] == lv).to_numpy()]
                ax.scatter(i + np.random.default_rng(i).uniform(-0.3, 0.3, len(y)), y, s=1, alpha=0.3,
                           color=npl.MODEL_STYLE["M3"]["color"], linewidths=0, rasterized=True)
            names = nc.VARIABLES[v].get("levels", {})
            def level_name(lv):
                key = int(lv) if isinstance(lv, (int, float, np.number)) and float(lv).is_integer() else lv
                return str(names.get(key, lv)).replace("NYHA ", "").replace("CCS ", "")   # class shown in the axis title
            ticks = [level_name(lv) for lv in shown]
            slant = len(shown) > 2 and max(map(len, ticks)) > 4    # slanted so neighbouring names never touch
            ax.set_xticks(range(len(shown)), ticks, fontsize=5, rotation=30 if slant else 0,
                          ha="right" if slant else "center", rotation_mode="anchor")
        ax.axhline(0, color="grey", lw=0.5)
        ax.set_xlabel(textwrap.fill(label(v), 22), fontsize=6.5)
        ax.tick_params(labelsize=5.5)
    fig.supylabel("SHAP value (log-odds)", fontsize=7.5)
    fig.suptitle(f"XGBoost, {npl.STAGE_TITLES[stage].lower()} model: the 10 most important predictors\n"
                 "(2017-19 sample; continuous predictors shown over their 1st-99th percentile)", x=0.01, ha="left",
                 fontsize=8)
    for ext in ("pdf", "png"):
        fig.savefig(FIG_DIR / f"SF12b_shap_dependence_top10_{stage}.{ext}", dpi=300)
    plt.close(fig)
log(f"SF12b: top-10 dependence plots for 3 stages; levels left out (fewer than {MIN_LEVEL} sampled patients): "
    f"{len(dropped_levels)}")

# %% [markdown]
# ## 3. Ridge (M2): odds ratios and spline curves (SF11), from the locked final models

# %%
or_rows, curve_rows = [], []
for stage in STAGES:
    b = pickle.load(open(LOCK / f"{stage}_M2.pkl", "rb"))
    Xd, yd, _ = nm.get_stage_data(stage, df=dev, verbose=False)
    Fd = b["imputer"].transform(Xd)
    Zd = np.asarray(b["inputs"].transform(Fd), float)
    names = list(b["inputs"].get_feature_names_out())
    glm = sm.GLM(yd.to_numpy(), sm.add_constant(Zd), family=sm.families.Binomial()).fit()
    lam = b["settings"]["settings"]["lambda"]
    lp_gap = np.abs(sm.add_constant(Zd) @ glm.params - b["fitted"].decision_function(Zd)).max()
    log(f"{stage}: ridge lambda {lam:.3g}; unpenalised GLM vs locked ridge, largest linear-predictor difference "
        f"{lp_gap:.1e} (log-odds)")
    sd = b["inputs"].sd_
    for j, n in enumerate(names):
        if "_rcs" in n or f"{n}_rcs1" in names:          # spline variables: shown as curves (SF11)
            continue
        beta, se = glm.params[j + 1] / sd[j], glm.bse[j + 1] / sd[j]
        or_rows.append({"stage": stage, "term": n, "odds_ratio": round(float(np.exp(beta)), 3),
                        "ci_low": round(float(np.exp(beta - 1.96 * se)), 3), "ci_high": round(float(np.exp(beta + 1.96 * se)), 3)})
    for v in sorted({var_of(n) for n in names if "_rcs" in n}):
        cols = [names.index(v)] + [i for i, n in enumerate(names) if var_of(n) == v and "_rcs" in n]
        x = Fd[v].to_numpy(float)
        grid = np.linspace(*np.percentile(x, [5, 95]), 40)
        B = nm.rcs_basis(np.r_[grid, np.median(x)], b["inputs"].knots_[v])
        lp = (B / sd[cols]) @ b["fitted"].coef_[0][cols]
        for g, l in zip(grid, lp[:-1] - lp[-1]):
            curve_rows.append({"stage": stage, "variable": v, "value": round(float(g), 2), "odds_ratio_vs_median": round(float(np.exp(l)), 4)})
ors, curves = pd.DataFrame(or_rows), pd.DataFrame(curve_rows)

# Figure 6 (decision 159): the ridge spline curve (top) beside the XGBoost SHAP dependence plot
# (bottom) for age, creatinine and bypass time, intra-operative (end-of-surgery) models; both on the log-odds scale,
# same x range per column (1st-99th percentile of the 2017-19 SHAP sample), so the two methods can
# be compared directly. Ridge: log odds ratio relative to the development median.
F6_VARS = ["Age", "Creatinine", "CPB"]
b6 = pickle.load(open(LOCK / "end_of_surgery_M2.pkl", "rb"))
Xd6, _, _ = nm.get_stage_data("end_of_surgery", df=dev, verbose=False)
Fd6 = b6["imputer"].transform(Xd6)
names6 = list(b6["inputs"].get_feature_names_out())
S6, F6 = shap_store["end_of_surgery"]
fig, axes = plt.subplots(2, 3, figsize=(npl.DOUBLE_COLUMN_WIDTH, 4.6), layout="constrained", sharex="col")
for c, v in enumerate(F6_VARS):
    x_s = F6[v].to_numpy(float)
    lo, hi = np.percentile(x_s, [1, 99])
    grid = np.linspace(lo, hi, 80)
    cols = [names6.index(v)] + [i for i, n in enumerate(names6) if var_of(n) == v and "_rcs" in n]
    B = nm.rcs_basis(np.r_[grid, np.median(Fd6[v].to_numpy(float))], b6["inputs"].knots_[v])
    lp = (B / b6["inputs"].sd_[cols]) @ b6["fitted"].coef_[0][cols]
    ax = axes[0, c]
    ax.plot(grid, lp[:-1] - lp[-1], color=npl.MODEL_STYLE["M2"]["color"], lw=1.6)
    ax.axhline(0, color="grey", lw=0.5)
    ax.set_title("ABC"[c], loc="left", fontweight="bold", fontsize=10)
    ax = axes[1, c]
    k = (x_s >= lo) & (x_s <= hi)
    ax.scatter(x_s[k], S6[v][k], s=1, alpha=0.3, color=npl.MODEL_STYLE["M3"]["color"], linewidths=0, rasterized=True)
    ax.axhline(0, color="grey", lw=0.5)
    ax.set_xlabel(label(v))
    ax.set_title("DEF"[c], loc="left", fontweight="bold", fontsize=10)
axes[0, 0].set_ylabel("Ridge regression:\nlog odds ratio vs median", fontsize=7)
axes[1, 0].set_ylabel("XGBoost:\nSHAP value (log-odds)", fontsize=7)
for ext in ("pdf", "png"):
    fig.savefig(FIG_DIR / f"F6_ridge_vs_shap.{ext}", dpi=300)
plt.close(fig)
cv = curves[curves["stage"] == "early_postop"]
vs = sorted(cv["variable"].unique())
fig, axes = plt.subplots(1, len(vs), figsize=(npl.DOUBLE_COLUMN_WIDTH, 2.2), layout="constrained", squeeze=False)
for ax, v in zip(axes[0], vs):
    c = cv[cv["variable"] == v]
    ax.plot(c["value"], c["odds_ratio_vs_median"], color=npl.MODEL_STYLE["M2"]["color"])
    ax.axhline(1, color="grey", lw=0.5)
    ax.set_xlabel(label(v))
axes[0][0].set_ylabel("Odds ratio vs median")
fig.suptitle("Ridge regression (splines), early post-op model: continuous predictors (5th-95th percentile)", x=0.01, ha="left", fontsize=8)
for ext in ("pdf", "png"):
    fig.savefig(FIG_DIR / f"SF11_ridge_spline_curves.{ext}", dpi=300)
plt.close(fig)

# %% [markdown]
# ## 4. Save (aggregate only)

# %%
imp.round(4).to_csv(OUT_DIR / "shap_importance_by_variable.csv", index=False)
grp.round(4).to_csv(OUT_DIR / "shap_importance_clinical_groups.csv", index=False)
ranked_groups.round(4).to_csv(OUT_DIR / "shap_ranking_grouped.csv", index=False)
ors.to_csv(OUT_DIR / "SF11_ridge_odds_ratios.csv", index=False)
curves.to_csv(OUT_DIR / "SF11_ridge_spline_curves.csv", index=False)
log(f"Run finished {datetime.now():%Y-%m-%d %H:%M}")
(OUT_DIR / "log.txt").write_text("\n".join(LOG) + "\n")

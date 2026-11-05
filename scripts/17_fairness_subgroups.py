# %% [markdown]
# # Step 17: fairness and subgroups, and calibration by year (decisions 36, 98, 149, 150)
#
# Uses the 2017-19 risks of the locked main models saved in Step 12 (no model refitted).
# Models: XGBoost (headline at every stage), ridge splines (M2) and EuroSCORE II (M0), at the
# three prediction stages. Groups: sex; age band (<60, 60-69, 70-79, 80+); urgency
# (elective, urgent, emergency or salvage). Per group: patients, events, AUC,
# observed/expected ratio, calibration-in-the-large and slope, and the false-negative and
# false-positive rates when the top 10% and top 20% of predicted risk are flagged, the
# cut-off being set once over ALL 2017-19 patients (decision 149, as in use). 95% CIs from
# 2,000 resamples of all patients (decision 96; the cut-off is recomputed in each
# resample); differences from a reference group (men; 60-69; elective) from the same
# resamples, descriptive (decision 150). Groups with fewer than 100 long stays:
# descriptive only (rule 10). Patients with sex or urgency not recorded are left out of
# that factor only. Calibration by year: drawn from the Step 12 by-year results.

# %%
import importlib
import os
import sys
from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

CODE_DIR = os.environ.get("NACSA_CODE_DIR", "/content/drive/MyDrive/nacsa/code")
sys.path.insert(0, CODE_DIR)
import nacsa_common as nc
import nacsa_modelling as nm
import nacsa_plots as npl
for mod in (nc, nm, npl):
    importlib.reload(mod)

OUT_DIR = nc.OUTPUTS_DIR / "17_fairness_subgroups"
FIG_DIR = OUT_DIR / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)
RISKS = nc.PROJECT_DIR / "work" / "validation"
STAGES = ["preop", "end_of_surgery", "early_postop"]
MODELS = {"M3": "XGBoost", "M2": "Ridge splines", "M0": "EuroSCORE II"}
SHARES = (0.10, 0.20)
MIN_EVENTS = 100                                              # rule 10
REFERENCE = {"Sex": "Male", "Age band": "60-69", "Urgency": "Elective"}
LOG, CHECKS = [], []


def log(msg):
    print(msg, flush=True)
    LOG.append(msg)


def check(name, ok, detail=""):
    CHECKS.append((name, bool(ok)))
    log(f"[{'PASS' if ok else 'FAIL'}] {name}" + (f": {detail}" if detail else ""))
    if not ok:
        raise AssertionError(f"Check failed: {name}")


# %% [markdown]
# ## 1. Patients, groups and the saved risks

# %%
val = nc.load_clean(period="validation", allow_validation=True, verbose=False)
_, y_ref, extras = nm.get_stage_data("preop", period="validation", allow_validation=True, df=val, verbose=False)
y = y_ref.to_numpy()
ids = extras["record_id"].to_numpy()
groups = {
    "Sex": extras["Gender"].map({0: "Male", 1: "Female"}),
    "Age band": extras["age_band"],
    "Urgency": extras["Urgency"].map({1: "Elective", 2: "Urgent", 3: "Emergency or salvage", 4: "Emergency or salvage"}),
}
for f, g in groups.items():
    n_missing = int(g.isna().sum())
    log(f"{f}: groups {sorted(g.dropna().unique())}; not recorded: {'<10' if 0 < n_missing < 10 else n_missing}")
groups = {f: g.to_numpy(dtype=object) for f, g in groups.items()}
risk = {}
for st in STAGES:
    r = pd.read_csv(RISKS / f"risks_{st}.csv.gz").set_index("record_id")
    check(f"{st}: saved Step 12 risks cover every 2017-19 patient", r.index.isin(ids).all() and len(r) == len(ids))
    risk[st] = {m: r.loc[ids, m].to_numpy() for m in MODELS}


def oe_citl_slope(yy, pp):
    cs = nm.calibration_intercept_slope(yy, pp)
    return yy.mean() / pp.mean(), cs["CITL"], cs["slope"]


def measures(yy, pp, ww):
    """All group measures for one model; ww = flag weights per share (global cut-off)."""
    out = {"AUC": roc_auc_score(yy, pp) if 0 < yy.sum() < len(yy) else np.nan}
    out["O/E"], out["CITL"], out["slope"] = oe_citl_slope(yy, pp)
    for s, w in ww.items():
        out[f"FNR_top{int(100 * s)}"] = 1 - (w * yy).sum() / yy.sum()
        out[f"FPR_top{int(100 * s)}"] = (w * (1 - yy)).sum() / (1 - yy).sum()
    return out


def all_measures(idx):
    """{(stage, model, factor, group): measures} on the patients idx (a resample or all)."""
    res = {}
    yy = y[idx]
    for st in STAGES:
        for m in MODELS:
            pp = risk[st][m][idx]
            ww = {s: nm.flag_weights(pp, s) for s in SHARES}       # cut-off over ALL patients (decision 149)
            for f, g in groups.items():
                gg = g[idx]
                for lv in pd.unique(g[pd.notna(g)]):
                    k = gg == lv
                    res[(st, m, f, lv)] = measures(yy[k], pp[k], {s: w[k] for s, w in ww.items()})
    return res


# %% [markdown]
# ## 2. Estimates and 2,000 resamples (same resamples for every model and group)

# %%
point = all_measures(np.arange(len(y)))
reps = [all_measures(nm.bootstrap_indices(len(y), b)) for b in range(nm.BOOT_REPLICATES)]
log(f"{nm.BOOT_REPLICATES} resamples done")
rows, diffs = [], []
for (st, m, f, lv), est in point.items():
    k = groups[f] == lv
    n, ev = int(k.sum()), int(y[k].sum())
    for meas, e in est.items():
        draws = np.array([r[(st, m, f, lv)][meas] for r in reps], float)
        draws = draws[np.isfinite(draws)]
        lo, hi = np.percentile(draws, [2.5, 97.5])
        rows.append({"stage": st, "model": MODELS[m], "factor": f, "group": lv, "patients": n, "events": ev,
                     "descriptive_only": ev < MIN_EVENTS, "measure": meas, "estimate": e, "ci_low": lo, "ci_high": hi,
                     "resamples_used": len(draws)})
        ref = REFERENCE[f]
        if lv != ref:
            d = np.array([r[(st, m, f, lv)][meas] - r[(st, m, f, ref)][meas] for r in reps], float)
            d = d[np.isfinite(d)]
            dl, dh = np.percentile(d, [2.5, 97.5])
            diffs.append({"stage": st, "model": MODELS[m], "factor": f, "group": lv, "reference": ref, "measure": meas,
                          "difference": e - point[(st, m, f, ref)][meas], "ci_low": dl, "ci_high": dh})
table, diff = pd.DataFrame(rows), pd.DataFrame(diffs)
check("every group and measure has an estimate and a 95% CI", table[["estimate", "ci_low", "ci_high"]].notna().all().all())
small = table[table["descriptive_only"]][["factor", "group"]].drop_duplicates()
log(f"Groups with fewer than {MIN_EVENTS} long stays (descriptive only): "
    f"{', '.join(small['group']) if len(small) else 'none'}")
check("group sizes add up to the recorded patients of each factor",
      all(table[(table["factor"] == f) & (table["stage"] == "preop") & (table["model"] == "XGBoost") & (table["measure"] == "AUC")]
          ["patients"].sum() == int(pd.notna(g).sum()) for f, g in groups.items()))

# %% [markdown]
# ## 3. Figures: SF13a subgroups (XGBoost), SF13b calibration by year

# %%
order = [("Sex", "Male"), ("Sex", "Female"), ("Age band", "<60"), ("Age band", "60-69"), ("Age band", "70-79"),
         ("Age band", "80+"), ("Urgency", "Elective"), ("Urgency", "Urgent"), ("Urgency", "Emergency or salvage")]
panels = [("AUC", "AUC"), ("O/E", "Observed / expected"), ("FNR_top20", "Missed long stays\n(flag top 20%)"),
          ("FPR_top20", "Flagged short stays\n(flag top 20%)")]
colours = {"preop": "#1b9e77", "end_of_surgery": "#d95f02", "early_postop": "#7570b3"}
x3 = table[table["model"] == "XGBoost"].set_index(["stage", "factor", "group", "measure"])
fig, axes = plt.subplots(1, 4, figsize=(npl.DOUBLE_COLUMN_WIDTH, 3.6), layout="constrained", sharey=True)
for ax, (meas, title) in zip(axes, panels):
    for j, st in enumerate(STAGES):
        for i, (f, lv) in enumerate(order[::-1]):
            r = x3.loc[(st, f, lv, meas)]
            yy = i + (1 - j) * 0.22
            ax.errorbar(r["estimate"], yy, xerr=[[r["estimate"] - r["ci_low"]], [r["ci_high"] - r["estimate"]]], fmt="o",
                        ms=2.5, lw=0.8, color=colours[st], mfc="white" if r["descriptive_only"] else colours[st],
                        label=npl.STAGE_TITLES[st] if (i == 0) else None)
    if meas == "O/E":
        ax.axvline(1, color="grey", lw=0.5)
    ax.set_title(title, fontsize=7)
    ax.tick_params(labelsize=6)
axes[0].set_yticks(range(len(order)), [lv for _, lv in order[::-1]], fontsize=6)
fig.legend(*axes[0].get_legend_handles_labels(), loc="outside lower center", ncol=3, fontsize=6.5, frameon=False)
fig.suptitle("XGBoost on 2017-19 patients by subgroup (95% CI; open circles: fewer than 100 long stays)",
             x=0.01, ha="left", fontsize=8)
for ext in ("pdf", "png"):
    fig.savefig(FIG_DIR / f"SF13a_subgroups.{ext}", dpi=300)
plt.close(fig)

by_year = pd.read_csv(nc.OUTPUTS_DIR / "12_temporal_validation" / "performance_by_year.csv")
by_year = by_year[by_year["what"].isin(MODELS) & by_year["stage"].isin(STAGES)]
fig, axes = plt.subplots(2, 3, figsize=(npl.DOUBLE_COLUMN_WIDTH, 4.0), layout="constrained", sharex=True)
for c, st in enumerate(STAGES):
    for r, (meas, lab, ref) in enumerate([("CITL", "Calibration-in-the-large", 0), ("slope", "Calibration slope", 1)]):
        ax = axes[r, c]
        for k, (m, name) in enumerate(MODELS.items()):
            d = by_year[(by_year["stage"] == st) & (by_year["what"] == m) & (by_year["measure"] == meas)].sort_values("year")
            xx = d["year"] + (k - 1) * 0.12
            ax.errorbar(xx, d["estimate"], yerr=[d["estimate"] - d["ci_low"], d["ci_high"] - d["estimate"]], fmt="o-",
                        ms=2.5, lw=0.8, color=npl.MODEL_STYLE[m]["color"], label=name if (r == 0 and c == 0) else None)
        ax.axhline(ref, color="grey", lw=0.5)
        ax.set_xticks([2017, 2018, 2019])
        ax.tick_params(labelsize=6)
        if c == 0:
            ax.set_ylabel(lab, fontsize=7)
        if r == 0:
            ax.set_title(npl.STAGE_TITLES[st], loc="left", fontweight="bold", fontsize=8)
fig.legend(*axes[0, 0].get_legend_handles_labels(), loc="outside lower center", ncol=3, fontsize=6.5, frameon=False)
for ext in ("pdf", "png"):
    fig.savefig(FIG_DIR / f"SF13b_calibration_by_year.{ext}", dpi=300)
plt.close(fig)
log("Figures SF13a and SF13b saved")

# %% [markdown]
# ## 4. Save (aggregate only; every group has hundreds of patients)

# %%
check("no group with fewer than 10 patients or 10 events is reported", (table["patients"] >= 10).all() and
      (table["events"] >= 10).all() and ((table["patients"] - table["events"]) >= 10).all())
table.round(4).to_csv(OUT_DIR / "ST17_fairness_subgroups.csv", index=False)
diff.round(4).to_csv(OUT_DIR / "ST17_differences_from_reference.csv", index=False)
log(f"Checks passed: {sum(ok for _, ok in CHECKS)} of {len(CHECKS)}")
log(f"Run finished {datetime.now():%Y-%m-%d %H:%M}")
(OUT_DIR / "log.txt").write_text("\n".join(LOG) + "\n")

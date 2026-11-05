# %% [markdown]
# # Step 16d: ST16 and the sensitivity forest plot (decision 148)
#
# Collects the main analysis (Step 12) and S1-S6 (Steps 16a-16c) into one long table,
# ST16 (analysis, stage, model, measure, estimate, 95% CI), and draws one figure: the
# 2017-19 AUC with 95% CI per analysis, one panel per stage, for XGBoost (the headline
# model; main, S1-S4, S6) and ridge splines (main, S5). Descriptive: S1-S4 use other
# outcomes and S6 other patients, so their AUCs are not directly comparable with the
# main one; S5 and S6 also carry paired differences in their own tables.

# %%
import importlib
import os
import sys
from datetime import datetime

import matplotlib.pyplot as plt
import pandas as pd

CODE_DIR = os.environ.get("NACSA_CODE_DIR", "/content/drive/MyDrive/nacsa/code")
sys.path.insert(0, CODE_DIR)
import nacsa_common as nc
import nacsa_plots as npl
for mod in (nc, npl):
    importlib.reload(mod)

OUT_DIR = nc.OUTPUTS_DIR / "16d_sensitivity_summary"
FIG_DIR = OUT_DIR / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)
O = nc.OUTPUTS_DIR
STAGES = ["preop", "end_of_surgery", "early_postop"]
COLS = ["analysis", "stage", "model", "measure", "estimate", "ci_low", "ci_high"]
LOG = []


def log(msg):
    print(msg, flush=True)
    LOG.append(msg)


# %% [markdown]
# ## 1. One long table

# %%
parts = []
main = pd.read_csv(O / "12_temporal_validation" / "performance_2017_19.csv")
main = main[~main["what"].str.contains(" - ")].rename(columns={"what": "model"}).assign(
    analysis="Main (2017-19, all patients)")
parts.append(main)

s14 = pd.read_csv(O / "16a_outcome_sensitivity" / "ST16_S1_S4_performance.csv")
s14 = s14[~s14["what"].str.contains(r"\|| - ", regex=True)].copy()
s14["stage"] = s14["what"].str.rsplit("_", n=1).str[0]
s14["model"] = s14["what"].str.rsplit("_", n=1).str[1]
names = {"S1": "S1 composite (stay >10 d or death)", "S2": "S2 all patients", "S3": "S3 cut-off >7 d",
         "S4": "S4 cut-off >14 d"}
s14["analysis"] = s14["analysis"].map(names)
parts.append(s14)

s5 = pd.read_csv(O / "16b_mice_sensitivity" / "ST16_S5_performance.csv")
s5 = s5[~s5["what"].str.contains(r"\|| - ", regex=True)].copy()
s5["analysis"] = s5["what"].str.split("_").str[0].map({"S5": "S5 multiple imputation", "main": "Main (2017-19, all patients)"})
s5["model"] = s5["what"].str.split("_").str[1]
parts.append(s5[s5["analysis"].str.startswith("S5")])

s6 = pd.read_csv(O / "16c_complete_case_sensitivity" / "ST16_S6_performance.csv")
s6 = s6[~s6["what"].str.contains(r"\|| - ", regex=True)].copy()
s6["analysis"] = s6["what"].str.split("_").str[0].map({"S6": "S6 complete cases",
                                                       "main": "Main models on S6 patients"})
s6["model"] = s6["what"].str.split("_").str[1]
parts.append(s6)

st16 = pd.concat([p[COLS] for p in parts], ignore_index=True)
st16 = st16[st16["stage"].isin(STAGES + ["complication_aware"])]
st16.to_csv(OUT_DIR / "ST16_sensitivity.csv", index=False)
log(f"ST16: {len(st16):,} rows, {st16['analysis'].nunique()} analyses")

# %% [markdown]
# ## 2. Forest plot of AUC (supplement)

# %%
rows = [("Main (2017-19, all patients)", "M3"), ("S1 composite (stay >10 d or death)", "M3"), ("S2 all patients", "M3"),
        ("S3 cut-off >7 d", "M3"), ("S4 cut-off >14 d", "M3"), ("Main models on S6 patients", "M3"),
        ("S6 complete cases", "M3"), ("Main (2017-19, all patients)", "M2"), ("S5 multiple imputation", "M2")]
auc = st16[st16["measure"] == "AUC"].set_index(["analysis", "stage", "model"])
fig, axes = plt.subplots(1, 3, figsize=(npl.DOUBLE_COLUMN_WIDTH, 3.4), layout="constrained", sharey=True)
for ax, stage in zip(axes, STAGES):
    for i, (a, m) in enumerate(rows[::-1]):
        if (a, stage, m) not in auc.index:
            continue
        r = auc.loc[(a, stage, m)]
        ax.errorbar(r["estimate"], i, xerr=[[r["estimate"] - r["ci_low"]], [r["ci_high"] - r["estimate"]]], fmt="o",
                    ms=3, lw=1, capsize=2, color=npl.MODEL_STYLE[m]["color"])
    ax.set_title(npl.STAGE_TITLES[stage], loc="left", fontweight="bold")
    ax.set_xlabel("AUC, 2017-19 (95% CI)")
axes[0].set_yticks(range(len(rows)), [f"{a} - {'XGBoost' if m == 'M3' else 'ridge splines'}" for a, m in rows[::-1]],
                   fontsize=6)
for ext in ("pdf", "png"):
    fig.savefig(FIG_DIR / f"SF19_sensitivity_auc.{ext}", dpi=300)
plt.close(fig)
log("SF19 forest plot saved")
log(f"Run finished {datetime.now():%Y-%m-%d %H:%M}")
(OUT_DIR / "log.txt").write_text("\n".join(LOG) + "\n")

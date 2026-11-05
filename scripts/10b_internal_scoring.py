# %% [markdown]
# # Step 10b: internal performance of all six models (2012-16 out-of-fold)
#
# The first time any performance is computed (decision 139): every model was built
# before this step. Development data only; the 2017-19 data stay locked.
#
# 1. Every measure for every model and stage with paired 95% bootstrap intervals
#    (2,000 resamples, the same for all models and stages; decision 96): ST9.
# 2. AUC in each of the 5 outer folds (decision 96e).
# 3. Headline model per prediction stage by the decision 88 rule (AUC difference
#    vs ridge with 95% CI above zero, ICI not higher, net benefit not lower at
#    10/15/20/30%; ties: AUC then log loss, rule 6), with the Monte Carlo safety
#    rule of decision 140. The complication-aware model is a ceiling analysis,
#    outside the selection (decision 90).
# 4. Internal stage comparisons (supporting; the verdict is from 2017-19, Step 12).
# 5. Development figures SF15-SF18 (the three prediction stages).

# %%
import importlib
import os
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import log_loss, roc_auc_score

CODE_DIR = os.environ.get("NACSA_CODE_DIR", "/content/drive/MyDrive/nacsa/code")
sys.path.insert(0, CODE_DIR)
import nacsa_common as nc
import nacsa_modelling as nm
import nacsa_plots as npl
import nacsa_scoring as ns
for mod in (nc, nm, npl, ns):
    importlib.reload(mod)

OUT_DIR = nc.OUTPUTS_DIR / "10b_internal_scoring"
FIG_DIR = OUT_DIR / "figures"
REP_DIR = nc.PROJECT_DIR / "work" / "scoring"          # replicate values: large, kept on the machine
for d in (OUT_DIR, FIG_DIR, REP_DIR):
    d.mkdir(parents=True, exist_ok=True)
MODELS, ML = ns.MODELS, ["M3", "M4", "M5"]
PREDICTION_STAGES = ["preop", "end_of_surgery", "early_postop"]
NB_KEYS = [f"NB_{t}" for t in (10, 15, 20, 30)]
LOG = []


def log(msg):
    print(msg, flush=True)
    LOG.append(msg)


t_start = time.time()

# %% [markdown]
# ## 1. Score the four stages side by side (separate programs)

# %%
env = {**os.environ, "PYTHONPATH": CODE_DIR, "OMP_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1"}
procs = {s: subprocess.Popen([sys.executable, str(Path(CODE_DIR) / "nacsa_scoring.py"), s, str(REP_DIR)], env=env,
                             stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
         for s in nm.STAGES if not (REP_DIR / f"replicates_{s}.npz").exists()}
for s, p in procs.items():
    out, err = p.communicate()
    if p.returncode != 0:
        raise RuntimeError(f"scoring {s} failed:\n{err}")
log(f"Bootstrap scoring of 4 stages x 6 models x 2,000 resamples: {(time.time() - t_start) / 60:.1f} min")

scores = pd.concat([pd.read_csv(REP_DIR / f"scores_{s}.csv") for s in nm.STAGES], ignore_index=True)
reps = {s: dict(np.load(REP_DIR / f"replicates_{s}.npz")) for s in nm.STAGES}
data = {s: ns.load_stage(s) for s in nm.STAGES}


def est(stage, what, measure):
    r = scores[(scores["stage"] == stage) & (scores["what"] == what) & (scores["measure"] == measure)].iloc[0]
    return r["estimate"], r["ci_low"], r["ci_high"]


# %% [markdown]
# ## 2. AUC in each outer fold, and pooled log loss (rule 6 tie-break)

# %%
fold_rows, logloss = [], {}
for s in nm.STAGES:
    y, preds, rounds = data[s]
    for m in MODELS:
        logloss[(s, m)] = log_loss(y, np.clip(preds[m], nm.LOGIT_EPS, 1 - nm.LOGIT_EPS))
        for rnd in sorted(set(rounds)):
            k = rounds == rnd
            fold_rows.append({"stage": s, "model": m, "fold": rnd, "AUC": round(roc_auc_score(y[k], preds[m][k]), 4)})
folds_auc = pd.DataFrame(fold_rows).pivot_table(index=["stage", "model"], columns="fold", values="AUC").reset_index()

# %% [markdown]
# ## 3. Headline model per prediction stage (decision 88) with the safety rule (decision 140)

# %%
def mc_check(diff, stage_i, model_i, B=1000):
    """Decision 140: Monte Carlo SE of the interval edge nearer zero (bootstrap the bootstrap)."""
    lo, hi = np.percentile(diff, [2.5, 97.5])
    pct, edge = (2.5, lo) if abs(lo) <= abs(hi) else (97.5, hi)
    rng = np.random.default_rng([nm.BOOT_SEED, 140, stage_i, model_i])
    q = [np.percentile(diff[rng.integers(0, len(diff), len(diff))], pct) for _ in range(B)]
    return edge, float(np.std(q))


def auc_diff_10000(y, a, b):
    d = np.empty(10_000)
    for r in range(10_000):
        i = nm.bootstrap_indices(len(y), r)
        d[r] = roc_auc_score(y[i], a[i]) - roc_auc_score(y[i], b[i])
    return np.percentile(d, [2.5, 97.5])


sel_rows, headline = [], {}
for si, s in enumerate(PREDICTION_STAGES):
    y, preds, _ = data[s]
    ici_m2 = est(s, "M2", "ICI")[0]
    nb_m2 = {k: est(s, "M2", k)[0] for k in NB_KEYS}
    passed = []
    for mi, m in enumerate(ML):
        d_est, d_lo, d_hi = est(s, f"{m} - M2", "AUC")
        edge, mc_se = mc_check(reps[s][f"{m}|AUC"] - reps[s]["M2|AUC"], si, mi)
        rerun = abs(edge) < 2 * mc_se
        if rerun:
            d_lo, d_hi = auc_diff_10000(y, preds[m], preds["M2"])
        auc_ok = d_lo > 0
        ici_ok = est(s, m, "ICI")[0] <= ici_m2
        nb_ok = all(est(s, m, k)[0] >= nb_m2[k] for k in NB_KEYS)
        sel_rows.append({"stage": s, "model": m, "AUC_diff_vs_M2": round(d_est, 4), "ci_low": round(d_lo, 4),
                         "ci_high": round(d_hi, 4), "edge_mc_se": round(mc_se, 5), "rerun_10000": rerun,
                         "AUC_higher": auc_ok, "ICI": round(est(s, m, "ICI")[0], 4), "ICI_M2": round(ici_m2, 4),
                         "ICI_not_higher": ici_ok,
                         "NB_not_lower_all_4": nb_ok, "passes": auc_ok and ici_ok and nb_ok})
        if auc_ok and ici_ok and nb_ok:
            passed.append(m)
    if passed:                                      # rule 6: highest AUC, then lower log loss (3 decimals)
        passed.sort(key=lambda m: (-round(est(s, m, "AUC")[0], 3), logloss[(s, m)]))
        headline[s] = passed[0]
    else:
        headline[s] = "M2"
    log(f"{s}: headline model {headline[s]} ({npl.MODEL_STYLE[headline[s]]['label']}); "
        f"machine-learning models passing all three tests: {passed or 'none'}")
selection = pd.DataFrame(sel_rows)

# %% [markdown]
# ## 4. Internal stage comparisons (supporting; same patients at every stage)

# %%
pairs = [("preop", "end_of_surgery"), ("end_of_surgery", "early_postop"), ("preop", "early_postop")]
stage_rows = []
for a, b in pairs:
    for label, ma, mb in [("best vs best", headline[a], headline[b])] + [("same model", m, m) for m in MODELS]:
        d = reps[b][f"{mb}|AUC"] - reps[a][f"{ma}|AUC"]
        stage_rows.append({"from": a, "to": b, "comparison": label, "model_from": ma, "model_to": mb,
                           "AUC_gain": round(est(b, mb, "AUC")[0] - est(a, ma, "AUC")[0], 4),
                           "ci_low": round(float(np.percentile(d, 2.5)), 4),
                           "ci_high": round(float(np.percentile(d, 97.5)), 4)})
stage_cmp = pd.DataFrame(stage_rows)

# %% [markdown]
# ## 5. Development figures SF15-SF18 (three prediction stages)

# %%
by_stage = {s: data[s][:2] for s in PREDICTION_STAGES}
ci = {s: {m: est(s, m, "AUC") for m in MODELS} for s in PREDICTION_STAGES}
best_ml = {s: max(ML, key=lambda m: est(s, m, "AUC")[0]) for s in PREDICTION_STAGES}
npl.stage_figure("roc", {s: (y, p, {"ci": ci[s]}) for s, (y, p) in by_stage.items()}, FIG_DIR / "SF15_roc_dev_B",
                 title="ROC curves, 2012-16 out-of-fold, all models (version B)")
npl.stage_figure("roc", {s: (y, {m: p[m] for m in ("M0", "M2", best_ml[s])},
                             {"ci": {m: ci[s][m] for m in ("M0", "M2", best_ml[s])}}) for s, (y, p) in by_stage.items()},
                 FIG_DIR / "SF15_roc_dev_A", title="ROC curves, 2012-16 out-of-fold (version A)")
for kind, name in [("pr", "SF16_pr_dev"), ("calibration", "SF17_calibration_dev"), ("decision", "SF18_decision_dev")]:
    npl.stage_figure(kind, by_stage, FIG_DIR / name, title=f"2012-16 out-of-fold: {kind}")

# %% [markdown]
# ## 6. Save (aggregate only)

# %%
st9 = scores.copy()
for c in ("estimate", "ci_low", "ci_high"):
    st9[c] = st9[c].round(4)
st9.to_csv(OUT_DIR / "ST9_internal_performance.csv", index=False)
folds_auc.to_csv(OUT_DIR / "auc_by_outer_fold.csv", index=False)
selection.to_csv(OUT_DIR / "model_selection.csv", index=False)
stage_cmp.to_csv(OUT_DIR / "stage_comparison_internal.csv", index=False)
pd.DataFrame([{"stage": s, "model": m, "log_loss": round(v, 4)} for (s, m), v in logloss.items()]) \
    .to_csv(OUT_DIR / "log_loss.csv", index=False)
log(f"Total time: {(time.time() - t_start) / 60:.1f} min")
log(f"Run finished {datetime.now():%Y-%m-%d %H:%M}; pandas {pd.__version__}, numpy {np.__version__}")
(OUT_DIR / "log.txt").write_text("\n".join(LOG) + "\n")

# %% [markdown]
# # Step 12: temporal validation on 2017-19 (opened once; decisions 25, 87, 96, 97, 140, 141)
#
# 1. The 24 locked final models (Step 11) are checked against their fingerprints and
#    applied unchanged to all 2017-19 patients.
# 2. Every model and stage scored with 2,000 paired bootstrap resamples, overall and by
#    year (decision 97), in separate processes side by side.
# 3. Stage verdicts (decision 87): stage adds value if the AUC gain has a 95% CI above
#    zero AND net benefit is higher than the earlier stage's model, flag-all and flag-none
#    at 10, 15, 20 and 30%. Best vs best (headline models of decision 88) is primary; the
#    same algorithm across stages is secondary; each stage is also compared with
#    EuroSCORE II. Safety rule of decision 140 on every verdict AUC interval.
# 4. Updating scenario (decision 97): each model recalibrated on 2017 patients
#    (recalibration-in-the-large; logistic recalibration) and tested on 2018-19, with the
#    frozen model on 2018-19 for comparison.
# 5. Figures (version A: EuroSCORE II, ridge, headline model; version B: all), calibration
#    axis 0-70% (decision 141).

# %%
import hashlib
import importlib
import os
import pickle
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy.special import expit, logit
from sklearn.metrics import roc_auc_score

CODE_DIR = os.environ.get("NACSA_CODE_DIR", "/content/drive/MyDrive/nacsa/code")
sys.path.insert(0, CODE_DIR)
import nacsa_common as nc
import nacsa_modelling as nm
import nacsa_plots as npl
for mod in (nc, nm, npl):
    importlib.reload(mod)

OUT_DIR = nc.OUTPUTS_DIR / "12_temporal_validation"
FIG_DIR = OUT_DIR / "figures"
WORK = nc.PROJECT_DIR / "work" / "validation"            # patient-level risks and npz inputs: machine only
LOCK_DIR = nc.PROJECT_DIR / "work" / "final_models"
for d in (OUT_DIR, FIG_DIR, WORK):
    d.mkdir(parents=True, exist_ok=True)
MODELS = ["M0", "M1", "M2", "M3", "M4", "M5"]
PRED_STAGES = ["preop", "end_of_surgery", "early_postop"]
NB_KEYS = [f"NB_{t}" for t in (10, 15, 20, 30)]
LOG = []


def log(msg):
    print(msg, flush=True)
    LOG.append(msg)


t_start = time.time()

# %% [markdown]
# ## 1. Locked models, applied unchanged

# %%
lock = pd.read_csv(nc.OUTPUTS_DIR / "11_final_models" / "final_model_fingerprints.csv")
bad = [f for f, h in zip(lock["file"], lock["sha256"]) if hashlib.sha256((LOCK_DIR / f).read_bytes()).hexdigest() != h]
if bad:
    raise AssertionError(f"locked model files changed: {bad}")
log(f"[PASS] all {len(lock)} locked model files match their Step 11 fingerprints")

val = nc.load_clean(period="validation", allow_validation=True, verbose=False)
log(f"2017-19 patients loaded once for final evaluation (decision 25): {len(val):,} rows")


def predict_bundle(b, X, es2):
    if b["model"] == "M0":
        return b["wrapper"].predict(b["fitted"], es2)
    Z = np.asarray(b["inputs"].transform(b["imputer"].transform(X)), float)
    return b["fitted"].predict(Z) if hasattr(b["fitted"], "calibrator") else b["wrapper"].predict(b["fitted"], Z)


data = {}
for stage in nm.STAGES:
    X, y, extras = nm.get_stage_data(stage, period="validation", allow_validation=True, df=val, verbose=False)
    preds = {}
    for m in MODELS:
        with open(LOCK_DIR / f"{stage}_{m}.pkl", "rb") as f:
            preds[m] = predict_bundle(pickle.load(f), X, extras["ES2"].to_numpy(float))
    data[stage] = {"y": y.to_numpy(), "year": extras["Year"].to_numpy(), "p": preds}
    pd.DataFrame({"record_id": extras["record_id"].to_numpy(), **preds}).to_csv(WORK / f"risks_{stage}.csv.gz", index=False)
n_val = len(data["preop"]["y"])
log(f"[{'PASS' if all(np.all((p >= 0) & (p <= 1)) for d in data.values() for p in d['p'].values()) else 'FAIL'}] "
    f"risks for {n_val:,} patients x 6 models x 4 stages, all between 0 and 1")

# %% [markdown]
# ## 2. Score overall, by year, and the updating scenario (separate programs, 4 at a time)

# %%
def recalibrate(p_fit, y_fit, p_new, method):
    """Decision 97: fitted on 2017 patients, applied to 2018-19, on the logit of the risk."""
    lp_fit = logit(np.clip(p_fit, nm.LOGIT_EPS, 1 - nm.LOGIT_EPS))
    lp_new = logit(np.clip(p_new, nm.LOGIT_EPS, 1 - nm.LOGIT_EPS))
    if method == "citl":
        a = sm.GLM(y_fit, np.ones((len(y_fit), 1)), family=sm.families.Binomial(), offset=lp_fit).fit().params[0]
        return expit(a + lp_new)
    a, b = sm.GLM(y_fit, sm.add_constant(lp_fit), family=sm.families.Binomial()).fit().params
    return expit(a + b * lp_new)


tasks, update_rows = [], []
for stage, d in data.items():
    np.savez(WORK / f"all_{stage}.npz", y=d["y"], ref=np.array("M2"), **{f"p_{m}": p for m, p in d["p"].items()})
    tasks.append((WORK / f"all_{stage}.npz", WORK / f"scores_all_{stage}"))
    for yr in (2017, 2018, 2019):
        k = d["year"] == yr
        np.savez(WORK / f"y{yr}_{stage}.npz", y=d["y"][k], **{f"p_{m}": p[k] for m, p in d["p"].items()})
        tasks.append((WORK / f"y{yr}_{stage}.npz", WORK / f"scores_y{yr}_{stage}"))
    fit, new = d["year"] == 2017, d["year"] >= 2018
    up = {}
    for m, p in d["p"].items():
        up[f"{m}_frozen"] = p[new]
        up[f"{m}_citl"] = recalibrate(p[fit], d["y"][fit], p[new], "citl")
        up[f"{m}_logistic"] = recalibrate(p[fit], d["y"][fit], p[new], "logistic")
    np.savez(WORK / f"update_{stage}.npz", y=d["y"][new], **{f"p_{k}": v for k, v in up.items()})
    tasks.append((WORK / f"update_{stage}.npz", WORK / f"scores_update_{stage}"))
env = {**os.environ, "PYTHONPATH": CODE_DIR, "OMP_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1"}
pending, running = [t for t in tasks if not Path(f"{t[1]}.csv").exists()], []
while pending or running:
    while pending and len(running) < 4:
        npz, prefix = pending.pop(0)
        running.append((prefix, subprocess.Popen([sys.executable, str(Path(CODE_DIR) / "nacsa_scoring.py"), "--file",
                                                  str(npz), str(prefix)], env=env, stderr=subprocess.PIPE, text=True)))
    time.sleep(5)
    for item in list(running):
        if item[1].poll() is not None:
            running.remove(item)
            if item[1].returncode != 0:
                raise RuntimeError(f"scoring failed for {item[0]}:\n{item[1].stderr.read()}")
log(f"Bootstrap scoring ({len(tasks)} sets): {(time.time() - t_start) / 60:.0f} min")


def table(name):
    t = pd.concat([pd.read_csv(WORK / f"scores_{name}_{s}.csv").assign(stage=s) for s in nm.STAGES], ignore_index=True)
    return t[["stage", "what", "measure", "estimate", "ci_low", "ci_high"]]


overall = table("all")
by_year = pd.concat([table(f"y{yr}").assign(year=yr) for yr in (2017, 2018, 2019)], ignore_index=True)
updating = table("update")
reps = {s: dict(np.load(WORK / f"scores_all_{s}_replicates.npz")) for s in nm.STAGES}


def est(stage, what, measure):
    r = overall[(overall["stage"] == stage) & (overall["what"] == what) & (overall["measure"] == measure)].iloc[0]
    return r["estimate"]


# %% [markdown]
# ## 3. Stage verdicts (decision 87) with the safety rule (decision 140)

# %%
sel = pd.read_csv(nc.OUTPUTS_DIR / "10b_internal_scoring" / "model_selection.csv")
st9 = pd.read_csv(nc.OUTPUTS_DIR / "10b_internal_scoring" / "ST9_internal_performance.csv")
ll = pd.read_csv(nc.OUTPUTS_DIR / "10b_internal_scoring" / "log_loss.csv").set_index(["stage", "model"])["log_loss"]
headline = {}
for s in PRED_STAGES:
    passed = sel[(sel["stage"] == s) & sel["passes"]]["model"].tolist()
    auc = lambda m: st9[(st9["stage"] == s) & (st9["what"] == m) & (st9["measure"] == "AUC")]["estimate"].iloc[0]
    headline[s] = sorted(passed, key=lambda m: (-round(auc(m), 3), ll[(s, m)]))[0] if passed else "M2"
log(f"Headline models (decision 88, from Step 10b): {headline}")


def auc_interval(s_a, m_a, s_b, m_b, tag):
    d = reps[s_b][f"{m_b}|AUC"] - reps[s_a][f"{m_a}|AUC"]
    lo, hi = np.percentile(d, [2.5, 97.5])
    pct, edge = (2.5, lo) if abs(lo) <= abs(hi) else (97.5, hi)
    rng = np.random.default_rng([nm.BOOT_SEED, 141, tag])
    se = np.std([np.percentile(d[rng.integers(0, len(d), len(d))], pct) for _ in range(1000)])
    if abs(edge) < 2 * se:                                  # decision 140: rerun with 10,000
        ya, yb = data[s_a]["y"], data[s_b]["y"]
        dd = np.empty(10_000)
        for r in range(10_000):
            i = nm.bootstrap_indices(len(ya), r)
            dd[r] = roc_auc_score(yb[i], data[s_b]["p"][m_b][i]) - roc_auc_score(ya[i], data[s_a]["p"][m_a][i])
        lo, hi = np.percentile(dd, [2.5, 97.5])
        return lo, hi, True
    return lo, hi, False


rows, tag = [], 0
comparisons = [("preop", "end_of_surgery"), ("end_of_surgery", "early_postop"), ("preop", "early_postop")]
for a, b in comparisons:
    for kind, ma, mb in [("best vs best", headline[a], headline[b])] + [("same algorithm", m, m) for m in MODELS[1:]]:
        tag += 1
        lo, hi, rerun = auc_interval(a, ma, b, mb, tag)
        nb_ok = all(est(b, mb, k) > est(a, ma, k) and est(b, mb, k) > est(b, mb, k.replace("NB_", "NB_all_"))
                    and est(b, mb, k) > 0 for k in NB_KEYS)
        rows.append({"from": a, "to": b, "comparison": kind, "model_from": ma, "model_to": mb,
                     "AUC_gain": round(est(b, mb, "AUC") - est(a, ma, "AUC"), 4), "ci_low": round(lo, 4),
                     "ci_high": round(hi, 4), "rerun_10000": rerun, "AUC_gain_clear": lo > 0,
                     "NB_improves_all_4": nb_ok, "adds_value": lo > 0 and nb_ok})
for s in PRED_STAGES:
    tag += 1
    lo, hi, rerun = auc_interval(s, "M0", s, headline[s], tag)
    nb_ok = all(est(s, headline[s], k) > est(s, "M0", k) and est(s, headline[s], k) > est(s, headline[s], k.replace("NB_", "NB_all_"))
                and est(s, headline[s], k) > 0 for k in NB_KEYS)
    rows.append({"from": s, "to": s, "comparison": "vs EuroSCORE II", "model_from": "M0", "model_to": headline[s],
                 "AUC_gain": round(est(s, headline[s], "AUC") - est(s, "M0", "AUC"), 4), "ci_low": round(lo, 4),
                 "ci_high": round(hi, 4), "rerun_10000": rerun, "AUC_gain_clear": lo > 0,
                 "NB_improves_all_4": nb_ok, "adds_value": lo > 0 and nb_ok})
verdicts = pd.DataFrame(rows)
log(f"Stage verdicts computed: {len(verdicts)} comparisons; safety-rule reruns: {int(verdicts['rerun_10000'].sum())}")

# %% [markdown]
# ## 4. Figures and save (aggregate only)

# %%
by_stage = {s: (data[s]["y"], data[s]["p"]) for s in PRED_STAGES}
pick = {s: ["M0", "M2", headline[s]] for s in PRED_STAGES}
ci = {s: {m: tuple(overall[(overall["stage"] == s) & (overall["what"] == m) & (overall["measure"] == "AUC")]
                   [["estimate", "ci_low", "ci_high"]].iloc[0]) for m in MODELS} for s in PRED_STAGES}
for version, sub in (("A", pick), ("B", {s: MODELS for s in PRED_STAGES})):
    npl.stage_figure("roc", {s: (y, {m: p[m] for m in sub[s]}, {"ci": {m: ci[s][m] for m in sub[s]}})
                             for s, (y, p) in by_stage.items()}, FIG_DIR / f"val_roc_{version}",
                     title=f"ROC curves, 2017-19 temporal validation (version {version})")
    for kind in ("pr", "calibration", "decision"):
        npl.stage_figure(kind, {s: (y, {m: p[m] for m in sub[s]}) for s, (y, p) in by_stage.items()},
                         FIG_DIR / f"val_{kind}_{version}", title=f"2017-19 temporal validation: {kind} (version {version})")
for t, name in ((overall, "performance_2017_19"), (by_year, "performance_by_year"), (updating, "updating_2018_19")):
    t.round({"estimate": 4, "ci_low": 4, "ci_high": 4}).to_csv(OUT_DIR / f"{name}.csv", index=False)
verdicts.to_csv(OUT_DIR / "stage_verdicts.csv", index=False)
log(f"Total time: {(time.time() - t_start) / 60:.0f} min")
log(f"Run finished {datetime.now():%Y-%m-%d %H:%M}")
(OUT_DIR / "log.txt").write_text("\n".join(LOG) + "\n")

# %% [markdown]
# # Step 16a: sensitivity analyses S1-S4, other outcome definitions (decisions 99, 145)
#
# S1 composite (stay over 10 days or in-hospital death), S2 all patients (deaths counted
# by their actual stay), S3 survivors with a cut-off over 7 days, S4 over 14 days.
# For each analysis and each of the three prediction stages, exactly as Step 14:
# XGBoost (the headline model at every stage, decision 88) is tuned with the 30-trial
# search on the 'final' 75/25 split of 2012-16, fitted on all 2012-16 patients and
# locked; EuroSCORE II is recalibrated on 2012-16. Both are tested once on 2017-19 with
# 2,000 paired bootstrap resamples: performance per stage, stage-to-stage AUC
# differences (with the decision 140 Monte Carlo safety rule) and XGBoost vs EuroSCORE II.
# Main-vs-sensitivity differences are descriptive (different outcomes, so not paired).

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
from sklearn.metrics import roc_auc_score

CODE_DIR = os.environ.get("NACSA_CODE_DIR", "/content/drive/MyDrive/nacsa/code")
sys.path.insert(0, CODE_DIR)
import nacsa_common as nc
import nacsa_modelling as nm
import nacsa_tuning as nt
for mod in (nc, nm, nt):
    importlib.reload(mod)

OUT_DIR = nc.OUTPUTS_DIR / "16a_outcome_sensitivity"
WORK = nc.PROJECT_DIR / "work" / "sensitivity_outcomes"     # models and patient-level risks: machine only
for d in (OUT_DIR, WORK):
    d.mkdir(parents=True, exist_ok=True)
STAGES = ["preop", "end_of_surgery", "early_postop"]
ANALYSES = {"S1": ("composite", 10), "S2": ("all", 10), "S3": ("survivors", 7), "S4": ("survivors", 14)}
COMPARISONS = [("preop", "end_of_surgery"), ("end_of_surgery", "early_postop"), ("preop", "early_postop")]
LOG, CHECKS = [], []


def log(msg):
    print(msg, flush=True)
    LOG.append(msg)


def check(name, ok, detail=""):
    CHECKS.append((name, bool(ok)))
    log(f"[{'PASS' if ok else 'FAIL'}] {name}" + (f": {detail}" if detail else ""))
    if not ok:
        raise AssertionError(f"Check failed: {name}")


def prepare(stage, X, fit_rows, apply_rows):
    imp = nm.StageImputer(stage).fit(X[fit_rows])
    rec = nm.ModelInputs("tree").fit(imp.transform(X[fit_rows]))
    Z = lambda rows: np.asarray(rec.transform(imp.transform(X[rows])), float)
    return imp, rec, Z(fit_rows), Z(apply_rows)


t_start = time.time()

# %% [markdown]
# ## 1. Tune, fit and lock XGBoost and EuroSCORE II for every analysis and stage

# %%
units, jobs = {}, []
for s, (outcome, cutoff) in ANALYSES.items():
    dev = nc.load_clean(period="development", outcome=outcome, verbose=False)
    check(f"{s}: only 2012-16 patients loaded (validation lock)", dev["Year"].between(2012, 2016).all(), f"{len(dev):,} rows")
    for stage in STAGES:
        X, y, extras = nm.get_stage_data(stage, outcome=outcome, cutoff=cutoff, df=dev, verbose=False)
        yv = y.to_numpy()
        roles = nm.attach_folds(extras)["final"]
        tr, tu = (roles == "train").to_numpy(), (roles == "tune").to_numpy()
        _, _, Z_tr, Z_tu = prepare(stage, X, tr, tu)
        allp = np.ones(len(yv), bool)
        imp, rec, Z_all, _ = prepare(stage, X, allp, allp)
        units[(s, stage)] = {"imp": imp, "rec": rec, "Z": Z_all, "y": yv, "es2": extras["ES2"].to_numpy(float)}
        jobs.append(dict(model="M3", stage=stage, round_name="final", Z_tr=Z_tr, y_tr=yv[tr], Z_tu=Z_tu, y_tu=yv[tu],
                         out_dir=nc.OUTPUTS_DIR / "tuning" / f"sensitivity_{s}", n_jobs=1))
        log(f"{s} {stage}: {len(yv):,} patients, {yv.sum():,} events ({tr.sum():,} / {tu.sum():,} on the tuning split)")
records = {(j["out_dir"].name.split("_")[-1], r["stage"]): r for j, r in zip(jobs, nt.run_studies(jobs, n_processes=4))}
log(f"{len(jobs)} XGBoost searches done ({(time.time() - t_start) / 60:.0f} min so far)")
del jobs

lock_rows, bundles, settings_rows = [], {}, []
for (s, stage), u in units.items():
    r = records[(s, stage)]
    check(f"{s} {stage}: search finished without failure", r["status"] == "ok")
    w = nm.XGBoostModel(n_jobs=4)
    bundles[(s, stage, "M3")] = {"imputer": u["imp"], "inputs": u["rec"], "wrapper": w,
                                 "fitted": nm.fit_with_calibration(w, u["Z"], u["y"], r["best"]["settings"],
                                                                   r["best"]["calibration"])}
    m0 = nm.EuroScoreModel()
    bundles[(s, stage, "M0")] = {"wrapper": m0, "fitted": m0.fit(u["es2"], u["y"])}
    settings_rows.append({"analysis": s, "stage": stage, "calibration": r["best"]["calibration"],
                          "trees": r["best"]["steps"], **r["best"]["params"]})
for (s, stage, m), b in bundles.items():
    path = WORK / f"{s}_{stage}_{m}.pkl"
    with open(path, "wb") as f:
        pickle.dump(b, f)
    lock_rows.append({"analysis": s, "stage": stage, "model": m, "file": path.name,
                      "sha256": hashlib.sha256(path.read_bytes()).hexdigest()})
lock = pd.DataFrame(lock_rows)
check("24 models fitted and locked (4 analyses x 3 stages x XGBoost and EuroSCORE II); no retries",
      len(lock) == 24 and len(nm.FIT_EVENTS) == 0, f"retries {len(nm.FIT_EVENTS)}")

# %% [markdown]
# ## 2. Test once on 2017-19 (paired bootstrap, separate programs)

# %%
def predict(b, X, es2):
    if isinstance(b["wrapper"], nm.EuroScoreModel):
        return b["wrapper"].predict(b["fitted"], es2)
    Z = np.asarray(b["inputs"].transform(b["imputer"].transform(X)), float)
    return b["fitted"].predict(Z) if hasattr(b["fitted"], "calibrator") else b["wrapper"].predict(b["fitted"], Z)


data, tasks, counts = {}, [], []
for s, (outcome, cutoff) in ANALYSES.items():
    val = nc.load_clean(period="validation", outcome=outcome, allow_validation=True, verbose=False)
    preds, ids, y_ref = {}, None, None
    for stage in STAGES:
        X, y, extras = nm.get_stage_data(stage, period="validation", outcome=outcome, cutoff=cutoff,
                                         allow_validation=True, df=val, verbose=False)
        if ids is None:
            ids, y_ref = extras["record_id"].to_numpy(), y.to_numpy()
        same = np.array_equal(ids, extras["record_id"].to_numpy()) and np.array_equal(y_ref, y.to_numpy())
        check(f"{s} {stage}: same 2017-19 patients and outcomes at every stage", same)
        preds[f"{stage}_M3"] = predict(bundles[(s, stage, "M3")], X, extras["ES2"])
        preds[f"{stage}_M0"] = predict(bundles[(s, stage, "M0")], X, extras["ES2"])
    counts.append({"analysis": s, "outcome": outcome, "cutoff_days": cutoff, "patients_2017_19": len(y_ref),
                   "events_2017_19": int(y_ref.sum())})
    pairs = [f"{b}_M3|{a}_M3" for a, b in COMPARISONS] + [f"{st}_M3|{st}_M0" for st in STAGES]
    np.savez(WORK / f"val_{s}.npz", y=y_ref, pairs=np.array(pairs), **{f"p_{k}": v for k, v in preds.items()})
    data[s] = {"y": y_ref, "p": preds}
    tasks.append((WORK / f"val_{s}.npz", WORK / f"scores_{s}"))
env = {**os.environ, "PYTHONPATH": CODE_DIR, "OMP_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1"}
procs = [subprocess.Popen([sys.executable, str(Path(CODE_DIR) / "nacsa_scoring.py"), "--file", str(a), str(b)],
                          env=env, stderr=subprocess.PIPE, text=True) for a, b in tasks]
for p in procs:
    if p.wait() != 0:
        raise RuntimeError(p.stderr.read())
scores = pd.concat([pd.read_csv(f"{b}.csv").assign(analysis=s) for (a, b), s in zip(tasks, ANALYSES)], ignore_index=True)
log(f"[PASS] S1-S4 scored on 2017-19 ({len(scores):,} rows)")

# %% [markdown]
# ## 3. Stage-to-stage AUC differences with the decision 140 safety rule

# %%
rows = []
for k, s in enumerate(ANALYSES):
    reps = dict(np.load(WORK / f"scores_{s}_replicates.npz"))
    y = data[s]["y"]
    for j, (a, b) in enumerate(COMPARISONS):
        d = reps[f"{b}_M3|AUC"] - reps[f"{a}_M3|AUC"]
        lo, hi = np.percentile(d, [2.5, 97.5])
        pct, edge = (2.5, lo) if abs(lo) <= abs(hi) else (97.5, hi)
        rng = np.random.default_rng([nm.BOOT_SEED, 145, k, j])
        se = np.std([np.percentile(d[rng.integers(0, len(d), len(d))], pct) for _ in range(1000)])
        rerun = abs(edge) < 2 * se
        if rerun:                                           # CI limit within Monte Carlo error of zero
            dd = np.empty(10_000)
            for r in range(10_000):
                i = nm.bootstrap_indices(len(y), r)
                dd[r] = roc_auc_score(y[i], data[s]["p"][f"{b}_M3"][i]) - roc_auc_score(y[i], data[s]["p"][f"{a}_M3"][i])
            lo, hi = np.percentile(dd, [2.5, 97.5])
        gain = roc_auc_score(y, data[s]["p"][f"{b}_M3"]) - roc_auc_score(y, data[s]["p"][f"{a}_M3"])
        rows.append({"analysis": s, "from": a, "to": b, "AUC_gain": round(gain, 4), "ci_low": round(lo, 4),
                     "ci_high": round(hi, 4), "rerun_10000": rerun, "CI_above_zero": bool(lo > 0)})
stage_gain = pd.DataFrame(rows)
log(f"Stage-to-stage comparisons: {len(stage_gain)}; reruns with 10,000 resamples: {int(stage_gain['rerun_10000'].sum())}")

# %% [markdown]
# ## 4. Save (aggregate only; models and risks stay on the machine)

# %%
scores = scores[["analysis", "what", "measure", "estimate", "ci_low", "ci_high"]]   # 'what' = stage_model or a pair
scores.round({"estimate": 4, "ci_low": 4, "ci_high": 4}).to_csv(OUT_DIR / "ST16_S1_S4_performance.csv", index=False)
stage_gain.to_csv(OUT_DIR / "ST16_S1_S4_stage_comparisons.csv", index=False)
pd.DataFrame(counts).to_csv(OUT_DIR / "ST16_S1_S4_patients_events.csv", index=False)
pd.DataFrame(settings_rows).to_csv(OUT_DIR / "S1_S4_xgboost_settings.csv", index=False)
lock.to_csv(OUT_DIR / "S1_S4_fingerprints.csv", index=False)
log(f"Checks passed: {sum(ok for _, ok in CHECKS)} of {len(CHECKS)}")
log(f"Total time: {(time.time() - t_start) / 60:.0f} min")
log(f"Run finished {datetime.now():%Y-%m-%d %H:%M}")
(OUT_DIR / "log.txt").write_text("\n".join(LOG) + "\n")

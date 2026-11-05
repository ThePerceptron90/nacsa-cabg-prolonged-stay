# %% [markdown]
# # Step 14: training-period comparison (decisions 18, 69, 81, 102)
#
# Training sets: A = 2012-16, full predictors (the locked main models of Step 11);
# B = 2012-16, all-era predictors; C = 1996-2016, all-era predictors. Models: ridge
# splines (M2) and XGBoost (M3), the three main stages. B and C get their own imputer
# (decision 81), are tuned on a 75/25 split exactly as in Step 11 (B: the 'final' split
# of the fold file; C: the same method over 1996-2016, seed 20261007), fitted on the
# whole set and locked. All are tested once on the same 2017-19 patients with 2,000
# paired bootstrap resamples; comparisons B-A, C-A, C-B reported descriptively.

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
from sklearn.model_selection import train_test_split

CODE_DIR = os.environ.get("NACSA_CODE_DIR", "/content/drive/MyDrive/nacsa/code")
sys.path.insert(0, CODE_DIR)
import nacsa_common as nc
import nacsa_modelling as nm
import nacsa_tuning as nt
for mod in (nc, nm, nt):
    importlib.reload(mod)

OUT_DIR = nc.OUTPUTS_DIR / "14_training_periods"
WORK = nc.PROJECT_DIR / "work" / "training_periods"
LOCK_A = nc.PROJECT_DIR / "work" / "final_models"
for d in (OUT_DIR, WORK):
    d.mkdir(parents=True, exist_ok=True)
STAGES = ["preop", "end_of_surgery", "early_postop"]
LOG = []


def log(msg):
    print(msg, flush=True)
    LOG.append(msg)


def prepare(stage, X, fit_rows, apply_rows, recipes=("spline", "tree")):
    imp = nm.StageImputer(stage, all_era=True).fit(X[fit_rows])
    F_fit, F_apply = imp.transform(X[fit_rows]), imp.transform(X[apply_rows])
    rec = {r: nm.ModelInputs(r).fit(F_fit) for r in recipes}
    Z = {r: (np.asarray(rec[r].transform(F_fit), float), np.asarray(rec[r].transform(F_apply), float)) for r in recipes}
    return imp, rec, Z


t_start = time.time()

# %% [markdown]
# ## 1. Sets B and C: tune, fit, lock

# %%
units, jobs = {}, []
for set_name, period in (("B", "development"), ("C", ["older", "development"])):
    df = nc.load_clean(period=period, verbose=False)
    assert df["Year"].between(1996, 2016).all(), "no 2017-19 patient may enter training"
    for stage in STAGES:
        X, y, extras = nm.get_stage_data(stage, all_era=True, df=df, verbose=False)
        yv = y.to_numpy()
        if set_name == "B":
            roles = nm.attach_folds(extras)["final"]
            tr, tu = (roles == "train").to_numpy(), (roles == "tune").to_numpy()
        else:
            idx = np.arange(len(yv))
            a, b = train_test_split(idx, test_size=0.25, stratify=yv, random_state=nm.FOLD_SEED)
            tr, tu = np.isin(idx, a), np.isin(idx, b)
        _, _, Z_in = prepare(stage, X, tr, tu)
        allp = np.ones(len(yv), bool)
        imp, rec, Z_all = prepare(stage, X, allp, allp)
        ridge = nt.tune_ridge(Z_in["spline"][0], yv[tr], Z_in["spline"][1], yv[tu])
        units[(set_name, stage)] = {"imp": imp, "rec": rec, "Z": Z_all, "y": yv, "ridge": ridge["best"]["settings"]}
        jobs.append(dict(model="M3", stage=stage, round_name="final", Z_tr=Z_in["tree"][0], y_tr=yv[tr],
                         Z_tu=Z_in["tree"][1], y_tu=yv[tu], out_dir=nc.OUTPUTS_DIR / "tuning" / f"set_{set_name}", n_jobs=1))
        log(f"set {set_name} {stage}: {len(yv):,} patients ({tr.sum():,} / {tu.sum():,} on the tuning split); "
            f"ridge lambda {ridge['best']['settings']['lambda']:.3g}")
xgb = {(j["out_dir"].name[-1], r["stage"]): r for j, r in zip(jobs, nt.run_studies(jobs, n_processes=4))}
log(f"6 XGBoost searches done ({(time.time() - t_start) / 60:.0f} min so far)")

lock_rows, bundles = [], {}
for (set_name, stage), u in units.items():
    m2 = nm.RidgeModel()
    bundles[(set_name, stage, "M2")] = {"imputer": u["imp"], "inputs": u["rec"]["spline"], "wrapper": m2,
                                        "fitted": m2.fit(u["Z"]["spline"][0], u["y"], u["ridge"])}
    r = xgb[(set_name, stage)]
    w = nm.XGBoostModel(n_jobs=4)
    bundles[(set_name, stage, "M3")] = {"imputer": u["imp"], "inputs": u["rec"]["tree"], "wrapper": w,
                                        "fitted": nm.fit_with_calibration(w, u["Z"]["tree"][0], u["y"],
                                                                          r["best"]["settings"], r["best"]["calibration"])}
for (set_name, stage, m), b in bundles.items():
    path = WORK / f"set{set_name}_{stage}_{m}.pkl"
    with open(path, "wb") as f:
        pickle.dump(b, f)
    lock_rows.append({"set": set_name, "stage": stage, "model": m, "file": path.name,
                      "sha256": hashlib.sha256(path.read_bytes()).hexdigest()})
lock = pd.DataFrame(lock_rows)
log(f"[{'PASS' if len(lock) == 12 and len(nm.FIT_EVENTS) == 0 else 'FAIL'}] 12 models (sets B, C) fitted and locked; "
    f"retries {len(nm.FIT_EVENTS)}")

# %% [markdown]
# ## 2. Test A, B and C once on 2017-19 (paired bootstrap, separate programs)

# %%
val = nc.load_clean(period="validation", allow_validation=True, verbose=False)


def predict(b, X):
    Z = np.asarray(b["inputs"].transform(b["imputer"].transform(X)), float)
    return b["fitted"].predict(Z) if hasattr(b["fitted"], "calibrator") else b["wrapper"].predict(b["fitted"], Z)


tasks = []
for stage in STAGES:
    X_full, y, _ = nm.get_stage_data(stage, period="validation", allow_validation=True, df=val, verbose=False)
    X_era, _, _ = nm.get_stage_data(stage, period="validation", allow_validation=True, all_era=True, df=val, verbose=False)
    preds = {}
    for m in ("M2", "M3"):
        with open(LOCK_A / f"{stage}_{m}.pkl", "rb") as f:
            preds[f"A_{m}"] = predict(pickle.load(f), X_full)
        for s in ("B", "C"):
            preds[f"{s}_{m}"] = predict(bundles[(s, stage, m)], X_era)
    pairs = [f"{b}_{m}|{a}_{m}" for m in ("M2", "M3") for b, a in (("B", "A"), ("C", "A"), ("C", "B"))]
    np.savez(WORK / f"val_{stage}.npz", y=y.to_numpy(), pairs=np.array(pairs), **{f"p_{k}": v for k, v in preds.items()})
    tasks.append((WORK / f"val_{stage}.npz", WORK / f"scores_{stage}"))
env = {**os.environ, "PYTHONPATH": CODE_DIR, "OMP_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1"}
procs = [subprocess.Popen([sys.executable, str(Path(CODE_DIR) / "nacsa_scoring.py"), "--file", str(a), str(b)],
                          env=env, stderr=subprocess.PIPE, text=True) for a, b in tasks]
for p in procs:
    if p.wait() != 0:
        raise RuntimeError(p.stderr.read())
scores = pd.concat([pd.read_csv(f"{b}.csv").assign(stage=s) for (a, b), s in zip(tasks, STAGES)], ignore_index=True)
scores = scores[["stage", "what", "measure", "estimate", "ci_low", "ci_high"]].round({"estimate": 4, "ci_low": 4, "ci_high": 4})
scores.to_csv(OUT_DIR / "training_period_comparison.csv", index=False)
lock.to_csv(OUT_DIR / "sets_BC_fingerprints.csv", index=False)
log(f"[PASS] A, B and C scored on 2017-19 ({len(scores):,} rows)")
log(f"Total time: {(time.time() - t_start) / 60:.0f} min")
log(f"Run finished {datetime.now():%Y-%m-%d %H:%M}")
(OUT_DIR / "log.txt").write_text("\n".join(LOG) + "\n")

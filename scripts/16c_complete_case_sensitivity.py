# %% [markdown]
# # Step 16c: sensitivity analysis S6, complete cases (decisions 74, 147)
#
# All six models (M0-M5) at the three prediction stages, developed only on 2012-16
# patients with every predictor of that stage recorded, exactly as Step 11: tuned on
# the 'final' 75/25 split (restricted to complete cases), fitted on all complete-case
# 2012-16 patients and locked. Tested once on complete-case 2017-19 patients, side by
# side with the locked main models (single imputation) on the same patients, with
# 2,000 paired bootstrap resamples (S6 model vs main model, per algorithm).

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

CODE_DIR = os.environ.get("NACSA_CODE_DIR", "/content/drive/MyDrive/nacsa/code")
sys.path.insert(0, CODE_DIR)
import nacsa_common as nc
import nacsa_modelling as nm
import nacsa_tuning as nt
for mod in (nc, nm, nt):
    importlib.reload(mod)

OUT_DIR = nc.OUTPUTS_DIR / "16c_complete_case_sensitivity"
WORK = nc.PROJECT_DIR / "work" / "sensitivity_complete_cases"   # models and risks: machine only
LOCK_MAIN = nc.PROJECT_DIR / "work" / "final_models"
TUNE_DIR = nc.OUTPUTS_DIR / "tuning" / "sensitivity_S6"
for d in (OUT_DIR, WORK):
    d.mkdir(parents=True, exist_ok=True)
STAGES = ["preop", "end_of_surgery", "early_postop"]
RECIPES = ("linear", "spline", "tree", "mlp")
ML = {"M3": ("tree", nm.XGBoostModel), "M4": ("tree", nm.RandomForestModel), "M5": ("mlp", nm.MLPModel)}
MODELS = ["M0", "M1", "M2", "M3", "M4", "M5"]
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
    imp = nm.StageImputer(stage).fit(X[fit_rows])            # complete data: nothing to fill, kept for one pipeline
    recipes = {r: nm.ModelInputs(r).fit(imp.transform(X[fit_rows])) for r in RECIPES}
    Z = {r: (np.asarray(recipes[r].transform(imp.transform(X[fit_rows])), float),
             np.asarray(recipes[r].transform(imp.transform(X[apply_rows])), float)) for r in RECIPES}
    return imp, recipes, Z


def predict_bundle(b, X, es2):
    if b["model"] == "M0":
        return b["wrapper"].predict(b["fitted"], es2)
    Z = np.asarray(b["inputs"].transform(b["imputer"].transform(X)), float)
    return b["fitted"].predict(Z) if hasattr(b["fitted"], "calibrator") else b["wrapper"].predict(b["fitted"], Z)


t_start = time.time()
dev = nc.load_clean(period="development", verbose=False)
check("only 2012-16 patients loaded (validation lock)", dev["Year"].between(2012, 2016).all(), f"{len(dev):,} rows")

# %% [markdown]
# ## 1. Complete-case inputs and searches on the final split

# %%
units, jobs, counts = {}, [], []
for stage in STAGES:
    X, y, extras = nm.get_stage_data(stage, df=dev, verbose=False)
    cc = X.notna().all(axis=1).to_numpy()
    X, y, extras = X[cc], y[cc], extras[cc]
    roles = nm.attach_folds(extras)["final"]
    tr, tu = (roles == "train").to_numpy(), (roles == "tune").to_numpy()
    allp = np.ones(len(y), bool)
    _, _, Z_in = prepare(stage, X, tr, tu)
    imp, rec, Z_all = prepare(stage, X, allp, allp)
    yv = y.to_numpy()
    units[stage] = {"y": yv, "es2": extras["ES2"].to_numpy(float), "imp": imp, "rec": rec, "Z": Z_all}
    nt.run_study("M2", stage, "final", Z_in["spline"][0], yv[tr], Z_in["spline"][1], yv[tu], TUNE_DIR)
    jobs += [dict(model=m, stage=stage, round_name="final", Z_tr=Z_in[r][0], y_tr=yv[tr], Z_tu=Z_in[r][1],
                  y_tu=yv[tu], out_dir=TUNE_DIR, n_jobs=1) for m, (r, _) in ML.items()]
    counts.append({"stage": stage, "set": "2012-16 complete cases", "patients": len(yv), "events": int(yv.sum()),
                   "share_of_all_patients": round(cc.mean(), 3)})
    log(f"{stage}: {len(yv):,} complete-case patients ({cc.mean():.1%} of 2012-16), {tr.sum():,} / {tu.sum():,} on the final split")
jobs.sort(key=lambda j: {"M5": 0, "M4": 1, "M3": 2}[j["model"]])
records = {(r["model"], r["stage"]): r for r in nt.run_studies(jobs, n_processes=4)}
log(f"9 searches done ({(time.time() - t_start) / 60:.0f} min so far)")
del jobs

# %% [markdown]
# ## 2. Final complete-case fits, saved and fingerprinted

# %%
lock_rows, bundles = [], {}
for stage, u in units.items():
    y = u["y"]
    m0 = nm.EuroScoreModel()
    fits = {"M0": (m0, m0.fit(u["es2"], y), None)}
    m1 = nm.LogisticModel()
    fits["M1"] = (m1, m1.fit(u["Z"]["linear"][0], y), "linear")
    ridge = nt.run_study("M2", stage, "final", None, None, None, None, TUNE_DIR)
    m2 = nm.RidgeModel()
    fits["M2"] = (m2, m2.fit(u["Z"]["spline"][0], y, ridge["best"]["settings"]), "spline")
    for m, (recipe, cls) in ML.items():
        r = records[(m, stage)]
        check(f"{stage} {m}: search finished without failure", r["status"] == "ok")
        fits[m] = (cls(n_jobs=4), nm.fit_with_calibration(cls(n_jobs=4), u["Z"][recipe][0], y, r["best"]["settings"],
                                                          r["best"]["calibration"]), recipe)
    for m, (wrapper, fitted, recipe) in fits.items():
        b = {"model": m, "stage": stage, "imputer": u["imp"] if recipe else None,
             "inputs": u["rec"][recipe] if recipe else None, "wrapper": wrapper, "fitted": fitted}
        bundles[(stage, m)] = b
        path = WORK / f"S6_{stage}_{m}.pkl"
        with open(path, "wb") as f:
            pickle.dump(b, f)
        lock_rows.append({"stage": stage, "model": m, "file": path.name,
                          "sha256": hashlib.sha256(path.read_bytes()).hexdigest()})
lock = pd.DataFrame(lock_rows)
check("18 complete-case models fitted and locked; no retries", len(lock) == 18 and len(nm.FIT_EVENTS) == 0,
      f"retries {len(nm.FIT_EVENTS)}")

# %% [markdown]
# ## 3. Test on complete-case 2017-19 patients, with the main models on the same patients

# %%
val = nc.load_clean(period="validation", allow_validation=True, verbose=False)
tasks = []
for stage in STAGES:
    X, y, extras = nm.get_stage_data(stage, period="validation", allow_validation=True, df=val, verbose=False)
    cc = X.notna().all(axis=1).to_numpy()
    X, y, es2 = X[cc], y[cc].to_numpy(), extras["ES2"][cc]
    counts.append({"stage": stage, "set": "2017-19 complete cases", "patients": len(y), "events": int(y.sum()),
                   "share_of_all_patients": round(cc.mean(), 3)})
    preds = {}
    for m in MODELS:
        preds[f"S6_{m}"] = predict_bundle(bundles[(stage, m)], X, es2)
        with open(LOCK_MAIN / f"{stage}_{m}.pkl", "rb") as f:
            preds[f"main_{m}"] = predict_bundle(pickle.load(f), X, es2)
    np.savez(WORK / f"val_{stage}.npz", y=y, pairs=np.array([f"S6_{m}|main_{m}" for m in MODELS]),
             **{f"p_{k}": v for k, v in preds.items()})
    tasks.append((stage, WORK / f"val_{stage}.npz", WORK / f"scores_{stage}"))
env = {**os.environ, "PYTHONPATH": CODE_DIR, "OMP_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1"}
procs = [subprocess.Popen([sys.executable, str(Path(CODE_DIR) / "nacsa_scoring.py"), "--file", str(a), str(b)],
                          env=env, stderr=subprocess.PIPE, text=True) for _, a, b in tasks]
for p in procs:
    if p.wait() != 0:
        raise RuntimeError(p.stderr.read())
scores = pd.concat([pd.read_csv(f"{b}.csv").assign(stage=s) for s, a, b in tasks], ignore_index=True)
scores = scores[["stage", "what", "measure", "estimate", "ci_low", "ci_high"]]
log(f"[PASS] S6 and main models scored on complete-case 2017-19 patients ({len(scores):,} rows)")

# %% [markdown]
# ## 4. Save (aggregate only)

# %%
scores.round({"estimate": 4, "ci_low": 4, "ci_high": 4}).to_csv(OUT_DIR / "ST16_S6_performance.csv", index=False)
pd.DataFrame(counts).to_csv(OUT_DIR / "ST16_S6_patients_events.csv", index=False)
pd.DataFrame([{"stage": st, "model": m, "calibration": records[(m, st)]["best"]["calibration"],
               "trees_or_epochs": records[(m, st)]["best"]["steps"], **records[(m, st)]["best"]["params"]}
              for st in STAGES for m in ML]).to_csv(OUT_DIR / "S6_settings_ml.csv", index=False)
lock.to_csv(OUT_DIR / "S6_fingerprints.csv", index=False)
log(f"Checks passed: {sum(ok for _, ok in CHECKS)} of {len(CHECKS)}")
log(f"Total time: {(time.time() - t_start) / 60:.0f} min")
log(f"Run finished {datetime.now():%Y-%m-%d %H:%M}")
(OUT_DIR / "log.txt").write_text("\n".join(LOG) + "\n")

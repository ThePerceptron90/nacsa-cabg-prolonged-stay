# %% [markdown]
# # Step 11: final models on all 2012-16 patients, locked (decision 142)
#
# For each stage: imputer and model inputs fitted on the 'final' inner 75% (for
# tuning) and on all 2012-16 patients (for the final fit); ridge grid and the
# 30-trial searches on the final 75/25 split (decisions 92, 95, 129-134); each model
# fitted on all 2012-16 patients with its calibration step (decision 126); M0
# recalibrated on everyone. Every final model (imputer, inputs, model, calibrator)
# is saved as one file per model and stage and its sha256 logged (the lock).
# No performance is computed on 2012-16 (decision 92). 2017-19 stays locked.

# %%
import hashlib
import importlib
import os
import pickle
import sys
import time
from datetime import datetime

import numpy as np
import pandas as pd

CODE_DIR = os.environ.get("NACSA_CODE_DIR", "/content/drive/MyDrive/nacsa/code")
sys.path.insert(0, CODE_DIR)
import nacsa_common as nc
import nacsa_modelling as nm
import nacsa_tuning as nt
for mod in (nc, nm, nt):
    importlib.reload(mod)

OUT_DIR = nc.OUTPUTS_DIR / "11_final_models"
LOCK_DIR = nc.PROJECT_DIR / "work" / "final_models"          # model files: never on GitHub
TUNE_DIR = nc.OUTPUTS_DIR / "tuning" / "primary"
for d in (OUT_DIR, LOCK_DIR):
    d.mkdir(parents=True, exist_ok=True)
RECIPES = ("linear", "spline", "tree", "mlp")
ML = {"M3": ("tree", nm.XGBoostModel), "M4": ("tree", nm.RandomForestModel), "M5": ("mlp", nm.MLPModel)}
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
    F_fit, F_apply = imp.transform(X[fit_rows]), imp.transform(X[apply_rows])
    recipes = {r: nm.ModelInputs(r).fit(F_fit) for r in RECIPES}
    Z = {r: (np.asarray(recipes[r].transform(F_fit), float), np.asarray(recipes[r].transform(F_apply), float))
         for r in RECIPES}
    return imp, recipes, Z


t_start = time.time()
dev = nc.load_clean(period="development", verbose=False)
check("only 2012-16 patients loaded (validation lock)", dev["Year"].between(2012, 2016).all(), f"{len(dev):,} rows")

# %% [markdown]
# ## 1. Inputs and searches on the final split (all stages, searches side by side)

# %%
units, jobs = {}, []
for stage in nm.STAGES:
    X, y, extras = nm.get_stage_data(stage, df=dev, verbose=False)
    roles = nm.attach_folds(extras)["final"]
    tr, tu = (roles == "train").to_numpy(), (roles == "tune").to_numpy()
    allp = np.ones(len(y), bool)
    _, _, Z_in = prepare(stage, X, tr, tu)
    imp_all, rec_all, Z_all = prepare(stage, X, allp, allp)
    yv = y.to_numpy()
    units[stage] = {"X": X, "y": yv, "es2": extras["ES2"].to_numpy(float), "imp": imp_all, "rec": rec_all, "Z": Z_all}
    nt.run_study("M2", stage, "final", Z_in["spline"][0], yv[tr], Z_in["spline"][1], yv[tu], TUNE_DIR)
    jobs += [dict(model=m, stage=stage, round_name="final", Z_tr=Z_in[r][0], y_tr=yv[tr], Z_tu=Z_in[r][1],
                  y_tu=yv[tu], out_dir=TUNE_DIR, n_jobs=1) for m, (r, _) in ML.items()]
    log(f"{stage}: inputs ready ({tr.sum():,} / {tu.sum():,} on the final split; {len(yv):,} for the final fit)")
jobs.sort(key=lambda j: {"M5": 0, "M4": 1, "M3": 2}[j["model"]])
t0 = time.time()
records = {(r["model"], r["stage"]): r for r in nt.run_studies(jobs, n_processes=4)}
log(f"12 searches finished in {(time.time() - t0) / 60:.0f} min")
del jobs

# %% [markdown]
# ## 2. Final fits on all 2012-16 patients, saved and fingerprinted (the lock)

# %%
def save(obj, name):
    path = LOCK_DIR / f"{name}.pkl"
    with open(path, "wb") as f:
        pickle.dump(obj, f)
    return hashlib.sha256(path.read_bytes()).hexdigest()


lock_rows, failed = [], []
for stage, u in units.items():
    y = u["y"]
    m0 = nm.EuroScoreModel()
    fits = {"M0": (m0, m0.fit(u["es2"], y), None)}
    m1 = nm.LogisticModel()
    fits["M1"] = (m1, m1.fit(u["Z"]["linear"][0], y), "linear")
    ridge = nt.run_study("M2", stage, "final", None, None, None, None, TUNE_DIR)   # loads the grid saved above
    m2 = nm.RidgeModel()
    fits["M2"] = (m2, m2.fit(u["Z"]["spline"][0], y, ridge["best"]["settings"]), "spline")
    for m, (recipe, cls) in ML.items():
        r = records[(m, stage)]
        if r["status"] != "ok":
            failed.append(f"{m} {stage}")
            continue
        fits[m] = (cls(n_jobs=4), nm.fit_with_calibration(cls(n_jobs=4), u["Z"][recipe][0], y, r["best"]["settings"],
                                                          r["best"]["calibration"]), recipe)
    for m, (wrapper, fitted, recipe) in fits.items():
        bundle = {"model": m, "stage": stage, "imputer": u["imp"] if recipe else None,
                  "inputs": u["rec"][recipe] if recipe else None, "wrapper": wrapper, "fitted": fitted,
                  "settings": (records[(m, stage)]["best"] if m in ML else ridge["best"] if m == "M2" else None),
                  "created": datetime.now().isoformat(timespec="seconds")}
        lock_rows.append({"stage": stage, "model": m, "file": f"{stage}_{m}.pkl", "sha256": save(bundle, f"{stage}_{m}")})
    log(f"{stage}: {len(fits)} final models saved")
lock = pd.DataFrame(lock_rows)

# %% [markdown]
# ## 3. Checks (no performance; decision 92)

# %%
check("all 24 final models fitted and saved; no failed searches", len(lock) == 24 and not failed, f"{len(lock)} files")
check("no retries or fallbacks in the final fits (decision 133)",
      len(nm.FIT_EVENTS) == 0 and all(len(u["imp"].fallbacks_) == 0 for u in units.values()))


def predict_bundle(b, X, es2):
    if b["model"] == "M0":
        return b["wrapper"].predict(b["fitted"], es2)
    Z = np.asarray(b["inputs"].transform(b["imputer"].transform(X)), float)
    return b["fitted"].predict(Z) if hasattr(b["fitted"], "calibrator") else b["wrapper"].predict(b["fitted"], Z)


# reload every locked file: fingerprint unchanged, and it predicts valid risks (first 2,000 patients)
ok_reload = True
for _, row in lock.iterrows():
    b = pickle.load(open(LOCK_DIR / row["file"], "rb"))
    u = units[row["stage"]]
    idx = np.arange(2000)
    p_file = predict_bundle(b, u["X"].iloc[idx], u["es2"][idx])
    ok_reload &= bool(np.all((p_file >= 0) & (p_file <= 1))) and hashlib.sha256(
        (LOCK_DIR / row["file"]).read_bytes()).hexdigest() == row["sha256"]
check("locked files reload, fingerprints match, risks between 0 and 1", ok_reload)
s = units["preop"]
again = nm.RidgeModel().fit(s["Z"]["spline"][0], s["y"], nt.run_study("M2", "preop", "final", None, None, None, None,
                                                                      TUNE_DIR)["best"]["settings"])
first = pickle.load(open(LOCK_DIR / "preop_M2.pkl", "rb"))["fitted"]
check("refitting gives an identical model (ridge, pre-op)", np.array_equal(again.coef_, first.coef_))

# %% [markdown]
# ## 4. Save (fingerprints and settings; no patient data)

# %%
lock.to_csv(OUT_DIR / "final_model_fingerprints.csv", index=False)
settings = pd.DataFrame([{"stage": st, "model": m, "calibration": records[(m, st)]["best"]["calibration"],
                          "trees_or_epochs": records[(m, st)]["best"]["steps"],
                          **{k: v for k, v in records[(m, st)]["best"]["params"].items()}}
                         for st in nm.STAGES for m in ML])
settings.to_csv(OUT_DIR / "final_settings_ml.csv", index=False)
log(f"Checks passed: {sum(ok for _, ok in CHECKS)} of {len(CHECKS)}")
log(f"Total time: {(time.time() - t_start) / 60:.0f} min")
log(f"Run finished {datetime.now():%Y-%m-%d %H:%M}")
(OUT_DIR / "log.txt").write_text("\n".join(LOG) + "\n")

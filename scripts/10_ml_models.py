# %% [markdown]
# # Step 10: machine-learning models (M3 XGBoost, M4 random forest, M5 neural network)
#
# Same structure as Step 9: the 5 outer rounds, all four stages, 2012-16 patients
# only (the 2017-19 data stay locked).
#
# 1. Step 9's fitted imputers and recipes are reloaded and applied; every prepared
#    table must match its Step 9 fingerprint, or the run stops (decision 138).
# 2. The 30-trial searches (decisions 95, 125, 127, 128, 129-134), side by side in
#    4 single-threaded processes; searches already saved are reused (decision 135).
# 3. Each winner refitted on the round's learning part with its calibration step
#    (decision 126; 4 threads), then the scored fold is predicted: out-of-fold
#    risks saved in $NACSA_DIR/work/oof/ (patient-level, never uploaded).
#
# Technical checks only (decision 139): no performance measure is computed here.

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

OUT_DIR = nc.OUTPUTS_DIR / "10_ml_models"
WORK_DIR = nc.PROJECT_DIR / "work"
INPUT_DIR, OOF_DIR = WORK_DIR / "inputs", WORK_DIR / "oof"
TUNE_DIR = nc.OUTPUTS_DIR / "tuning" / "primary"
FINGERPRINTS = nc.OUTPUTS_DIR / "09_baseline_models" / "input_fingerprints.csv"
OUT_DIR.mkdir(parents=True, exist_ok=True)
ROUNDS = [f"round_{k}" for k in range(1, nm.N_OUTER + 1)]
RECIPES = ("linear", "spline", "tree", "mlp")
MODELS = {"M3": ("tree", nm.XGBoostModel), "M4": ("tree", nm.RandomForestModel), "M5": ("mlp", nm.MLPModel)}
REFIT_THREADS = 4
LOG, CHECKS = [], []


def log(msg):
    print(msg, flush=True)
    LOG.append(msg)


def check(name, ok, detail=""):
    CHECKS.append((name, bool(ok)))
    log(f"[{'PASS' if ok else 'FAIL'}] {name}" + (f": {detail}" if detail else ""))
    if not ok:
        raise AssertionError(f"Check failed: {name}")


def fingerprint(a):
    return hashlib.sha256(np.ascontiguousarray(np.asarray(a, dtype=np.float64)).tobytes()).hexdigest()


t_start = time.time()
dev = nc.load_clean(period="development", verbose=False)
check("only 2012-16 patients loaded (validation lock)", dev["Year"].between(2012, 2016).all(), f"{len(dev):,} rows")
expected = pd.read_csv(FINGERPRINTS).set_index(["stage", "round", "part", "recipe"])["sha256"]

# %% [markdown]
# ## 1. Reload Step 9's prepared inputs and verify them

# %%
units, n_matched = [], 0
for stage in nm.STAGES:
    X, y, extras = nm.get_stage_data(stage, df=dev, verbose=False)
    folds = nm.attach_folds(extras)
    for rnd in ROUNDS:
        roles = folds[rnd]
        tr, tu = (roles == "train").to_numpy(), (roles == "tune").to_numpy()
        learn, score = (roles != "score").to_numpy(), (roles == "score").to_numpy()
        with open(INPUT_DIR / f"{stage}_{rnd}.pkl", "rb") as f:
            saved = pickle.load(f)
        Z = {}
        for side, imp, recipes, rows in (("inner", saved["imputer_inner"], saved["recipes_inner"], (tr, tu)),
                                         ("learning", saved["imputer_learning"], saved["recipes_learning"], (learn, score))):
            F_fit, F_apply = imp.transform(X[rows[0]]), imp.transform(X[rows[1]])
            for r in RECIPES:
                pair = (np.asarray(recipes[r].transform(F_fit), float), np.asarray(recipes[r].transform(F_apply), float))
                for use, a in zip(("fit", "apply"), pair):
                    if fingerprint(a) != expected[(stage, rnd, f"{side}_{use}", r)]:
                        raise AssertionError(f"prepared table differs from Step 9: {stage} {rnd} {side}_{use} {r}")
                    n_matched += 1
                if r in ("tree", "mlp"):                    # the only recipes M3-M5 use; others verified, then dropped
                    Z[(side, r)] = pair
        yv = y.to_numpy()
        units.append({"stage": stage, "round": rnd, "Z": Z, "y_tr": yv[tr], "y_tu": yv[tu], "y_learn": yv[learn],
                      "ids": extras["record_id"].to_numpy()[score]})
check("every prepared table identical to Step 9 (fingerprints)", n_matched == len(expected),
      f"{n_matched} of {len(expected)} tables")

# %% [markdown]
# ## 2. The searches, side by side (slowest models first)

# %%
jobs = [dict(model=m, stage=u["stage"], round_name=u["round"], Z_tr=u["Z"][("inner", MODELS[m][0])][0],
             y_tr=u["y_tr"], Z_tu=u["Z"][("inner", MODELS[m][0])][1], y_tu=u["y_tu"], out_dir=TUNE_DIR, n_jobs=1)
        for m in ("M5", "M4", "M3") for u in units]
reused = [f"{j['model']} {j['stage']} {j['round_name']}" for j in jobs
          if nt.study_path(TUNE_DIR, j["model"], j["stage"], j["round_name"]).exists()]
log(f"{len(jobs)} searches; already saved and reused: {len(reused)} ({'; '.join(reused) or 'none'})")
t0 = time.time()
records = nt.run_studies(jobs, n_processes=4)
log(f"Searches finished in {(time.time() - t0) / 3600:.2f} h")
rec = {(r["model"], r["stage"], r["round"]): r for r in records}
del jobs                                             # free the copies of the inputs

# %% [markdown]
# ## 3. Refit each winner on the learning part; predict the scored fold

# %%
oofs, rows = [], []
t0 = time.time()
for u in units:
    for m, (recipe, cls) in MODELS.items():
        r = rec[(m, u["stage"], u["round"])]
        if r["status"] != "ok":                          # decision 109 rule 5 / decision 133
            log(f"{m} {u['stage']} {u['round']}: every trial failed - model failed for this stage")
            continue
        Z_L, Z_S = u["Z"][("learning", recipe)]
        fit = nm.fit_with_calibration(cls(n_jobs=REFIT_THREADS), Z_L, u["y_learn"], r["best"]["settings"],
                                      r["best"]["calibration"])
        oofs.append(pd.DataFrame({"record_id": u["ids"], "stage": u["stage"], "round": u["round"], "model": m,
                                  "risk": fit.predict(Z_S)}))
        rows.append({"model": m, "stage": u["stage"], "round": u["round"], "trials": r["n_trials"],
                     "failed_trials": r["n_failed"], "calibration_kept": r["best"]["calibration"],
                     "trees_or_epochs_of_winner": r["best"]["steps"], "search_minutes": round(r.get("seconds_total", 0) / 60, 1),
                     "trees_at_cap": sum((t.get("steps") or 0) >= 2000 for t in r["trials"]) if m == "M3" else "",
                     "epochs_at_limit": sum((t.get("steps") or 0) >= 200 for t in r["trials"]) if m == "M5" else ""})
log(f"Refits and predictions finished in {(time.time() - t0) / 60:.1f} min")
oof = pd.concat(oofs, ignore_index=True)
oof.to_csv(OOF_DIR / "oof_ml.csv.gz", index=False)
summary = pd.DataFrame(rows)

# %% [markdown]
# ## 4. Technical checks (no performance measures; decision 139)

# %%
n_primary = len(units[0]["y_learn"]) + len(units[0]["ids"])
counts = oof.groupby(["stage", "model"])["record_id"].agg(["size", "nunique"])
check("every patient has exactly one out-of-fold risk per model and stage",
      len(counts) == 12 and (counts["size"] == n_primary).all() and (counts["nunique"] == n_primary).all(),
      f"{len(counts)} model-stage sets of {n_primary:,} patients")
check("all risks between 0 and 1, no gaps", oof["risk"].between(0, 1).all() and oof["risk"].notna().all())
baseline = pd.read_csv(OOF_DIR / "oof_baseline.csv.gz")
check("same patients and folds as the baseline models (Step 9)",
      set(map(tuple, oof[["record_id", "stage", "round"]].drop_duplicates().to_numpy()))
      == set(map(tuple, baseline[["record_id", "stage", "round"]].drop_duplicates().to_numpy())))
check("no failed trials, no retries in the refits (decision 133)",
      summary["failed_trials"].sum() == 0 and len(nm.FIT_EVENTS) == 0,
      f"failed trials {int(summary['failed_trials'].sum())}; refit retries {len(nm.FIT_EVENTS)}")
log(f"XGBoost trials at the 2,000-tree cap: {int(summary.loc[summary['model'] == 'M3', 'trees_at_cap'].sum())} "
    f"of {int(summary.loc[summary['model'] == 'M3', 'trials'].sum())}; "
    f"network trials at the 200-epoch limit: {int(summary.loc[summary['model'] == 'M5', 'epochs_at_limit'].sum())} "
    f"of {int(summary.loc[summary['model'] == 'M5', 'trials'].sum())}")
log("Calibration kept by the winners: " + "; ".join(
    f"{m}: " + ", ".join(f"{c} {n}" for c, n in summary[summary['model'] == m]['calibration_kept'].value_counts().items())
    for m in MODELS))
u = units[0]
# rerun in a separate program, exactly as the real searches ran (same thread settings)
again = nt.run_studies([dict(model="M3", stage=u["stage"], round_name=u["round"], Z_tr=u["Z"][("inner", "tree")][0],
                             y_tr=u["y_tr"], Z_tu=u["Z"][("inner", "tree")][1], y_tu=u["y_tu"],
                             out_dir=WORK_DIR / "rerun_check", n_jobs=1)], n_processes=2)[0]
import shutil
shutil.rmtree(WORK_DIR / "rerun_check", ignore_errors=True)
first = rec[("M3", u["stage"], u["round"])]
check("rerunning a search gives identical trials and winner",
      [(t["params"], t["steps"], t["log_loss"]) for t in again["trials"]]
      == [(t["params"], t["steps"], t["log_loss"]) for t in first["trials"]],
      f"M3 {u['stage']} {u['round']}")

# %% [markdown]
# ## 5. Save outputs (aggregate only; no performance)

# %%
summary.to_csv(OUT_DIR / "search_summary.csv", index=False)
log(f"Checks passed: {sum(ok for _, ok in CHECKS)} of {len(CHECKS)}")
log(f"Total time: {(time.time() - t_start) / 3600:.2f} h")
log(f"Run finished {datetime.now():%Y-%m-%d %H:%M}; optuna {nt.optuna.__version__}, numpy {np.__version__}")
(OUT_DIR / "log.txt").write_text("\n".join(LOG) + "\n")

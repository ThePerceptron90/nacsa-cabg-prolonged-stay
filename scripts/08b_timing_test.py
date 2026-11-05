# %% [markdown]
# # Step 8.6: timing test (decision 135)
#
# One real round, run exactly as the real analysis will run it, to project how long
# the full analysis takes on this machine (decision 95: if too long, any cut applies
# to every model equally). Scope (decision 135): outer round 1, the largest stage
# (complication-aware, 40 predictors), 2012-16 patients only; the 2017-19 data stay
# locked. Time grows with the number of predictors, so this is an upper bound.
#
# 1. Imputer and model inputs fitted on the inner 75%.
# 2. Ridge grid and the 30-trial searches for XGBoost, random forest and the neural
#    network, side by side in 4 processes (decision 129), each single-threaded.
#    The searches are saved in outputs/tuning/primary/ and reused by the real run
#    if nothing changes (decision 135).
# 3. Each winner refitted on the whole learning part with its calibration step
#    (decision 126), plus EuroSCORE II (M0) and logistic regression (M1).
# 4. Projection of the full run.
#
# Only time is measured. No model performance is computed or shown, so the test
# cannot influence any modelling choice.

# %%
import importlib
import json
import os
import resource
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

OUT_DIR = nc.OUTPUTS_DIR / "08b_timing_test"
OUT_DIR.mkdir(parents=True, exist_ok=True)
TUNE_DIR = nc.OUTPUTS_DIR / "tuning" / "primary"       # shared with the real run (Steps 9-11)
STAGE, ROUND, N_PROCESSES = "complication_aware", "round_1", 4
LOG, ROWS = [], []


def log(msg):
    print(msg)
    LOG.append(msg)


def step(name, seconds, note=""):
    ROWS.append({"step": name, "seconds": round(seconds, 1), "note": note})
    log(f"{name}: {seconds:,.1f} s {note}")


t_start = time.time()

# %% [markdown]
# ## 1. Data, folds, imputer and model inputs (inner 75%)

# %%
t0 = time.time()
X, y, extras = nm.get_stage_data(STAGE, verbose=False)
roles = nm.attach_folds(extras)[ROUND]
tr, tu, learn = (roles == "train").to_numpy(), (roles == "tune").to_numpy(), (roles != "score").to_numpy()
step("load data and folds", time.time() - t0,
     f"({len(y):,} patients; inner 75% {tr.sum():,}, inner 25% {tu.sum():,}, learning part {learn.sum():,})")

t0 = time.time()
imp = nm.StageImputer(STAGE).fit(X[tr])
F_tr, F_tu = imp.transform(X[tr]), imp.transform(X[tu])
step("imputer, inner 75%", time.time() - t0, f"(fallbacks: {len(imp.fallbacks_)})")

t0 = time.time()
Z = {}
for recipe in ("linear", "spline", "tree", "mlp"):
    mi = nm.ModelInputs(recipe).fit(F_tr)
    Z[recipe] = (np.asarray(mi.transform(F_tr), float), np.asarray(mi.transform(F_tu), float))
step("model inputs (4 recipes), inner 75%", time.time() - t0,
     "(" + ", ".join(f"{r} {Z[r][0].shape[1]} columns" for r in Z) + ")")

# %% [markdown]
# ## 2. The searches, side by side
#
# Longest first (random forest, neural network), so the 4 processes stay busy.

# %%
y_tr, y_tu = y.to_numpy()[tr], y.to_numpy()[tu]
jobs = [dict(model=m, stage=STAGE, round_name=ROUND, Z_tr=Z[r][0], y_tr=y_tr, Z_tu=Z[r][1], y_tu=y_tu,
             out_dir=TUNE_DIR, n_jobs=1)
        for m, r in [("M4", "tree"), ("M5", "mlp"), ("M3", "tree"), ("M2", "spline")]]
already = [j["model"] for j in jobs if nt.study_path(TUNE_DIR, j["model"], STAGE, ROUND).exists()]
log(f"Searches already saved (reused, not timed again): {already or 'none'}")
t0 = time.time()
records = {r["model"]: r for r in nt.run_studies(jobs, n_processes=N_PROCESSES)}
search_wall = time.time() - t0
step("all searches, side by side (wall clock)", search_wall, f"({N_PROCESSES} processes)")

trial_rows = []
for m, rec in records.items():
    step(f"search {m}", rec.get("seconds_total", float("nan")), f"({rec['n_trials']} trials or grid values)")
    for t in rec["trials"]:
        trial_rows.append({"model": m, "seconds": t.get("seconds"), "steps": t.get("steps"), "state": t.get("state", "COMPLETE")})
trials = pd.DataFrame(trial_rows)

# %% [markdown]
# ## 3. Final refit of each model on the whole learning part

# %%
t0 = time.time()
imp_L = nm.StageImputer(STAGE).fit(X[learn])
F_L, F_S = imp_L.transform(X[learn]), imp_L.transform(X[~learn])
step("imputer, learning part", time.time() - t0)

t0 = time.time()
ZL = {}
for recipe in ("linear", "spline", "tree", "mlp"):
    mi = nm.ModelInputs(recipe).fit(F_L)
    ZL[recipe] = (np.asarray(mi.transform(F_L), float), np.asarray(mi.transform(F_S), float))
step("model inputs (4 recipes), learning part", time.time() - t0)
y_L = y.to_numpy()[learn]

t0 = time.time()
m0 = nm.EuroScoreModel()
m0.predict(m0.fit(extras["ES2"].to_numpy()[learn], y_L), extras["ES2"].to_numpy()[~learn])
step("refit M0 EuroSCORE II", time.time() - t0)

t0 = time.time()
m1 = nm.LogisticModel()
m1.predict(m1.fit(ZL["linear"][0], y_L), ZL["linear"][1])
step("refit M1 logistic", time.time() - t0)

t0 = time.time()
m2 = nm.RidgeModel()
m2.predict(m2.fit(ZL["spline"][0], y_L, records["M2"]["best"]["settings"]), ZL["spline"][1])
step("refit M2 ridge", time.time() - t0)

for m, wrapper, recipe in [("M3", nm.XGBoostModel(n_jobs=4), "tree"), ("M4", nm.RandomForestModel(n_jobs=4), "tree"),
                           ("M5", nm.MLPModel(), "mlp")]:
    best = records[m]["best"]
    t0 = time.time()
    fit = nm.fit_with_calibration(wrapper, ZL[recipe][0], y_L, best["settings"], best["calibration"])
    fit.predict(ZL[recipe][1])
    step(f"refit {m} with calibration step", time.time() - t0,
         f"(calibration '{best['calibration']}': {'1 fit' if best['calibration'] == 'none' else '6 fits'})")
# predictions are discarded: no performance is computed in this test

# %% [markdown]
# ## 4. What happened inside the searches, and the projection

# %%
peak_mb = max(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
              resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss) / 1024
log(f"Peak memory of one process: {peak_mb:,.0f} MB")
log(f"Retries or failed fits in this process (decision 133): {len(nm.FIT_EVENTS)}")
for m in ("M3", "M4", "M5"):
    d = trials[trials["model"] == m]
    q = d["seconds"].quantile([0.25, 0.5, 0.75]).round(1).tolist()
    note = f"{m}: seconds per trial, quartiles {q[0]} / {q[1]} / {q[2]}; failed trials {int((d['state'] == 'FAIL').sum())}"
    if m == "M3":
        note += f"; trees reached: median {int(d['steps'].median())}, at the 2,000 cap in {int((d['steps'] >= 2000).sum())} of {len(d)}"
    if m == "M5":
        note += f"; epochs: median {int(d['steps'].median())}, at the 200 limit in {int((d['steps'] >= 200).sum())} of {len(d)}"
    log(note)

per_round_stage = {
    "imputers and inputs": sum(r["seconds"] for r in ROWS if r["step"].startswith(("imputer", "model inputs"))),
    "searches (wall, 4 processes)": search_wall,
    "final refits": sum(r["seconds"] for r in ROWS if r["step"].startswith("refit")),
}
unit = sum(per_round_stage.values())
n_units = 6 * 4                                   # 6 rounds (5 outer + final) x 4 stages
search_cpu = sum(rec.get("seconds_total", 0) for rec in records.values())
packed = (per_round_stage["imputers and inputs"] + per_round_stage["final refits"]) * n_units + search_cpu * n_units / N_PROCESSES
boot_min = 6 * 4 * 2 * 1.6                        # models x stages x datasets x ~1.6 min (Step 8.4 timing)
log(f"One round and stage: {unit / 60:,.1f} min (imputers/inputs {per_round_stage['imputers and inputs'] / 60:.1f}, "
    f"searches {search_wall / 60:.1f}, refits {per_round_stage['final refits'] / 60:.1f}); search CPU time {search_cpu / 60:,.1f} min")
log(f"Projection, main analysis (24 rounds x stages, upper bound from the largest stage):")
log(f"  one after another:            {unit * n_units / 3600:,.1f} h")
log(f"  searches packed on 4 cores:   {packed / 3600:,.1f} h")
log(f"  plus bootstraps (about):      {boot_min / 60:,.1f} h")
log("Not included: sensitivity outcomes and the training-period comparison (listed separately in the report)")

ts = pd.DataFrame(ROWS)
ts.to_csv(OUT_DIR / "timing_steps.csv", index=False)
trials.groupby("model")["seconds"].describe(percentiles=[0.25, 0.5, 0.75]).drop(columns=["min", "max"]).round(2) \
    .to_csv(OUT_DIR / "timing_trials.csv")
log(f"Total test time: {(time.time() - t_start) / 60:,.1f} min")
log(f"Run finished {datetime.now():%Y-%m-%d %H:%M}; optuna {nt.optuna.__version__}, numpy {np.__version__}")
(OUT_DIR / "log.txt").write_text("\n".join(LOG) + "\n")

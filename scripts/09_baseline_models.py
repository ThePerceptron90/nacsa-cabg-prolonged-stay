# %% [markdown]
# # Step 9: baseline models (M0 EuroSCORE II, M1 logistic, M2 ridge splines)
#
# Decisions 137-139. The 5 outer rounds of the nested cross-validation, 2012-16
# patients only (the 2017-19 data stay locked), all four stages:
#
# 1. In each round and stage: imputer and the four model-input recipes fitted on
#    the inner 75% (for tuning) and on the whole learning part (for the round's
#    fit). The fitted objects (no patient rows) are saved in $NACSA_DIR/work/inputs/
#    for Step 10, with a sha256 fingerprint of every prepared table (decision 138).
# 2. M2: ridge penalty chosen from the decision 123 grid on the inner split (saved
#    as a study file, decision 132), then refitted on the learning part.
#    M1: fitted on the learning part. M0: EuroSCORE II recalibrated on the learning
#    part (missing EuroSCORE II filled with the training median, decision 124).
# 3. Each patient's out-of-fold risk (from the round in which they were scored)
#    saved in $NACSA_DIR/work/oof/ (patient-level: never committed or uploaded).
#
# Technical checks only (decision 139): no performance measure is computed here.
# All six models are scored together after Step 10.

# %%
import hashlib
import importlib
import json
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

OUT_DIR = nc.OUTPUTS_DIR / "09_baseline_models"
WORK_DIR = nc.PROJECT_DIR / "work"
INPUT_DIR, OOF_DIR = WORK_DIR / "inputs", WORK_DIR / "oof"
TUNE_DIR = nc.OUTPUTS_DIR / "tuning" / "primary"
for d in (OUT_DIR, INPUT_DIR, OOF_DIR):
    d.mkdir(parents=True, exist_ok=True)
ROUNDS = [f"round_{k}" for k in range(1, nm.N_OUTER + 1)]
RECIPES = ("linear", "spline", "tree", "mlp")
LOG, CHECKS = [], []


def log(msg):
    print(msg)
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
check("only 2012-16 patients loaded (validation lock)", dev["Year"].between(2012, 2016).all(),
      f"{len(dev):,} rows, years {sorted(dev['Year'].unique())[0]}-{sorted(dev['Year'].unique())[-1]}")

# %% [markdown]
# ## 1-3. Every round and stage
#
# `prepare()` fits the imputer and recipes on one set of patients and applies them
# to another; `run_round()` does one round and stage end to end and returns the
# out-of-fold risks of the scored patients.

# %%
def prepare(stage, X, fit_rows, apply_rows):
    imp = nm.StageImputer(stage).fit(X[fit_rows])
    F_fit, F_apply = imp.transform(X[fit_rows]), imp.transform(X[apply_rows])
    recipes, Z = {}, {}
    for r in RECIPES:
        recipes[r] = nm.ModelInputs(r).fit(F_fit)
        Z[r] = (np.asarray(recipes[r].transform(F_fit), float), np.asarray(recipes[r].transform(F_apply), float))
    return imp, recipes, Z


def run_round(stage, rnd, X, y, extras, roles, save=True):
    tr, tu = (roles == "train").to_numpy(), (roles == "tune").to_numpy()
    learn, score = (roles != "score").to_numpy(), (roles == "score").to_numpy()
    y = y.to_numpy()
    imp_in, rec_in, Z_in = prepare(stage, X, tr, tu)            # inner 75% -> inner 25%
    imp_L, rec_L, Z_L = prepare(stage, X, learn, score)          # learning part -> scored fold
    prints = [{"stage": stage, "round": rnd, "part": f"{side}_{use}", "recipe": r, "rows": Z[r][i].shape[0],
               "columns": Z[r][i].shape[1], "sha256": fingerprint(Z[r][i])}
              for side, Z in (("inner", Z_in), ("learning", Z_L)) for r in RECIPES
              for i, use in ((0, "fit"), (1, "apply"))]
    if save:
        with open(INPUT_DIR / f"{stage}_{rnd}.pkl", "wb") as f:
            pickle.dump({"imputer_inner": imp_in, "recipes_inner": rec_in,
                         "imputer_learning": imp_L, "recipes_learning": rec_L}, f)
    # M2: ridge penalty on the inner split (saved study), then refit on the learning part
    study = nt.run_study("M2", stage, rnd, Z_in["spline"][0], y[tr], Z_in["spline"][1], y[tu], TUNE_DIR) if save \
        else nt.tune_ridge(Z_in["spline"][0], y[tr], Z_in["spline"][1], y[tu])
    m2 = nm.RidgeModel()
    p2 = m2.predict(m2.fit(Z_L["spline"][0], y[learn], study["best"]["settings"]), Z_L["spline"][1])
    m1 = nm.LogisticModel()
    p1 = m1.predict(m1.fit(Z_L["linear"][0], y[learn]), Z_L["linear"][1])
    m0 = nm.EuroScoreModel()
    es2 = extras["ES2"].to_numpy(float)
    p0 = m0.predict(m0.fit(es2[learn], y[learn]), es2[score])
    ids = extras["record_id"].to_numpy()[score]
    oof = pd.concat([pd.DataFrame({"record_id": ids, "stage": stage, "round": rnd, "model": m, "risk": p})
                     for m, p in (("M0", p0), ("M1", p1), ("M2", p2))], ignore_index=True)
    setting = {"stage": stage, "round": rnd, "ridge_lambda": study["best"]["settings"]["lambda"],
               "edge_rule": study["best"]["edge"], "grid_values_tried": study["n_trials"],
               "imputer_fallbacks": len(imp_in.fallbacks_) + len(imp_L.fallbacks_),
               "es2_filled_scored": m0.n_filled_}
    return oof, prints, setting


oofs, prints, settings = [], [], []
for stage in nm.STAGES:
    X, y, extras = nm.get_stage_data(stage, df=dev, verbose=False)
    folds = nm.attach_folds(extras)
    for rnd in ROUNDS:
        t0 = time.time()
        o, p, s = run_round(stage, rnd, X, y, extras, folds[rnd])
        oofs.append(o)
        prints += p
        settings.append(s)
        log(f"{stage} {rnd}: {time.time() - t0:.0f} s; ridge lambda {s['ridge_lambda']:.3g} (edge: {s['edge_rule']})")
oof = pd.concat(oofs, ignore_index=True)
oof.to_csv(OOF_DIR / "oof_baseline.csv.gz", index=False)

# %% [markdown]
# ## 4. Technical checks (no performance measures; decision 139)

# %%
n_primary = len(nm.get_stage_data("preop", df=dev, verbose=False)[1])
counts = oof.groupby(["stage", "model"])["record_id"].agg(["size", "nunique"])
check("every patient has exactly one out-of-fold risk per model and stage",
      (counts["size"] == n_primary).all() and (counts["nunique"] == n_primary).all(),
      f"{len(counts)} model-stage sets of {n_primary:,} patients")
check("all risks between 0 and 1, no gaps", oof["risk"].between(0, 1).all() and oof["risk"].notna().all())
check("every fit converged first time (decision 133); no imputer fallbacks",
      len(nm.FIT_EVENTS) == 0 and all(s["imputer_fallbacks"] == 0 for s in settings),
      f"{len(nm.FIT_EVENTS)} retries or failures")
st = pd.DataFrame(settings)
log("Ridge penalty (lambda) chosen per stage and round: " +
    "; ".join(f"{s}: {', '.join(f'{v:.3g}' for v in st.loc[st['stage'] == s, 'ridge_lambda'])}" for s in nm.STAGES))
log(f"Edge rule fired in {int((st['edge_rule'] != 'none').sum())} of {len(st)} rounds: "
    f"{st.loc[st['edge_rule'] != 'none', ['stage', 'round', 'edge_rule']].to_dict('records')}")
log(f"EuroSCORE II filled with the training median in scored patients: "
    f"{nc.suppress(int(st.groupby('round')['es2_filled_scored'].first().sum()))} over the 5 scored folds")

X, y, extras = nm.get_stage_data("preop", df=dev, verbose=False)
again, prints_again, _ = run_round("preop", "round_1", X, y, extras, nm.attach_folds(extras)["round_1"], save=False)
first = oof[(oof["stage"] == "preop") & (oof["round"] == "round_1")].reset_index(drop=True)
check("rerunning a round gives identical risks and identical prepared inputs",
      first[["record_id", "model"]].equals(again[["record_id", "model"]]) and np.array_equal(first["risk"], again["risk"])
      and [p["sha256"] for p in prints if p["stage"] == "preop" and p["round"] == "round_1"]
      == [p["sha256"] for p in prints_again])
saved = sorted(INPUT_DIR.glob("*.pkl"))
sizes = [f.stat().st_size for f in saved]
check("fitted imputers and recipes saved for Step 10; files small (no patient tables inside)",
      len(saved) == 20 and max(sizes) < 1_000_000,
      f"{len(saved)} files, largest {max(sizes) / 1024:,.0f} KB (one patient-level table would be several MB)")

# %% [markdown]
# ## 5. Save outputs (aggregate only)

# %%
# per-round counts of filled EuroSCORE II are often 1-9 and, with the total in the log, could
# be worked back (secondary disclosure): only the total is shared
st.drop(columns="es2_filled_scored").to_csv(OUT_DIR / "ridge_settings.csv", index=False)
fp = pd.DataFrame(prints)
fp.to_csv(OUT_DIR / "input_fingerprints.csv", index=False)          # full list: kept on the machine for Step 10
digest = (fp.groupby(["stage", "round"], sort=False)                # compact digest: shared on Drive
          .apply(lambda g: pd.Series({"tables": len(g), "digest_sha256":
                                      hashlib.sha256("".join(g["sha256"]).encode()).hexdigest()}))
          .reset_index())
digest.to_csv(OUT_DIR / "input_fingerprints_digest.csv", index=False)
log(f"Checks passed: {sum(ok for _, ok in CHECKS)} of {len(CHECKS)}")
log(f"Total time: {(time.time() - t_start) / 60:.1f} min")
log(f"Run finished {datetime.now():%Y-%m-%d %H:%M}; pandas {pd.__version__}, numpy {np.__version__}")
(OUT_DIR / "log.txt").write_text("\n".join(LOG) + "\n")

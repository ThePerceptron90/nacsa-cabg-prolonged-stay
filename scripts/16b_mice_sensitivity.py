# %% [markdown]
# # Step 16b: sensitivity analysis S5, multiple imputation (decisions 78, 146)
#
# Regression models M1 and M2 at all four stages. Ten (or more, decision 78 rule) proper imputations (bootstrap
# resample + random draws, outcome not used; code/nacsa_mice.py) of 2012-16 and, with
# the same imputation models, of 2017-19; models fitted on each imputed 2012-16 set,
# risks averaged over the 10 matching imputed 2017-19 sets; M2 keeps the main
# analysis's tuned lambda. M1 coefficients pooled with Rubin's rules and the fraction
# of missing information (FMI) checked against 0.21 (decision 78). Performance on
# 2017-19 compared with the main (single-imputation) M1 and M2 on the same patients
# with 2,000 paired bootstrap resamples.

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

CODE_DIR = os.environ.get("NACSA_CODE_DIR", "/content/drive/MyDrive/nacsa/code")
sys.path.insert(0, CODE_DIR)
import nacsa_common as nc
import nacsa_modelling as nm
for mod in (nc, nm):
    importlib.reload(mod)

OUT_DIR = nc.OUTPUTS_DIR / "16b_mice_sensitivity"
WORK = nc.PROJECT_DIR / "work" / "sensitivity_mice"           # patient-level risks: machine only
MAIN = nc.PROJECT_DIR / "work" / "validation"                 # Step 12 risks of the locked main models
for d in (OUT_DIR, WORK):
    d.mkdir(parents=True, exist_ok=True)
FMI_LIMIT = 0.21                                              # decision 78
LOG, CHECKS = [], []


def log(msg):
    print(msg, flush=True)
    LOG.append(msg)


def check(name, ok, detail=""):
    CHECKS.append((name, bool(ok)))
    log(f"[{'PASS' if ok else 'FAIL'}] {name}" + (f": {detail}" if detail else ""))
    if not ok:
        raise AssertionError(f"Check failed: {name}")


t_start = time.time()
env = {**os.environ, "PYTHONPATH": CODE_DIR, "OMP_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1"}

# %% [markdown]
# ## 1. Imputations per stage (10 to start; more where the decision 78 rule asks)
#
# Each imputation is saved separately. After pooling, if any coefficient's fraction of
# missing information (FMI) is above 0.21, that stage gets more imputations, to the
# number von Hippel's quadratic rule gives for its largest FMI (m = 1 + 0.5 x (FMI / 0.05)^2,
# decision 78), and is pooled again; repeated until the rule is met (at most 200).

# %%
M_START, M_MAX = 10, 200


def run_imputations(wanted):
    """wanted: {stage: m}. Runs every missing imputation file, 4 programs at a time."""
    jobs = []
    for st, m in wanted.items():
        todo = [k for k in range(m) if not (WORK / f"mice_{st}_k{k}.npz").exists()]
        for i in range(0, len(todo), 5):
            jobs.append((st, todo[i], todo[min(i + 5, len(todo)) - 1] + 1))
    running = []
    while jobs or running:
        while jobs and len(running) < 4:
            st, a, b = jobs.pop(0)
            running.append((st, subprocess.Popen([sys.executable, str(Path(CODE_DIR) / "nacsa_mice.py"), st, str(WORK),
                                                  str(a), str(b)], env=env, stdout=subprocess.DEVNULL,
                                                 stderr=subprocess.PIPE, text=True)))
        time.sleep(5)
        for st, p in list(running):
            if p.poll() is not None:
                if p.returncode != 0:
                    raise RuntimeError(f"{st}: {p.stderr.read()}")
                running.remove((st, p))


def pool(st, m):
    """Rubin's rules over imputations 0..m-1 of one stage; also the averaged risks."""
    d = [np.load(WORK / f"mice_{st}_k{k}.npz") for k in range(m)]
    assert all(np.array_equal(x["names"], d[0]["names"]) and np.array_equal(x["record_id"], d[0]["record_id"]) for x in d)
    Q, U = np.array([x["coef"] for x in d]), np.array([x["se"] for x in d]) ** 2
    qbar, ubar, b = Q.mean(0), U.mean(0), Q.var(0, ddof=1)
    t = ubar + (1 + 1 / m) * b
    table = pd.DataFrame({"stage": st, "imputations": m, "term": d[0]["names"], "estimate": qbar, "se_pooled": np.sqrt(t),
                          "odds_ratio": np.exp(qbar), "ci_low": np.exp(qbar - 1.96 * np.sqrt(t)),
                          "ci_high": np.exp(qbar + 1.96 * np.sqrt(t)), "fmi": (1 + 1 / m) * b / t})
    risks = {mod: np.mean([x[f"p_{mod}"] for x in d], axis=0) for mod in ("M1", "M2")}
    return table, risks, d[0]["record_id"], d[0]["y"]


m_used = {st: M_START for st in nm.STAGES}
while True:
    run_imputations(m_used)
    pooled = {st: pool(st, m) for st, m in m_used.items()}
    more = {}
    for st, (table, _, _, _) in pooled.items():
        fmi = table["fmi"].max()
        need = int(np.ceil(1 + 0.5 * (fmi / 0.05) ** 2))
        if fmi > FMI_LIMIT and need > m_used[st]:
            more[st] = min(need, M_MAX)
            log(f"{st}: largest FMI {fmi:.3f} with m = {m_used[st]} (term {table.loc[table['fmi'].idxmax(), 'term']}); "
                f"rule gives m = {need}; running {more[st]}")
    if not more or all(m_used[st] >= M_MAX for st in more):
        break
    m_used.update(more)
log(f"Imputations used per stage: {m_used} ({(time.time() - t_start) / 60:.0f} min)")

rubin = pd.concat([pooled[st][0] for st in nm.STAGES], ignore_index=True)
for st in nm.STAGES:
    log(f"{st}: largest FMI {pooled[st][0]['fmi'].max():.3f} with m = {m_used[st]}")
check("each stage has at least the number of imputations von Hippel's rule asks for its largest FMI (decision 78)",
      all(m_used[st] >= np.ceil(1 + 0.5 * (pooled[st][0]["fmi"].max() / 0.05) ** 2) or pooled[st][0]["fmi"].max() <= FMI_LIMIT
          for st in nm.STAGES))

# %% [markdown]
# ## 2. S5 vs main analysis on the same 2017-19 patients (paired bootstrap)

# %%
tasks = []
for s in nm.STAGES:
    _, risks, ids, y = pooled[s]
    main = pd.read_csv(MAIN / f"risks_{s}.csv.gz").set_index("record_id").loc[ids]
    check(f"{s}: every 2017-19 patient matched to the main analysis's risks", len(main) == len(y))
    np.savez(WORK / f"compare_{s}.npz", y=y, pairs=np.array(["S5_M1|main_M1", "S5_M2|main_M2"]),
             p_S5_M1=risks["M1"], p_S5_M2=risks["M2"], p_main_M1=main["M1"].to_numpy(), p_main_M2=main["M2"].to_numpy())
    tasks.append((s, WORK / f"compare_{s}.npz", WORK / f"scores_{s}"))
procs = [subprocess.Popen([sys.executable, str(Path(CODE_DIR) / "nacsa_scoring.py"), "--file", str(a), str(b)],
                          env=env, stderr=subprocess.PIPE, text=True) for _, a, b in tasks]
for p in procs:
    if p.wait() != 0:
        raise RuntimeError(p.stderr.read())
scores = pd.concat([pd.read_csv(f"{b}.csv").assign(stage=s) for s, a, b in tasks], ignore_index=True)
scores = scores[["stage", "what", "measure", "estimate", "ci_low", "ci_high"]]
log(f"[PASS] S5 and main M1/M2 scored on 2017-19 ({len(scores):,} rows)")

# %% [markdown]
# ## 3. Save (aggregate only)

# %%
scores.round({"estimate": 4, "ci_low": 4, "ci_high": 4}).to_csv(OUT_DIR / "ST16_S5_performance.csv", index=False)
rubin.round(4).to_csv(OUT_DIR / "ST16_S5_pooled_M1_coefficients.csv", index=False)
log(f"Checks passed: {sum(ok for _, ok in CHECKS)} of {len(CHECKS)}")
log(f"Total time: {(time.time() - t_start) / 60:.0f} min")
log(f"Run finished {datetime.now():%Y-%m-%d %H:%M}")
(OUT_DIR / "log.txt").write_text("\n".join(LOG) + "\n")

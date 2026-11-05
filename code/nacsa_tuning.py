"""
NACSA PLoS project | tuning harness (Step 8.5f)
=============================================================================
Runs the hyperparameter search for one model, one stage and one round (a
"study") and saves it. Agreed rules:

    decision 95   30 trials per model; score = log loss on the inner 25%
    decision 126  the calibration step is part of every trial (cross-fitted score)
    decision 134  every trial scores all three calibration options on its model and
                  keeps one (ties to 3 decimals -> the simpler); TPE samples only
                  the model's own settings
    decision 129  trials of one study run one after another with a fixed seed
                  made from the project seed + stage + round + model; different
                  studies may run side by side in separate processes
    decision 130  no pruner: every trial runs to its end
    decision 131  Optuna's TPE sampler with its default settings
    decision 132  each finished study is saved as one JSON file (seed, every
                  trial, winning settings); a rerun skips studies already saved
    decision 133  a trial that raises an error is recorded as failed and the
                  search goes on (it counts toward the 30); if every trial fails,
                  the study's status is 'failed' (decision 109 rule 5)

The harness never loads data: it receives the inner 75% / 25% arrays already
imputed and coded, so the validation lock (decision 25) is kept by the caller,
which loads 2012-16 data only.
"""
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import optuna

import nacsa_modelling as nm

optuna.logging.set_verbosity(optuna.logging.WARNING)

MODEL_INDEX = {"M2": 2, "M3": 3, "M4": 4, "M5": 5}


def study_seed(stage, round_name, model):
    """Fixed seed of one study (decision 129): the project seed combined with the
    stage, the round and the model, so every study differs but never changes."""
    stage_i = nm.STAGES.index(stage)
    round_i = 0 if round_name == "final" else int(round_name.split("_")[1])
    return int(np.random.SeedSequence([nm.MODEL_SEED, stage_i, round_i, MODEL_INDEX[model]]).generate_state(1)[0])


def suggest(trial, tuned):
    """One trial's settings, drawn from a search space in config/search_spaces.json."""
    params = {}
    for name, spec in tuned.items():
        if "choices" in spec:
            params[name] = trial.suggest_categorical(name, spec["choices"])
        elif spec.get("type") == "int":
            params[name] = trial.suggest_int(name, spec["low"], spec["high"], log=spec.get("log", False))
        else:
            params[name] = trial.suggest_float(name, spec["low"], spec["high"], log=spec.get("log", False))
    return params


def settings_from(wrapper, params, steps):
    """The settings a wrapper's fit() needs: the parameters plus, for XGBoost and the
    neural network, the number of trees or epochs reached in the trial."""
    settings = {"params": {k: v for k, v in params.items() if k != "calibration"}}
    if wrapper.steps_key:
        settings[wrapper.steps_key] = int(steps)
    return settings


def tune_optuna(wrapper, Z_tr, y_tr, Z_tu, y_tu, seed, n_trials=None):
    """One study: n_trials TPE trials, one after another. Returns the record that
    is saved as JSON (decision 132)."""
    space = nm.SEARCH_SPACES[wrapper.space]
    n_trials = n_trials or space["trials"]
    Z_tr, y_tr, Z_tu, y_tu = map(np.asarray, (Z_tr, y_tr, Z_tu, y_tu))

    def objective(trial):
        params = suggest(trial, space["tuned"])
        try:
            fitted, steps = wrapper.fit_trial(Z_tr, y_tr, Z_tu, y_tu, params)
            # decision 134: all three calibration options scored on this model; rule-7 tie rule
            method, loss, losses = nm.choose_calibration(wrapper.predict(fitted, Z_tu), y_tu)
        except Exception as e:                               # decision 133: recorded, search goes on
            trial.set_user_attr("error", f"{type(e).__name__}: {e}"[:500])
            raise
        trial.set_user_attr("steps", int(steps))
        trial.set_user_attr("calibration", method)
        trial.set_user_attr("calibration_losses", losses)
        return loss

    sampler = optuna.samplers.TPESampler(seed=seed)          # defaults otherwise (decision 131)
    study = optuna.create_study(direction="minimize", sampler=sampler)
    t0 = time.time()
    # one after another (decision 129); a failed trial still counts toward n_trials (decision 133)
    study.optimize(objective, n_trials=n_trials, n_jobs=1, catch=(Exception,))
    trials = [{"number": t.number, "params": t.params, "state": t.state.name,
               "steps": t.user_attrs.get("steps"), "log_loss": t.value,
               "calibration": t.user_attrs.get("calibration"),
               "calibration_losses": t.user_attrs.get("calibration_losses"),
               "error": t.user_attrs.get("error"), "seconds": round(t.duration.total_seconds(), 2)}
              for t in study.trials]
    done = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
    record = {"model": wrapper.name, "space": wrapper.space, "seed": seed, "n_trials": n_trials,
              "n_failed": n_trials - len(done), "n_train": int(len(y_tr)), "n_tune": int(len(y_tu)),
              "trials": trials, "seconds": round(time.time() - t0, 1), "versions": {"optuna": optuna.__version__}}
    if not done:                                             # every trial failed: rule 5
        record.update(status="failed", best=None)
        return record
    best = study.best_trial                                  # lowest score among completed trials
    record.update(status="ok", best={
        "number": best.number, "params": best.params, "steps": best.user_attrs["steps"], "log_loss": best.value,
        "settings": settings_from(wrapper, best.params, best.user_attrs["steps"]),
        "calibration": best.user_attrs["calibration"]})
    return record


def tune_ridge(Z_tr, y_tr, Z_tu, y_tu):
    """The ridge grid (decision 123) saved in the same format as an Optuna study."""
    best, grid = nm.RidgeModel().tune(Z_tr, y_tr, Z_tu, y_tu)
    trials = [{"number": i, "params": {"lambda": r["lambda"]}, "steps": 0, "log_loss": r["log_loss"]}
              for i, r in enumerate(grid.to_dict("records"))]
    return {"model": "M2", "space": "M2_ridge", "seed": None, "n_trials": len(trials), "status": "ok",
            "n_train": int(len(y_tr)), "n_tune": int(len(y_tu)),
            "best": {"params": {"lambda": best["lambda"]}, "edge": best["edge"],
                     "log_loss": float(grid.loc[grid["lambda"] == best["lambda"], "log_loss"].iloc[0]),
                     "settings": {"lambda": best["lambda"]}, "calibration": "none"},
            "trials": trials}


def study_path(out_dir, model, stage, round_name):
    return Path(out_dir) / f"{model}_{stage}_{round_name}.json"


def run_study(model, stage, round_name, Z_tr, y_tr, Z_tu, y_tu, out_dir, n_trials=None, n_jobs=1):
    """Tune one study and save it; if its file already exists, load it instead
    (a rerun after a crash only does the missing studies, decision 132)."""
    path = study_path(out_dir, model, stage, round_name)
    if path.exists():
        record = json.loads(path.read_text())
        record["loaded_from_file"] = True
        return record
    t0 = time.time()
    if model == "M2":
        record = tune_ridge(Z_tr, y_tr, Z_tu, y_tu)
    else:
        wrapper = {"M3": nm.XGBoostModel, "M4": nm.RandomForestModel, "M5": nm.MLPModel}[model](n_jobs=n_jobs)
        record = tune_optuna(wrapper, Z_tr, y_tr, Z_tu, y_tu, study_seed(stage, round_name, model), n_trials)
    record.update(stage=stage, round=round_name, n_jobs=n_jobs, seconds_total=round(time.time() - t0, 1))
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(record, indent=1))
    os.replace(tmp, path)                                   # a half-written file is never left behind
    return record


def run_studies(jobs, n_processes=4):
    """Several studies side by side (decision 129). Each study runs as a fresh,
    separate Python program, not a forked copy of this one: a forked copy of a
    process that has already used XGBoost's thread pool can freeze (seen in the
    Step 8.5f test). Each job's arrays go to a temporary file next to its output,
    deleted as soon as the study is saved. `jobs` is a list of run_study keyword
    dicts; the saved records come back in job order."""
    if n_processes == 1:
        return [run_study(**job) for job in jobs]
    env = {**os.environ, "PYTHONPATH": os.pathsep.join([str(Path(__file__).parent), os.environ.get("PYTHONPATH", "")]),
           # one thread for numpy's linear algebra in each search process, so 4 searches
           # share 4 cores without competing (and arithmetic order never varies)
           "OMP_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1", "MKL_NUM_THREADS": "1"}
    pending, running, tmp_dirs = list(enumerate(jobs)), [], []
    while pending or running:
        while pending and len(running) < n_processes:
            i, job = pending.pop(0)
            if study_path(job["out_dir"], job["model"], job["stage"], job["round_name"]).exists():
                continue                                      # already saved: nothing to run
            tmp = Path(job["out_dir"]) / f".job_{job['model']}_{job['stage']}_{job['round_name']}"
            tmp.mkdir(parents=True, exist_ok=True)
            tmp_dirs.append(tmp)
            np.savez(tmp / "arrays.npz", **{k: np.asarray(job[k]) for k in ("Z_tr", "y_tr", "Z_tu", "y_tu")})
            meta = {k: (str(v) if k == "out_dir" else v) for k, v in job.items() if k not in ("Z_tr", "y_tr", "Z_tu", "y_tu")}
            (tmp / "job.json").write_text(json.dumps(meta))
            proc = subprocess.Popen([sys.executable, str(Path(__file__)), str(tmp)], env=env,
                                    stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
            running.append((i, proc, tmp))
        time.sleep(0.2)
        for item in list(running):
            i, proc, tmp = item
            if proc.poll() is not None:
                running.remove(item)
                out, err = proc.communicate()
                shutil.rmtree(tmp, ignore_errors=True)
                if proc.returncode != 0:
                    raise RuntimeError(f"study {jobs[i]['model']} {jobs[i]['stage']} {jobs[i]['round_name']} failed:\n{err}")
    for tmp in tmp_dirs:
        shutil.rmtree(tmp, ignore_errors=True)
    return [json.loads(study_path(j["out_dir"], j["model"], j["stage"], j["round_name"]).read_text()) for j in jobs]


if __name__ == "__main__":                               # worker: one study from run_studies
    job_dir = Path(sys.argv[1])
    meta = json.loads((job_dir / "job.json").read_text())
    arrays = np.load(job_dir / "arrays.npz")
    run_study(**meta, **{k: arrays[k] for k in arrays.files})

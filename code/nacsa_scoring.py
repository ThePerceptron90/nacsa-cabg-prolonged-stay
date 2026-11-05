"""
NACSA PLoS project | scoring worker (Step 10b)
=============================================================================
Scores one stage's pooled 2012-16 out-of-fold risks for all six models with the
paired bootstrap of decision 96 (2,000 resamples, the same for every model and
stage), and saves the table and every resample's values. Run as a separate
program per stage, so the four stages are scored side by side:

    python nacsa_scoring.py <stage> <output folder>

Development data only: the 2017-19 data stay locked (decision 25).
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

import nacsa_common as nc
import nacsa_modelling as nm

MODELS = ["M0", "M1", "M2", "M3", "M4", "M5"]
OOF_DIR = nc.PROJECT_DIR / "work" / "oof"


def load_stage(stage):
    """Outcome, out-of-fold risks per model, and the round each patient was scored
    in, all aligned to the stage's patient order."""
    dev = nc.load_clean(period="development", verbose=False)
    _, y, extras = nm.get_stage_data(stage, df=dev, verbose=False)
    oof = pd.concat([pd.read_csv(OOF_DIR / "oof_baseline.csv.gz"), pd.read_csv(OOF_DIR / "oof_ml.csv.gz")])
    oof = oof[oof["stage"] == stage]
    ids = extras["record_id"].to_numpy()
    preds = {m: oof[oof["model"] == m].set_index("record_id").loc[ids, "risk"].to_numpy() for m in MODELS}
    rounds = oof[oof["model"] == "M0"].set_index("record_id").loc[ids, "round"].to_numpy()
    return y.to_numpy(), preds, rounds


def score_stage(stage, out_dir):
    y, preds, _ = load_stage(stage)
    table, reps = nm.paired_bootstrap(y, preds, nm.performance, pairs=[(m, "M2") for m in MODELS if m != "M2"])
    table.insert(0, "stage", stage)
    table.to_csv(out_dir / f"scores_{stage}.csv", index=False)
    np.savez_compressed(out_dir / f"replicates_{stage}.npz",
                        **{f"{m}|{k}": v for m in MODELS for k, v in reps[m].items()})


def score_file(npz_path, out_prefix):
    """Step 12: score risks saved in an npz file (keys 'y', 'p_<name>' and optionally
    'ref' = the model every other is compared with); writes <prefix>.csv and
    <prefix>_replicates.npz."""
    d = np.load(npz_path, allow_pickle=False)
    preds = {k[2:]: d[k] for k in d.files if k.startswith("p_")}
    ref = str(d["ref"]) if "ref" in d.files else None
    pairs = [(m, ref) for m in preds if ref and m != ref]
    if "pairs" in d.files:                                  # explicit "a|b" pairs (Step 14)
        pairs = [tuple(str(x).split("|")) for x in d["pairs"]]
    table, reps = nm.paired_bootstrap(d["y"], preds, nm.performance, pairs=pairs)
    table.to_csv(f"{out_prefix}.csv", index=False)
    np.savez_compressed(f"{out_prefix}_replicates.npz", **{f"{m}|{k}": v for m in preds for k, v in reps[m].items()})


if __name__ == "__main__":
    if sys.argv[1] == "--file":
        score_file(sys.argv[2], sys.argv[3])
    else:
        score_stage(sys.argv[1], Path(sys.argv[2]))

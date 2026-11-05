"""
NACSA PLoS project | S5 multiple-imputation worker (Step 16b; decisions 78, 146)
=============================================================================
For one stage and a range of imputation numbers k: each imputation k is a proper
draw (MiceImputer: bootstrap + random draws, outcome not used, seed fixed by k) of
the 2012-16 patients and, with the same imputation models, of the 2017-19
patients. On the imputed 2012-16 set, M1 (plain logistic) and M2 (ridge splines
with the main analysis's tuned lambda) are fitted and applied to the matching
imputed 2017-19 set. M1 coefficients and standard errors (unpenalised GLM on the
same design, per unit of each predictor's coding) are kept for Rubin's rules.
Each imputation is saved as its own file, so more imputations can be added later
without redoing earlier ones (decision 78 rule). Run as a separate program:

    python nacsa_mice.py <stage> <work folder> <first k> <last k + 1>

Writes <work>/mice_<stage>_k<k>.npz (patient-level risks: machine only).
"""
import pickle
import sys
from pathlib import Path

import numpy as np
import statsmodels.api as sm

import nacsa_common as nc
import nacsa_modelling as nm

LOCK = nc.PROJECT_DIR / "work" / "final_models"


def run(stage, work, k_from, k_to):
    dev = nc.load_clean(period="development", verbose=False)
    val = nc.load_clean(period="validation", allow_validation=True, verbose=False)
    Xd, yd, _ = nm.get_stage_data(stage, df=dev, verbose=False)
    Xv, yv, ev = nm.get_stage_data(stage, period="validation", allow_validation=True, df=val, verbose=False)
    yd = yd.to_numpy()
    lam = pickle.load(open(LOCK / f"{stage}_M2.pkl", "rb"))["settings"]["settings"]
    for k in range(k_from, k_to):
        imp = nm.MiceImputer(stage, seed=[nm.MODEL_SEED, 146, k]).fit(Xd)
        Fd, Fv = imp.transform(Xd), imp.transform(Xv)
        assert not Fd.isna().any().any() and not Fv.isna().any().any()
        lin = nm.ModelInputs("linear").fit(Fd)
        Zd, Zv = np.asarray(lin.transform(Fd), float), np.asarray(lin.transform(Fv), float)
        m1 = nm.LogisticModel()
        p1 = m1.predict(m1.fit(Zd, yd), Zv)
        spl = nm.ModelInputs("spline").fit(Fd)
        m2 = nm.RidgeModel()
        p2 = m2.predict(m2.fit(np.asarray(spl.transform(Fd), float), yd, lam), np.asarray(spl.transform(Fv), float))
        glm = sm.GLM(yd, sm.add_constant(Zd), family=sm.families.Binomial()).fit()
        sd = np.r_[1.0, lin.sd_]                            # per unit of the original coding
        np.savez(Path(work) / f"mice_{stage}_k{k}.npz", record_id=ev["record_id"].to_numpy(), y=yv.to_numpy(),
                 p_M1=p1, p_M2=p2, coef=(glm.params / sd)[1:], se=(glm.bse / sd)[1:],
                 names=np.array(lin.get_feature_names_out(), dtype=str))
        print(f"{stage}: imputation {k} done", flush=True)


if __name__ == "__main__":
    run(sys.argv[1], sys.argv[2], int(sys.argv[3]), int(sys.argv[4]))

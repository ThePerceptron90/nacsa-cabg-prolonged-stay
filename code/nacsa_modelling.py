"""
NACSA PLoS project | shared modelling tools (Step 8 onwards)
=============================================================================
Everything the modelling steps share lives here, so every model gets the same
patients, the same folds, the same preprocessing and the same tested metrics:

    8.1  get_stage_data()   patients, predictors and outcome for one stage
    8.2  make_folds(), load_folds(), attach_folds()   fixed cross-validation folds
    8.3  StageImputer   stage-ordered deterministic chained-equation imputer
         ModelInputs    coding, splines and scaling per model family (recipes)
    8.4  calibration_curve(), calibration_errors()   lowess curve, ICI/E50/E90
         performance()   AUC, PR-AUC, CITL, slope, Brier, net benefit, top 10%/20% (+ ICI/E50/E90)
         paired_bootstrap()   shared-resample percentile CIs and differences
    8.5  EuroScoreModel (M0), LogisticModel (M1), RidgeModel (M2): tune / fit / predict
         XGBoostModel (M3), RandomForestModel (M4), MLPModel (M5); search spaces from
         config/search_spaces.json; the Optuna search itself is in nacsa_tuning.py
         Calibrator, crossfit_calibration_loss(), fit_with_calibration()   calibration step

(The remaining metrics and the 8.5 model wrappers are added here as they are
agreed.)

nacsa_common.py (Steps 1-6) is deliberately left untouched, so those steps
still rerun byte for byte (decision 111, Step 8 choice 2A).
"""
import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy.special import expit, logit
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.exceptions import ConvergenceWarning
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import average_precision_score, log_loss, roc_auc_score
from sklearn.model_selection import StratifiedKFold, train_test_split
from statsmodels.nonparametric.smoothers_lowess import lowess

import nacsa_common as nc

# =============================================================================
# 8.1 STAGE DATA
# -----------------------------------------------------------------------------
# One function answers "give me the data for stage X". It returns three parts:
#
#   X       the predictors of that stage only (26 / 34 / 36 / 40 columns),
#           Step 5 recoding applied, codes still raw, missing values still
#           missing (imputation and coding happen inside the pipeline, 8.3)
#   y       the outcome, 1 = prolonged stay, 0 = not
#   extras  columns needed later but NEVER used as predictors: EuroSCORE II
#           (M0 only), year, sex / age band / urgency (fairness analysis) and
#           the record id (to match folds and predictions)
#
# Keeping extras apart from X means EuroSCORE II, year or the outcome can never
# slip into a model by accident.
#
# Outcome cut-offs (Step 8 choice 1A): the primary outcome is a post-operative
# stay of more than 10 whole days among patients discharged alive. Sensitivity
# analyses S3 and S4 use more than 7 and more than 14 days, built exactly the
# same way (survivors only, whole days, as in Step 3).
# =============================================================================
STAGES = list(nc.PREDICTOR_SETS)                 # preop, end_of_surgery, early_postop, complication_aware
EXTRA_COLUMNS = ["record_id", "Year", "ES2", "Gender", "age_band", "Urgency"]
ALLOWED_CUTOFFS = (7, 10, 14)                     # 10 primary; 7 and 14 for S3 and S4 (decision 99)


def get_stage_data(stage, period="development", outcome=nc.PRIMARY_OUTCOME, cutoff=nc.LOS_CUTOFF,
                   allow_validation=False, all_era=False, df=None, verbose=True):
    """Return (X, y, extras) for one stage.

    stage             "preop", "end_of_surgery", "early_postop" or "complication_aware"
    period, outcome,
    allow_validation  passed to nc.load_clean (the validation lock applies)
    cutoff            7, 10 (primary) or 14 days; only with outcome="survivors"
    all_era           True for the all-era predictor sets (training-period comparison)
    df                an already loaded table (from nc.load_clean) to avoid re-reading
                      the file; it must have been loaded with the same period/outcome
    """
    if stage not in STAGES:
        raise ValueError(f"Unknown stage '{stage}'; choose from {STAGES}")
    if cutoff not in ALLOWED_CUTOFFS:
        raise ValueError(f"cutoff must be one of {ALLOWED_CUTOFFS}")
    if cutoff != nc.LOS_CUTOFF and outcome != "survivors":
        raise ValueError("Other cut-offs are defined for the survivors outcome only (S3, S4)")

    if df is None:
        df = nc.load_clean(period=period, outcome=outcome, allow_validation=allow_validation,
                           verbose=verbose)
    df = nc.prepare_for_model(df)

    # Outcome: y from load_clean is the 10-day version; rebuild for 7 or 14 days
    # exactly as Step 3 did (survivors only, whole days). Every survivor kept by
    # load_clean has a known whole-day stay, so no patient is gained or lost.
    if cutoff == nc.LOS_CUTOFF:
        y = df["y"].astype(int)
    else:
        assert df["los_days"].notna().all() and (df["Mortality"] != 1).all()
        y = (df["los_days"] > cutoff).astype(int)
    y = y.rename("y")

    cols = nc.predictors(stage, all_era=all_era)
    X = df[cols].copy()
    extras = df[EXTRA_COLUMNS].copy()

    # Guard: nothing that is not an agreed predictor may be in X
    forbidden = {"y", "ES2", "Year", "period", "record_id", "los_days", "PostLOS", "Mortality",
                 "age_band"} | set(nc.OUTCOMES.values())
    leaked = forbidden & set(X.columns)
    assert not leaked, f"non-predictor columns in X: {leaked}"
    return X, y, extras


# =============================================================================
# 8.2 FIXED CROSS-VALIDATION FOLDS (decision 112)
# -----------------------------------------------------------------------------
# Made once, saved, and used by every model, stage and outcome, so that model
# comparisons are like-for-like (paired) and not partly luck of the split.
#
#   Who      every 2012-16 patient usable in any outcome definition: survivors
#            with a known stay (primary, S3, S4), in-hospital deaths and
#            patients whose discharge status was not recorded (S1 composite,
#            S2 all patients). Each analysis takes its subset.
#   Balance  four groups: survivor with long stay (>10 days), survivor with
#            short stay, died in hospital, discharge status not recorded. So
#            every fold has ~15.9% events in the primary analysis and similar
#            shares of the other groups.
#   Outer    5 folds (decision 92); each patient is scored once, in its fold.
#   Inner    within each outer round, the 4 learning folds are split 75/25
#            (stratified the same way): "train" (75%) fits the model, "tune"
#            (25%) chooses settings and fits the calibration step; the scoring
#            fold is never used for either.
#   Final    one 75/25 split of all patients, to tune the final models that
#            are tested on 2017-19.
#   Seed     FOLD_SEED for the outer folds; FOLD_SEED + k for the inner split
#            of round k (k = 1..5); FOLD_SEED + 6 for the final split. One set
#            of folds only (no repeated cross-validation, decision 92).
#
# The file lists record ids, so it is patient-level: it stays in processed/
# (never committed, never on Drive). Anyone can rebuild it exactly from the
# seed; its sha256 is logged.
# =============================================================================
FOLD_SEED = 20261007
N_OUTER = 5
INNER_TUNE_SHARE = 0.25
FOLD_FILE = nc.PROJECT_DIR / "processed" / "folds_development.csv"
STRATA = {0: "survivor, short stay", 1: "survivor, long stay", 2: "died in hospital",
          3: "discharge status not recorded"}


def fold_population(verbose=True):
    """All 2012-16 patients who belong to at least one outcome definition, with their stratum.

    That is: survivors with a known stay (primary, S3, S4), in-hospital deaths
    (S1, S2) and patients whose discharge status was not recorded but whose
    stay is known (S2; and S1 when the stay was over 10 days).
    """
    df = nc.load_clean(period="development", outcome=None, verbose=verbose)
    known = df[list(nc.OUTCOMES.values())].notna().any(axis=1)
    df = df[known]
    stratum = np.select([df["Mortality"] == 1, df["Mortality"].isna()], [2, 3],
                        default=df["outcome_survivors"].fillna(-1)).astype(int)
    assert (stratum >= 0).all(), "a recorded survivor without a survivor outcome"
    return pd.DataFrame({"record_id": df["record_id"].to_numpy(), "stratum": stratum})


def make_folds(pop=None, seed=FOLD_SEED):
    """Assign outer folds and inner/final train-tune roles. Returns one row per patient.

    Columns: record_id, stratum, outer_fold (1-5), round_1 .. round_5 and final,
    where round_k is "score" (the outer fold k), "train" or "tune".
    """
    if pop is None:
        pop = fold_population(verbose=False)
    pop = pop.sort_values("record_id").reset_index(drop=True)      # order-independent result
    assert pop["record_id"].is_unique
    folds = pop.copy()
    folds["outer_fold"] = 0
    skf = StratifiedKFold(n_splits=N_OUTER, shuffle=True, random_state=seed)
    for k, (_, score_idx) in enumerate(skf.split(pop, pop["stratum"]), start=1):
        folds.loc[score_idx, "outer_fold"] = k

    def inner_roles(mask, s):
        idx = np.flatnonzero(mask)
        tr, tu = train_test_split(idx, test_size=INNER_TUNE_SHARE, random_state=s,
                                  stratify=folds["stratum"].to_numpy()[idx])
        roles = np.full(len(folds), "", dtype=object)
        roles[tr], roles[tu] = "train", "tune"
        return roles

    for k in range(1, N_OUTER + 1):
        roles = inner_roles(folds["outer_fold"].to_numpy() != k, seed + k)
        roles[folds["outer_fold"].to_numpy() == k] = "score"
        folds[f"round_{k}"] = roles
    folds["final"] = inner_roles(np.ones(len(folds), bool), seed + N_OUTER + 1)
    return folds


def save_folds(folds, path=FOLD_FILE):
    path.parent.mkdir(parents=True, exist_ok=True)
    folds.to_csv(path, index=False)
    return path


def load_folds(path=FOLD_FILE):
    """Read the saved folds; stop if missing (run Step 8 first)."""
    if not path.exists():
        raise FileNotFoundError(f"{path} not found: run scripts/08_infrastructure.py (part 8.2) first")
    return pd.read_csv(path)


def attach_folds(extras, folds=None):
    """Fold columns for the patients of one analysis, aligned to its rows (via record_id)."""
    if folds is None:
        folds = load_folds()
    out = extras[["record_id"]].merge(folds, on="record_id", how="left", validate="one_to_one")
    if out["outer_fold"].isna().any():
        raise ValueError("some patients have no fold: were the folds made on a different population?")
    out.index = extras.index
    return out.drop(columns="record_id")


# =============================================================================
# 8.3a STAGE-ORDERED DETERMINISTIC IMPUTER (decisions 77, 81-85, 113, 114)
# -----------------------------------------------------------------------------
# Fills missing predictor values by chained equations, one completed dataset:
#
#   - each incomplete predictor is predicted from the others with a model
#     matched to its type (decision 113): logistic regression for yes/no,
#     multinomial logistic regression for categories with more than two
#     levels (ordered ones too, A1), linear regression for numbers
#     (decision 114; creatinine on the log scale, filled numbers capped at
#     the range seen in the training patients, no rounding);
#   - deterministic (decision 82): a category gap gets its most likely
#     category, a number gets its predicted value;
#   - the outcome is never used (decision 81): fit() ignores y;
#   - stage-ordered (decision 85): predictors are filled block by block
#     (pre-op, then intra-op, early post-op, late post-op), each block only
#     from its own and earlier blocks, so pre-op values are identical in every
#     stage's model;
#   - fitted on training patients only and applied unchanged to new patients
#     (decision 84); it stores equations (coefficients), medians/most common
#     values and ranges, never patient rows (deployable, decision 77).
#
# The loop: start from the median / most common value, then repeatedly
# re-predict each incomplete predictor of the block from the current values
# of the others, until the filled values stop changing (at most 10 rounds,
# decision 114). The models of the last round are kept: once the loop has
# settled they are the models of the completed training data. New patients
# are filled by the same loop using those kept models. Predictors with no gap
# in the training patients get no model (in this extract the same predictors
# are complete in 2012-16 and 2017-19, Step 6b); a gap in one of them at
# prediction time stops with an error rather than being filled silently.
# =============================================================================
LOG_SCALE = {"Creatinine"}                       # imputed on the log scale (decision 114, 1A)
MAX_ROUNDS = 10                                  # decision 114
NUMERIC_TOL = 1e-3                               # stop when no filled number moves by more than
                                                 # 0.001 standard deviations and no category changes


def stage_blocks(stage, all_era=False):
    """The predictor blocks of a stage, in time order (decision 85); all_era=True uses
    the all-era sets of training sets B and C (decisions 18, 69, 81)."""
    sets = nc.ALL_ERA_SETS if all_era else nc.PREDICTOR_SETS
    order = [s for s in ["preop", "end_of_surgery", "early_postop", "complication_aware"] if s in sets]
    blocks, seen = [], []
    for s in order[:order.index(stage) + 1]:
        new = [v for v in sets[s] if v not in seen]
        blocks.append(new)
        seen += new
    return blocks


def _is_numeric(var):
    return nc.VARIABLES[var]["type"] == "continuous"


class StageImputer(BaseEstimator, TransformerMixin):
    """Deterministic, stage-ordered chained-equation imputer for one stage's X."""

    def __init__(self, stage="preop", max_rounds=MAX_ROUNDS, tol=NUMERIC_TOL, all_era=False):
        self.stage = stage
        self.max_rounds = max_rounds
        self.tol = tol
        self.all_era = all_era

    # ---- design matrix used as input to every imputation model ------------
    def _design(self, D, cols):
        """Inputs for an imputation model: numbers standardised (creatinine logged),
        categories as indicator columns (first level dropped), fixed by the training data."""
        parts = []
        for c in cols:
            if _is_numeric(c):
                x = D[c].astype(float)
                if c in LOG_SCALE:
                    x = np.log(x)
                parts.append(((x - self.centre_[c]) / self.scale_[c]).to_numpy()[:, None])
            else:
                lv = self.levels_[c]
                parts.append(np.column_stack([(D[c] == l).to_numpy(float) for l in lv[1:]])
                             if len(lv) > 1 else np.zeros((len(D), 0)))
        return np.hstack(parts) if parts else np.zeros((len(D), 0))

    def _target(self, D, v):
        x = D[v].astype(float)
        return np.log(x) if v in LOG_SCALE else x

    def _predict(self, model, v, Z):
        if _is_numeric(v):
            p = model.predict(Z)
            if v in LOG_SCALE:
                p = np.exp(p)
            return np.clip(p, self.low_[v], self.high_[v])       # decision 114, 2A
        return model.predict(Z)                                   # most likely category

    def _fit_model(self, D, v, cols, rows):
        Z, t = self._design(D.loc[rows], cols), D.loc[rows, v]
        if _is_numeric(v):
            return LinearRegression().fit(Z, self._target(D.loc[rows], v))
        if t.nunique() < 2:                                       # only one level seen: constant
            return _Constant(t.iloc[0])
        # newton-cholesky: same (convex) solution as the default solver, far fewer
        # iterations when patients greatly outnumber inputs; scikit-learn's default
        # mild penalty (C = 1) and tolerance. Retries per decision 133; if every
        # attempt fails, rule 8: this variable is filled with its training most
        # common value (logged in fallbacks_ and in FIT_EVENTS).
        try:
            return fit_logistic(1.0, Z, t.to_numpy(), where=f"imputer {self.stage}: {v}", tol=1e-4)
        except FitFailed:
            if v not in self.fallbacks_:                      # listed once, whatever the round
                self.fallbacks_.append(v)
            return _Constant(self.start_[v])

    def _changed(self, v, old, new):
        if _is_numeric(v):
            return np.max(np.abs(self._target(pd.DataFrame({v: new}), v).to_numpy()
                                 - self._target(pd.DataFrame({v: old}), v).to_numpy())
                          / self.scale_[v], initial=0) > self.tol
        return bool((np.asarray(old) != np.asarray(new)).any())

    # ---- fitting ------------------------------------------------------------
    def fit(self, X, y=None):
        """Fit on training patients. y is accepted (pipeline convention) and ignored."""
        X = X.copy()
        self.blocks_ = stage_blocks(self.stage, getattr(self, "all_era", False))
        self.columns_ = [c for b in self.blocks_ for c in b]
        assert list(X.columns) == self.columns_, "columns differ from the stage's predictor list"

        self.start_, self.centre_, self.scale_, self.low_, self.high_, self.levels_ = {}, {}, {}, {}, {}, {}
        for c in self.columns_:
            obs = X[c].dropna()
            if _is_numeric(c):
                t = self._target(obs.to_frame(), c)
                self.start_[c] = float(obs.median())
                self.centre_[c], self.scale_[c] = float(t.mean()), float(t.std()) or 1.0
                self.low_[c], self.high_[c] = float(obs.min()), float(obs.max())
            else:
                self.start_[c] = obs.mode().iloc[0]
                self.levels_[c] = sorted(obs.unique().tolist(), key=str)

        D = X.fillna(self.start_)
        miss = X.isna()
        self.models_, self.rounds_used_, self.converged_, self.fallbacks_ = {}, [], [], []
        for b, block in enumerate(self.blocks_):
            inputs = [c for blk in self.blocks_[:b + 1] for c in blk]
            todo = sorted([v for v in block if miss[v].any()], key=lambda v: miss[v].mean())
            rounds, converged = 0, True
            for rounds in range(1, self.max_rounds + 1):
                changed = False
                for v in todo:
                    cols = [c for c in inputs if c != v]
                    m = self._fit_model(D, v, cols, ~miss[v])
                    new = self._predict(m, v, self._design(D.loc[miss[v]], cols))
                    changed |= self._changed(v, D.loc[miss[v], v], new)
                    D.loc[miss[v], v] = new
                    self.models_[v] = (cols, m)              # last round's model is kept
                if not changed:
                    break
            else:
                converged = not todo
            self.rounds_used_.append(rounds if todo else 0)
            self.converged_.append(converged)
        return self

    # ---- applying -------------------------------------------------------------
    def transform(self, X):
        X = X[self.columns_].copy()
        D = X.fillna(self.start_)
        miss = X.isna()
        no_model = [v for v in self.columns_ if miss[v].any() and v not in self.models_]
        if no_model:
            raise ValueError(f"gaps in predictors that were complete in the training patients: {no_model}")
        for block in self.blocks_:
            todo = [v for v in block if miss[v].any()]
            for _ in range(self.max_rounds):
                changed = False
                for v in todo:
                    cols, m = self.models_[v]
                    new = self._predict(m, v, self._design(D.loc[miss[v]], cols))
                    changed |= self._changed(v, D.loc[miss[v], v], new)
                    D.loc[miss[v], v] = new
                if not changed:
                    break
        return D


class MiceImputer(StageImputer):
    """S5 (decisions 78, 146): one 'proper' multiple-imputation draw with the same
    imputation models as StageImputer, as in mice's norm.boot / logreg.boot:
    the models are fitted on a bootstrap resample of the training patients
    (parameter uncertainty), and gaps get random draws instead of best guesses
    (numbers: prediction plus normal noise with the model's residual SD, on the
    log scale for creatinine, then capped as in the main imputer; categories: a
    level drawn from the predicted probabilities). With random draws the loop
    cannot settle, so it runs a fixed 5 rounds (mice default maxit = 5). The
    outcome is never used. One instance = one imputation; seed fixes it."""

    def __init__(self, stage="preop", seed=0, rounds=5, all_era=False):
        super().__init__(stage=stage, max_rounds=rounds, tol=-1.0, all_era=all_era)   # tol < 0: never 'settled'
        self.seed = seed
        self.rounds = rounds

    def _fit_model(self, D, v, cols, rows):
        m = super()._fit_model(D, v, cols, rows)
        if _is_numeric(v):
            Z = self._design(D.loc[rows], cols)
            resid = self._target(D.loc[rows], v).to_numpy() - m.predict(Z)
            self.sigma_[v] = float(np.sqrt(np.sum(resid ** 2) / max(len(resid) - Z.shape[1] - 1, 1)))
        return m

    def _predict(self, model, v, Z):
        if _is_numeric(v):
            p = model.predict(Z) + self.rng_.normal(0.0, self.sigma_[v], len(Z))
            if v in LOG_SCALE:
                p = np.exp(p)
            return np.clip(p, self.low_[v], self.high_[v])
        if isinstance(model, _Constant):
            return model.predict(Z)
        prob = model.predict_proba(Z)
        pick = (prob.cumsum(axis=1) < self.rng_.random(len(Z))[:, None]).sum(axis=1)
        return model.classes_[np.minimum(pick, prob.shape[1] - 1)]

    def fit(self, X, y=None):
        self.rng_ = np.random.default_rng(self.seed)
        self.sigma_ = {}
        boot = X.iloc[self.rng_.integers(0, len(X), len(X))].reset_index(drop=True)
        return super().fit(boot)


class _Constant:
    """A 'model' that always returns one training value: the one level seen in
    training, or the most common value when the model cannot be fitted (rule 8)."""

    def __init__(self, value):
        self.value = value

    def predict(self, Z):
        # the value's own type (e.g. float for numerically coded categories), so it
        # can be written back into the column; dtype=object was refused by pandas 3
        return np.full(len(Z), self.value)


# =============================================================================
# 8.3b MODEL INPUTS AFTER IMPUTATION (decisions 29, 89, 91, 101, 102, 115)
# -----------------------------------------------------------------------------
# One transformer, four recipes, all fitted on training patients only:
#
#   "linear"  M1   numbers as one column; yes/no 0/1; categories one column per
#                  level except the clinical reference level; all standardised
#   "spline"  M2   as "linear" but each number becomes a 4-knot restricted cubic
#                  spline (3 columns; knots at the 5/35/65/95th percentiles;
#                  duplicate knots dropped); all standardised
#   "tree"    M3, M4  numbers unchanged; yes/no 0/1; ordered categories as their
#                  ordered codes; unordered categories one-hot (one column per
#                  level); no scaling
#   "mlp"     M5   the "tree" columns, then all standardised (decision 115, 2A)
#
# No interaction terms (decision 89). A category level never seen in training
# stops with an error instead of becoming a silent all-zero row.
# =============================================================================
RECIPES = ("linear", "spline", "tree", "mlp")
KNOT_QUANTILES = (0.05, 0.35, 0.65, 0.95)          # Harrell's default for 4 knots (decision 101)


def rcs_basis(x, knots):
    """Restricted cubic spline basis (Harrell; Hmisc rcspline.eval, norm = 2).

    Returns len(knots) - 1 columns: x itself and len(knots) - 2 non-linear
    terms. The fitted curve is cubic between knots, smooth at the knots, and a
    straight line below the first and above the last knot.
    """
    x = np.asarray(x, dtype=float)
    t = np.asarray(knots, dtype=float)
    k = len(t)
    if k < 3:
        return x[:, None]
    pos = lambda u: np.maximum(u, 0) ** 3
    norm = (t[-1] - t[0]) ** 2
    cols = [x]
    for j in range(k - 2):
        cols.append((pos(x - t[j])
                     - pos(x - t[-2]) * (t[-1] - t[j]) / (t[-1] - t[-2])
                     + pos(x - t[-1]) * (t[-2] - t[j]) / (t[-1] - t[-2])) / norm)
    return np.column_stack(cols)


class ModelInputs(BaseEstimator, TransformerMixin):
    """Turn an imputed stage table into the columns one model family uses."""

    def __init__(self, recipe="spline"):
        self.recipe = recipe

    def fit(self, X, y=None):
        """Fit on imputed training patients. y is accepted (pipeline convention) and ignored."""
        if self.recipe not in RECIPES:
            raise ValueError(f"recipe must be one of {RECIPES}")
        if X.isna().any().any():
            raise ValueError("ModelInputs expects imputed data (run StageImputer first)")
        self.columns_ = list(X.columns)
        self.levels_, self.knots_, self.knots_dropped_ = {}, {}, {}
        for c in self.columns_:
            t = nc.VARIABLES[c]["type"]
            if t in ("ordinal", "nominal"):
                seen = sorted(X[c].unique().tolist(), key=str)
                self.levels_[c] = seen
            elif t == "continuous" and self.recipe == "spline":
                q = np.quantile(X[c].to_numpy(float), KNOT_QUANTILES)
                uq = np.unique(q)
                self.knots_[c] = uq
                self.knots_dropped_[c] = len(q) - len(uq)
        Z = self._encode(X)
        self.feature_names_ = list(Z.columns)
        if self.recipe in ("linear", "spline", "mlp"):
            self.mean_ = Z.mean().to_numpy()
            sd = Z.std(ddof=0).to_numpy()
            self.constant_columns_ = [n for n, s in zip(self.feature_names_, sd) if s == 0]
            self.sd_ = np.where(sd == 0, 1.0, sd)
        return self

    def _encode(self, X):
        parts = {}
        for c in self.columns_:
            meta = nc.VARIABLES[c]
            t = meta["type"]
            x = X[c]
            if t in ("ordinal", "nominal"):
                unseen = set(x.unique().tolist()) - set(self.levels_[c])
                if unseen:
                    raise ValueError(f"{c}: level(s) {unseen} not seen in training")
            if t == "continuous":
                if self.recipe == "spline":
                    B = rcs_basis(x, self.knots_[c])
                    parts[c] = B[:, 0]
                    for j in range(1, B.shape[1]):
                        parts[f"{c}_rcs{j}"] = B[:, j]
                else:
                    parts[c] = x.to_numpy(float)
            elif t == "binary":
                parts[c] = x.to_numpy(float)
            elif self.recipe in ("tree", "mlp") and t == "ordinal":
                parts[c] = x.to_numpy(float)                      # ordered codes
            else:
                drop_ref = self.recipe in ("linear", "spline")    # regression: reference level dropped
                for lv in self.levels_[c]:
                    if drop_ref and lv == meta["ref"]:
                        continue
                    parts[f"{c}={lv}"] = (x == lv).to_numpy(float)
        return pd.DataFrame(parts, index=X.index)

    def transform(self, X):
        Z = self._encode(X[self.columns_])
        assert list(Z.columns) == self.feature_names_
        if self.recipe in ("linear", "spline", "mlp"):
            Z = (Z - self.mean_) / self.sd_
        return Z

    def get_feature_names_out(self, input_features=None):
        return np.asarray(self.feature_names_, dtype=object)


# =============================================================================
# 8.4 METRICS (decisions 86, 87, 96, 98, 110, 116), added part by part
# -----------------------------------------------------------------------------
# 8.4a Calibration curve and ICI / E50 / E90 (decision 116)
#
# The smoothed calibration curve is lowess exactly as Austin & Steyerberg 2019
# (R56) ran it in R: lowess(Y ~ P, iter = 0) with R's defaults, span f = 2/3
# and delta = 0.01 x range of P. statsmodels' lowess is the same algorithm;
# its own defaults differ (it = 3, delta = 0), so both are set explicitly.
#   ICI  mean   |smoothed observed - predicted| over patients
#   E50  median of the same differences
#   E90  90th percentile
#   Emax largest difference (stored, not a headline measure)
# =============================================================================
LOWESS_FRAC = 2 / 3
LOWESS_DELTA_SHARE = 0.01


def calibration_curve(y, p):
    """Smoothed observed risk at each patient's predicted risk (lowess, R56 settings)."""
    y, p = np.asarray(y, float), np.asarray(p, float)
    delta = LOWESS_DELTA_SHARE * (p.max() - p.min())
    return lowess(y, p, frac=LOWESS_FRAC, it=0, delta=delta, return_sorted=False)


def calibration_errors(y, p):
    """ICI, E50, E90 and Emax from the lowess calibration curve (decisions 110, 116)."""
    p = np.asarray(p, float)
    gap = np.abs(calibration_curve(y, p) - p)
    return {"ICI": float(gap.mean()), "E50": float(np.median(gap)),
            "E90": float(np.quantile(gap, 0.9)), "Emax": float(gap.max())}


# -----------------------------------------------------------------------------
# 8.4b Discrimination, calibration, overall and clinical-utility measures
#      (decisions 86, 87, 98, 117) and the paired bootstrap (decision 96)
#
#   AUC                  area under the ROC curve
#   PR-AUC               average precision (decision 117), shown next to the event rate
#   CITL                 calibration-in-the-large: intercept of a logistic regression
#                        of y with the model's log-odds as a fixed offset (0 ideal)
#   slope                calibration slope: coefficient of the model's log-odds in a
#                        logistic regression of y (1 ideal; < 1 = too extreme)
#   Brier, scaled Brier  mean squared error; 1 - Brier / Brier of always predicting
#                        the event rate of the data being scored
#   net benefit          (TP - FP x t/(1-t)) / n at threshold t, a patient flagged
#                        when predicted risk >= t; plus "treat all" and "treat none"
#   top 10% / 20%        sensitivity, specificity, PPV, NPV when the highest-risk
#                        10% / 20% are flagged; ties at the cut-off share the last
#                        places equally (decision 118)
#
# Log-odds need risks strictly between 0 and 1; tree models and the isotonic
# calibration step can give exactly 0 or 1, so risks are clipped to
# [LOGIT_EPS, 1 - LOGIT_EPS] for CITL and slope only.
# -----------------------------------------------------------------------------
LOGIT_EPS = 1e-8
NB_THRESHOLDS = (0.10, 0.15, 0.20, 0.30)          # key thresholds (decision 98)


def calibration_intercept_slope(y, p, tol=1e-10, max_iter=50):
    """Calibration-in-the-large (offset model) and calibration slope.

    Fitted by Newton-Raphson, the standard way a logistic regression is solved,
    written out because these fits have only one or two numbers and the general
    statsmodels routine is about 20 times slower; the Step 8 tests check that both
    give the same answer.
    """
    y = np.asarray(y, float)
    lp = logit(np.clip(np.asarray(p, float), LOGIT_EPS, 1 - LOGIT_EPS))
    # CITL: intercept a with lp as a fixed offset
    a = 0.0
    for _ in range(max_iter):
        mu = expit(lp + a)
        step = np.sum(y - mu) / np.sum(mu * (1 - mu))
        a += step
        if abs(step) < tol:
            break
    # slope: intercept and coefficient of lp
    beta = np.zeros(2)
    Xd = np.column_stack([np.ones_like(lp), lp])
    for _ in range(max_iter):
        mu = expit(Xd @ beta)
        w = mu * (1 - mu)
        step = np.linalg.solve(Xd.T @ (Xd * w[:, None]), Xd.T @ (y - mu))
        beta += step
        if np.max(np.abs(step)) < tol:
            break
    return {"CITL": float(a), "slope": float(beta[1])}


def calibration_intercept_slope_statsmodels(y, p):
    """Reference version with statsmodels GLM (used only to test the fast version)."""
    y = np.asarray(y, float)
    lp = logit(np.clip(np.asarray(p, float), LOGIT_EPS, 1 - LOGIT_EPS))
    citl = sm.GLM(y, np.ones((len(y), 1)), family=sm.families.Binomial(), offset=lp).fit().params[0]
    slope = sm.GLM(y, sm.add_constant(lp), family=sm.families.Binomial()).fit().params[1]
    return {"CITL": float(citl), "slope": float(slope)}


def brier_scores(y, p):
    y, p = np.asarray(y, float), np.asarray(p, float)
    brier = float(np.mean((p - y) ** 2))
    ybar = y.mean()
    return {"Brier": brier, "scaled_Brier": 1 - brier / (ybar * (1 - ybar))}


def net_benefit(y, p, t):
    """Net benefit of flagging patients with predicted risk >= t."""
    y, p = np.asarray(y), np.asarray(p, float)
    n = len(y)
    flag = p >= t
    tp, fp = np.sum(flag & (y == 1)), np.sum(flag & (y == 0))
    return float(tp / n - fp / n * t / (1 - t))


def net_benefit_treat_all(y, t):
    prev = float(np.mean(y))
    return prev - (1 - prev) * t / (1 - t)


TOP_SHARES = (0.10, 0.20)                         # flag the top 10% / 20% of risk (decision 98)


def flag_weights(p, share):
    """How much each patient counts as flagged when the top `share` of risk is flagged.

    Patients above the cut-off count 1, below 0. Patients tied AT the cut-off share
    the remaining places equally (decision 118), so the weights add up to exactly
    share x n for every model; with no ties this is the ordinary top-k flag.
    """
    p = np.asarray(p, float)
    n_flag = share * len(p)
    order = np.sort(p)[::-1]
    k = int(np.ceil(n_flag)) - 1
    cut = order[max(k, 0)]
    above = p > cut
    tied = p == cut
    w = above.astype(float)
    w[tied] = (n_flag - above.sum()) / tied.sum()
    return w


def classification_from_weights(y, w):
    """Sensitivity, specificity, PPV and NPV from (possibly fractional) flag weights."""
    y = np.asarray(y)
    tp, fp = float(np.sum(w * (y == 1))), float(np.sum(w * (y == 0)))
    events, non = float(np.sum(y == 1)), float(np.sum(y == 0))
    flagged = tp + fp
    fn, tn = events - tp, non - fp
    return {"sensitivity": tp / events if events else np.nan,
            "specificity": tn / non if non else np.nan,
            "PPV": tp / flagged if flagged else np.nan,
            "NPV": tn / (tn + fn) if (tn + fn) else np.nan}


def performance(y, p, thresholds=NB_THRESHOLDS):
    """Every single-number measure for one model on one dataset."""
    out = {"n": int(len(y)), "events": int(np.sum(y)), "event_rate": float(np.mean(y)),
           "AUC": float(roc_auc_score(y, p)), "PR_AUC": float(average_precision_score(y, p))}
    out.update(calibration_intercept_slope(y, p))
    out.update(brier_scores(y, p))
    out.update(calibration_errors(y, p))
    for t in thresholds:
        out[f"NB_{int(round(100 * t))}"] = net_benefit(y, p, t)
        out[f"NB_all_{int(round(100 * t))}"] = net_benefit_treat_all(y, t)
    for s in TOP_SHARES:
        for k, v in classification_from_weights(y, flag_weights(p, s)).items():
            out[f"top{int(round(100 * s))}_{k}"] = v
    return out


# -----------------------------------------------------------------------------
# Paired percentile bootstrap (decision 96): 2,000 resamples of patients with
# replacement; the SAME resamples for every model scored on a dataset (paired),
# made from a fixed seed so any model can be added later and gets identical
# resamples; models are not refitted; 95% limits are the 2.5th and 97.5th
# percentiles; differences between models are taken within each resample.
# -----------------------------------------------------------------------------
BOOT_REPLICATES = 2000
BOOT_SEED = 20261007


def bootstrap_indices(n, b, seed=BOOT_SEED):
    """Patient indices of resample b (0-based), identical for every model."""
    return np.random.default_rng([seed, n, b]).integers(0, n, n)


def paired_bootstrap(y, preds, metric, replicates=BOOT_REPLICATES, seed=BOOT_SEED, pairs=()):
    """Point estimates and 95% percentile CIs per model, and for paired differences.

    y       outcomes (0/1)
    preds   dict model -> predicted risks, aligned with y
    metric  function (y, p) -> dict of measures (e.g. performance) or a single float
    pairs   [(a, b), ...]: differences metric(a) - metric(b), taken within each resample

    Returns (table, replicates): table has one row per model or difference and
    measure, with estimate, ci_low, ci_high; replicates holds every resample's values.
    """
    y = np.asarray(y)
    names = list(preds)
    P = {m: np.asarray(preds[m], float) for m in names}

    def as_dict(v):
        return v if isinstance(v, dict) else {"value": v}

    point = {m: as_dict(metric(y, P[m])) for m in names}
    measures = [k for k, v in point[names[0]].items() if isinstance(v, float)]
    reps = {m: {k: np.empty(replicates) for k in measures} for m in names}
    for b in range(replicates):
        idx = bootstrap_indices(len(y), b, seed)
        for m in names:
            r = as_dict(metric(y[idx], P[m][idx]))
            for k in measures:
                reps[m][k][b] = r[k]
    rows = []
    for m in names:
        for k in measures:
            rows.append({"what": m, "measure": k, "estimate": point[m][k],
                         "ci_low": float(np.percentile(reps[m][k], 2.5)),
                         "ci_high": float(np.percentile(reps[m][k], 97.5))})
    for a, b_ in pairs:
        for k in measures:
            d = reps[a][k] - reps[b_][k]
            rows.append({"what": f"{a} - {b_}", "measure": k, "estimate": point[a][k] - point[b_][k],
                         "ci_low": float(np.percentile(d, 2.5)), "ci_high": float(np.percentile(d, 97.5))})
    return pd.DataFrame(rows), reps


# =============================================================================
# 8.5 MODEL WRAPPERS (decisions 91-93, 95, 102, 103, 123)
# -----------------------------------------------------------------------------
# Every model has the same three steps, so later steps can loop over models:
#
#   tune(Z_train, y_train, Z_tune, y_tune) -> settings + a log of every setting tried
#   fit(Z, y, settings)                    -> fitted model
#   predict(fitted, Z)                     -> predicted risk of a long stay
#
# Z is the output of ModelInputs (the model's recipe) after StageImputer, both
# fitted on the patients the model is fitted on. Tuning follows decision 92: fit
# on the inner 75% ("train"), score log loss on the inner 25% ("tune"); then the
# chosen settings are refitted on the whole learning part.
# =============================================================================
RIDGE_LAMBDAS = np.logspace(1, -5, 30)            # decision 123: 10 ... 0.00001, glmnet scale
RIDGE_EXTENSION = np.logspace(3, 1, 11)[:-1]      # edge rule: two more decades upward, used once


class FitFailed(RuntimeError):
    """Every fallback attempt of decision 133 failed."""


FIT_EVENTS = []          # retries and failures in this process (decision 133), for the report


def fit_logistic(C, Z, y, where="", tol=1e-10, attempts=None):
    """Logistic regression with the fixed fallbacks of decision 133 (config
    'fit_fallbacks'): newton-cholesky 500 iterations, then 5,000, then lbfgs 10,000.
    An attempt fails if it raises an error or warns that it did not converge.
    C=None means no penalty (M1); the intercept is never penalised. Retries are
    recorded in FIT_EVENTS; if every attempt fails, FitFailed is raised."""
    # scikit-learn >= 1.8: no penalty is C = inf; ridge (L2) is l1_ratio = 0
    attempts = attempts or SEARCH_SPACES["fit_fallbacks"]["regression_attempts"]
    Z, y, reasons = np.asarray(Z), np.asarray(y), []
    for k, a in enumerate(attempts, 1):
        m = LogisticRegression(C=np.inf if C is None else C, l1_ratio=0.0, solver=a["solver"],
                               max_iter=a["max_iter"], tol=tol)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always", ConvergenceWarning)
            try:
                m.fit(Z, y)
            except (ValueError, FloatingPointError, np.linalg.LinAlgError) as e:
                reasons.append(f"attempt {k} ({a['solver']}, {a['max_iter']}): {type(e).__name__}: {e}")
                continue
        if any(issubclass(c.category, ConvergenceWarning) for c in caught):
            reasons.append(f"attempt {k} ({a['solver']}, {a['max_iter']}): did not converge")
            continue
        if k > 1:
            FIT_EVENTS.append({"where": where, "succeeded_at": k, "reasons": reasons})
        return m
    FIT_EVENTS.append({"where": where, "succeeded_at": None, "reasons": reasons})
    raise FitFailed(f"{where}: all {len(attempts)} attempts failed: {reasons}")


class LogisticModel:
    """M1: plain logistic regression, recipe 'linear', no tuning."""
    name, recipe = "M1", "linear"

    def tune(self, Z_tr, y_tr, Z_tu, y_tu):
        return {}, pd.DataFrame()

    def fit(self, Z, y, settings=None):
        return fit_logistic(None, Z, y, where="M1")

    def predict(self, fitted, Z):
        return fitted.predict_proba(np.asarray(Z))[:, 1]


class RidgeModel:
    """M2: ridge logistic regression on the 'spline' recipe; lambda chosen from the
    decision 123 grid by inner log loss (ties -> larger lambda), with the edge rule."""
    name, recipe = "M2", "spline"

    @staticmethod
    def _C(lam, n):
        return 1.0 / (n * lam)                     # glmnet lambda -> scikit-learn C

    def _score_grid(self, lambdas, Z_tr, y_tr, Z_tu, y_tu):
        rows = []
        for lam in lambdas:
            m = fit_logistic(self._C(lam, len(y_tr)), Z_tr, y_tr, where=f"M2 lambda {lam:.3g}")
            rows.append({"lambda": float(lam), "log_loss": float(log_loss(y_tu, m.predict_proba(Z_tu)[:, 1]))})
        return rows

    def tune(self, Z_tr, y_tr, Z_tu, y_tu):
        Z_tr, Z_tu, y_tr, y_tu = map(np.asarray, (Z_tr, Z_tu, y_tr, y_tu))
        rows = self._score_grid(RIDGE_LAMBDAS, Z_tr, y_tr, Z_tu, y_tu)
        log = pd.DataFrame(rows)
        best = log.sort_values(["log_loss", "lambda"], ascending=[True, False]).iloc[0]
        edge = "none"
        if best["lambda"] == RIDGE_LAMBDAS.max():          # edge rule: extend upward once
            log = pd.concat([log, pd.DataFrame(self._score_grid(RIDGE_EXTENSION, Z_tr, y_tr, Z_tu, y_tu))],
                            ignore_index=True)
            best = log.sort_values(["log_loss", "lambda"], ascending=[True, False]).iloc[0]
            edge = "largest: grid extended"
        elif best["lambda"] == RIDGE_LAMBDAS.min():
            edge = "smallest: effectively unpenalised"
        return {"lambda": float(best["lambda"]), "edge": edge}, log

    def fit(self, Z, y, settings):
        return fit_logistic(self._C(settings["lambda"], len(y)), Z, y, where="M2")

    def predict(self, fitted, Z):
        return fitted.predict_proba(np.asarray(Z))[:, 1]


class EuroScoreModel:
    """M0: EuroSCORE II recalibrated on development data (decision 93): logistic
    regression of the outcome on the log-odds of EuroSCORE II (intercept and slope).
    Uses the 'extras' column ES2 (in %), not the predictors. Missing EuroSCORE II
    (decision 124): the recalibration is fitted on training patients with a recorded
    value; a patient without one receives the training median, so every patient
    gets a prediction. The number filled is kept in `n_filled_` for reporting."""
    name = "M0"

    @staticmethod
    def _lp(es2):
        return logit(np.clip(np.asarray(es2, float) / 100, LOGIT_EPS, 1 - LOGIT_EPS))[:, None]

    def tune(self, *args):
        return {}, pd.DataFrame()

    def fit(self, es2, y, settings=None):
        es2, y = np.asarray(es2, float), np.asarray(y)
        recorded = ~np.isnan(es2)
        fitted = fit_logistic(None, self._lp(es2[recorded]), y[recorded], where="M0")
        fitted.es2_median_ = float(np.median(es2[recorded]))
        fitted.n_missing_in_fit_ = int((~recorded).sum())
        return fitted

    def predict(self, fitted, es2):
        es2 = np.asarray(es2, float)
        missing = np.isnan(es2)
        self.n_filled_ = int(missing.sum())
        return fitted.predict_proba(self._lp(np.where(missing, fitted.es2_median_, es2)))[:, 1]


# -----------------------------------------------------------------------------
# Search spaces live in config/search_spaces.json, written before any tuning
# (decision 95). The code reads them, so code and registered space cannot drift.
# -----------------------------------------------------------------------------
SEARCH_SPACE_FILE = Path(__file__).resolve().parent.parent / "config" / "search_spaces.json"
SEARCH_SPACES = json.loads(SEARCH_SPACE_FILE.read_text())
_rg = SEARCH_SPACES["M2_ridge"]["lambda_grid"]
assert np.allclose(RIDGE_LAMBDAS, np.logspace(np.log10(_rg["from"]), np.log10(_rg["to"]), _rg["values"])), \
    "RIDGE_LAMBDAS differs from config/search_spaces.json"
MODEL_SEED = 20261007


class XGBoostModel:
    """M3: gradient-boosted trees (XGBoost) on the 'tree' recipe (decision 125).

    fit_trial() fits one setting on the inner 75% with early stopping on the inner
    25% (decision 95) and records the number of trees reached; fit() refits the
    chosen setting on the whole learning part with that number of trees. The
    Optuna search and the calibration step are added once the calibration split
    is agreed."""
    name, recipe = "M3", "tree"
    space, steps_key = "M3_xgboost", "n_trees"     # used by nacsa_tuning

    def __init__(self, n_jobs=4):
        self.fixed = SEARCH_SPACES["M3_xgboost"]["fixed"]
        self.n_jobs = n_jobs

    def _make(self, params, n_estimators, early_stopping):
        from xgboost import XGBClassifier
        tree_params = {k: v for k, v in params.items() if k != "calibration"}
        return XGBClassifier(
            n_estimators=n_estimators, early_stopping_rounds=early_stopping,
            objective=self.fixed["objective"], eval_metric=self.fixed["eval_metric"],
            tree_method=self.fixed["tree_method"], reg_alpha=self.fixed["reg_alpha"],
            gamma=self.fixed["gamma"], random_state=MODEL_SEED, n_jobs=self.n_jobs, **tree_params)

    def fit_trial(self, Z_tr, y_tr, Z_tu, y_tu, params):
        m = self._make(params, self.fixed["n_estimators_max"], self.fixed["early_stopping_rounds"])
        m.fit(np.asarray(Z_tr), np.asarray(y_tr), eval_set=[(np.asarray(Z_tu), np.asarray(y_tu))], verbose=False)
        return m, int(m.best_iteration) + 1

    def fit(self, Z, y, settings):
        m = self._make(settings["params"], settings["n_trees"], None)
        return m.fit(np.asarray(Z), np.asarray(y), verbose=False)

    def predict(self, fitted, Z):
        return fitted.predict_proba(np.asarray(Z))[:, 1]


# -----------------------------------------------------------------------------
# Calibration step for M3-M5 (decisions 93, 126)
#
#   none      the model's own risks
#   platt     logistic regression on the log-odds of the model's risk
#   isotonic  monotone step function of the risk, clipped to the observed range
#
# Inside a tuning trial (1A): the inner 25% is split into two stratified halves;
# the calibrator is fitted on one and scored on the other, then swapped; the two
# log losses are averaged. 'none' is scored on the whole 25%.
# Final model of a round (2A): refitted on the whole learning part; the calibrator
# is fitted on 5-fold out-of-fold predictions of the learning part (the design of
# scikit-learn's CalibratedClassifierCV(ensemble=False)).
# -----------------------------------------------------------------------------
CALIBRATION_METHODS = ("none", "platt", "isotonic")
CALIBRATION_SEED = MODEL_SEED
CALIBRATION_FOLDS = 5


class Calibrator:
    def __init__(self, method):
        if method not in CALIBRATION_METHODS:
            raise ValueError(f"method must be one of {CALIBRATION_METHODS}")
        self.method = method

    def fit(self, p, y):
        p, y = np.asarray(p, float), np.asarray(y)
        if self.method == "platt":
            self.model_ = fit_logistic(None, logit(np.clip(p, LOGIT_EPS, 1 - LOGIT_EPS))[:, None], y, where="Platt calibrator")
        elif self.method == "isotonic":
            self.model_ = IsotonicRegression(y_min=0, y_max=1, out_of_bounds="clip").fit(p, y)
        return self

    def transform(self, p):
        p = np.asarray(p, float)
        if self.method == "none":
            return p
        if self.method == "platt":
            return self.model_.predict_proba(logit(np.clip(p, LOGIT_EPS, 1 - LOGIT_EPS))[:, None])[:, 1]
        return self.model_.predict(p)


def _safe_log_loss(y, p):
    return float(log_loss(y, np.clip(p, LOGIT_EPS, 1 - LOGIT_EPS), labels=[0, 1]))


def tune_halves(y_tu, seed=CALIBRATION_SEED):
    """The two stratified halves of the inner 25% (fixed seed)."""
    idx = np.arange(len(y_tu))
    a, b = train_test_split(idx, test_size=0.5, random_state=seed, stratify=np.asarray(y_tu))
    return np.sort(a), np.sort(b)


def crossfit_calibration_loss(p_tu, y_tu, method, seed=CALIBRATION_SEED):
    """Trial score (decision 126, 1A): log loss of the calibrated risks, each half
    scored with a calibrator fitted on the other half; 'none' on the whole 25%."""
    p_tu, y_tu = np.asarray(p_tu, float), np.asarray(y_tu)
    if method == "none":
        return _safe_log_loss(y_tu, p_tu)
    a, b = tune_halves(y_tu, seed)
    losses = []
    for fit_idx, score_idx in ((a, b), (b, a)):
        cal = Calibrator(method).fit(p_tu[fit_idx], y_tu[fit_idx])
        losses.append(_safe_log_loss(y_tu[score_idx], cal.transform(p_tu[score_idx])))
    return float(np.mean(losses))


def pick_calibration(losses, decimals=3):
    """Decision 134 / rule 7: the option with the lowest log loss; options whose
    losses are equal when rounded to `decimals` places are tied, and the simpler
    one wins (none, then platt, then isotonic)."""
    best = min(round(losses[m], decimals) for m in CALIBRATION_METHODS)
    return next(m for m in CALIBRATION_METHODS if round(losses[m], decimals) == best)


def choose_calibration(p_tu, y_tu, seed=CALIBRATION_SEED):
    """Decision 134: score all three options on one model's inner-25% predictions
    (cross-fitted halves, decision 126) and keep one by the rule-7 tie rule.
    Returns (method, its log loss, all three losses)."""
    losses = {m: crossfit_calibration_loss(p_tu, y_tu, m, seed) for m in CALIBRATION_METHODS}
    method = pick_calibration(losses)
    return method, losses[method], losses


class CalibratedFit:
    """A fitted model plus its calibrator; predict() returns calibrated risks."""

    def __init__(self, wrapper, fitted, calibrator, oof_index=None):
        self.wrapper, self.fitted, self.calibrator, self.oof_index_ = wrapper, fitted, calibrator, oof_index

    def predict(self, Z):
        return self.calibrator.transform(self.wrapper.predict(self.fitted, Z))


def fit_with_calibration(wrapper, Z, y, settings, method, seed=CALIBRATION_SEED):
    """Final model of a round (decision 126, 2A)."""
    Z, y = np.asarray(Z), np.asarray(y)
    if method == "none":
        return CalibratedFit(wrapper, wrapper.fit(Z, y, settings), Calibrator("none"))
    oof = np.empty(len(y))
    folds = StratifiedKFold(n_splits=CALIBRATION_FOLDS, shuffle=True, random_state=seed)
    fold_of = np.empty(len(y), int)
    for k, (tr, te) in enumerate(folds.split(Z, y)):
        oof[te] = wrapper.predict(wrapper.fit(Z[tr], y[tr], settings), Z[te])
        fold_of[te] = k
    calibrator = Calibrator(method).fit(oof, y)
    return CalibratedFit(wrapper, wrapper.fit(Z, y, settings), calibrator, oof_index=fold_of)


class RandomForestModel:
    """M4: random forest on the 'tree' recipe (decision 127). No early stopping:
    fit_trial() fits on the inner 75% and returns the model (the number of trees
    is fixed); fit() refits the chosen setting on the whole learning part."""
    name, recipe = "M4", "tree"
    space, steps_key = "M4_random_forest", None

    def __init__(self, n_jobs=4):
        self.fixed = SEARCH_SPACES["M4_random_forest"]["fixed"]
        self.n_jobs = n_jobs

    def _make(self, params):
        from sklearn.ensemble import RandomForestClassifier
        tree_params = {k: v for k, v in params.items() if k != "calibration"}
        return RandomForestClassifier(
            n_estimators=self.fixed["n_estimators"], bootstrap=self.fixed["bootstrap"],
            max_depth=self.fixed["max_depth"], criterion=self.fixed["criterion"],
            random_state=MODEL_SEED, n_jobs=self.n_jobs, **tree_params)

    def fit_trial(self, Z_tr, y_tr, Z_tu, y_tu, params):
        return self._make(params).fit(np.asarray(Z_tr), np.asarray(y_tr)), self.fixed["n_estimators"]

    def fit(self, Z, y, settings):
        return self._make(settings["params"]).fit(np.asarray(Z), np.asarray(y))

    def predict(self, fitted, Z):
        # trees are fitted in parallel, but their votes are added on one core so the
        # order of addition (and so the last decimal place) never varies between runs
        fitted.set_params(n_jobs=1)
        p = fitted.predict_proba(np.asarray(Z))[:, 1]
        fitted.set_params(n_jobs=self.n_jobs)
        return p


class MLPModel:
    """M5: neural network (scikit-learn multilayer perceptron) on the 'mlp' recipe
    (decision 128). Every hidden layer has the same number of neurons; ReLU
    activation and the Adam optimiser. scikit-learn has no dropout: overfitting is
    held back by weight decay (alpha) and early stopping.

    Early stopping is our own loop, one pass through the data (epoch) at a time,
    watching log loss rather than scikit-learn's built-in accuracy. fit_trial()
    trains on the inner 75%, scores the inner 25% after every epoch, stops after
    `patience` epochs with no gain and keeps the best epoch's weights; it returns
    the number of epochs reached. fit() refits on the whole learning part for that
    many epochs (the same pattern as XGBoost's number of trees)."""
    name, recipe = "M5", "mlp"
    space, steps_key = "M5_mlp", "n_epochs"

    def __init__(self, n_jobs=1):                  # n_jobs accepted for a common interface; unused
        self.fixed = SEARCH_SPACES["M5_mlp"]["fixed"]

    def _make(self, params):
        from sklearn.neural_network import MLPClassifier
        p = {k: v for k, v in params.items() if k != "calibration"}
        return MLPClassifier(
            hidden_layer_sizes=(int(p["n_neurons"]),) * int(p["n_layers"]),
            alpha=p["alpha"], learning_rate_init=p["learning_rate"], batch_size=int(p["batch_size"]),
            activation=self.fixed["activation"], solver=self.fixed["solver"],
            # a RandomState object (not an integer) so each epoch draws a new batch order;
            # scikit-learn re-seeds an integer seed on every partial_fit call
            random_state=np.random.RandomState(MODEL_SEED))

    def _epoch(self, m, Z, y):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            m.partial_fit(Z, y, classes=np.array([0, 1]))

    def fit_trial(self, Z_tr, y_tr, Z_tu, y_tu, params):
        Z_tr, y_tr, Z_tu, y_tu = map(np.asarray, (Z_tr, y_tr, Z_tu, y_tu))
        m = self._make(params)
        best, best_epoch, best_weights, wait, history = np.inf, 0, None, 0, []
        for epoch in range(1, self.fixed["max_epochs"] + 1):
            self._epoch(m, Z_tr, y_tr)
            loss = _safe_log_loss(y_tu, m.predict_proba(Z_tu)[:, 1])
            history.append(loss)
            if loss < best - self.fixed["min_improvement"]:
                best, best_epoch, wait = loss, epoch, 0
                best_weights = ([c.copy() for c in m.coefs_], [b.copy() for b in m.intercepts_])
            else:
                wait += 1
                if wait >= self.fixed["patience"]:
                    break
        m.coefs_, m.intercepts_ = best_weights
        m.tune_losses_ = history                   # log loss on the inner 25% after each epoch
        return m, best_epoch

    def fit(self, Z, y, settings):
        Z, y = np.asarray(Z), np.asarray(y)
        m = self._make(settings["params"])
        for _ in range(int(settings["n_epochs"])):
            self._epoch(m, Z, y)
        return m

    def predict(self, fitted, Z):
        return fitted.predict_proba(np.asarray(Z))[:, 1]

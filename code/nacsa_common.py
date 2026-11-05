"""
nacsa_common.py  |  Shared helpers for every analysis step from Step 4 onwards
==============================================================================

WHY THIS FILE EXISTS
    Every later script needs the same things: load the clean data, check it is
    the file we expect, make a few derived variables, know which variables are
    which, and hide small counts. If each script did this its own way, they
    would slowly drift apart. So each rule is written ONCE, here, and every
    script imports it:

        import nacsa_common as nc
        dev = nc.load_clean()                 # development patients, primary outcome

WHERE IT LIVES
    MyDrive/nacsa/code/nacsa_common.py   (on your Google Drive)
    code/nacsa_common.py                 (in the GitHub repository)

WHAT IS INSIDE (in this order)
    1. Paths and fixed settings
    2. The expected schema (the 63 columns of nacsa_clean.parquet)
    3. The variable registry: type, stage, label and levels of every variable
    4. load_clean(): the only way scripts read the data
       (schema check, derived variables, validation lock, outcome filter)
    5. Privacy helpers: suppress(), n_pct(), pct_missing_text()

Nothing in this file changes the clean data on disk. It only reads it.
"""

import os
from pathlib import Path

import numpy as np
import pandas as pd

# =============================================================================
# 1. PATHS AND FIXED SETTINGS
# =============================================================================
# In Colab the project lives on Google Drive. When Claude runs the code in its
# own workspace, the environment variable NACSA_DIR points to the local copy.
PROJECT_DIR = Path(os.environ.get("NACSA_DIR", "/content/drive/MyDrive/nacsa"))
CLEAN_FILE = PROJECT_DIR / "processed" / "nacsa_clean.parquet"
CLEAN_FILE_CSV = PROJECT_DIR / "processed" / "nacsa_clean.csv.gz"   # same data, compressed CSV
OUTPUTS_DIR = PROJECT_DIR / "outputs"

EXPECTED_ROWS = 377_301      # rows in nacsa_clean.parquet after Step 3
LOS_CUTOFF = 10              # prolonged stay = more than 10 whole days
SMALL_CELL = 10              # counts from 1 to SMALL_CELL - 1 are shown as "<10"

PERIODS = {                  # calendar years in each period (decision 6)
    "older":       (1996, 2011),
    "development": (2012, 2016),
    "validation":  (2017, 2019),
}

AGE_BANDS = [-np.inf, 60, 70, 80, np.inf]                    # decision 36
AGE_BAND_LABELS = ["<60", "60-69", "70-79", "80+"]

# =============================================================================
# 2. THE EXPECTED SCHEMA
# -----------------------------------------------------------------------------
# These are the 63 columns Step 3 wrote, in the same order. load_clean() stops
# with an error if the file has different columns or a different number of
# rows: that would mean the cleaning was rerun differently, and every number in
# the paper would silently change.
# =============================================================================
EXPECTED_COLUMNS = (
    ["record_id", "Year", "period", "Country",
     "outcome_primary", "outcome_survivors", "Mortality", "los_days", "PostLOS"]
    + ["Age", "Gender", "Height", "Weight", "BMI", "Smoking", "Diabetes", "Hypertension",
       "PulmonaryDisease", "PVD", "Stroke", "NeuroDys", "PrevMI", "RecentMI", "UA", "PCI",
       "PrevSurgery", "PrevCABG", "CCS", "NYHA", "PreopHR", "CADExtent", "lv_impaired", "PAP",
       "PulHTN", "CardiogenicShock", "Inotropes", "VentilatedPreop", "CPS", "Urgency",
       "Cr200", "Creatinine", "crcl", "PreopDialysis", "iabp_preop"]
    + ["PumpCase", "CPB", "XClamp", "myo_protection", "DistalGrafts", "RCA", "LAD", "Diag", "Cx",
       "iabp_intraop"]
    + ["RTT", "rtt_bleeding", "rtt_other_cardiac", "rtt_wound", "PostopCVA", "PostopDialysis",
       "PostopDSWI", "iabp_postop"]
    + ["ES2"]
)

# =============================================================================
# 3. OUTCOMES
# -----------------------------------------------------------------------------
# The Step 3 file calls the composite outcome "outcome_primary". Since the
# primary outcome became option B (decision 1), that name is misleading, so
# load_clean() renames it to "outcome_composite". From Step 4 on, scripts ask
# for an outcome by its short name:
#
#   "survivors"  PRIMARY (option B): stay > 10 days among patients discharged
#                alive; in-hospital deaths excluded
#   "composite"  SENSITIVITY C: stay > 10 days or in-hospital death
#   "all"        SENSITIVITY A: stay > 10 days among all patients, deaths
#                counted by their actual length of stay
# =============================================================================
OUTCOMES = {
    "survivors": "outcome_survivors",
    "composite": "outcome_composite",
    "all":       "outcome_all",
}
PRIMARY_OUTCOME = "survivors"

# =============================================================================
# 4. THE VARIABLE REGISTRY
# -----------------------------------------------------------------------------
# One entry per candidate predictor (plus the EuroSCORE II benchmark).
#
#   type    "continuous"  a number (age, minutes, kg)
#           "binary"      yes/no, coded 1/0
#           "ordinal"     categories with a natural order (NYHA I < II < III < IV)
#           "nominal"     categories with no order (heart rhythm)
#   stage   "pre"   known before surgery           (all three models)
#           "intra" known at the end of surgery    (models 2 and 3)
#           "post"  known early after surgery      (model 3)
#           "benchmark"  EuroSCORE II, never a predictor (decision: M0 only)
#   label   wording used in tables
#   levels  code -> label, for categorical variables (in display order)
#   ref     reference level (the "normal" category regression compares with)
#   source  where the variable comes from (NICOR dataset field number, or how
#           we derived it); feeds the variable dictionary (ST1)
#
# Being in this list does NOT mean a variable will be a predictor. Overlapping
# variables (e.g. Creatinine / crcl / Cr200) are all listed now; Step 5 decides
# which ones go into the models (decisions 20 and 21).
# =============================================================================
YN = {0: "No", 1: "Yes"}

VARIABLES = {
    # ---- pre-operative --------------------------------------------------------
    "Age":              dict(type="continuous", stage="pre", label="Age, years", source="NICOR D1"),
    "Gender":           dict(type="binary", stage="pre", label="Female sex", levels=YN, ref=0,
                             source="NICOR 1.07 (0 male, 1 female)"),
    "Height":           dict(type="continuous", stage="pre", label="Height, cm", source="NICOR 2.37; 120-220 kept"),
    "Weight":           dict(type="continuous", stage="pre", label="Weight, kg", source="NICOR 2.38; 30-250 kept"),
    "BMI":              dict(type="continuous", stage="pre", label="Body mass index, kg/m2",
                             source="Derived: weight / height(m)^2; 12-70 kept"),
    "smoking_ever":     dict(type="binary", stage="pre", label="Ever smoked", levels=YN, ref=0,
                             source="Derived from NICOR 2.11: ex or current smoker = 1, never = 0 (decision 49)"),
    "diabetes_any":     dict(type="binary", stage="pre", label="Diabetes (any treatment)", levels=YN, ref=0,
                             source="Derived from NICOR 2.09: diet, oral or insulin = 1, none = 0 (decision 49)"),
    "Hypertension":     dict(type="binary", stage="pre", label="Hypertension", levels=YN, ref=0, source="NICOR 2.10"),
    "PulmonaryDisease": dict(type="binary", stage="pre", label="Chronic pulmonary disease", levels=YN, ref=0,
                             source="NICOR 2.13"),
    "PVD":              dict(type="binary", stage="pre", label="Extracardiac arteriopathy", levels=YN, ref=0,
                             source="NICOR 2.17"),
    "stroke_any":       dict(type="binary", stage="pre", label="Previous TIA or stroke", levels=YN, ref=0,
                             source="Derived from NICOR 2.14: TIA or any stroke = 1, none = 0 (decision 49)"),
    "NeuroDys":         dict(type="binary", stage="pre", label="Neurological dysfunction", levels=YN, ref=0,
                             source="NICOR 2.16"),
    "PrevMI":           dict(type="ordinal", stage="pre", label="Previous myocardial infarctions", ref=0,
                             levels={0: "None", 1: "One", 2: "Two or more"}, source="NICOR 2.03"),
    "RecentMI":         dict(type="binary", stage="pre", label="Myocardial infarction within 90 days", levels=YN,
                             ref=0, source="From NICOR 2.04 (EuroSCORE II definition)"),
    "UA":               dict(type="binary", stage="pre", label="Unstable angina", levels=YN, ref=0,
                             source="Older dataset field"),
    "PCI":              dict(type="nominal", stage="pre", label="Previous PCI", ref=0,
                             levels={0: "None", 1: "Within 24 hours", 2: "Same admission, over 24 hours",
                                     3: "Previous admission"}, source="NICOR 2.05"),
    "PrevSurgery":      dict(type="binary", stage="pre", label="Previous cardiac surgery", levels=YN, ref=0,
                             source="NICOR 2.07"),
    "PrevCABG":         dict(type="binary", stage="pre", label="Previous CABG", levels=YN, ref=0,
                             source="Older dataset field"),
    "CCS":              dict(type="ordinal", stage="pre", label="Angina (CCS class)", ref=0,
                             levels={0: "No angina", 1: "CCS 1", 2: "CCS 2", 3: "CCS 3", 4: "CCS 4"},
                             source="NICOR 2.01"),
    "NYHA":             dict(type="ordinal", stage="pre", label="Dyspnoea (NYHA class)", ref=1,
                             levels={1: "NYHA I", 2: "NYHA II", 3: "NYHA III", 4: "NYHA IV"}, source="NICOR 2.02"),
    "PreopHR":          dict(type="nominal", stage="pre", label="Pre-operative heart rhythm", ref=0,
                             levels={0: "Sinus rhythm", 1: "AF or flutter", 2: "Heart block or paced",
                                     3: "VF or VT", 4: "Other"}, source="NICOR 2.18"),
    "CADExtent":        dict(type="ordinal", stage="pre", label="Diseased coronary vessels", ref=3,
                             levels={0: "None", 1: "One", 2: "Two", 3: "Three"},
                             source="NICOR 2.21 (reference 3 vessels, decision 50)"),
    "lv_impaired":      dict(type="binary", stage="pre", label="Impaired LV function (EF below 50%)", levels=YN,
                             ref=0, source="Derived from LVEFC (NICOR 2.28): any code other than good (decision 17)"),
    "PAP":              dict(type="ordinal", stage="pre", label="Pulmonary artery pressure", ref=0,
                             levels={0: "Below 31 mmHg", 1: "31-55 mmHg", 2: "Above 55 mmHg"},
                             source="EuroSCORE II category"),
    "PulHTN":           dict(type="binary", stage="pre", label="Pulmonary hypertension", levels=YN, ref=0,
                             source="Older dataset field"),
    "CardiogenicShock": dict(type="binary", stage="pre", label="Cardiogenic shock", levels=YN, ref=0,
                             source="NICOR 2.30"),
    "Inotropes":        dict(type="binary", stage="pre", label="IV inotropes before anaesthesia", levels=YN,
                             ref=0, source="NICOR 2.31"),
    "VentilatedPreop":  dict(type="binary", stage="pre", label="Ventilated before surgery", levels=YN, ref=0,
                             source="NICOR 2.32"),
    "CPS":              dict(type="binary", stage="pre", label="Critical pre-operative state", levels=YN, ref=0,
                             source="EuroSCORE II composite"),
    "Urgency":          dict(type="ordinal", stage="pre", label="Operative urgency", ref=1,
                             levels={1: "Elective", 2: "Urgent", 3: "Emergency", 4: "Salvage"}, source="NICOR 2.35"),
    "Cr200":            dict(type="binary", stage="pre", label="Creatinine above 200 umol/L", levels=YN, ref=0,
                             source="Older dataset field"),
    "Creatinine":       dict(type="continuous", stage="pre", label="Creatinine, umol/L", source="NICOR 2.12.0"),
    "crcl":             dict(type="continuous", stage="pre", label="Creatinine clearance, mL/min",
                             source="Derived: Cockcroft-Gault from age, weight, sex and creatinine"),
    "PreopDialysis":    dict(type="nominal", stage="pre", label="Renal failure and dialysis", ref=0,
                             levels={0: "None", 1: "Dialysis, acute renal failure",
                                     2: "Dialysis, chronic renal failure",
                                     3: "Acute renal failure, no dialysis"}, source="NICOR 2.12.1"),
    "iabp_preop":       dict(type="binary", stage="pre", label="IABP before surgery", levels=YN, ref=0,
                             source="Derived from IABP timing code 1"),
    # ---- intra-operative ------------------------------------------------------
    "PumpCase":         dict(type="binary", stage="intra", label="On-pump (cardiopulmonary bypass)", levels=YN,
                             ref=1, source="NICOR 3.78 (1 on-pump, 0 off-pump)"),
    "CPB":              dict(type="continuous", stage="intra", label="Bypass time, min",
                             source="NICOR 3.85; 0 for off-pump"),
    "XClamp":           dict(type="continuous", stage="intra", label="Cross-clamp time, min",
                             source="NICOR 3.86; 0 for off-pump"),
    "myo_protection":   dict(type="nominal", stage="intra", label="Myocardial protection", ref="cardioplegia",
                             levels={"cardioplegia": "Cardioplegia", "no_cardioplegia": "On-pump, no cardioplegia",
                                     "off_pump": "Off-pump"},
                             source="Derived from PumpCase and NICOR 3.79"),
    "DistalGrafts":     dict(type="continuous", stage="intra", label="Distal anastomoses, n", source="NICOR 3.14"),
    "RCA":              dict(type="binary", stage="intra", label="Graft to right coronary territory", levels=YN,
                             ref=0, source="NICOR 3.15"),
    "LAD":              dict(type="binary", stage="intra", label="Graft to LAD territory", levels=YN, ref=0,
                             source="NICOR 3.15"),
    "Diag":             dict(type="binary", stage="intra", label="Graft to diagonal territory", levels=YN, ref=0,
                             source="NICOR 3.15"),
    "Cx":               dict(type="binary", stage="intra", label="Graft to circumflex territory", levels=YN, ref=0,
                             source="NICOR 3.15"),
    "iabp_intraop":     dict(type="binary", stage="intra", label="IABP inserted during surgery", levels=YN, ref=0,
                             source="Derived from IABP timing code 2"),
    # ---- early post-operative -------------------------------------------------
    "RTT":              dict(type="binary", stage="post", label="Return to theatre (any)", levels=YN, ref=0,
                             source="NICOR 4.01"),
    "rtt_bleeding":     dict(type="binary", stage="post", label="Return to theatre, bleeding or tamponade",
                             levels=YN, ref=0, source="Derived from NICOR 4.01 code 1"),
    "rtt_other_cardiac": dict(type="binary", stage="post", label="Return to theatre, other cardiac", levels=YN,
                              ref=0, source="Derived from NICOR 4.01 codes 2-4"),
    "rtt_wound":        dict(type="binary", stage="post", label="Return to theatre, wound", levels=YN, ref=0,
                             source="Derived from NICOR 4.01 codes 5-6"),
    "PostopCVA":        dict(type="ordinal", stage="post", label="Post-operative stroke", ref=0,
                             levels={0: "None", 1: "Transient", 2: "Permanent"}, source="NICOR 4.02"),
    "PostopDialysis":   dict(type="binary", stage="post", label="New post-operative dialysis", levels=YN, ref=0,
                             source="NICOR 4.03"),
    "PostopDSWI":       dict(type="binary", stage="post", label="Deep sternal wound infection", levels=YN, ref=0,
                             source="NICOR 4.01.1"),
    "iabp_postop":      dict(type="binary", stage="post", label="IABP inserted after surgery", levels=YN, ref=0,
                             source="Derived from IABP timing code 3"),
    # ---- benchmark --------------------------------------------------------------
    "ES2":              dict(type="continuous", stage="benchmark", label="EuroSCORE II, %",
                             source="Supplied in the extract"),
}


# Short names for figure axes and heatmaps, where the full label is too long.
# Tables use the full label; the raw column name (e.g. "PrevCABG") is kept
# only as a "code" column at the end of CSV files, so any number can be traced
# back to the data, and in the variable dictionary (ST1), which links the two.
SHORT_LABELS = {
    "Age": "Age", "Gender": "Female sex", "Height": "Height", "Weight": "Weight", "BMI": "BMI",
    "smoking_ever": "Ever smoked", "diabetes_any": "Diabetes", "Hypertension": "Hypertension",
    "PulmonaryDisease": "Chronic lung disease", "PVD": "Extracardiac arteriopathy",
    "stroke_any": "Previous TIA/stroke", "NeuroDys": "Neurological dysfunction",
    "PrevMI": "Previous MIs", "RecentMI": "MI within 90 days", "UA": "Unstable angina",
    "PCI": "Previous PCI", "PrevSurgery": "Previous cardiac surgery", "PrevCABG": "Previous CABG",
    "CCS": "CCS angina class", "NYHA": "NYHA class", "PreopHR": "Pre-op rhythm",
    "CADExtent": "Diseased vessels", "lv_impaired": "LVEF below 50%", "PAP": "PA pressure",
    "PulHTN": "Pulmonary hypertension", "CardiogenicShock": "Cardiogenic shock",
    "Inotropes": "Pre-op inotropes", "VentilatedPreop": "Pre-op ventilation",
    "CPS": "Critical pre-op state", "Urgency": "Urgency", "Cr200": "Creatinine >200",
    "Creatinine": "Creatinine", "crcl": "Creatinine clearance", "PreopDialysis": "Renal failure/dialysis",
    "iabp_preop": "Pre-op IABP", "PumpCase": "On-pump", "CPB": "Bypass time", "XClamp": "Cross-clamp time",
    "myo_protection": "Myocardial protection", "DistalGrafts": "Distal anastomoses",
    "RCA": "Graft: right coronary", "LAD": "Graft: LAD", "Diag": "Graft: diagonal", "Cx": "Graft: circumflex",
    "iabp_intraop": "Intra-op IABP", "RTT": "Return to theatre", "rtt_bleeding": "RTT: bleeding",
    "rtt_other_cardiac": "RTT: other cardiac", "rtt_wound": "RTT: wound", "PostopCVA": "Post-op stroke",
    "PostopDialysis": "Post-op dialysis", "PostopDSWI": "Deep sternal infection",
    "iabp_postop": "Post-op IABP", "ES2": "EuroSCORE II",
}
assert set(SHORT_LABELS) == set(VARIABLES), "every variable needs a short label"


def label(var):
    """Full label for tables, e.g. 'Previous CABG' instead of 'PrevCABG'."""
    return VARIABLES[var]["label"]


def short(var):
    """Short label for figure axes."""
    return SHORT_LABELS[var]


def variables(stage=None, types=None, include_benchmark=False):
    """List variable names, optionally filtered.

    stage  "pre", "intra", "post", or a list of them. Cumulative sets are easy:
           variables(["pre", "intra"]) = everything known at the end of surgery.
    types  e.g. ["binary", "nominal"]
    """
    stages = [stage] if isinstance(stage, str) else stage
    tps = [types] if isinstance(types, str) else types
    out = []
    for name, meta in VARIABLES.items():
        if meta["stage"] == "benchmark" and not include_benchmark:
            continue
        if stages is not None and meta["stage"] not in stages:
            continue
        if tps is not None and meta["type"] not in tps:
            continue
        out.append(name)
    return out


# =============================================================================
# 4b. MODEL INPUTS (agreed in Step 5, decisions 51 and 57-68)
# -----------------------------------------------------------------------------
# The candidate list above is everything we describe. The lists below are what
# each model is allowed to use. They are cumulative: each stage adds what has
# become known by its prediction moment.
#
#   preop               model 1, before surgery
#   end_of_surgery      model 2, on arrival in ICU
#   early_postop        model 3, by the end of post-op day 1-2 (early events only)
#   complication_aware  secondary model, adds later complications; explanatory
#                       only, NOT for prospective use (decision 66)
#
# Every variable left out of all four lists is in NOT_MODEL_INPUTS with the
# decision that removed it, so the paper can report why.
# =============================================================================
_PREOP = ["Age", "Gender", "BMI", "smoking_ever", "diabetes_any", "Hypertension", "PulmonaryDisease",
          "PVD", "stroke_any", "NeuroDys", "PrevMI", "RecentMI", "UA", "PCI", "PrevSurgery", "CCS",
          "NYHA", "PreopHR", "CADExtent", "lv_impaired", "PAP", "CardiogenicShock", "CPS", "Urgency",
          "Creatinine", "PreopDialysis"]
_INTRA = ["myo_protection", "CPB", "DistalGrafts", "RCA", "LAD", "Diag", "Cx", "iabp_intraop"]
_EARLY_POST = ["rtt_bleeding", "iabp_postop"]
_LATE_POST = ["rtt_other_cardiac", "rtt_wound", "PostopDialysis", "PostopCVA"]

PREDICTOR_SETS = {
    "preop":              _PREOP,
    "end_of_surgery":     _PREOP + _INTRA,
    "early_postop":       _PREOP + _INTRA + _EARLY_POST,
    "complication_aware": _PREOP + _INTRA + _EARLY_POST + _LATE_POST,
}
MODEL_LABELS = {
    "preop": "Model 1: pre-operative",
    "end_of_surgery": "Model 2: intra-operative (end of surgery)",
    "early_postop": "Model 3: early post-operative",
    "complication_aware": "Secondary: complication-aware (explanatory only)",
}

NOT_MODEL_INPUTS = {   # variable: reason (decision number)
    "Height":         "Any two of height, weight and BMI fix the third; BMI kept (59)",
    "Weight":         "Any two of height, weight and BMI fix the third; BMI kept (59)",
    "PrevCABG":       "Strict subset of previous cardiac surgery (61)",
    "PulHTN":         "Every 'yes' lies within the top PA pressure band (63)",
    "Inotropes":      "Component of critical pre-operative state (60)",
    "VentilatedPreop": "Component of critical pre-operative state (60)",
    "iabp_preop":     "Component of critical pre-operative state; 16.6% missing (60)",
    "Cr200":          "A cut of creatinine at 200 umol/L (62); kidney marker in the all-era sets only (69)",
    "crcl":           "Built from age, weight and sex already in the model; double-counts age (62)",
    "PumpCase":       "Fully contained in myocardial protection (51)",
    "XClamp":         "Overlaps bypass time (Spearman 0.89); detail captured by distal anastomoses (57)",
    "RTT":            "Combines all return reasons, including late ones (leakage) (58)",
    "PostopDSWI":     "Late event, 28.8% missing, overlaps wound return (66)",
}

# All-era predictor sets for the training-period comparison (decisions 18, 69).
# Training set A = 2012-2016 with the full sets above (the main models);
# B = 2012-2016 and C = 1996-2016, both with these all-era sets. Variables only
# recorded from 2011 (creatinine, renal failure/dialysis, both IABP variables)
# are left out; creatinine above 200, recorded since 1996, is the kidney marker
# here only. The comparison covers the three main models, not the secondary one.
_PREOP_ALL_ERA = [v for v in _PREOP if v not in ("Creatinine", "PreopDialysis")] + ["Cr200"]
_INTRA_ALL_ERA = [v for v in _INTRA if v != "iabp_intraop"]
_EARLY_POST_ALL_ERA = [v for v in _EARLY_POST if v != "iabp_postop"]
ALL_ERA_SETS = {
    "preop":          _PREOP_ALL_ERA,
    "end_of_surgery": _PREOP_ALL_ERA + _INTRA_ALL_ERA,
    "early_postop":   _PREOP_ALL_ERA + _INTRA_ALL_ERA + _EARLY_POST_ALL_ERA,
}
ALL_ERA_YEARS = (1996, 2016)    # training set C

# Safety check: every candidate is either a model input or listed with a reason
_inputs = set(PREDICTOR_SETS["complication_aware"])
assert not _inputs & set(NOT_MODEL_INPUTS), "a variable is both an input and excluded"
assert _inputs | set(NOT_MODEL_INPUTS) == set(variables(["pre", "intra", "post"])), \
    "every candidate must be a model input or listed in NOT_MODEL_INPUTS"

# Pre-specified rules (decision 68), checked in Step 5
MAX_PCT_MISSING = 30          # a predictor may have at most 30% missing in development data
MIN_LEVEL_PATIENTS = 100      # every level of a categorical predictor needs >= 100 development patients

# Levels used for modelling where they differ from the descriptive ones (decision 65)
MODEL_LEVELS = {
    "Urgency":   {1: "Elective", 2: "Urgent", 3: "Emergency or salvage"},
    "CADExtent": {1: "One", 2: "Two", 3: "Three"},
}


def prepare_for_model(df):
    """Apply the Step 5 modelling rules (decision 65) to a loaded table.

    - Urgency: salvage (4) merged into emergency (3) -> "Emergency or salvage"
    - Diseased vessels: "none" (0) set to missing (coding error: these
      patients received 1-4 grafts); imputed later with other missing values

    Returns a copy; the descriptive tables (Step 4) use the original codes.
    """
    df = df.copy()
    df["Urgency"] = df["Urgency"].replace(4, 3)
    df["CADExtent"] = df["CADExtent"].where(df["CADExtent"] != 0)
    return df


def predictors(model, all_era=False):
    """The predictor list for one model, e.g. predictors("preop").

    all_era=True gives the all-era version used for training sets B and C.
    """
    sets = ALL_ERA_SETS if all_era else PREDICTOR_SETS
    if model not in sets:
        raise ValueError(f"Unknown model '{model}'; choose from {list(sets)}")
    return list(sets[model])


# =============================================================================
# 5. LOADING
# =============================================================================
def check_schema(df):
    """Stop with a clear message if the clean file is not the one we expect."""
    problems = []
    if len(df) != EXPECTED_ROWS:
        problems.append(f"expected {EXPECTED_ROWS:,} rows, found {len(df):,}")
    missing = [c for c in EXPECTED_COLUMNS if c not in df.columns]
    extra = [c for c in df.columns if c not in EXPECTED_COLUMNS]
    if missing:
        problems.append(f"columns missing: {missing}")
    if extra:
        problems.append(f"unexpected columns: {extra}")
    if problems:
        raise ValueError(
            "nacsa_clean.parquet is not the file this code was written for:\n  - "
            + "\n  - ".join(problems)
            + "\nIf you reran Step 3 on purpose, update EXPECTED_ROWS / EXPECTED_COLUMNS "
              "in nacsa_common.py and rerun every later step.")


def add_derived(df):
    """Make the variables defined in Step 4 (decision 49 and the outcome names).

    Every rule keeps missing as missing: a patient with unknown diabetes status
    gets a missing diabetes_any, never a 0.
    """
    df = df.rename(columns={"outcome_primary": "outcome_composite"})

    # Sensitivity outcome A: stay over 10 days among everyone with a known stay
    los = df["los_days"]
    df["outcome_all"] = np.where(los.notna(), (los > LOS_CUTOFF).astype(float), np.nan)

    # Yes/no versions of three multi-level fields (originals stay in the data)
    for new, old in [("diabetes_any", "Diabetes"), ("stroke_any", "Stroke"), ("smoking_ever", "Smoking")]:
        df[new] = np.where(df[old].notna(), (df[old] > 0).astype(float), np.nan)

    # Age band for description and the fairness analysis (not a predictor)
    df["age_band"] = pd.cut(df["Age"], AGE_BANDS, right=False, labels=AGE_BAND_LABELS)
    return df


def load_clean(period="development", outcome=PRIMARY_OUTCOME, allow_validation=False,
               path=None, verbose=True):
    """Read nacsa_clean.parquet and return the rows a script needs.

    period            "development" (default), "validation", "older",
                      "modern" (development + validation), "all", or a list
    outcome           "survivors" (primary, default), "composite", "all",
                      or None to keep every row whatever its outcome
    allow_validation  must be True to receive any 2017-2019 patient
                      (the validation lock, decision 25)

    The returned table carries a small flow summary in df.attrs["flow"]: how
    many rows the period selection kept and how many were dropped because the
    outcome was missing. Figure 1 is built from these numbers.
    """
    if path is None:
        path = CLEAN_FILE if CLEAN_FILE.exists() else CLEAN_FILE_CSV
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Clean file not found: {path}\nIs Google Drive mounted? Did Step 3 run?")

    # Parquet in Colab; the compressed CSV copy holds the same 377,301 x 63 table
    df = pd.read_parquet(path) if path.suffix == ".parquet" else pd.read_csv(path, low_memory=False)
    check_schema(df)
    df = add_derived(df)

    # ---- which periods ----
    if period == "all":
        wanted = list(PERIODS)
    elif period == "modern":
        wanted = ["development", "validation"]
    elif isinstance(period, str):
        wanted = [period]
    else:
        wanted = list(period)
    unknown = [p for p in wanted if p not in PERIODS]
    if unknown:
        raise ValueError(f"Unknown period(s) {unknown}; choose from {list(PERIODS)}, 'modern' or 'all'")
    if "validation" in wanted and not allow_validation:
        raise PermissionError(
            "VALIDATION LOCK: 2017-2019 patients are only released with allow_validation=True.\n"
            "Use it for description (Step 4) and final evaluation (Steps 12-14) only, "
            "never for choosing predictors, tuning or any modelling decision.")
    df = df[df["period"].isin(wanted)].copy()
    flow = {"period": "+".join(wanted), "n_selected": len(df)}

    # ---- outcome filter ----
    if outcome is not None:
        if outcome not in OUTCOMES:
            raise ValueError(f"Unknown outcome '{outcome}'; choose from {list(OUTCOMES)} or None")
        col = OUTCOMES[outcome]
        has = df[col].notna()
        flow.update({
            "outcome": outcome,
            "n_died_excluded": int(((df["Mortality"] == 1) & ~has).sum()),
            "n_other_missing_excluded": int(((df["Mortality"] != 1) & ~has).sum()),
            "n_analysed": int(has.sum()),
            "n_events": int((df.loc[has, col] == 1).sum()),
        })
        df = df[has].copy()
        df["y"] = df[col].astype(int)          # the outcome for this analysis, always called y

    df.attrs["flow"] = flow
    if verbose:
        print("load_clean:", ", ".join(f"{k}={v:,}" if isinstance(v, int) else f"{k}={v}"
                                       for k, v in flow.items()))
    return df


# =============================================================================
# 6. PRIVACY HELPERS (NICOR small-number rules)
# -----------------------------------------------------------------------------
# Any count from 1 to 9 is shown as "<10". Percentages that come from such a
# count are hidden too, otherwise the count could be worked back out. No
# minimum or maximum values are ever reported.
# =============================================================================
def is_small(n):
    return 0 < n < SMALL_CELL


def suppress(n):
    """A count, or '<10' if it is between 1 and 9."""
    n = int(n)
    return f"<{SMALL_CELL}" if is_small(n) else n


def n_pct(n, denominator, digits=1):
    """'1,234 (12.3%)', or '<10' when the count is small."""
    n, denominator = int(n), int(denominator)
    if is_small(n):
        return f"<{SMALL_CELL}"
    if denominator == 0:
        return "-"
    return f"{n:,} ({100 * n / denominator:.{digits}f}%)"


def pct_missing_text(n_missing, n_total):
    """Percentage missing, but '<10 records' when only 1-9 records are missing."""
    n_missing, n_total = int(n_missing), int(n_total)
    if is_small(n_missing):
        return f"<{SMALL_CELL} records"
    return f"{100 * n_missing / n_total:.1f}" if n_total else "-"


SECONDARY = "suppressed"     # label for a count hidden only to protect a small one


def suppress_counts(counts):
    """Suppress a set of counts that add up to a shown total (e.g. the levels of
    one variable). If exactly one count is small, the next smallest is hidden
    too ("secondary suppression"), so the small one cannot be found by
    subtracting the others from the total. The small count shows as '<10'; the
    count hidden to protect it shows as 'suppressed' (it is NOT small).
    """
    counts = [int(c) for c in counts]
    small = {i for i, c in enumerate(counts) if is_small(c)}
    secondary = set()
    if len(small) == 1:
        others = [(c, i) for i, c in enumerate(counts) if i not in small and c > 0]
        if others:
            secondary.add(min(others)[1])
    return [f"<{SMALL_CELL}" if i in small else SECONDARY if i in secondary else c
            for i, c in enumerate(counts)]


def rate_text(events, n, digits=1):
    """Event rate in %, hidden when events or non-events are 1-9."""
    events, n = int(events), int(n)
    if n == 0:
        return "-"
    if is_small(events) or is_small(n - events):
        return f"<{SMALL_CELL} events or non-events"
    return f"{100 * events / n:.{digits}f}"

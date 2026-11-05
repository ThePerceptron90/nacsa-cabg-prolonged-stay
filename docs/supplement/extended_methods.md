# Supplementary Methods

These methods expand the main text. Every modelling choice was fixed before the 2017-19 evaluation data were opened; later choices concerned only how results were explained or reported and did not change any model. All choices are recorded, with their reasons, in the project's decisions log. Exhibits are cited as ST (supplementary table) and SF (supplementary figure).

## S1. Data source, study periods and outcome

**Data source and study population.** We used prospectively collected data from the National Adult Cardiac Surgery Audit (NACSA), which records all major adult cardiac operations performed in the United Kingdom. The extract comprised 380,293 records of isolated coronary artery bypass grafting (CABG) performed between 1996 and March 2019 (the last operation date in the extract). After removal of 2,992 exact duplicate records, 377,301 records remained. Patients operated on in 2012-2016 formed the development cohort (n = 79,437) and those operated on in 2017-2019 the temporal evaluation cohort (n = 32,902). Records from 1996-2011 (n = 264,962) were used only in the pre-specified comparison of training periods (S10), because several key predictors (pre-operative creatinine, renal replacement therapy and intra-aortic balloon pump use) were recorded for fewer than 15% of patients before 2011 (almost none before 2010) and the dataset definitions changed over time (SF1, SF2).

**Outcome.** The primary outcome was prolonged post-operative length of stay, defined as more than 10 days counted in whole days from the day of operation, among patients discharged alive. Patients who died in hospital were excluded (1,048 development, 418 evaluation), as were records with unknown survival status (1,433 and 348) or unknown length of stay (1,142 and 32) (Figure 1). Because survival to discharge is not known when the models are used, predictions are interpreted as the risk of prolonged stay conditional on survival. Sensitivity analyses used a composite of prolonged stay or in-hospital death, prolonged stay among all patients with deaths classified by their actual length of stay, and cut-offs of 7 and 14 days (S12). The 14-day cut-off is the definition used by the US Society of Thoracic Surgeons registry [R42]; published cut-offs range from more than 1 to more than 14 days, more than 7 days being common [R41].

**Data preparation.** Data were audited before any modelling (completeness by year, internal consistency of overlapping fields and plausibility of continuous values) using the NICOR dataset specification. Physiologically implausible values were set to missing rather than removing the patient. Derived fields supplied with the extract that contained imputed values (for example, creatinine clearance set to a fixed value when creatinine was missing) were discarded and recalculated from source fields; creatinine clearance was recalculated with the Cockcroft-Gault equation for description only (kidney function enters the models as serum creatinine and renal failure or dialysis status). Binary comorbidity fields in which missing values had been recorded as absent were analysed under that registry convention. Left ventricular function was analysed as preserved (ejection fraction 50% or more) or impaired, because the recorded categories did not reliably distinguish moderate from poor function. Intra-aortic balloon pump use was assigned to the pre-, intra- or post-operative model according to its recorded timing. All cleaning rules and the number of records affected are in ST3, and the columns not used, with reasons, in ST4.

**Validation lock.** The 2017-19 records could be loaded only through a deliberate switch in the analysis code, used for cohort description and for evaluating the locked models. No 2017-19 outcome influenced any modelling choice.

## S2. Prediction moments and predictors

Three models were developed, one for each moment at which a prediction could be made: **pre-operative** (26 predictors), **intra-operative**, applied at the end of surgery (34: the pre-operative predictors plus operative details) and **early post-operative** (36: adding early complications that are known within hours of surgery, namely return to theatre for bleeding or tamponade and post-operative balloon pump insertion). Each model uses only information recorded by its moment. A fourth, **complication-aware** model (40 predictors, adding later complications) was fitted as a ceiling analysis only: its predictors are not known at any usable moment, so it is reported in the supplement and takes no part in the stage comparison.

Predictors were chosen in advance from clinical knowledge and the registry dataset; no data-driven variable selection was used, because outcome-driven selection gives optimistic performance unless repeated inside every resample and may discard predictors with curved effects [R61-R63]. Pairs of related predictors (for example previous and recent myocardial infarction; unstable angina and Canadian Cardiovascular Society class) were kept when they carry different information (SF4). Every variable, its definition, coding, reference level, the stage models that use it and its missingness are listed in ST1, and category counts in ST2. For the training-period comparison, all-era predictor sets (25, 32 and 33 predictors) excluded predictors not recorded before 2012.

## S3. Missing predictor values

Missing predictor values were imputed with a chained-equation imputer written for this study, matched to each variable's type as in the standard mice software [R57]: logistic regression for yes/no predictors, multinomial logistic regression for categories with more than two levels, and linear regression for numbers (creatinine on the log scale; filled values capped at the range seen in the training patients). Imputation was **stage-ordered**: pre-operative predictors were filled from pre-operative information only, operative predictors from pre-operative and operative information, and so on, so that a patient's pre-operative values are identical in every stage's model. The loop started from the median or most common value and repeated until the filled values stopped changing (at most 10 rounds).

The imputer is **deterministic** (each gap receives its predicted value or most likely category), so that a patient entered twice receives the same risk. The **outcome was never used**, because it is unknown when the model is applied; omitting the outcome is preferred when predictors may be missing at deployment [R4], at the known cost of some attenuation of associations [R10]. The imputer was fitted on the training patients of every resample and applied unchanged to the held-out patients, and the final imputer, fitted on all 2012-16 patients, was applied unchanged to 2017-19 [R5]. It is part of every saved model. The evidence behind these choices (missingness by year, outcome rates when each predictor is missing, balloon pump recording, co-missingness and complete cases) is in ST5 and ST5b. Multiple imputation and complete-case analysis were pre-specified sensitivity analyses (S12). Machine-learning models received the same imputed data as the regressions, so that differences between algorithms reflect the algorithms rather than their missing-data handling.

## S4. Models and their inputs

Six models were fitted at every stage (inclusion rule and candidates in ST8b):

- **EuroSCORE II (recalibrated)**, the benchmark: logistic recalibration (intercept and slope) of the logit of the published EuroSCORE II risk on development data; a patient without a recorded EuroSCORE II received the training median.
- **Logistic regression** with straight-line terms, as a comparator.
- **Ridge-penalised logistic regression with restricted cubic splines**, the default headline model: four knots at the 5th, 35th, 65th and 95th percentiles of the training data for every continuous predictor [R46], and categorical predictors as indicator columns against a clinical reference level. All columns were standardised before fitting, as in the reference software glmnet [R43], so that the penalty does not depend on units.
- **XGBoost** (gradient-boosted trees).
- **Random forest** (bagged trees).
- **Multilayer perceptron** (neural network, scikit-learn).

Tree models received numbers unchanged, ordered categories as their ordered codes and unordered categories as indicator columns; the neural network received the same columns standardised. No interaction terms were specified for the regressions. Continuous predictors were never dichotomised [R47].

## S5. Tuning, calibration and internal validation

**Nested cross-validation.** The 2012-16 patients were divided once into five outer folds, stratified on outcome and discharge status (fixed seed). In each of the five rounds, four folds formed the learning part and one fold was held out. Within the learning part, one stratified 75/25 split was used for tuning; the winning settings were then refitted on the whole learning part and applied to the held-out fold. Pooling the five held-out folds gave out-of-fold predictions for every development patient, from which internal performance was estimated (ST9, SF15-SF18). The imputer and model inputs were refitted inside every learning part, so no information from held-out patients entered the models [R20].

**Hyperparameter searches.** XGBoost, random forest and the neural network were each tuned with 30 trials of Optuna's tree-structured Parzen estimator with default settings and a fixed seed per search [R25, R26, R60]; the objective was log loss on the 25% inner scoring part, trials ran one after another so the search is exactly reproducible, and no pruning was used. Search spaces were written in the code repository before any tuning and follow published benchmark spaces [R21, R58, R59] (ST8): XGBoost tuned learning rate, maximum depth, minimum child weight, row and column subsampling and the L2 penalty, with up to 2,000 trees and early stopping after 50 rounds without improvement; random forest (500 trees) tuned the share of predictors per split, minimum leaf size and the share of patients per tree; the neural network tuned the number of layers (1-4), neurons per layer, learning rate, batch size and weight decay, with up to 200 epochs and early stopping on the inner log loss. The ridge penalty was chosen from 30 values between 10 and 0.00001 (glmnet scale) by the same inner log loss. Every trial of every search is listed in ST8c.

**Calibration step.** Because tree ensembles in particular can produce distorted probabilities [R23], the three machine-learning models had a post-hoc calibration step (none, Platt scaling or isotonic regression) scored in every trial on held-out inner predictions; the option with the lowest log loss was kept, the simpler one when tied to three decimals. In the final fit, the calibrator was trained on five-fold cross-fitted predictions of the learning data and saved with the model. Regressions received no calibration step.

**Fit failures.** A regression that did not converge was refitted with more iterations and then with a more robust solver; a failed tuning trial counted towards the budget. No model failed in the main analysis.

**Final models.** For each stage, every model was tuned once more on a 75/25 split of all 2012-16 patients, fitted on all of them, saved and fingerprinted (SHA-256) before the 2017-19 data were opened (ST8).

## S6. Choice of headline model and judging stage-to-stage gains

**Headline model.** At each stage separately, ridge regression with splines was the default. A machine-learning model replaced it only if, on the 2012-16 out-of-fold predictions, its AUC was higher with the 95% CI of the difference above zero, its integrated calibration index was not higher, and its net benefit was not lower at any of the four key thresholds. If several passed, the highest internal AUC won. Regression was the default because machine learning has often shown no advantage over logistic regression for clinical prediction [R18]; the rule guards against choosing a model by chance, and selection is kept separate from evaluation [R20] (ST9).

**Stage-to-stage verdict.** A later prediction moment was judged to add value when, on 2017-19 patients, the AUC difference between the headline models of the two moments had a 95% CI above zero **and** net benefit was higher at all four key thresholds (10%, 15%, 20% and 30%) than both the earlier moment's model and the flag-all and flag-none strategies. Comparing the same algorithm across moments was a secondary analysis. Changes in the AUC of nested models were judged with bootstrap confidence intervals rather than significance tests designed for other purposes [R15, R16]; reclassification indices were not used [R17] (Table 2, ST11).

## S7. Performance measures and uncertainty

**Measures.** Discrimination: AUC; precision-recall AUC (average precision), shown next to the event rate. Calibration: calibration-in-the-large (intercept with the logit of the predicted risk as offset; ideal 0), calibration slope (ideal 1), and lowess-smoothed calibration curves with the integrated calibration index (ICI), E50 and E90 [R56], computed with lowess settings matching the original description (span 2/3, no robustness iterations). Overall accuracy: Brier score and scaled Brier score. Clinical usefulness: net benefit and decision curves [R29, R37-R40] between 10% and 30% threshold risk, chosen with clinical co-authors because the intended actions (a planning flag for beds and discharge support) are low-harm; classification measures (sensitivity, specificity, positive and negative predictive value) when the top 10% and top 20% of predicted risk are flagged, an equal workload for every model that needs no fixed risk cut-off. When patients tied at the cut-off, the remaining places were shared equally among them, so exactly 10% or 20% were flagged. The essential exhibits recommended for evaluating prediction models were reported [R14].

**Confidence intervals.** All 95% CIs come from 2,000 bootstrap resamples of patients with replacement, the same resamples for every model scored on a dataset (paired), with percentile limits [R27, R28, R31]. Differences between models or stages were taken within each resample. When a confidence limit that decided a verdict lay within two Monte Carlo standard errors of zero, the interval was recomputed with 10,000 resamples.

## S8. Temporal evaluation and model updating

The locked models were applied once to the 2017-19 patients after their fingerprints were checked. Performance is reported overall (Table 2, ST10, ST11, SF5-SF8) and by year (ST15, SF9, SF13b) to show drift [R36]. A pre-specified updating scenario re-estimated the intercept (and, separately, intercept and slope) on 2017 patients and evaluated the updated models on 2018-19 (ST12) [R32-R35].

## S9. Sample size

The minimum development sample size was calculated with the method of Riley et al. for binary outcomes [R1], assuming a Cox-Snell R2 derived from an anticipated C-statistic [R3]; the available patients were at least 9.7 times the requirement at every stage. The evaluation sample size was calculated for precise estimation of the observed/expected ratio, calibration slope and C-statistic [R2]; the 2017-19 cohort was 5.7 times the requirement (ST7).

## S10. Training-period comparison

To test whether older data help, ridge regression and XGBoost were also trained on two further training sets and tested on the same 2017-19 patients: B, 2012-16 with the all-era predictor sets; C, 1996-2016 with the all-era predictor sets. Each set had its own imputer, was tuned and fitted as in S5, and was compared with set A (the main models) with paired bootstrap differences, descriptively (ST14, SF10).

## S11. Explainability

The headline XGBoost models were explained with exact TreeSHAP values computed by XGBoost on the log-odds scale [R53, R54, R64], for a fixed random sample of 5,000 patients from 2017-19 (predictors only; outcomes not used). The SHAP values of each patient were checked to add up to the model's own log-odds. Importance is the mean absolute SHAP value per original predictor (indicator columns summed) and, because standard SHAP splits credit between related predictors [R55], also per pre-specified clinical group (SF14). Dependence plots show each predictor over its 1st-99th percentile only (SF12, SF12b). For the ridge regressions, odds ratios with 95% CIs are reported for categorical predictors (ST13) and continuous predictors are shown as spline curves relative to their median (SF11). Explanations describe what the models use for prediction, not causal effects.

## S12. Sensitivity analyses

Six pre-specified sensitivity analyses were run (ST16, SF19):

- **S1** composite outcome (stay over 10 days or in-hospital death) and **S2** all patients with a known stay (deaths counted by their stay);
- **S3** cut-off over 7 days and **S4** over 14 days;

for S1-S4, XGBoost was re-tuned and EuroSCORE II recalibrated for each outcome at each stage, fitted on all 2012-16 patients, locked and tested once on 2017-19, with stage-to-stage AUC differences;

- **S5** multiple imputation for the two regressions: 10 imputations drawn from the same imputation models, each refitted on a bootstrap resample of the development patients and filling gaps with random draws (prediction plus residual noise for numbers; a level drawn from the predicted probabilities for categories), outcome not used [R6-R8, R57, R67]; the 2017-19 data were imputed with the matching imputation models and each patient's risks averaged; coefficients were pooled with Rubin's rules. The number of imputations followed von Hippel's rule for a fraction of missing information up to 0.21 [R6]; where it was exceeded (the complication-aware stage), the number was increased to what the rule required (48);
- **S6** complete cases: all six models re-developed on patients with every predictor recorded and tested on complete-case 2017-19 patients alongside the main models on the same patients.

## S13. Fairness

Performance was compared across sex, age band (under 60, 60-69, 70-79, 80 and over) and urgency (elective, urgent, emergency or salvage) for XGBoost, ridge regression and EuroSCORE II at each stage (ST17, SF13a): AUC, observed/expected ratio, calibration-in-the-large and slope, and the rates of missed long stays and flagged short stays. The flagging cut-off was set once over all 2017-19 patients, as it would be in use, so that differences in error rates between groups reflect the model [R66]. Differences from a reference group (men; 60-69; elective) are reported with bootstrap CIs, descriptively. Ethnicity, deprivation and hospital were not available in the extract.

## S14. Reporting, software and reproducibility

The study is reported following TRIPOD+AI [R19] (ST18) and was self-assessed with PROBAST+AI [R65] (ST19). Analyses used Python 3.13 with pandas 3.0.5, NumPy 2.5.3, scikit-learn 1.9.1, XGBoost 3.4.1, Optuna 5.0.0 and statsmodels 0.15.0; versions are pinned in the code repository. Every random step used fixed seeds, every saved model was fingerprinted, and all exhibits in this supplement are produced by the analysis scripts from aggregate outputs. Counts of 1-9 patients are shown as '<10' (with secondary suppression where needed) and rates based on such counts are not shown, in line with the data provider's disclosure rules; minimum and maximum values are not reported. The data are held by NICOR and are not shareable by the authors.

## S15. Changes after the protocol was frozen

The protocol and statistical analysis plan were frozen on 7 October 2026 (protocol v1.0, repository commit e89dfa1), before any model was fitted to the study data. The changes made afterwards are listed here with their reasons (TRIPOD+AI item 18c). None was prompted by 2017-2019 results, and none changed an outcome, predictor, model, tuning criterion, performance measure or decision rule.

- **No pruning of tuning trials** (protocol 9.4 listed pruning). Every trial was run to its end and judged on its calibrated log loss. A pruner needs step-by-step scores, which XGBoost and the neural network give but random forest and ridge regression do not, so the algorithms would have been treated unequally; and step-by-step scores come before the calibration step, so pruning could remove a setting that is best after calibration. Poor runs were still cut short within a trial by early stopping.
- **Calibration step scored within every trial.** Rather than drawing the calibration option (none, Platt scaling or isotonic regression) at random for each trial, all three were scored on the same trained model in every trial and the best kept, with ties going to the simpler option. This is how the pre-specified tie rule (protocol section 12) could be applied at all.
- **Tuning trials run one after another within each search**, with separate searches run in parallel, so that a fixed seed gives exactly the same result on every run.
- **Neural network without dropout.** The scikit-learn multilayer perceptron has no dropout; overfitting was controlled by weight decay and early stopping instead.
- **Missing EuroSCORE II.** A few patients had no recorded EuroSCORE II; they received the training median so that every patient had a benchmark prediction and all comparisons were paired on identical patients (S4).
- **Presentation.**
  - The model the protocol calls the end-of-surgery model (Model 2) is called the intra-operative model in the paper, at a co-author's request; it is the same model, applied at the end of surgery.
  - ROC curves were added to the main text (Figure 2), with versions of every curve plot for the 2012-2016 out-of-fold predictions in the supplement (SF15-SF18).
  - The axis of every calibration plot runs from 0% to 70% instead of 0% to 60%, fixed before any 2017-2019 result, so that the high-risk region is not cut off.
  - The planned single Table 2 with a verdict panel was split into Table 2 (performance) and Table 3 (stage verdicts). Table 3 also shows, as descriptive information, net-benefit differences with paired bootstrap confidence intervals and the comparisons with EuroSCORE II.
  - SHAP beeswarm plots were moved from the supplement to the main text (Figure 5), and the dependence plots for age, creatinine and bypass time are shown in the main text next to the ridge-regression spline curves (Figure 6).

The multiple-imputation sensitivity analysis (S5) used more than 10 imputations at the complication-aware stage, as the protocol's pre-specified rule required (S12). This was an application of the protocol, not a change to it.

## S16. Further reporting items (TRIPOD+AI)

- **Outcome assessment (items 8b, 8c).** Length of stay is recorded routinely by hospital staff for the national audit. It is an objective count of days from operation to discharge, recorded without knowledge of any prediction model, so assessor qualifications and blinding do not arise.
- **Subjective predictors (item 9c).** Angina class, breathlessness class, urgency of operation and left ventricular function grade involve clinical judgement. They were recorded by the operating unit before or during surgery, before the outcome was known.
- **Class imbalance (item 13).** No oversampling, undersampling or class weights were used, so predicted probabilities reflect the natural outcome rate (about 15%); this is needed for calibrated risks and for net benefit.
- **Missing inputs in use (item 27a).** Each saved model contains its stage-ordered imputer, fitted on all 2012-2016 patients, so a missing predictor in a new patient is filled in the same deterministic way as in this study (S3). Inputs outside the development range should be checked before use.
- **Users and expertise (item 27b).** The intended users are bed managers, intensive care and ward teams, and discharge planners. They need no statistical expertise: the inputs are fields already recorded for the national audit, and the output is a risk of prolonged stay for planning, not for treatment decisions.
- **Patient and public involvement (item 19).** Patients and the public were not involved in the design, conduct or reporting of this study.

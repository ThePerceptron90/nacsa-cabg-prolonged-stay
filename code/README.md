# Shared helper module

`nacsa_common.py` is imported by every analysis step from Step 4 onwards. It holds:

- paths and fixed settings (length-of-stay cut-off, small-count threshold, periods, age bands)
- the expected schema of `nacsa_clean.parquet` (63 columns, 377,301 rows); loading stops on any mismatch
- `load_clean()`: the only way scripts read the data. It adds the Step 4 variables
  (`diabetes_any`, `stroke_any`, `smoking_ever`, `outcome_all`, `age_band`), renames the
  composite outcome to `outcome_composite`, applies the validation lock (2017-2019 only with
  `allow_validation=True`) and filters on the chosen outcome
- the variable registry `VARIABLES`: type, stage, label, levels, reference level and source of
  every candidate predictor and the EuroSCORE II benchmark
- privacy helpers: counts 1-9 shown as "<10", secondary suppression, rates hidden when events or
  non-events are 1-9

In Colab, copy it to `MyDrive/nacsa/code/nacsa_common.py`.

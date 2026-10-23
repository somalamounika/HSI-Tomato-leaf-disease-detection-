# Leaf model report (inoculated vs control)

Training time points: 2hr, d1, d2, d3. **bi is never used for training.**
Validation: leave-one-unit-out (group = run + unit).
A leave-one-time-point-out evaluation is also reported.
Runs map to different time points in this subset; no train-on-one-run/test-on-the-other split.

WARNING: leaf order within each scan is UNCONFIRMED. The leaf_map must be checked against the dataset's own metadata or confirmed with the dataset authors (Song Li, songli@vt.edu). Do not treat assumed unit/group labels as ground truth.

The source paper reported roughly 0.55–0.64 accuracy at the early/pre-symptomatic stage. Modest numbers at d1–d3 are expected, not a bug.

Selected model: **lda**
Chosen for highest out-of-fold balanced accuracy (0.500 if finite). Ties break toward the simpler model (logistic regression, then LDA, then LightGBM). No hyperparameter search was run on the test folds.

## Overall leave-one-unit-out metrics

| model | n | accuracy | balanced_accuracy | F1 | ROC AUC | permutation p (AUC) |
|---|---:|---:|---:|---:|---:|---:|
| logistic_regression | 24 | 0.458 | 0.458 | 0.381 | 0.458 | 0.6020 |
| lda | 24 | 0.500 | 0.500 | 0.455 | 0.451 | 0.5920 |
| lightgbm | 24 | 0.417 | 0.417 | 0.417 | 0.333 | 0.7861 |

## Leave-one-time-point-out (overall)

| model | n | accuracy | balanced_accuracy | F1 | ROC AUC |
|---|---:|---:|---:|---:|---:|
| logistic_regression | 24 | 0.583 | 0.583 | 0.500 | 0.590 |
| lda | 24 | 0.583 | 0.583 | 0.500 | 0.611 |
| lightgbm | 24 | 0.500 | 0.500 | 0.538 | 0.441 |

## Per time point (leave-one-unit-out)

| model | time_point | n | accuracy | balanced_accuracy | F1 | ROC AUC |
|---|---|---:|---:|---:|---:|---:|
| logistic_regression | d1 | 8 | 0.500 | 0.500 | 0.333 | 0.562 |
| logistic_regression | d2 | 8 | 0.500 | 0.500 | 0.500 | 0.438 |
| logistic_regression | d3 | 8 | 0.375 | 0.375 | 0.286 | 0.438 |
| lda | d1 | 8 | 0.500 | 0.500 | 0.333 | 0.562 |
| lda | d2 | 8 | 0.500 | 0.500 | 0.500 | 0.438 |
| lda | d3 | 8 | 0.500 | 0.500 | 0.500 | 0.438 |
| lightgbm | d1 | 8 | 0.500 | 0.500 | 0.500 | 0.250 |
| lightgbm | d2 | 8 | 0.500 | 0.500 | 0.500 | 0.438 |
| lightgbm | d3 | 8 | 0.250 | 0.250 | 0.250 | 0.312 |

## bi check (fold models applied to before-inoculation leaves)

Expected AUC of P(infected) between groups at bi is about 0.5.

| model | n_bi | AUC |
|---|---:|---:|
| logistic_regression | 0 | nan |
| lda | 0 | nan |
| lightgbm | 0 | nan |

## Features used

NDVI, GNDVI, NDRE, PRI, PSSRa, CTR2, SIPI, mND705, mSR705, ARI, REP, WI, NDVI_median, GNDVI_median, NDRE_median, PRI_median, PSSRa_median, CTR2_median, SIPI_median, mND705_median, mSR705_median, ARI_median, REP_median, WI_median, NDVI_std, NDVI_p5, NDVI_p95, PRI_std, PRI_p5, PRI_p95, NDVI_low_frac, RE_deriv_peak_nm, RE_deriv_peak_amp


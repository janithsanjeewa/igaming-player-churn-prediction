# Frozen final model specification

All model-development decisions were frozen before evaluating the held-out test period.

## Selected family and exact estimator settings

**XGBoost**; demographics **removed**;
calibration **raw**. Diagnostic threshold **0.50** only.
Selection evidence and uncertainty are documented in `final_validation_decision.md`.

Every estimator parameter (including defaults) is frozen below; `"NaN"` denotes
the numerical missing sentinel rather than a literal string passed to XGBoost.

```json
{
  "objective": "binary:logistic",
  "base_score": null,
  "booster": null,
  "callbacks": null,
  "colsample_bylevel": null,
  "colsample_bynode": null,
  "colsample_bytree": 0.9,
  "device": null,
  "early_stopping_rounds": null,
  "enable_categorical": true,
  "eval_metric": "logloss",
  "feature_types": null,
  "feature_weights": null,
  "gamma": 0.1,
  "grow_policy": null,
  "importance_type": "gain",
  "interaction_constraints": null,
  "learning_rate": 0.03,
  "max_bin": null,
  "max_cat_threshold": null,
  "max_cat_to_onehot": null,
  "max_delta_step": null,
  "max_depth": 3,
  "max_leaves": null,
  "min_child_weight": 10,
  "missing": "NaN",
  "monotone_constraints": null,
  "multi_strategy": null,
  "n_estimators": 400,
  "n_jobs": -1,
  "num_parallel_tree": null,
  "random_state": 42,
  "reg_alpha": 0.1,
  "reg_lambda": 10,
  "sampling_method": null,
  "scale_pos_weight": null,
  "subsample": 0.8,
  "tree_method": "hist",
  "validate_parameters": null,
  "verbosity": null
}
```

## Exact raw predictor set (42 columns)

- `active_days_30d`
- `bets_30d`
- `active_days_14d`
- `bets_14d`
- `active_days_7d`
- `bets_7d`
- `avg_bets_per_active_day_30d`
- `avg_bets_per_active_day_7d`
- `days_since_last_active_bet`
- `max_gap_between_active_days_30d`
- `mean_gap_between_active_days_30d`
- `stake_30d`
- `stake_14d`
- `stake_7d`
- `winnings_30d`
- `winnings_14d`
- `winnings_7d`
- `net_loss_30d`
- `net_loss_7d`
- `avg_stake_per_bet_30d`
- `avg_stake_per_active_day_30d`
- `max_daily_stake_30d`
- `median_daily_stake_30d`
- `std_daily_stake_30d`
- `active_days_7d_share`
- `bets_7d_share`
- `stake_7d_share`
- `bets_recent15`
- `bets_early15`
- `bets_change_15d`
- `bets_change_pct_15d`
- `stake_recent15`
- `stake_early15`
- `stake_change_15d`
- `stake_change_pct_15d`
- `active_days_recent15`
- `active_days_early15`
- `active_days_change_15d`
- `active_days_change_pct_15d`
- `days_since_registration`
- `days_since_first_pay`
- `days_since_first_casino_activity`

Identifiers, snapshot date and outcome are excluded. The machine-readable JSON
also records the original-TRAIN transformed-column names for artifact validation.
Feature definitions remain those in `docs/feature_dictionary.md`.

## Preprocessing and missing values

Numerical: median imputation learned from the fitting partition plus missing-value
indicators for columns missing during that fit. No scaling. Categorical:
excluded. Dense output preserves numerical zeros;
unlisted columns are dropped. No target encoding or class reweighting.
For the future safe pre-test refit, fit the same preprocessing algorithm afresh;
indicator columns/category levels may change with safe fitting data, but the raw
predictor set and treatment rules stay frozen.

## Probability mapping

Selected method: **raw**. Candidates were raw, TRAIN-only
purged-OOF sigmoid and isotonic; the exact fold/selection rules and aggregate
validation evidence are recorded in the decision report and CSVs. Validation
logistic intercept/slope fits were descriptive diagnostics, never applied maps.
If selected, keep the TRAIN-OOF calibration mapping frozen through the future
base-estimator pre-test refit; do not use validation/test labels to recalibrate.

## Temporal task definition

- Observation window: 30 inclusive calendar dates [S-29,S].
- Outcome: 60 calendar dates (S,S+60]. Churn=1 means no positive-bet day in this interval.
- Development TRAIN: 2005-03-31 through 2005-12-31.
- VALIDATION: 2006-03-31 through 2006-05-31.
- Development embargoes: January-February and June-July 2006.
- Sealed TEST snapshots: 2006-08-31 through 2006-11-30.
- Same-player overlap across temporal partitions is allowed by the existing known-player design.

## Evaluation and business policy

Primary discrimination summaries: average precision (PR-AUC) and ROC-AUC; primary
probability-quality measure: Brier. Also report log loss, diagnostic precision,
recall, F1 and specificity, and lift/recall at 10% and 20%. Uncertainty uses 1,000
whole-UserID cluster bootstrap replicates, seed 42, percentile 95% intervals.

Output probability/risk scores. Capacity scenarios: top 5/10/15/20/25/30%, with
10/20% as standard comparisons, not universal campaign budgets. Select ceil(n*f)
rows from descending scores, using stable original-panel order for ties. These
are observation-level evaluations; a future campaign must separately define
one-contact-per-player rules and eligibility.

Diagnostic confusion matrix threshold is **0.50**, frozen for test reporting.
The best validation F1 on the specified grid is a descriptive reference only.
No profit-maximizing threshold or ROI is claimed without economic/treatment data.

Risk bands: bottom 50%, 50th-80th, 80th-90th, top 10%. Descending-rank cuts use
ceil(n*.1), ceil(n*.2), ceil(n*.5), then n; bands are disjoint, label-free and
relative to the scored cohort. Identical scores can span bands. They are
presentation/prioritization tools, not clinical/regulatory categories.

## Future pre-test refit: not executed here

First test snapshot 2006-08-31 has observation start **2006-08-02**. Require
S+60 < 2006-08-02; the latest safe calendar date is **2006-06-02**, and the last
safe month-end is **2006-05-31**. Include eligible monthly snapshots from
**2005-03-31 through 2006-05-31**,
including January-February 2006; exclude June-July. This yields
**14,242 observations**, **3,651 players**,
and **35.9009%** churn. May's labels end 2006-07-30.

Fit the unchanged selected configuration/preprocessing to these safe snapshots
only in the next notebook. Keep any selected calibration map frozen as above.
Persist and verify the resulting artifact, then evaluate the sealed test once.
Report the predefined metrics, calibration/ROC/PR/gains/lift plots, confusion
matrix and cluster-bootstrap intervals. No post-test tuning is permitted.

## Known limitations

Historical data and temporal drift; repeated-player dependence and calendar shocks;
correlated/proxy predictors; validation-selection optimism; limited calibration
transportability; unavailable protected-group analysis and campaign economics.
The model predicts inactivity, not gambling harm or customer value. Associations
are not causal explanations or evidence that contacting players changes outcomes.
Governance review is required before operational use. TEST remains sealed.

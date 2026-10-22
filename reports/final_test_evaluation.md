# Final held-out test evaluation

The held-out test set was evaluated once after all model-family, feature, calibration, and business-policy decisions were frozen.

No model or policy decisions were changed in response to test results.

## A. Frozen model specification

Demographic-free XGBoost classifier with 42 historical numerical predictors. Country, Language, Gender, UserID, snapshot_date and churn_60d are not model inputs. The target is constructed separately. No calibration: probabilities are the raw classifier outputs. Diagnostic threshold remains **0.50**. Exact predictor definitions and all estimator defaults remain those in `final_model_specification.json` / `.md` and `docs/feature_dictionary.md`.

```json
{
  "n_estimators": 400,
  "learning_rate": 0.03,
  "max_depth": 3,
  "min_child_weight": 10,
  "subsample": 0.8,
  "colsample_bytree": 0.9,
  "reg_alpha": 0.1,
  "reg_lambda": 10,
  "gamma": 0.1,
  "random_state": 42,
  "objective": "binary:logistic",
  "eval_metric": "logloss",
  "tree_method": "hist"
}
```

Numerical preprocessing is median imputation with add_indicator=True, no scaling and dense output. The complete pipeline was fitted once to final-refit rows only. The same rule may learn different missing-indicator columns on the expanded safe history; it does not change raw feature definitions. Transformed refit and test values were finite. Test transformation did not change imputation statistics.

## B. Final refit period

**2005-03-31 through 2006-05-31**: **14,242 observations**, **3,651 players**, **35.9009%** inactivity. January-February 2006 are included after selection is frozen. June-July are excluded. Every refit snapshot satisfies S+60 < 2006-08-02. May 31 + 60 days = July 30, before the first test observation window begins August 2. The final fitted pipeline is local at `models/final_xgboost_churn_model.joblib` (Git-ignored). No test labels entered fitting.

## C. Held-out test period

Exactly the four snapshots **August 31, September 30, October 31 and November 30, 2006**: **2,838 observations**, **1,455 players**, **34.0733%** inactivity. No other snapshots enter test scoring or aggregate results. Panel shape, binary target, predictor availability and unique player-snapshot keys were checked before fitting; the data SHA-256 matches the frozen Day 10 artifact.

## D. Main test metrics

| metric | value |
| --- | --- |
| roc_auc | 0.748954 |
| average_precision | 0.551004 |
| log_loss | 0.545688 |
| brier_score | 0.186973 |
| accuracy | 0.696970 |
| balanced_accuracy | 0.645013 |
| precision | 0.564848 |
| recall | 0.481903 |
| f1 | 0.520089 |
| specificity | 0.808124 |
| true_positives | 466.000000 |
| false_positives | 359.000000 |
| true_negatives | 1512.000000 |
| false_negatives | 501.000000 |
| predicted_positives | 825.000000 |
| recall_at_top10pct | 0.176836 |
| precision_at_top10pct | 0.602113 |
| lift_at_top10pct | 1.767110 |
| recall_at_top20pct | 0.345398 |
| precision_at_top20pct | 0.588028 |
| lift_at_top20pct | 1.725775 |
| threshold | 0.500000 |
| mean_predicted_probability | 0.348123 |
| observed_churn_prevalence | 0.340733 |
| ece | 0.015567 |

ROC-AUC and average precision summarize ranking; Brier and log loss assess probability quality. Accuracy is supplementary. All binary metrics and the confusion matrix use the unchanged 0.50 threshold, with p >= 0.50 positive. No test threshold search was performed.

## E. Whole-player cluster-bootstrap uncertainty

| metric | estimate | bootstrap_mean | ci95_low | ci95_high | valid_replicates | undefined_replicates | requested_replicates | random_state |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| roc_auc | 0.748954 | 0.748706 | 0.730040 | 0.767841 | 1000 | 0 | 1000 | 42 |
| average_precision | 0.551004 | 0.552031 | 0.517576 | 0.586999 | 1000 | 0 | 1000 | 42 |
| brier_score | 0.186973 | 0.187107 | 0.180521 | 0.194018 | 1000 | 0 | 1000 | 42 |
| precision | 0.564848 | 0.565008 | 0.530098 | 0.600709 | 1000 | 0 | 1000 | 42 |
| recall | 0.481903 | 0.481815 | 0.451438 | 0.512001 | 1000 | 0 | 1000 | 42 |
| f1 | 0.520089 | 0.519949 | 0.492183 | 0.546057 | 1000 | 0 | 1000 | 42 |
| lift_at_top10pct | 1.767110 | 1.765133 | 1.598260 | 1.924276 | 1000 | 0 | 1000 | 42 |
| recall_at_top10pct | 0.176836 | 0.176795 | 0.159916 | 0.192588 | 1000 | 0 | 1000 | 42 |
| lift_at_top20pct | 1.725775 | 1.732001 | 1.623206 | 1.842778 | 1000 | 0 | 1000 | 42 |
| recall_at_top20pct | 0.345398 | 0.346649 | 0.324985 | 0.368815 | 1000 | 0 | 1000 | 42 |

1,000 replicates, random_state=42. Each draw samples 1,455 whole UserID clusters with replacement, keeping every snapshot and repeated-cluster multiplicity. Row counts can vary. Capacity counts and prevalence are recalculated per draw. Intervals are the 2.5th and 97.5th percentiles, conditional on the fixed fitted model. They do not capture training uncertainty, earlier model selection, shared calendar shocks or future temporal drift. Undefined metrics are excluded per metric and valid counts are reported. ROC-AUC requires both classes; AP requires at least one positive; capacity recall/lift require positives. Diagnostic precision/recall/F1 use the existing zero_division=0 convention. Brier remains defined in single-class draws.

## F. Validation versus test generalization

| metric | frozen_validation | final_test | test_minus_validation |
| --- | --- | --- | --- |
| roc_auc | 0.744785 | 0.748954 | 0.004169 |
| average_precision | 0.565764 | 0.551004 | -0.014760 |
| brier_score | 0.190772 | 0.186973 | -0.003800 |
| lift_at_top10pct | 1.793262 | 1.767110 | -0.026152 |
| recall_at_top10pct | 0.179819 | 0.176836 | -0.002983 |
| lift_at_top20pct | 1.700386 | 1.725775 | 0.025389 |
| recall_at_top20pct | 0.340233 | 0.345398 | 0.005165 |

**Broadly stable**, descriptively rather than as a pass/fail verdict. Test ROC-AUC improves by 0.0042 and Brier improves by 0.0038; average precision declines by 0.0148. Top-10% targeting is slightly weaker, while top-20% targeting is slightly stronger. Test prevalence is lower than validation (34.07% versus 35.39%), affecting precision and average precision; that does not fully explain or remove the observed AP decline. The test model was refitted on more safe history, so the comparison mixes population/time change with the planned refit. No inferential validation-versus-test difference interval or new pass/fail threshold is claimed, and the selection is not reconsidered.

## G. Fixed-budget targeting

| target_fraction | selected_observations | actual_fraction | churn_observations_captured | precision | recall | lift |
| --- | --- | --- | --- | --- | --- | --- |
| 0.050000 | 142 | 0.050035 | 80 | 0.563380 | 0.082730 | 1.653437 |
| 0.100000 | 284 | 0.100070 | 171 | 0.602113 | 0.176836 | 1.767110 |
| 0.150000 | 426 | 0.150106 | 258 | 0.605634 | 0.266805 | 1.777444 |
| 0.200000 | 568 | 0.200141 | 334 | 0.588028 | 0.345398 | 1.725775 |
| 0.250000 | 710 | 0.250176 | 416 | 0.585915 | 0.430196 | 1.719574 |
| 0.300000 | 852 | 0.300211 | 479 | 0.562207 | 0.495346 | 1.649992 |

The frozen 5/10/15/20/25/30% capacities use ceil(n*f), stable original-panel input-order ties, and no label-based tie-breaking. Precision is the observed inactivity rate among selected observations; recall is the share of all inactivity outcomes captured; lift is precision / overall test prevalence. These are player-snapshot observations, not deduplicated campaign recipients. No optimal capacity, financial threshold, retention treatment effect, customer value or ROI is inferred.

## H. Frozen percentile risk bands

| band | observations | actual_fraction | average_predicted_risk | actual_churn_rate | lift |
| --- | --- | --- | --- | --- | --- |
| Very high risk (top 10%) | 284 | 0.100070 | 0.612180 | 0.602113 | 1.767110 |
| High risk (80th-90th) | 284 | 0.100070 | 0.558605 | 0.573944 | 1.684439 |
| Medium risk (50th-80th) | 851 | 0.299859 | 0.468183 | 0.460635 | 1.351893 |
| Low risk (bottom 50%) | 1419 | 0.500000 | 0.181147 | 0.169838 | 0.498449 |

Bands use TEST score ranking only: bottom 50%, 50th-80th, 80th-90th and top 10%. Descending-rank cuts use ceil(n*.1), ceil(n*.2), ceil(n*.5), then n. They are disjoint relative prioritization bands; tied scores can span bands. They are not clinical or regulatory categories. The test labels are used only to describe outcomes after band assignment.

## I. Raw test calibration and score overlap

Mean predicted probability **34.8123%**, observed inactivity **34.0733%**: mean overprediction **0.739 percentage points**. Brier **0.186973**, log loss **0.545688**, ECE **0.015567**.

ECE uses the same ten fixed equal-width bins as Day 10: [0,.1), ... [.9,1], including 1.0 in the last bin. ECE = sum(n_bin/n * abs(mean probability - observed frequency)); empty bins contribute zero. The reliability figure includes bin counts and marks sparse bins. No calibration model, slope map or test-label-derived mapping was fitted; raw probabilities remain frozen.

The class-conditional density plot shows substantial overlap: both active and inactive outcomes occur on each side of 0.50. The false positives and false negatives in the confusion matrix demonstrate that these scores cannot deterministically identify an individual outcome. No additional cutoff was selected from the distribution plot.

## J. Monthly descriptive stability

| snapshot_date | observations | unique_players | churn_prevalence | mean_predicted_risk | roc_auc | average_precision | brier_score | lift_at_top10pct | targeted_top10pct |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 2006-08-31 | 773 | 773 | 0.367400 | 0.356764 | 0.733147 | 0.566076 | 0.197025 | 1.640078 | 78 |
| 2006-09-30 | 709 | 709 | 0.328632 | 0.340287 | 0.773037 | 0.569036 | 0.177618 | 1.885752 | 71 |
| 2006-10-31 | 667 | 667 | 0.304348 | 0.347600 | 0.742053 | 0.506405 | 0.182922 | 1.863539 | 67 |
| 2006-11-30 | 689 | 689 | 0.358491 | 0.347000 | 0.750362 | 0.566271 | 0.189243 | 1.697941 | 69 |

September has the strongest ROC-AUC (0.7730); August the weakest (0.7331). October has the lowest observed prevalence (30.43%) and average precision (0.5064); predicted mean risk remains approximately 34-36% across months. Top-10% lift ranges from 1.6401 to 1.8858, based on 67-78 selected rows per month. Each monthly snapshot contains one row per player, but players repeat across months. These descriptive differences have no month-specific uncertainty adjustment and do not justify post-test tuning.

## K. Limitations and final governance

- The label is no positive-bet activity during the 60 days immediately after a snapshot. It predicts extended inactivity, not permanent departure; reactivation after the outcome window remains possible.
- The source is historical bwin Virtual Casino Gambling data (2005-2007), not a present-day operational sample. Temporal drift and historical cohort selection limit transferability.
- Same players can appear in refit and test, following the frozen future behavior of known players design; results do not establish unseen-player generalization. Player-cluster intervals do not account for shared calendar shocks.
- Validation was reused for model-family, feature and calibration decisions. The single held-out evaluation estimates the frozen procedure; it is not permission to iterate against this now-used test period.
- Correlated predictors and model explanations are associative, not causal. Demographics are excluded, but behavioral proxies can remain. No protected-group outcome analysis or fairness guarantee is available.
- Risk scoring does not measure gambling harm, customer value, causal retention impact or responsible-gambling status; it must not drive automated adverse decisions. Governance review, recent-data validation, monitoring and a retraining plan are required before real deployment.
- Capacity results count snapshots, not unique recipients. Campaign deduplication, contact costs and eligibility remain separate requirements.
- Test probabilities were generated in one predict_proba call from one final refit. Notebook reruns load the ignored local checkpoint, not a new model or prediction vector. Only aggregate reports and anonymous figures are exported under reports/. No commit or push was performed.

The held-out test set was evaluated once after all model-family, feature, calibration, and business-policy decisions were frozen.

No model or policy decisions were changed in response to test results.

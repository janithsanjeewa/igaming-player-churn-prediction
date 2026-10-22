# Final validation decision: Day 10

## Decision and scope

**Final model family: XGBoost.** Its stronger overall validation discrimination and probability quality, much smaller train-validation gaps, and established SHAP explanations outweigh Random Forest's small observed top-budget advantage. No financial utility function or universal contact capacity is assumed. This is the final development/validation decision before one sealed test evaluation.

Variant: **demographics removed**. Probability policy: **raw**. The exact specification and all estimator parameters are frozen in `final_model_specification.md` and `final_model_specification.json`. Logistic Regression remains the historical baseline.

## Validation reproduction and generalization

Original TRAIN: 2005-03-31 through 2005-12-31 (10,380 observations). Validation: 2006-03-31 through 2006-05-31 (2,184 observations, 1,310 players). All saved candidate metrics reproduce within 0.00005. No test predictor matrix or predictions were created.

| model | roc_auc | average_precision | log_loss | brier_score | recall_at_top10pct | precision_at_top10pct | lift_at_top10pct | recall_at_top20pct | precision_at_top20pct | lift_at_top20pct |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Random Forest | 0.7415 | 0.5594 | 0.5535 | 0.1911 | 0.1772 | 0.6256 | 1.7675 | 0.3441 | 0.6087 | 1.7198 |
| XGBoost | 0.7464 | 0.5630 | 0.5515 | 0.1906 | 0.1746 | 0.6164 | 1.7417 | 0.3376 | 0.5973 | 1.6875 |

Historical train-minus-validation ROC-AUC gaps: RF 0.0763, XGBoost 0.0398; average-precision gaps: RF 0.1265, XGBoost 0.0675. These favor XGBoost's generalization behavior, but are descriptive and do not establish future performance. Logistic baseline: ROC-AUC 0.7295, AP 0.5342, Brier 0.1963. XGBoost adds a dependency versus sklearn RF, but both use the same practical train-fitted preprocessing and are already implemented and explainable.

Raw final selected variant train/validation metrics (before any probability mapping):

| metric | train | validation | train_minus_validation |
| --- | --- | --- | --- |
| roc_auc | 0.7838 | 0.7448 | 0.0390 |
| average_precision | 0.6243 | 0.5658 | 0.0585 |
| brier_score | 0.1796 | 0.1908 | -0.0112 |

## Fixed-budget business comparison

| target_fraction | selected_observations | actual_fraction | churn_observations_captured | precision | recall | lift | model |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 0.0500 | 110 | 0.0504 | 72 | 0.6545 | 0.0931 | 1.8493 | Random Forest |
| 0.1000 | 219 | 0.1003 | 137 | 0.6256 | 0.1772 | 1.7675 | Random Forest |
| 0.1500 | 328 | 0.1502 | 196 | 0.5976 | 0.2536 | 1.6883 | Random Forest |
| 0.2000 | 437 | 0.2001 | 266 | 0.6087 | 0.3441 | 1.7198 | Random Forest |
| 0.2500 | 546 | 0.2500 | 324 | 0.5934 | 0.4191 | 1.6766 | Random Forest |
| 0.3000 | 656 | 0.3004 | 374 | 0.5701 | 0.4838 | 1.6108 | Random Forest |
| 0.0500 | 110 | 0.0504 | 73 | 0.6636 | 0.0944 | 1.8750 | XGBoost |
| 0.1000 | 219 | 0.1003 | 135 | 0.6164 | 0.1746 | 1.7417 | XGBoost |
| 0.1500 | 328 | 0.1502 | 203 | 0.6189 | 0.2626 | 1.7486 | XGBoost |
| 0.2000 | 437 | 0.2001 | 261 | 0.5973 | 0.3376 | 1.6875 | XGBoost |
| 0.2500 | 546 | 0.2500 | 326 | 0.5971 | 0.4217 | 1.6869 | XGBoost |
| 0.3000 | 656 | 0.3004 | 382 | 0.5823 | 0.4942 | 1.6453 | XGBoost |

Targeting units are player-snapshot observations, not unique campaign recipients. Each capacity selects ceil(n * fraction) observations; ties preserve original-panel input order without consulting labels. Precision is the observed inactivity rate among selected rows; recall is the share of inactivity outcomes captured; lift is precision divided by overall validation prevalence. A capacity must be chosen from actual operational resources, not inferred from these metrics.

## Paired cluster-bootstrap uncertainty

| metric | observed_difference | bootstrap_mean_difference | ci95_low | ci95_high | proportion_xgboost_better | proportion_tied | direction | valid_replicates | skipped_replicates | seed |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| roc_auc | 0.0049 | 0.0049 | -0.0017 | 0.0116 | 0.9310 | 0.0000 | higher is better | 1000 | 0 | 42 |
| average_precision | 0.0037 | 0.0032 | -0.0100 | 0.0172 | 0.6720 | 0.0000 | higher is better | 1000 | 0 | 42 |
| brier_score | -0.0005 | -0.0005 | -0.0025 | 0.0014 | 0.6800 | 0.0000 | lower is better | 1000 | 0 | 42 |
| lift_at_top10pct | -0.0258 | -0.0110 | -0.1312 | 0.1178 | 0.3810 | 0.0940 | higher is better | 1000 | 0 | 42 |
| recall_at_top10pct | -0.0026 | -0.0011 | -0.0131 | 0.0118 | 0.3810 | 0.0940 | higher is better | 1000 | 0 | 42 |
| lift_at_top20pct | -0.0323 | -0.0265 | -0.1019 | 0.0447 | 0.2170 | 0.0500 | higher is better | 1000 | 0 | 42 |
| recall_at_top20pct | -0.0065 | -0.0053 | -0.0204 | 0.0089 | 0.2170 | 0.0500 | higher is better | 1000 | 0 | 42 |

Each replicate draws the same number of distinct-player clusters with replacement, retains every validation snapshot for each sampled player (including repeated copies when a player is drawn twice), and evaluates both models on identical draws. Rank capacities and prevalence are recomputed in each replicate. Total row counts can vary. Intervals are 2.5th/97.5th percentiles of XGBoost-minus-RF differences; Brier favors negative differences. The proportion better is a descriptive bootstrap frequency, not a posterior probability, p-value, or proof of superiority. Intervals condition on the fitted models and selected validation period; they do not include model-selection or training uncertainty.

- lift_at_top10pct: XGB minus RF -0.02580, 95% CI [-0.13118, +0.11778]; contains zero: the observed RF advantage is not clearly separated from validation uncertainty.
- recall_at_top10pct: XGB minus RF -0.00259, 95% CI [-0.01315, +0.01180]; contains zero: the observed RF advantage is not clearly separated from validation uncertainty.
- lift_at_top20pct: XGB minus RF -0.03233, 95% CI [-0.10191, +0.04470]; contains zero: the observed RF advantage is not clearly separated from validation uncertainty.
- recall_at_top20pct: XGB minus RF -0.00647, 95% CI [-0.02041, +0.00894]; contains zero: the observed RF advantage is not clearly separated from validation uncertainty.

## One demographic sensitivity analysis

Retrained the exact selected XGBoost settings once on original TRAIN with Country, Language and Gender removed. No tuning or alternative feature search. Before running, practical-loss tolerances were fixed at ROC-AUC/AP 0.005, Brier increase 0.002, lift decrease 0.05 and recall decrease 0.01 at both 10% and 20%. These are declared project judgment limits, not profit, fairness or statistical thresholds. Removal is preferred only if every metric meets its tolerance.

| metric | full_xgboost | demographic_free_xgboost | difference_free_minus_full | absolute_difference | predictive_loss | practical_loss_tolerance | within_tolerance |
| --- | --- | --- | --- | --- | --- | --- | --- |
| roc_auc | 0.7464 | 0.7448 | -0.0016 | 0.0016 | 0.0016 | 0.0050 | True |
| average_precision | 0.5630 | 0.5658 | 0.0027 | 0.0027 | -0.0027 | 0.0050 | True |
| brier_score | 0.1906 | 0.1908 | 0.0002 | 0.0002 | 0.0002 | 0.0020 | True |
| lift_at_top10pct | 1.7417 | 1.7933 | 0.0516 | 0.0516 | -0.0516 | 0.0500 | True |
| recall_at_top10pct | 0.1746 | 0.1798 | 0.0052 | 0.0052 | -0.0052 | 0.0100 | True |
| lift_at_top20pct | 1.6875 | 1.7004 | 0.0129 | 0.0129 | -0.0129 | 0.0500 | True |
| recall_at_top20pct | 0.3376 | 0.3402 | 0.0026 | 0.0026 | -0.0026 | 0.0100 | True |

All losses fall within the declared tolerances; freeze the demographic-free variant for simpler inputs and reduced direct demographic dependence.

This sensitivity does not establish group fairness, remove behavioral proxies, or quantify protected-group outcomes. Demographic exclusion is not proof of fairness. Prior Day 9 SHAP findings explain the full model; they are not automatically exact attributions for a changed feature set. The paired bootstrap above compares the original full-feature candidates; it does not quantify uncertainty for the demographic ablation or prove the final variant superior to RF.

## Calibration assessment and TRAIN-only experiment

Leading variant raw Brier 0.190772, log loss 0.551661, mean probability 33.2823%, observed prevalence 35.3938%, ECE 0.026161. Descriptive validation logistic calibration intercept +0.1162, slope 1.0069; ideal values are 0 and 1. This regression is an assessment only: it is never applied as a calibration mapping.

ECE uses ten fixed equal-width probability bins [0,.1), ... [.9,1]; ECE = sum(n_bin/n * abs(mean probability - observed frequency)). Empty bins contribute zero. Logistic calibration diagnostics fit outcome on clipped logit(probability), clipping to [1e-6,1-1e-6] for finite logits. Calibration curves can be noisy in sparse high-score bins.

The calibration experiment uses five expanding temporal folds within TRAIN. Every fitting snapshot's 60-day label horizon ends strictly before the scoring snapshot's 30-day observation window starts, stronger than a 60-day snapshot gap. Preprocessing is independently fitted within each fitting fold. Repeated players across time remain permitted by the original known-player design.

| fold | fit_start | fit_end | outcome_end | score_date | score_observation_start | gap_snapshot_days | fit_observations | score_observations |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | 2005-03-31 00:00:00 | 2005-05-31 00:00:00 | 2005-07-30 00:00:00 | 2005-08-31 00:00:00 | 2005-08-02 00:00:00 | 92 | 4141 | 871 |
| 2 | 2005-03-31 00:00:00 | 2005-06-30 00:00:00 | 2005-08-29 00:00:00 | 2005-09-30 00:00:00 | 2005-09-01 00:00:00 | 92 | 5017 | 1043 |
| 3 | 2005-03-31 00:00:00 | 2005-07-31 00:00:00 | 2005-09-29 00:00:00 | 2005-10-31 00:00:00 | 2005-10-02 00:00:00 | 92 | 5836 | 902 |
| 4 | 2005-03-31 00:00:00 | 2005-08-31 00:00:00 | 2005-10-30 00:00:00 | 2005-11-30 00:00:00 | 2005-11-01 00:00:00 | 91 | 6707 | 873 |
| 5 | 2005-03-31 00:00:00 | 2005-09-30 00:00:00 | 2005-11-29 00:00:00 | 2005-12-31 00:00:00 | 2005-12-02 00:00:00 | 92 | 7750 | 855 |

Pooled OOF support: 4,544 rows, 2,108 players, 34.35% inactivity. Sigmoid fits a nonnegative-slope two-parameter logistic map of OOF logits. Isotonic is allowed only with at least 1,000 OOF observations, 100 per class and 100 distinct scores. Both mappings are fitted exclusively on pooled TRAIN-OOF labels and predictions. The exact frozen base estimator is then refitted to complete original TRAIN before validation scoring. This is not the future pre-test refit.

| method | brier_score | log_loss | ece | mean_probability | observed_prevalence | calibration_intercept | calibration_slope | roc_auc | average_precision |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| raw | 0.1908 | 0.5517 | 0.0262 | 0.3328 | 0.3539 | 0.1162 | 1.0069 | 0.7448 | 0.5658 |
| sigmoid | 0.1904 | 0.5517 | 0.0233 | 0.3441 | 0.3539 | 0.1129 | 1.1377 | 0.7448 | 0.5658 |
| isotonic | 0.1908 | 0.5516 | 0.0269 | 0.3444 | 0.3539 | 0.0951 | 1.1075 | 0.7437 | 0.5546 |

Prespecified selection rule: retain raw unless a mapping improves Brier by at least 0.001 and log loss by at least 0.002, with ECE worsening no more than 0.002. Prefer sigmoid if within Brier 0.0005 and log loss 0.001 of the best eligible method. These are practical tolerances, not significance tests. Isotonic can introduce ties and endpoint scores; ranking changes are not evidence of improved discrimination. Earlier TRAIN-OOF and later validation distributions can differ, and selecting a mapping using validation adds selection optimism.

**Selected calibration: raw.** The mappings do not provide sufficiently consistent material gains under the declared rule, so retain raw scores.

## Business policy and diagnostic thresholds

Primary output is the final model's probability/risk score for extended inactivity. Use ranking to prioritize within a real capacity: top 10% and top 20% are standard reporting scenarios; 5/15/25/30% remain alternatives. No contact costs, customer lifetime value, treatment effect, incremental retention or ROI are known. No probability threshold is claimed to maximize profit. Campaign design must separately handle repeated-player contacts, eligibility and governance.

Freeze **0.50** as the diagnostic confusion-matrix threshold for the next notebook. It is not a deployment or financial optimum.

| threshold | targeted_observations | targeting_percentage | precision | recall | f1 | specificity | churn_rate_targeted | lift |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 0.2000 | 1517 | 69.4597 | 0.4667 | 0.9159 | 0.6183 | 0.4266 | 0.4667 | 1.3186 |
| 0.2500 | 1411 | 64.6062 | 0.4819 | 0.8797 | 0.6227 | 0.4819 | 0.4819 | 1.3616 |
| 0.3000 | 1287 | 58.9286 | 0.4926 | 0.8202 | 0.6155 | 0.5372 | 0.4926 | 1.3918 |
| 0.3500 | 1141 | 52.2436 | 0.5101 | 0.7529 | 0.6082 | 0.6038 | 0.5101 | 1.4412 |
| 0.4000 | 971 | 44.4597 | 0.5273 | 0.6624 | 0.5872 | 0.6747 | 0.5273 | 1.4898 |
| 0.4500 | 757 | 34.6612 | 0.5469 | 0.5356 | 0.5412 | 0.7569 | 0.5469 | 1.5452 |
| 0.5000 | 514 | 23.5348 | 0.5914 | 0.3933 | 0.4724 | 0.8512 | 0.5914 | 1.6710 |
| 0.5500 | 310 | 14.1941 | 0.6194 | 0.2484 | 0.3546 | 0.9164 | 0.6194 | 1.7499 |
| 0.6000 | 133 | 6.0897 | 0.6842 | 0.1177 | 0.2009 | 0.9702 | 0.6842 | 1.9331 |
| 0.6500 | 21 | 0.9615 | 0.5714 | 0.0155 | 0.0302 | 0.9936 | 0.5714 | 1.6145 |
| 0.7000 | 1 | 0.0458 | 1.0000 | 0.0013 | 0.0026 | 1.0000 | 1.0000 | 2.8254 |
| 0.7500 | 0 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 1.0000 | 0.0000 | nan |
| 0.8000 | 0 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 1.0000 | 0.0000 | nan |

The maximum-F1 threshold on this **prespecified grid** is 0.25 (F1 0.6227); this is a validation diagnostic reference only, not a replacement for the frozen 0.50 diagnostic or an exhaustive optimum over all possible scores.

## Risk bands

| band | observations | actual_fraction | average_predicted_risk | actual_churn_rate | lift |
| --- | --- | --- | --- | --- | --- |
| Very high risk (top 10%) | 219 | 0.1003 | 0.6116 | 0.6347 | 1.7933 |
| High risk (80th-90th) | 218 | 0.0998 | 0.5453 | 0.5688 | 1.6071 |
| Medium risk (50th-80th) | 655 | 0.2999 | 0.4458 | 0.4641 | 1.3113 |
| Low risk (bottom 50%) | 1092 | 0.5000 | 0.1667 | 0.1886 | 0.5330 |

Four bands partition the ranked population: bottom 50%, 50th-80th, 80th-90th and top 10%. Starting from descending score order, boundaries use ceil(n*.1), ceil(n*.2), ceil(n*.5), then n. Ties use stable original-panel order and can split identical scores across bands. Bands are relative to the scoring cohort, not fixed probability cutoffs; mean scores and observed rates need not be perfectly monotonic. They are presentation/prioritization tools, not clinical or regulatory categories. Recompute ranks within each future eligible campaign cohort without labels; deduplication requires a separate operational policy.

Final selected variant/policy capacity results (the RF/XGB comparison above uses the original full-feature candidates):

| target_fraction | selected_observations | actual_fraction | churn_observations_captured | precision | recall | lift | model |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 0.0500 | 110 | 0.0504 | 75 | 0.6818 | 0.0970 | 1.9264 | Final selected XGBoost |
| 0.1000 | 219 | 0.1003 | 139 | 0.6347 | 0.1798 | 1.7933 | Final selected XGBoost |
| 0.1500 | 328 | 0.1502 | 201 | 0.6128 | 0.2600 | 1.7314 | Final selected XGBoost |
| 0.2000 | 437 | 0.2001 | 263 | 0.6018 | 0.3402 | 1.7004 | Final selected XGBoost |
| 0.2500 | 546 | 0.2500 | 320 | 0.5861 | 0.4140 | 1.6559 | Final selected XGBoost |
| 0.3000 | 656 | 0.3004 | 373 | 0.5686 | 0.4825 | 1.6065 | Final selected XGBoost |

## Exact pre-test refit protocol: defined, not executed

First test snapshot: 2006-08-31; inclusive 30-day observation starts 2006-08-02. Require S + 60 days < 2006-08-02, so the latest admissible calendar snapshot is **2006-06-02**. For month-end snapshots the last safe one is **2006-05-31** (outcome ends 2006-07-30).

| snapshot_date | outcome_end | safe_for_future_refit |
| --- | --- | --- |
| 2006-01-31 00:00:00 | 2006-04-01 00:00:00 | True |
| 2006-02-28 00:00:00 | 2006-04-29 00:00:00 | True |
| 2006-05-31 00:00:00 | 2006-07-30 00:00:00 | True |
| 2006-06-30 00:00:00 | 2006-08-29 00:00:00 | False |
| 2006-07-31 00:00:00 | 2006-09-29 00:00:00 | False |

January and February 2006 may join the future refit because their original embargo role separated development train/validation; their outcomes are available before the first test observation period. June and July remain excluded. No unsafe rows are used for the aggregate counts below.

| snapshot_date | observations | unique_players | churn_rate | outcome_end |
| --- | --- | --- | --- | --- |
| 2005-03-31 00:00:00 | 1805 | 1805 | 0.4017 | 2005-05-30 00:00:00 |
| 2005-04-30 00:00:00 | 1314 | 1314 | 0.3927 | 2005-06-29 00:00:00 |
| 2005-05-31 00:00:00 | 1022 | 1022 | 0.3679 | 2005-07-30 00:00:00 |
| 2005-06-30 00:00:00 | 876 | 876 | 0.3413 | 2005-08-29 00:00:00 |
| 2005-07-31 00:00:00 | 819 | 819 | 0.3284 | 2005-09-29 00:00:00 |
| 2005-08-31 00:00:00 | 871 | 871 | 0.3008 | 2005-10-30 00:00:00 |
| 2005-09-30 00:00:00 | 1043 | 1043 | 0.3490 | 2005-11-29 00:00:00 |
| 2005-10-31 00:00:00 | 902 | 902 | 0.3636 | 2005-12-30 00:00:00 |
| 2005-11-30 00:00:00 | 873 | 873 | 0.3643 | 2006-01-29 00:00:00 |
| 2005-12-31 00:00:00 | 855 | 855 | 0.3380 | 2006-03-01 00:00:00 |
| 2006-01-31 00:00:00 | 862 | 862 | 0.3561 | 2006-04-01 00:00:00 |
| 2006-02-28 00:00:00 | 816 | 816 | 0.3517 | 2006-04-29 00:00:00 |
| 2006-03-31 00:00:00 | 803 | 803 | 0.3836 | 2006-05-30 00:00:00 |
| 2006-04-30 00:00:00 | 736 | 736 | 0.3315 | 2006-06-29 00:00:00 |
| 2006-05-31 00:00:00 | 645 | 645 | 0.3426 | 2006-07-30 00:00:00 |

Planned final refit: **2005-03-31 through 2006-05-31**, **14,242 observations**, **3,651 players**, **35.9009%** inactivity. This notebook reports eligibility only and does not perform this pre-test refit.

Next notebook: verify the frozen specification and data hash, select only eligible snapshots by the strict date rule, fit the frozen preprocessing and estimator on that pre-test data, and retain the already frozen TRAIN-OOF calibration mapping if selected (otherwise no mapping). Missing indicators and category levels may be relearned by the same preprocessing algorithm on safe pre-test data; raw predictor definitions/hyperparameters remain fixed. Freeze the resulting fitted artifact before making test predictions.

## One-time held-out evaluation plan

Score only 2006-08-31 through 2006-11-30 once after the freeze/refit. Compute ROC-AUC; PR-AUC as average precision; Brier; log loss; precision, recall, F1 and specificity at diagnostic 0.50; lift/recall at top 10% and top 20%. Use 1,000 seeded (42) whole-UserID cluster bootstrap replicates for 95% percentile intervals. Ranking ties use stable original-panel order. Produce the calibration plot (same ten bins), ROC curve, PR curve, cumulative gains/lift and confusion matrix. Freeze risk bands/capacity definitions; do not choose a new threshold, calibrator, feature set or family after inspecting test outcomes. If results disappoint, report them; any later development requires genuinely new holdout data. No post-test tuning is allowed.

## Limitations and status

- Validation has been reused across model-family, feature-variant and calibration choices, creating selection optimism; bootstrap intervals do not correct it.
- Repeated-player clustering respects dependence within a player but not shared calendar shocks; only three validation months are available.
- Historical 2005-2006 data, cohort drift, correlated features and unverified demographic history limit generalization; no causal retention benefit or ROI is established.
- Scores concern extended inactivity, not permanent departure, gambling harm, problem gambling, customer value or causal behavioral explanations.
- Protected-group outcome analysis is unavailable; neither ablation nor SHAP is a fairness audit.
- Calibration trained on earlier OOF periods may not transfer to a final pre-test refit or future populations.
- TEST remains completely sealed. No pre-test final refit, test transformation, test predictions, probability inspection, or test evaluation occurred. No player-level predictions were exported; original artifacts were preserved. No commit or push was performed.

All model-development decisions were frozen before evaluating the held-out test period.

# Day 9: SHAP explainability of the provisional XGBoost candidate

## Model and population

Frozen Day 8 `initial_conservative` XGBoost pipeline, 400 trees, learning rate 0.03, depth 3, min_child_weight 10, subsample 0.8, colsample_bytree 0.9, alpha 0.1, lambda 10, gamma 0.1, seed 42. No fitting, tuning, feature changes, calibration or final threshold selection occurred. Random Forest remains the targeting challenger.

Validation: 2,184 player-snapshots from 2006-03-31 through 2006-05-31; 1,310 distinct players; observed inactivity prevalence 35.39%. SHAP sample: 1,000 validation rows (uniform, without replacement, random_state=42). No test observations were transformed, predicted, explained or evaluated. Training remains 2005-03-31 through 2005-12-31; test remains sealed at 2006-08-31 through 2006-11-30; the original embargo windows are preserved.

Reproduced full-validation ROC-AUC **0.746382**, PR-AUC (average precision) **0.563016**.

TreeExplainer uses tree_path_dependent perturbation, the stored training path counts as its implicit reference, and raw model output (log-odds). There is no separately sampled background. This reference is not the validation prevalence. For every explained row, sigmoid(base value + sum of SHAP) reproduces its model probability within numerical tolerance. Positive SHAP pushes toward churn; negative SHAP pushes away. Contributions are not probability percentage points.

## Global explanation

| feature | mean_abs_shap | feature_group |
| --- | --- | --- |
| days_since_last_active_bet | 0.3443 | recency |
| days_since_first_casino_activity | 0.2446 | lifecycle |
| winnings_7d | 0.2139 | monetary |
| active_days_30d | 0.1979 | activity |
| net_loss_30d | 0.1019 | monetary |
| net_loss_7d | 0.0698 | monetary |
| winnings_14d | 0.0632 | monetary |
| days_since_first_pay | 0.0609 | lifecycle |
| active_days_recent15 | 0.0594 | trend |
| active_days_early15 | 0.0522 | trend |
| mean_gap_between_active_days_30d | 0.0501 | activity |
| bets_30d | 0.0413 | betting volume |
| winnings_30d | 0.0376 | monetary |
| Language = 2 | 0.0367 | demographics |
| stake_recent15 | 0.0362 | trend |
| max_daily_stake_30d | 0.0318 | monetary |
| avg_bets_per_active_day_30d | 0.0311 | betting volume |
| active_days_7d_share | 0.0303 | trend |
| avg_bets_per_active_day_7d | 0.0267 | betting volume |
| days_since_registration | 0.0259 | lifecycle |

The global table ranks transformed columns; category levels and missingness flags remain distinct. Red in the beeswarm denotes a higher transformed value, blue a lower value. For dummy columns these mean present/absent, and for missingness flags they mean missing/not missing. Numeric color values after preprocessing include median imputations.

## Feature groups

| feature_group | mean_abs_shap | share |
| --- | --- | --- |
| monetary | 0.5993 | 0.2829 |
| recency | 0.3443 | 0.1625 |
| lifecycle | 0.3315 | 0.1565 |
| activity | 0.3013 | 0.1422 |
| trend | 0.2983 | 0.1408 |
| demographics | 0.1212 | 0.0572 |
| betting volume | 0.1197 | 0.0565 |
| missingness | 0.0030 | 0.0014 |

Group scores sum per-column mean absolute SHAP. They describe attribution mass, not explained variance or percentages of predicted risk. Correlated and larger groups can accumulate attribution; signed contributions can cancel within a row. Recent/early half-window predictors and change/share predictors are grouped as trend; missingness always has its own group.

## Direction and nonlinear patterns

- **days_since_last_active_bet**: low observed bin [0, 1] (n=219) has mean SHAP -0.831; high bin [22, 29] (n=167) has +0.366 log-odds. Higher values have higher average contributions in this low/high comparison. Bin means are monotonic across this sample, but this does not establish a globally monotonic response.
- **active_days_30d**: low observed bin [1, 2] (n=502) has mean SHAP +0.154; high bin [9, 30] (n=158) has -0.421 log-odds. Higher values have lower average contributions in this low/high comparison. Bin means are monotonic across this sample, but this does not establish a globally monotonic response.
- **winnings_7d**: low observed bin [0, 44.6] (n=667) has mean SHAP +0.147; high bin [656, 5.72e+04] (n=167) has -0.384 log-odds. Higher values have lower average contributions in this low/high comparison. Bin means are monotonic across this sample, but this does not establish a globally monotonic response.
- **max_daily_stake_30d**: low observed bin [0, 20] (n=170) has mean SHAP +0.025; high bin [2.61e+03, 1.01e+05] (n=167) has -0.081 log-odds. Higher values have lower average contributions in this low/high comparison. Intermediate bin means reverse direction: the observed pattern is nonlinear/nonmonotonic.
- **days_since_first_casino_activity**: low observed bin [0, 149] (n=168) has mean SHAP +0.013; high bin [448, 484] (n=163) has -0.318 log-odds. Higher values have lower average contributions in this low/high comparison. Intermediate bin means reverse direction: the observed pattern is nonlinear/nonmonotonic.

These are marginal patterns in the observed validation sample, affected by other predictors and missingness. Quantile edges are descriptive bins, not learned operational thresholds. The dependence plots retain all finite values (monetary axes use a signed logarithmic scale); no cutoff or causal effect is inferred. Rows with a missing plotted value or color are excluded from that scatter; bin summaries separately record missing feature values.

## Targeted interaction diagnostics

| feature_1 | feature_2 | first_low | first_high | second_low | second_high | mean_mixed_margin | mean_abs_mixed_margin | fraction_nonzero | n |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| days_since_last_active_bet | active_days_30d | 2.0000 | 18.0000 | 1.0000 | 6.0000 | 0.2877 | 0.2877 | 1.0000 | 1000 |
| days_since_last_active_bet | bets_30d | 2.0000 | 18.0000 | 10.0000 | 297.2500 | -0.0000 | 0.0000 | 0.0000 | 1000 |
| active_days_30d | days_since_first_casino_activity | 1.0000 | 6.0000 | 218.0000 | 432.5000 | -0.0216 | 0.0220 | 1.0000 | 1000 |
| winnings_7d | active_days_30d | 0.0000 | 171.1465 | 1.0000 | 6.0000 | -0.1071 | 0.1073 | 1.0000 | 1000 |

These are small 2x2 fitted log-odds contrasts at each feature's validation Q25/Q75, averaged over validation sample contexts. A mixed difference is f(high,high) - f(high,low) - f(low,high) + f(low,low). A nonzero contrast supports non-additivity of the fitted function at those synthetic points; a near-zero contrast does not rule out interactions elsewhere. Positive mean contrast means the joint high/high log-odds response exceeds the sum of the two separate changes. No full interaction matrix or SHAP interaction values were computed. Dependence-plot color alone does not prove an interaction. Holding correlated and derived predictors fixed can create inconsistent/off-support combinations, so these diagnostics do not establish real-player or causal effects.

## Local explanations (diagnostic threshold 0.50)

One example nearest to the within-cell median predicted probability is selected for each outcome within the SHAP sample. This is an illustrative selection rule, not a claim of population representativeness. Raw player records, identifiers, row indices, dates and local predictor values are omitted from exports. Waterfalls show anonymous feature contributions in log-odds.

| case | outcome | probability | actual | increasing_risk | decreasing_risk |
| --- | --- | --- | --- | --- | --- |
| Validation Case A | TP | 0.5729 | 1 | days_since_last_active_bet (+0.343 log-odds); winnings_7d (+0.166 log-odds); active_days_30d (+0.129 log-odds) | days_since_first_casino_activity (-0.291 log-odds); days_since_registration (-0.026 log-odds); net_loss_7d (-0.024 log-odds) |
| Validation Case B | FP | 0.5560 | 0 | days_since_last_active_bet (+0.210 log-odds); winnings_7d (+0.151 log-odds); bets_30d (+0.149 log-odds) | days_since_first_casino_activity (-0.241 log-odds); net_loss_30d (-0.152 log-odds); avg_stake_per_bet_30d (-0.045 log-odds) |
| Validation Case C | TN | 0.1996 | 0 | active_days_30d (+0.099 log-odds); net_loss_30d (+0.089 log-odds); days_since_first_pay (+0.061 log-odds) | days_since_first_casino_activity (-0.396 log-odds); winnings_7d (-0.360 log-odds); mean_gap_between_active_days_30d (-0.147 log-odds) |
| Validation Case D | FN | 0.3738 | 1 | active_days_30d (+0.271 log-odds); net_loss_7d (+0.163 log-odds); days_since_first_pay (+0.087 log-odds) | days_since_last_active_bet (-0.727 log-odds); days_since_first_casino_activity (-0.224 log-odds); avg_bets_per_active_day_7d (-0.055 log-odds) |

The same risk-increasing associations can occur in both TP/FP cases, and risk-decreasing associations in both TN/FN cases: an explanation describes the score, not why the observed outcome occurred. The 0.50 diagnostic does not select a deployment threshold.

## Consistency with Day 8 permutation importance

| feature | shap_rank | permutation_rank | notes |
| --- | --- | --- | --- |
| days_since_last_active_bet | 1 | 1 | Both top 10 |
| days_since_first_casino_activity | 2 | 4 | Both top 10 |
| winnings_7d | 3 | 2 | Both top 10 |
| active_days_30d | 4 | 3 | Both top 10 |
| net_loss_30d | 5 | 15 | Different attribution/AP disruption; correlated features may redistribute importance |
| net_loss_7d | 6 | 45 | Different attribution/AP disruption; correlated features may redistribute importance |
| winnings_14d | 7 | 22 | Different attribution/AP disruption; correlated features may redistribute importance |
| days_since_first_pay | 8 | 6 | Both top 10 |
| active_days_recent15 | 9 | 8 | Both top 10 |
| active_days_early15 | 10 | 13 | Ranks within 5 positions |
| mean_gap_between_active_days_30d | 11 | 10 | Ranks within 5 positions |
| Language | 12 | 32 | Different attribution/AP disruption; correlated features may redistribute importance |
| bets_30d | 13 | 25 | Different attribution/AP disruption; correlated features may redistribute importance |
| winnings_30d | 14 | 16 | Ranks within 5 positions |
| stake_recent15 | 15 | 11 | Ranks within 5 positions |
| Country | 16 | 14 | Ranks within 5 positions |
| max_daily_stake_30d | 17 | 5 | Different attribution/AP disruption; correlated features may redistribute importance |
| avg_bets_per_active_day_30d | 18 | 44 | Different attribution/AP disruption; correlated features may redistribute importance |
| active_days_7d_share | 19 | 31 | Different attribution/AP disruption; correlated features may redistribute importance |
| avg_bets_per_active_day_7d | 20 | 20 | Ranks within 5 positions |

Top-10 overlap: 6/10; source-level Spearman rank correlation 0.626. The five earlier drivers have SHAP source ranks: days_since_last_active_bet 1, winnings_7d 3, active_days_30d 4, days_since_first_casino_activity 2, max_daily_stake_30d 17.

SHAP partly challenges the earlier ordering; inspect the source ranks above.

For comparability, signed SHAP contributions of every raw feature's descendants (including its missing indicator and category levels) are summed within each row, then absolute-valued and averaged. This differs from the transformed/group tables. Permutation importance measures full-validation average-precision loss after shuffling a raw predictor (five repeats); SHAP measures contribution magnitude in a 1,000-row sample. Correlated predictors can share SHAP credit; permutation can disrupt correlated structure, and nonlinear interactions redistribute contributions. Small negative AP drops reflect noise/redundancy, not protective effects. Neither ranking is absolute truth.

## Demographics and missingness

- **demographics**: 5.72% of summed mean absolute SHAP; highest column `Language = 2` is rank 14 (0.03672 log-odds). At least one column appears in the top 20; review its role.
- **missingness**: 0.14% of summed mean absolute SHAP; highest column `max_gap_between_active_days_30d (missing)` is rank 51 (0.00303 log-odds). No column appears in the top 20.

| feature | mean_abs_shap | shap_rank | permutation_rank |
| --- | --- | --- | --- |
| Language | 0.0448 | 12 | 32 |
| Country | 0.0326 | 16 | 14 |
| Gender | 0.0010 | 44 | 37 |

Missingness indicators with nonzero mean absolute SHAP:

| feature | mean_abs_shap | shap_rank |
| --- | --- | --- |
| max_gap_between_active_days_30d (missing) | 0.0030 | 51 |

Importance is reported neutrally: low demographic attribution does not establish fairness, and proxy effects can remain in behavioral predictors. Country/Language/Gender require governance review before operational use; this notebook performs no subgroup fairness audit or feature removal. Structural missingness may encode player behavior (for example, insufficient activity to define a gap) or source-data availability. A zero indicator attribution does not mean missingness is irrelevant: median-imputed values and correlated predictors may carry it indirectly.

## Responsible interpretation and limitations

The model predicts extended inactivity: no positive-bet day in the 60 days after the snapshot. It does **not** diagnose gambling harm, identify problem gambling, determine customer value, or prove why a player churns. Model explanations describe associations used by the model, not causal behavioral explanations.

- Correlated/derived predictors make individual attributions reference-dependent; tree-path SHAP does not resolve confounding.
- Repeated-player dependence means sampled snapshots are not independent; the analysis is not evidence of unseen-player generalization.
- This historical 2005-2006 dataset may not represent current products, regulation or player populations; demographic history is not independently verified.
- Model-family selection already used validation, so performance and interpretations can have validation-selection optimism. No uncertainty intervals or causal inference are provided.
- Uniform sampling and quantile summaries can underrepresent rare categories/extremes. Local cases are illustrative, not a representative survey.
- Synthetic interaction contrasts may be outside observed joint support and do not constitute exact SHAP interaction attribution.
- Test remains sealed; probabilities remain uncalibrated; no final operational threshold was chosen.

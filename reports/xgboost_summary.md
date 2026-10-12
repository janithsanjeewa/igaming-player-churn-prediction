# Day 8 XGBoost: initial_conservative

Configuration: n_estimators=400, learning_rate=0.03, max_depth=3, min_child_weight=10, subsample=0.8, colsample_bytree=0.9, reg_lambda=10, reg_alpha=0.1, gamma=0.1; random_state=42, n_jobs=-1, objective=binary:logistic, eval_metric=logloss, tree_method=hist.

Train ROC-AUC 0.7862; PR-AUC/AP 0.6305.

Validation ROC-AUC 0.7464; PR-AUC/AP 0.5630; Brier 0.1906.

At diagnostic threshold 0.50: precision 0.6058; recall 0.4036; F1 0.4845.

Top10%: precision 0.6164; recall 17.46%; lift 1.7417.

Top20%: precision 0.5973; recall 33.76%; lift 1.6875.

Train-validation gaps: ROC-AUC 0.0398; AP 0.0675. RF gaps are 0.0763 and 0.1265. XGBoost generalizes better by these resubstitution-gap diagnostics (similarity tolerances: 0.01 ROC, 0.02 AP); temporal drift also contributes.

Metric leaders: roc_auc: XGBoost; average_precision: XGBoost; lift_at_top10pct: Random Forest; brier_score: XGBoost.

Differences XGBoost minus Logistic Regression: roc_auc +0.0168; average_precision +0.0288; brier_score -0.0057; precision +0.0584; recall +0.1423; f1 +0.1307; lift_at_top10pct +0.1548; recall_at_top10pct +0.0155; lift_at_top20pct +0.1099; recall_at_top20pct +0.0220.

Differences XGBoost minus Random Forest: roc_auc +0.0049; average_precision +0.0037; brier_score -0.0005; precision -0.0083; recall +0.0660; f1 +0.0487; lift_at_top10pct -0.0258; recall_at_top10pct -0.0026; lift_at_top20pct -0.0323; recall_at_top20pct -0.0065. Lower Brier is better.

Top five validation permutation predictors: days_since_last_active_bet, winnings_7d, active_days_30d, days_since_first_casino_activity, max_daily_stake_30d. Correlated features may share or mask predictive importance; repeat SD is not population uncertainty and importance is not causal.

Error analysis: XGB TP/FP median recency 20/19 days (RF 20/19); XGB TP/FP median active days 1/1 (RF 1/1). XGB FN/TN median recency 9/3, bets 46/106 (RF recency 10/3, bets 44/100). These are descriptive groups, not individual explanations.

Calibration: mean predicted risk 33.30%, observed churn 35.39%, Brier 0.1906. Average risk is underpredicted. Lowest-risk bin: predicted 3.38%, observed 0.91%; highest-risk bin: predicted 61.70%, observed 61.64%. Middle-bin deviations can differ; the mean alone does not establish under/overconfidence. Raw reliability and Brier need review before calibration.

Reliability interpretation: observed churn exceeds predicted risk in 7 of 10 bins. Risk is especially underpredicted in several middle bins, while the highest-risk bin is close to the diagonal. The lowest-risk bin overpredicts churn. This is mixed miscalibration, rather than uniform underconfidence or overconfidence. XGB's mean is farther below prevalence than RF's 34.48%, despite a slightly lower Brier score; Brier alone does not establish superior calibration.

Provisional model-family recommendation: **XGBoost** under the predeclared AP/ROC/business-ranking rule, using generalization gaps and Brier to resolve near ties. This is a development recommendation, not a test-validated final winner.

The XGB-versus-RF AP difference is +0.0037, a modest advantage whose robustness has not been established. RF retains stronger top10/top20 lift and capture, whereas XGB has higher overall ROC-AUC/AP, smaller resubstitution gaps and slightly lower Brier. These results support a provisional overall XGB choice under the stated priorities; a fixed campaign capacity may favor RF. Do not claim statistical superiority or select an operational policy from these diagnostics.

Limitations: validation-selection optimism, one temporal validation window, repeated-player dependence, correlated features and uncalibrated probabilities. The primary split is unchanged. No SHAP or operational threshold has been selected.

The model is saved locally to Git-ignored models/xgboost_model.joblib. Only aggregate tables/figures are tracked. Test remains sealed; the recommendation is provisional until one final locked test-set evaluation. No commit or push.
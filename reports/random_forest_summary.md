## Selected Random Forest: depth8_leaf20_half



Configuration: n_estimators=500, max_depth=8, min_samples_leaf=20, max_features=0.5, class_weight=None

Train ROC-AUC 0.8178; AP 0.6859.

Validation ROC-AUC 0.7415; AP 0.5594; Brier 0.1911.

At 0.50: precision 0.6141; recall 0.3376; F1 0.4357.

Top 10%: lift 1.7675; recall 17.72%.

Top 20%: lift 1.7198; recall 34.41%.

Train–validation gaps: ROC-AUC 0.0763; AP 0.1265.

Mean raw probability 34.48%; observed validation churn 35.39%. Average risk is underpredicted.

Top permutation features: days_since_last_active_bet, std_daily_stake_30d, max_daily_stake_30d, days_since_first_casino_activity, winnings_30d. Recency has the largest AP drop; several smaller importances overlap within repeat variability, so their exact order is uncertain.

Logistic comparison (RF minus LR): roc_auc +0.0119; average_precision +0.0252; brier_score -0.0052; precision +0.0667; recall +0.0763; f1 +0.0820; lift_at_top10pct +0.1806; recall_at_top10pct +0.0181; lift_at_top20pct +0.1422; recall_at_top20pct +0.0285. Lower Brier is better; every requested comparison metric improves here.

The unrestricted initial forest scored train ROC-AUC 1.0000 / AP 1.0000, but validation ROC-AUC 0.7128 / AP 0.5154. Constraints substantially reduce this overfit, though the selected forest still has a meaningful gap.

Weighted sensitivity: AP 0.5559, precision 0.5057, recall 0.7503, Brier 0.1973. Weighting increases diagnostic recall while reducing precision and worsening Brier; primary remains unweighted.

Error analysis: TP/FP median recency is 20/19 days, and both groups have median 1/1 active days. Their similar low activity shows why threshold-positive observations are hard to separate. FN median recency is 10 days versus TN 3; FN median bets are 44 versus TN 100. These are descriptive, correlated group summaries, not causes.

Reliability: the highest-risk decile predicts 57.82% against 62.56% observed; the lowest-risk decile predicts 4.48% against 1.37%. This suggests some underconfidence (probabilities compressed toward the base rate), especially in the high-risk tail, with mixed deviations in middle bins. It does not support uniform overconfidence. RF improves Brier versus LR and its average probability is closer to prevalence; the actual results do not support assuming RF is worse calibrated. Brier also reflects discrimination, so it is not a pure calibration measure. No calibration is fitted.

Ranking material-improvement criterion met: True. No final winner before XGBoost.

Concerns: overfitting, correlated predictors, probability calibration, validation selection optimism and repeated-player dependence.

Model saved locally; tracked outputs are aggregate only. Test remains sealed. No production threshold or calibration selected.
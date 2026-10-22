# Model card: final iGaming inactivity-risk XGBoost

## Purpose and intended use

Predict 60-day operational inactivity risk for eligible historically active casino
player-snapshots. Intended for portfolio research, prioritization and churn-risk
analysis with human governance. A probability is not a certainty or evidence of
why an outcome occurred. This artifact has not been validated for live deployment.

Not intended for diagnosing gambling harm, responsible-gambling classification,
causal interpretation, automated adverse decisions or customer-value scoring.

## Data provenance and population

Historical bwin Virtual Casino Gambling source files supplied locally:
`RawDataSet1_DemographicsCasinoTXT.zip`, `RawDataSet2_DailyAggregCasinoTXT.zip`
and `Codebook_for_Virtual_Casino_Gambling.pdf`. The existing Day 1 validation
records daily activity from **2005-02-01 through 2007-02-28**. Source files and
cleaned/model panels remain Git-ignored; this card makes no new claim about
licensing or representativeness. See notebooks 01-05 and the feature dictionary
for cleaning, target construction and source assumptions.

The processed panel has 18,472 eligible monthly player-snapshots, 4,025 players
and 48 columns before predictor selection. Eligibility requires positive Bets
within the 30-day history and complete history/outcome coverage. No activity
outside a source's observed coverage is assumed to prove permanent departure.

## Task, inputs and preprocessing

Observation: 30 inclusive dates [S-29,S]. Target `churn_60d=1` means no Bets>0
day in (S,S+60]; zero means at least one such day. Reactivation after day 60 is
possible. The model predicts extended inactivity rather than lifetime churn.

**42 historical numerical predictors**: recency, activity frequency/gaps, betting
volume, stake/winnings/net loss, recent trends and lifecycle durations. Exact
names/definitions are frozen in `final_model_specification.json` and
`docs/feature_dictionary.md`. Country, Language and Gender are excluded; UserID,
snapshot_date and the target are never predictors. Behavioral proxies can remain.

Median imputation with training-learned missing indicators, no scaling, dense
output. Preprocessing and the frozen XGBoost (400 trees, depth 3, learning rate
0.03, seed 42; full settings in the specification) were fitted once on safe
pre-test data only. Raw probabilities, no calibrator, diagnostic threshold 0.50.

## Development, refit and evaluation

Development TRAIN March-December 2005; validation March-May 2006. All model-family,
feature, calibration and policy decisions were frozen before test evaluation.
Final refit March 31, 2005-May 31, 2006: **14,242 observations**, **3,651 players**,
**35.9009%** inactivity. January/February join after selection; June/July remain
excluded. All fitting labels end before the first test observation begins August 2.

Held-out TEST August-November 2006: **2,838 observations**, **1,455 players**,
**34.0733%** inactivity. It was scored once. Original known-player
temporal evaluation permits repeated players across partitions and months.

## Performance

- ROC-AUC **0.7490**, average precision **0.5510**.
- Brier **0.1870**, log loss **0.5457**, ECE **0.0156**.
- At 0.50: precision **56.48%**, recall **48.19%**, F1 **0.5201**.
- Top 10%: lift **1.7671**, recall **17.68%**.
- Top 20%: lift **1.7258**, recall **34.54%**.

See `final_test_bootstrap_intervals.csv` for 1,000 whole-player bootstrap percentile
intervals and `final_test_evaluation.md` for calibration, monthly variation, all
metrics and validation differences. Overall performance is descriptively broadly
stable; AP is lower than validation. Accuracy is not the primary metric.

## Policy and limitations

Rank capacities 5/10/15/20/25/30% and four frozen percentile risk bands support
prioritization, not regulatory categories. The 0.50 diagnostic threshold is not
a financial optimum. No causal retention benefit, contact economics or ROI is known.
Counts refer to snapshots rather than unique campaign recipients.

Historical data, temporal drift, selection of active players, correlated predictors,
source-data availability and repeated-player dependence limit generalization.
There is no protected-group outcome audit, causal analysis, unseen-player test or
fairness guarantee. Source demographic/lifecycle history has known availability
assumptions. Test calibration is descriptive and was not used to choose a calibrator.

Before real deployment, obtain current representative data and governance approval;
validate current population performance and calibration; monitor drift, missingness,
score distributions, delayed outcomes, monthly calibration and capacity precision/
recall/lift. Define campaign deduplication and human review, responsible use,
monitoring triggers and a retraining schedule with genuinely new holdouts.
The evaluated historical test is now used and must not become a tuning set.

## Artifacts and governance

Local model: `models/final_xgboost_churn_model.joblib` (Git-ignored).
Fitted model SHA-256: `3e75723784423b430b42dcb37ac9e9606c20f1b496dc36b35ba36d9d568b6c85`.
Frozen specification SHA-256: `7242eca09f8105dc402fb256d6ff9070dcc4c4544ec173f5d87b180da3feda3b`.
Aggregate evidence is tracked separately from ignored data and private scoring
checkpoints. No public player-level predictions are exported.

The held-out test set was evaluated once after all model-family, feature, calibration, and business-policy decisions were frozen.

No model or policy decisions were changed in response to test results.

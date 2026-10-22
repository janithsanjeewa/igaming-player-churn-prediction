"""Aggregate reporting of the single frozen test evaluation; no model calls."""
from __future__ import annotations

import json
import platform

import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, precision_recall_curve, roc_curve

from src.explainability import markdown_table
from src.final_test import export_aggregate_tables, sha256

GENERALIZATION = (
    '**Broadly stable**, descriptively rather than as a pass/fail verdict. '
    'Test ROC-AUC improves by 0.0042 and Brier improves by 0.0038; average precision '
    'declines by 0.0148. Top-10% targeting is slightly weaker, while top-20% targeting '
    'is slightly stronger. Test prevalence is lower than validation (34.07% versus '
    '35.39%), affecting precision and average precision; that does not fully explain '
    'or remove the observed AP decline. The test model was refitted on more safe '
    'history, so the comparison mixes population/time change with the planned refit. '
    'No inferential validation-versus-test difference interval or new pass/fail '
    'threshold is claimed, and the selection is not reconsidered.'
)


def create_final_figures(c):
    import matplotlib.pyplot as plt
    y, p = c['raw']['y'], c['raw']['probabilities']
    m = c['metrics'].iloc[0]
    directory = c['root'] / 'reports/figures'
    directory.mkdir(exist_ok=True)
    def save(name):
        plt.gcf().savefig(directory / name, dpi=180, bbox_inches='tight')
        plt.close('all')
    fpr, tpr, _ = roc_curve(y, p)
    plt.figure(figsize=(6, 5))
    plt.plot(fpr, tpr, label=f'XGBoost (ROC-AUC {m.roc_auc:.3f})')
    plt.plot([0, 1], [0, 1], '--', color='gray', label='Random ranking')
    plt.xlabel('False positive rate')
    plt.ylabel('True positive rate')
    plt.title('Held-out test: ROC curve')
    plt.legend()
    save('final_test_roc_curve.png')
    precision, recall, _ = precision_recall_curve(y, p)
    plt.figure(figsize=(6, 5))
    plt.plot(recall, precision, label=f'XGBoost (AP {m.average_precision:.3f})')
    plt.axhline(y.mean(), color='gray', linestyle='--', label=f'Prevalence {y.mean():.1%}')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Held-out test: precision-recall curve')
    plt.legend()
    save('final_test_pr_curve.png')
    curve = c['calibration'].loc[lambda d: d.observations.gt(0)]
    fig, axes = plt.subplots(2, 1, figsize=(6.5, 7), gridspec_kw={'height_ratios': [3, 1]})
    axes[0].plot([0, 1], [0, 1], '--', color='gray', label='Ideal calibration')
    axes[0].plot(curve.mean_probability, curve.observed_prevalence, 'o-', label='Raw probabilities')
    for row in curve.loc[curve.observations.lt(30)].itertuples():
        axes[0].annotate(f'n={row.observations}', (row.mean_probability, row.observed_prevalence),
                         xytext=(4, -12), textcoords='offset points', fontsize=8)
    axes[0].set(xlim=(0, 1), ylim=(0, 1), ylabel='Observed inactivity rate',
                title=f'Held-out test: reliability (ECE {m.ece:.3f})')
    axes[0].legend()
    axes[1].bar((c['calibration'].lower+c['calibration'].upper)/2,
                c['calibration'].observations, width=.09)
    axes[1].set(xlim=(0, 1), xlabel='Predicted risk (10 fixed equal-width bins)', ylabel='Rows')
    fig.tight_layout()
    save('final_test_calibration.png')
    matrix = confusion_matrix(y, p >= .50, labels=[0, 1])
    fig, ax = plt.subplots(figsize=(6, 5))
    dots = ax.imshow(matrix, cmap='Blues')
    for i in range(2):
        for j in range(2):
            ax.text(j, i, f'{matrix[i,j]:,}', ha='center', va='center', fontsize=18,
                    color='white' if matrix[i,j] > matrix.max()/2 else 'black')
    ax.set(xticks=[0, 1], yticks=[0, 1], xticklabels=['Active (0)', 'Inactive (1)'],
           yticklabels=['Active (0)', 'Inactive (1)'], xlabel='Predicted label', ylabel='Actual label',
           title='Held-out test: diagnostic threshold 0.50')
    fig.colorbar(dots, ax=ax, label='Observations')
    save('final_test_confusion_matrix.png')
    # Full rank curves are display summaries, not new capacity policy candidates.
    order = np.argsort(-p, kind='stable')
    fraction = np.arange(1, len(y)+1)/len(y)
    captured = np.cumsum(y[order]) / y.sum()
    lift = (np.cumsum(y[order])/np.arange(1, len(y)+1))/y.mean()
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    axes[0].plot(np.r_[0, fraction], np.r_[0, captured], label='XGBoost')
    axes[0].plot([0, 1], [0, 1], '--', color='gray', label='Random targeting')
    axes[0].scatter(c['targeting'].actual_fraction, c['targeting'].recall, s=25)
    axes[0].set(xlabel='Fraction targeted', ylabel='Fraction of churn outcomes captured', title='Cumulative gains')
    axes[0].legend()
    axes[1].plot(fraction, lift, label='XGBoost')
    axes[1].axhline(1, color='gray', linestyle='--', label='Random targeting')
    axes[1].scatter(c['targeting'].actual_fraction, c['targeting'].lift, s=25)
    axes[1].set(xlabel='Fraction targeted', ylabel='Lift versus test prevalence', title='Cumulative lift')
    axes[1].legend()
    fig.suptitle('Held-out test: frozen capacity policy')
    fig.tight_layout()
    save('final_test_lift_curve.png')
    bands = c['risk_bands'].iloc[::-1]
    fig, ax = plt.subplots(figsize=(9, 4.5))
    bars = ax.bar(bands.band, bands.actual_churn_rate, color='tab:blue')
    ax.bar_label(bars, labels=[f'{value:.1%}' for value in bands.actual_churn_rate], padding=3)
    ax.axhline(y.mean(), color='gray', linestyle='--', label=f'Overall {y.mean():.1%}')
    ax.set(ylabel='Observed 60-day inactivity rate', title='Held-out test: frozen percentile risk bands')
    plt.xticks(rotation=12, ha='right')
    ax.legend()
    save('final_test_risk_bands.png')
    plt.figure(figsize=(7, 4.5))
    for label, color in [(0, 'tab:blue'), (1, 'tab:orange')]:
        plt.hist(p[y == label], bins=np.linspace(0, 1, 26), density=True, histtype='step', linewidth=2,
                 color=color, label=f'Actual inactivity = {label}')
    plt.axvline(.50, color='gray', linestyle='--', label='Frozen diagnostic 0.50')
    plt.xlim(0, 1)
    plt.xlabel('Raw predicted inactivity probability')
    plt.ylabel('Density within actual outcome group')
    plt.title('Held-out test: score overlap by actual outcome')
    plt.legend()
    save('final_test_probability_distribution.png')


def final_report(c):
    raw, m = c['raw'], c['metrics'].iloc[0]
    refit, test = raw['refit_summary'], raw['test_summary']
    spec = raw['specification']
    parameters = {k: spec['estimator_parameters'][k] for k in ['n_estimators', 'learning_rate', 'max_depth',
        'min_child_weight', 'subsample', 'colsample_bytree', 'reg_alpha', 'reg_lambda', 'gamma',
        'random_state', 'objective', 'eval_metric', 'tree_method']}
    metric_table = c['metrics'].T.reset_index()
    metric_table.columns = ['metric', 'value']
    intervals = c['intervals']
    lines = ['# Final held-out test evaluation', '',
        'The held-out test set was evaluated once after all model-family, feature, calibration, and business-policy decisions were frozen.', '',
        'No model or policy decisions were changed in response to test results.', '',
        '## A. Frozen model specification', '',
        'Demographic-free XGBoost classifier with 42 historical numerical predictors. '
        'Country, Language, Gender, UserID, snapshot_date and churn_60d are not model '
        'inputs. The target is constructed separately. No calibration: probabilities '
        'are the raw classifier outputs. Diagnostic threshold remains **0.50**. '
        'Exact predictor definitions and all estimator defaults remain those in '
        '`final_model_specification.json` / `.md` and `docs/feature_dictionary.md`.', '',
        '```json', json.dumps(parameters, indent=2), '```', '',
        'Numerical preprocessing is median imputation with add_indicator=True, no scaling '
        'and dense output. The complete pipeline was fitted once to final-refit rows '
        'only. The same rule may learn different missing-indicator columns on the expanded '
        'safe history; it does not change raw feature definitions. Transformed refit and '
        'test values were finite. Test transformation did not change imputation statistics.', '',
        '## B. Final refit period', '',
        f'**{refit["start"]} through {refit["end"]}**: **{refit["observations"]:,} '
        f'observations**, **{refit["unique_players"]:,} players**, **{refit["churn_rate"]:.4%}** inactivity. '
        'January-February 2006 are included after selection is frozen. June-July are '
        'excluded. Every refit snapshot satisfies S+60 < 2006-08-02. May 31 + 60 '
        'days = July 30, before the first test observation window begins August 2. '
        'The final fitted pipeline is local at `models/final_xgboost_churn_model.joblib` '
        '(Git-ignored). No test labels entered fitting.', '',
        '## C. Held-out test period', '',
        f'Exactly the four snapshots **August 31, September 30, October 31 and '
        f'November 30, 2006**: **{test["observations"]:,} observations**, '
        f'**{test["unique_players"]:,} players**, **{test["churn_prevalence"]:.4%}** inactivity. '
        'No other snapshots enter test scoring or aggregate results. Panel shape, binary '
        'target, predictor availability and unique player-snapshot keys were checked '
        'before fitting; the data SHA-256 matches the frozen Day 10 artifact.', '',
        '## D. Main test metrics', '', markdown_table(metric_table, digits=6), '',
        'ROC-AUC and average precision summarize ranking; Brier and log loss assess '
        'probability quality. Accuracy is supplementary. All binary metrics and the '
        'confusion matrix use the unchanged 0.50 threshold, with p >= 0.50 positive. '
        'No test threshold search was performed.', '',
        '## E. Whole-player cluster-bootstrap uncertainty', '', markdown_table(intervals, digits=6), '',
        '1,000 replicates, random_state=42. Each draw samples 1,455 whole UserID clusters '
        'with replacement, keeping every snapshot and repeated-cluster multiplicity. '
        'Row counts can vary. Capacity counts and prevalence are recalculated per draw. '
        'Intervals are the 2.5th and 97.5th percentiles, conditional on the fixed fitted '
        'model. They do not capture training uncertainty, earlier model selection, '
        'shared calendar shocks or future temporal drift. Undefined metrics are excluded '
        'per metric and valid counts are reported. ROC-AUC requires both classes; AP '
        'requires at least one positive; capacity recall/lift require positives. '
        'Diagnostic precision/recall/F1 use the existing zero_division=0 convention. '
        'Brier remains defined in single-class draws.', '',
        '## F. Validation versus test generalization', '', markdown_table(c['comparison'], digits=6), '',
        GENERALIZATION, '',
        '## G. Fixed-budget targeting', '', markdown_table(c['targeting'], digits=6), '',
        'The frozen 5/10/15/20/25/30% capacities use ceil(n*f), stable original-panel '
        'input-order ties, and no label-based tie-breaking. Precision is the observed '
        'inactivity rate among selected observations; recall is the share of all '
        'inactivity outcomes captured; lift is precision / overall test prevalence. '
        'These are player-snapshot observations, not deduplicated campaign recipients. '
        'No optimal capacity, financial threshold, retention treatment effect, customer '
        'value or ROI is inferred.', '',
        '## H. Frozen percentile risk bands', '', markdown_table(c['risk_bands'], digits=6), '',
        'Bands use TEST score ranking only: bottom 50%, 50th-80th, 80th-90th and top '
        '10%. Descending-rank cuts use ceil(n*.1), ceil(n*.2), ceil(n*.5), then n. '
        'They are disjoint relative prioritization bands; tied scores can span bands. '
        'They are not clinical or regulatory categories. The test labels are used '
        'only to describe outcomes after band assignment.', '',
        '## I. Raw test calibration and score overlap', '',
        f'Mean predicted probability **{m.mean_predicted_probability:.4%}**, observed '
        f'inactivity **{m.observed_churn_prevalence:.4%}**: mean overprediction '
        f'**{100*(m.mean_predicted_probability-m.observed_churn_prevalence):.3f} percentage points**. '
        f'Brier **{m.brier_score:.6f}**, log loss **{m.log_loss:.6f}**, '
        f'ECE **{m.ece:.6f}**.', '',
        'ECE uses the same ten fixed equal-width bins as Day 10: [0,.1), ... [.9,1], '
        'including 1.0 in the last bin. ECE = sum(n_bin/n * abs(mean probability '
        '- observed frequency)); empty bins contribute zero. The reliability figure '
        'includes bin counts and marks sparse bins. No calibration model, slope map '
        'or test-label-derived mapping was fitted; raw probabilities remain frozen.', '',
        'The class-conditional density plot shows substantial overlap: both active and '
        'inactive outcomes occur on each side of 0.50. The false positives and false '
        'negatives in the confusion matrix demonstrate that these scores cannot '
        'deterministically identify an individual outcome. No additional cutoff '
        'was selected from the distribution plot.', '',
        '## J. Monthly descriptive stability', '', markdown_table(c['monthly'], digits=6), '',
        'September has the strongest ROC-AUC (0.7730); August the weakest (0.7331). '
        'October has the lowest observed prevalence (30.43%) and average precision '
        '(0.5064); predicted mean risk remains approximately 34-36% across months. '
        'Top-10% lift ranges from 1.6401 to 1.8858, based on 67-78 selected rows per '
        'month. Each monthly snapshot contains one row per player, but players repeat '
        'across months. These descriptive differences have no month-specific '
        'uncertainty adjustment and do not justify post-test tuning.', '',
        '## K. Limitations and final governance', '',
        '- The label is no positive-bet activity during the 60 days immediately after '
        'a snapshot. It predicts extended inactivity, not permanent departure; '
        'reactivation after the outcome window remains possible.\n'
        '- The source is historical bwin Virtual Casino Gambling data (2005-2007), '
        'not a present-day operational sample. Temporal drift and historical cohort '
        'selection limit transferability.\n'
        '- Same players can appear in refit and test, following the frozen future '
        'behavior of known players design; results do not establish unseen-player '
        'generalization. Player-cluster intervals do not account for shared calendar shocks.\n'
        '- Validation was reused for model-family, feature and calibration decisions. '
        'The single held-out evaluation estimates the frozen procedure; it is not '
        'permission to iterate against this now-used test period.\n'
        '- Correlated predictors and model explanations are associative, not causal. '
        'Demographics are excluded, but behavioral proxies can remain. No protected-group '
        'outcome analysis or fairness guarantee is available.\n'
        '- Risk scoring does not measure gambling harm, customer value, causal '
        'retention impact or responsible-gambling status; it must not drive automated '
        'adverse decisions. Governance review, recent-data validation, monitoring '
        'and a retraining plan are required before real deployment.\n'
        '- Capacity results count snapshots, not unique recipients. Campaign '
        'deduplication, contact costs and eligibility remain separate requirements.\n'
        '- Test probabilities were generated in one predict_proba call from one '
        'final refit. Notebook reruns load the ignored local checkpoint, not a new '
        'model or prediction vector. Only aggregate reports and anonymous figures '
        'are exported under reports/. No commit or push was performed.', '',
        'The held-out test set was evaluated once after all model-family, feature, calibration, and business-policy decisions were frozen.', '',
        'No model or policy decisions were changed in response to test results.', '']
    return '\n'.join(lines)


def model_card(c):
    raw, m = c['raw'], c['metrics'].iloc[0]
    return f'''# Model card: final iGaming inactivity-risk XGBoost

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
**{m.observed_churn_prevalence:.4%}** inactivity. It was scored once. Original known-player
temporal evaluation permits repeated players across partitions and months.

## Performance

- ROC-AUC **{m.roc_auc:.4f}**, average precision **{m.average_precision:.4f}**.
- Brier **{m.brier_score:.4f}**, log loss **{m.log_loss:.4f}**, ECE **{m.ece:.4f}**.
- At 0.50: precision **{m.precision:.2%}**, recall **{m.recall:.2%}**, F1 **{m.f1:.4f}**.
- Top 10%: lift **{m.lift_at_top10pct:.4f}**, recall **{m.recall_at_top10pct:.2%}**.
- Top 20%: lift **{m.lift_at_top20pct:.4f}**, recall **{m.recall_at_top20pct:.2%}**.

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
Fitted model SHA-256: `{raw['model_sha256']}`.
Frozen specification SHA-256: `{raw['fingerprint']['specification_sha256']}`.
Aggregate evidence is tracked separately from ignored data and private scoring
checkpoints. No public player-level predictions are exported.

The held-out test set was evaluated once after all model-family, feature, calibration, and business-policy decisions were frozen.

No model or policy decisions were changed in response to test results.
'''


def export_final_report(c):
    import sklearn
    import xgboost
    export_aggregate_tables(c)
    create_final_figures(c)
    reports = c['root'] / 'reports'
    (reports / 'final_test_evaluation.md').write_text(final_report(c), encoding='utf-8')
    (reports / 'model_card.md').write_text(model_card(c), encoding='utf-8')
    raw = c['raw']
    metadata = dict(**raw['fingerprint'], final_model_sha256=raw['model_sha256'],
        refit=raw['refit_summary'], test=raw['test_summary'],
        predictor_count=len(raw['specification']['predictors']),
        transformed_features=raw['transformed_features'], demographics_retained=False,
        calibration='raw', diagnostic_threshold=.50, model_fit_calls=raw['fit_calls'],
        test_predict_proba_calls=raw['test_probability_calls'], bootstrap_replicates=1000,
        random_state=42, player_level_predictions_exported=False,
        post_test_model_or_policy_changes=False, test_status='evaluated once; no longer unused holdout',
        python=platform.python_version(), sklearn=sklearn.__version__, xgboost=xgboost.__version__)
    (reports / 'final_test_run_metadata.json').write_text(json.dumps(metadata, indent=2)+'\n', encoding='utf-8')
    if sha256(reports / 'final_model_specification.json') != raw['fingerprint']['specification_sha256']:
        raise RuntimeError('Frozen specification changed after test scoring.')

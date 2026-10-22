"""Orchestrate Day 10 validation work and freeze reviewable local artifacts.

Run: python -m src.final_validation (from the repository root).
No test frame is constructed or passed to an estimator. The sole full TRAIN
ablation and the purged calibration-fold fits are explicitly authorized here.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from src.business_policy import (ABLATION_TOLERANCES, COMPARISON_METRICS,
    ablation_comparison, apply_calibrator, calibration_curve_table, calibration_metrics,
    capacity_table, diagnostic_threshold_table, fit_original_train, fit_train_calibrators,
    generate_train_oof, paired_cluster_bootstrap, policy_probabilities, refit_protocol,
    risk_band_table, score_metrics, select_calibration, verify_specification)
from src.explainability import load_candidate, markdown_table
from src.features import CATEGORICAL_COLUMNS, PREDICTOR_COLUMNS, TARGET_COLUMN
from src.modelling import modelling_partition


def load_inputs(root):
    root = Path(root).resolve()
    context = load_candidate(root)
    panel = pd.read_csv(root / 'data/processed/model_panel_60d.csv', parse_dates=['snapshot_date'])
    context['train'] = modelling_partition(panel, 'train')
    context['refit_summary'], context['refit_snapshots'], context['refit_boundary'] = refit_protocol(panel)
    del panel
    context['rf'] = joblib.load(root / 'models/random_forest.joblib')
    context['artifact_hashes'] = {name: hashlib.sha256((root / 'models' / name).read_bytes()).hexdigest()
                               for name in ['random_forest.joblib', 'xgboost_model.joblib']}
    context['rf_probabilities'] = policy_probabilities(context['rf'], context['validation'], 'validation')
    context['xgb_probabilities'] = policy_probabilities(context['pipeline'], context['validation'], 'validation')
    y = context['validation'][TARGET_COLUMN]
    context['candidate_metrics'] = pd.DataFrame([
        dict(model='Random Forest', **score_metrics(y, context['rf_probabilities'])),
        dict(model='XGBoost', **score_metrics(y, context['xgb_probabilities']))])
    for name, report in [('Random Forest', 'random_forest'), ('XGBoost', 'xgboost')]:
        saved = pd.read_csv(root / f'reports/{report}_validation_metrics.csv').iloc[0]
        observed = context['candidate_metrics'].set_index('model').loc[name]
        for metric in COMPARISON_METRICS:
            np.testing.assert_allclose(observed[metric], saved[metric], atol=5e-5, rtol=0)
    return context


def compare_candidates(context, replicates=1000):
    c = context
    y = c['validation'][TARGET_COLUMN]
    c['targeting'] = capacity_table(y, {'Random Forest': c['rf_probabilities'], 'XGBoost': c['xgb_probabilities']})
    c['bootstrap'] = paired_cluster_bootstrap(c['validation'], c['rf_probabilities'], c['xgb_probabilities'], replicates)
    # Exactly one untuned full-TRAIN demographic ablation; preserve original models.
    c['demographic_free_model'] = fit_original_train(c['pipeline'], c['train'], demographics=False)
    c['demographic_free_probabilities'] = policy_probabilities(c['demographic_free_model'], c['validation'],
                                                             'validation', demographics=False)
    c['ablation'] = ablation_comparison(y, c['xgb_probabilities'], c['demographic_free_probabilities'])
    c['demographics_retained'] = not bool(c['ablation'].within_tolerance.all())
    c['selected_model'] = c['pipeline'] if c['demographics_retained'] else c['demographic_free_model']
    c['selected_raw_probabilities'] = (c['xgb_probabilities'] if c['demographics_retained'] else
                                       c['demographic_free_probabilities'])
    # Family decision considers the entire evidence, not a single small metric edge.
    c['selected_family'] = 'XGBoost'
    return c


def assess_calibration(context):
    c = context
    y = c['validation'][TARGET_COLUMN]
    oof, c['calibration_folds'] = generate_train_oof(c['pipeline'], c['train'], c['demographics_retained'])
    c['oof_summary'] = dict(observations=len(oof), unique_players=oof.UserID.nunique(),
                            churn_rate=float(oof.label.mean()), folds=len(c['calibration_folds']))
    c['calibrators'] = fit_train_calibrators(oof)
    del oof  # Player-level OOF probabilities remain in memory only, never exported.
    # Follow the declared protocol: frozen estimator refitted to complete original TRAIN.
    # This is not the future pre-test refit. It never includes Jan-May 2006.
    c['calibration_base_model'] = fit_original_train(c['pipeline'], c['train'], c['demographics_retained'])
    p = policy_probabilities(c['calibration_base_model'], c['validation'], 'validation', c['demographics_retained'])
    np.testing.assert_allclose(p, c['selected_raw_probabilities'], atol=1e-7, rtol=0)
    c['selected_model'] = c['calibration_base_model']
    c['calibration_predictions'] = {'raw': p, **{name: apply_calibrator(mapping, p)
        for name, mapping in c['calibrators'].items()}}
    c['calibration_comparison'] = pd.DataFrame([dict(method=name, **calibration_metrics(y, values))
        for name, values in c['calibration_predictions'].items()])
    c['calibration_curves'] = pd.concat([calibration_curve_table(y, values).assign(method=name)
        for name, values in c['calibration_predictions'].items()], ignore_index=True)
    c['calibration_choice'] = select_calibration(c['calibration_comparison'])
    c['final_probabilities'] = c['calibration_predictions'][c['calibration_choice']]
    c['thresholds'] = diagnostic_threshold_table(y, c['final_probabilities'])
    c['risk_bands'] = risk_band_table(y, c['final_probabilities'])
    c['final_targeting'] = capacity_table(y, {'Final selected XGBoost': c['final_probabilities']})
    train_p = policy_probabilities(c['selected_model'], c['train'], 'train', c['demographics_retained'])
    training = score_metrics(c['train'][TARGET_COLUMN], train_p)
    heldout = score_metrics(y, c['selected_raw_probabilities'])
    c['selected_generalization'] = pd.DataFrame([dict(metric=metric, train=training[metric],
        validation=heldout[metric], train_minus_validation=training[metric]-heldout[metric])
        for metric in ['roc_auc', 'average_precision', 'brier_score']])
    return c


def json_safe(value):
    if isinstance(value, dict):
        return {key: json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    if isinstance(value, (np.integer, np.floating)):
        value = value.item()
    if isinstance(value, float) and not np.isfinite(value):
        return 'NaN' if np.isnan(value) else str(value)
    return value


def freeze_specification(context):
    c = context
    pipeline = c['selected_model']
    pre = pipeline['preprocessing']
    spec = dict(selected_family=c['selected_family'],
        estimator_parameters=json_safe(pipeline['model'].get_params()),
        predictors=pre.feature_names_in_.tolist(),
        transformed_features=pre.get_feature_names_out().tolist(),
        demographics_retained=c['demographics_retained'],
        calibration_choice=c['calibration_choice'], diagnostic_threshold=.50,
        threshold_role='diagnostic only, not financially optimal',
        capacity_fractions=[.05, .10, .15, .20, .25, .30],
        primary_capacity_reporting=[.10, .20],
        capacity_rounding='ceil(n * fraction), stable original-panel input order breaks score ties',
        risk_bands='bottom 50%; 50th-80th; 80th-90th; top 10%, disjoint rank slices',
        observation_days=30, outcome_days=60,
        churn_definition='No positive-bet day in the 60 calendar dates after snapshot S: (S,S+60].',
        original_train=['2005-03-31', '2005-12-31'], validation=['2006-03-31', '2006-05-31'],
        development_embargo_months=['2006-01', '2006-02', '2006-06', '2006-07'],
        test=['2006-08-31', '2006-11-30'],
        preprocessing=dict(numeric='train-median imputation plus training-learned missing indicators; no scaling',
            categorical='train-most-frequent imputation; one-hot, unknown categories ignored' if c['demographics_retained'] else 'excluded',
            output='dense, preserving legitimate numerical zeros; remainder dropped'),
        primary_metrics=['average_precision', 'roc_auc', 'brier_score'],
        refit_protocol=c['refit_summary'],
        calibration_refit_protocol='If a calibrator is selected, retain the frozen TRAIN-OOF mapping; do not refit it on validation or test labels.',
        freeze_statement='All model-development decisions were frozen before evaluating the held-out test period.',
        test_predictions_generated=False)
    verify_specification(spec, pipeline)
    c['specification'] = spec
    return spec


def create_figures(c):
    import matplotlib.pyplot as plt
    from matplotlib.ticker import MaxNLocator
    directory = c['root'] / 'reports/figures'
    directory.mkdir(exist_ok=True)
    def save(name):
        plt.gcf().savefig(directory / name, dpi=160, bbox_inches='tight')
        plt.close('all')
    for metric in ['lift', 'recall']:
        plt.figure(figsize=(8, 4.5))
        for model, table in c['targeting'].groupby('model', sort=False):
            plt.plot(table.target_fraction*100, table[metric], '.-', label=model)
        if metric == 'lift':
            plt.axhline(1, color='gray', linestyle=':', linewidth=1)
        plt.xlabel('Targeting capacity (% of validation observations)')
        plt.ylabel('Lift versus validation prevalence' if metric == 'lift' else 'Share of churn observations captured')
        plt.title('Fixed-budget comparison: frozen Day 8 candidates')
        plt.legend()
        save(f'policy_{metric}_by_capacity.png')
    # Separate axes retain native units for AUC, Brier, lift and recall.
    table = c['bootstrap']
    fig, axes = plt.subplots(4, 2, figsize=(11, 10))
    for ax, row in zip(axes.flat, table.itertuples()):
        ax.hlines(0, row.ci95_low, row.ci95_high, linewidth=3)
        ax.plot(row.observed_difference, 0, 'o', color='tab:orange', label='Observed difference')
        ax.plot(row.bootstrap_mean_difference, 0, 'x', color='black', label='Bootstrap mean')
        ax.axvline(0, color='gray', linestyle=':')
        ax.set_title(row.metric + ' (' + ('negative favors XGB' if row.metric == 'brier_score' else 'positive favors XGB') + ')', fontsize=10)
        ax.set_yticks([])
        ax.xaxis.set_major_locator(MaxNLocator(nbins=5))
        ax.tick_params(axis='x', labelsize=9)
        ax.set_xlabel('XGBoost minus Random Forest')
    axes.flat[-1].axis('off')
    handles, labels = axes.flat[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='lower right')
    fig.suptitle('Paired player-cluster bootstrap: 95% percentile intervals')
    fig.tight_layout(rect=(0, .02, 1, .96))
    save('policy_bootstrap_intervals.png')
    plt.figure(figsize=(6.5, 6))
    plt.plot([0, 1], [0, 1], '--', color='gray', label='Ideal calibration')
    for method, curve in c['calibration_curves'].groupby('method', sort=False):
        nonempty = curve.loc[curve.observations.gt(0)]
        plt.plot(nonempty.mean_probability, nonempty.observed_prevalence, '.-', label=method)
        for row in nonempty.loc[nonempty.observations.lt(20)].itertuples():
            plt.annotate(f'n={row.observations}', (row.mean_probability, row.observed_prevalence),
                         xytext=(4, -12), textcoords='offset points', fontsize=8)
    plt.xlabel('Mean predicted risk (equal-width bins)')
    plt.ylabel('Observed 60-day inactivity rate')
    plt.title('Validation calibration: TRAIN-derived mappings')
    plt.xlim(0, 1)
    plt.ylim(0, 1)
    plt.legend()
    save('policy_calibration_comparison.png')
    bands = c['risk_bands'].iloc[::-1]
    plt.figure(figsize=(9, 4.5))
    plt.bar(bands.band, bands.actual_churn_rate)
    plt.axhline(c['validation'][TARGET_COLUMN].mean(), color='gray', linestyle='--', label='Validation prevalence')
    plt.ylabel('Observed 60-day inactivity rate')
    plt.title('Validation risk bands: final selected scoring policy')
    plt.xticks(rotation=12, ha='right')
    plt.legend()
    save('policy_risk_band_churn.png')


def decision_report(c):
    raw = c['calibration_comparison'].set_index('method').loc['raw']
    diagnostic = c['thresholds'].sort_values(['f1', 'threshold'], ascending=[False, True]).iloc[0]
    boot = c['bootstrap'].set_index('metric')
    include = c['demographics_retained']
    lines = ['# Final validation decision: Day 10', '',
        '## Decision and scope', '',
        '**Final model family: XGBoost.** Its stronger overall validation discrimination '
        'and probability quality, much smaller train-validation gaps, and established SHAP '
        'explanations outweigh Random Forest\'s small observed top-budget advantage. '
        'No financial utility function or universal contact capacity is assumed. '
        'This is the final development/validation decision before one sealed test evaluation.', '',
        f'Variant: **{"demographics retained" if include else "demographics removed"}**. '
        f'Probability policy: **{c["calibration_choice"]}**. The exact specification and '
        'all estimator parameters are frozen in `final_model_specification.md` and '
        '`final_model_specification.json`. Logistic Regression remains the historical baseline.', '',
        '## Validation reproduction and generalization', '',
        f'Original TRAIN: 2005-03-31 through 2005-12-31 ({len(c["train"]):,} observations). '
        f'Validation: 2006-03-31 through 2006-05-31 ({len(c["validation"]):,} observations, '
        f'{c["validation"].UserID.nunique():,} players). All saved candidate metrics '
        'reproduce within 0.00005. No test predictor matrix or predictions were created.', '',
        markdown_table(c['candidate_metrics']), '',
        'Historical train-minus-validation ROC-AUC gaps: RF 0.0763, XGBoost 0.0398; '
        'average-precision gaps: RF 0.1265, XGBoost 0.0675. These favor XGBoost\'s '
        'generalization behavior, but are descriptive and do not establish future performance. '
        'Logistic baseline: ROC-AUC 0.7295, AP 0.5342, Brier 0.1963. XGBoost adds '
        'a dependency versus sklearn RF, but both use the same practical train-fitted '
        'preprocessing and are already implemented and explainable.', '',
        'Raw final selected variant train/validation metrics (before any probability mapping):', '',
        markdown_table(c['selected_generalization']), '',
        '## Fixed-budget business comparison', '', markdown_table(c['targeting']), '',
        'Targeting units are player-snapshot observations, not unique campaign recipients. '
        'Each capacity selects ceil(n * fraction) observations; ties preserve original-panel '
        'input order without consulting labels. Precision is the observed inactivity rate '
        'among selected rows; recall is the share of inactivity outcomes captured; lift '
        'is precision divided by overall validation prevalence. A capacity must be chosen '
        'from actual operational resources, not inferred from these metrics.', '',
        '## Paired cluster-bootstrap uncertainty', '', markdown_table(c['bootstrap']), '',
        'Each replicate draws the same number of distinct-player clusters with replacement, '
        'retains every validation snapshot for each sampled player (including repeated copies '
        'when a player is drawn twice), and evaluates both models on identical draws. '
        'Rank capacities and prevalence are recomputed in each replicate. Total row counts '
        'can vary. Intervals are 2.5th/97.5th percentiles of XGBoost-minus-RF differences; '
        'Brier favors negative differences. The proportion better is a descriptive bootstrap '
        'frequency, not a posterior probability, p-value, or proof of superiority. '
        'Intervals condition on the fitted models and selected validation period; they '
        'do not include model-selection or training uncertainty.', '']
    for metric in ['lift_at_top10pct', 'recall_at_top10pct', 'lift_at_top20pct', 'recall_at_top20pct']:
        row = boot.loc[metric]
        conclusion = ('contains zero: the observed RF advantage is not clearly separated from validation uncertainty'
                      if row.ci95_low <= 0 <= row.ci95_high else
                      'excludes zero in this conditional analysis; selection optimism and temporal uncertainty remain')
        lines += [f'- {metric}: XGB minus RF {row.observed_difference:+.5f}, '
                  f'95% CI [{row.ci95_low:+.5f}, {row.ci95_high:+.5f}]; {conclusion}.']
    lines += ['', '## One demographic sensitivity analysis', '',
        'Retrained the exact selected XGBoost settings once on original TRAIN with Country, '
        'Language and Gender removed. No tuning or alternative feature search. Before running, '
        'practical-loss tolerances were fixed at ROC-AUC/AP 0.005, Brier increase 0.002, '
        'lift decrease 0.05 and recall decrease 0.01 at both 10% and 20%. These are '
        'declared project judgment limits, not profit, fairness or statistical thresholds. '
        'Removal is preferred only if every metric meets its tolerance.', '',
        markdown_table(c['ablation']), '',
        ('At least one tolerance is exceeded; retain demographics and require governance '
         'review before deployment.' if include else
         'All losses fall within the declared tolerances; freeze the demographic-free variant '
         'for simpler inputs and reduced direct demographic dependence.'), '',
        'This sensitivity does not establish group fairness, remove behavioral proxies, '
        'or quantify protected-group outcomes. Demographic exclusion is not proof of fairness. '
        'Prior Day 9 SHAP findings explain the full model; they are not automatically exact '
        'attributions for a changed feature set. The paired bootstrap above compares the '
        'original full-feature candidates; it does not quantify uncertainty for the '
        'demographic ablation or prove the final variant superior to RF.', '',
        '## Calibration assessment and TRAIN-only experiment', '',
        f'Leading variant raw Brier {raw.brier_score:.6f}, log loss {raw.log_loss:.6f}, '
        f'mean probability {raw.mean_probability:.4%}, observed prevalence '
        f'{raw.observed_prevalence:.4%}, ECE {raw.ece:.6f}. Descriptive validation '
        f'logistic calibration intercept {raw.calibration_intercept:+.4f}, slope '
        f'{raw.calibration_slope:.4f}; ideal values are 0 and 1. This regression is an '
        'assessment only: it is never applied as a calibration mapping.', '',
        'ECE uses ten fixed equal-width probability bins [0,.1), ... [.9,1]; '
        'ECE = sum(n_bin/n * abs(mean probability - observed frequency)). Empty bins '
        'contribute zero. Logistic calibration diagnostics fit outcome on clipped '
        'logit(probability), clipping to [1e-6,1-1e-6] for finite logits. Calibration '
        'curves can be noisy in sparse high-score bins.', '',
        'The calibration experiment uses five expanding temporal folds within TRAIN. '
        'Every fitting snapshot\'s 60-day label horizon ends strictly before the scoring '
        'snapshot\'s 30-day observation window starts, stronger than a 60-day snapshot '
        'gap. Preprocessing is independently fitted within each fitting fold. Repeated '
        'players across time remain permitted by the original known-player design.', '',
        markdown_table(c['calibration_folds']), '',
        f'Pooled OOF support: {c["oof_summary"]["observations"]:,} rows, '
        f'{c["oof_summary"]["unique_players"]:,} players, '
        f'{c["oof_summary"]["churn_rate"]:.2%} inactivity. Sigmoid fits a nonnegative-slope '
        'two-parameter logistic map of OOF logits. Isotonic is allowed only with at least '
        '1,000 OOF observations, 100 per class and 100 distinct scores. Both mappings '
        'are fitted exclusively on pooled TRAIN-OOF labels and predictions. The exact '
        'frozen base estimator is then refitted to complete original TRAIN before '
        'validation scoring. This is not the future pre-test refit.', '',
        markdown_table(c['calibration_comparison']), '',
        'Prespecified selection rule: retain raw unless a mapping improves Brier by at '
        'least 0.001 and log loss by at least 0.002, with ECE worsening no more than '
        '0.002. Prefer sigmoid if within Brier 0.0005 and log loss 0.001 of the best '
        'eligible method. These are practical tolerances, not significance tests. '
        'Isotonic can introduce ties and endpoint scores; ranking changes are not evidence '
        'of improved discrimination. Earlier TRAIN-OOF and later validation distributions '
        'can differ, and selecting a mapping using validation adds selection optimism.', '',
        f'**Selected calibration: {c["calibration_choice"]}.** '
        + ('The mappings do not provide sufficiently consistent material gains under the '
           'declared rule, so retain raw scores.' if c['calibration_choice'] == 'raw' else
           'The selected TRAIN-derived mapping meets the declared probability-quality rule. '
           'Its parameters are frozen; do not fit it on validation or test outcomes.'), '',
        '## Business policy and diagnostic thresholds', '',
        'Primary output is the final model\'s probability/risk score for extended inactivity. '
        'Use ranking to prioritize within a real capacity: top 10% and top 20% are '
        'standard reporting scenarios; 5/15/25/30% remain alternatives. No contact costs, '
        'customer lifetime value, treatment effect, incremental retention or ROI are known. '
        'No probability threshold is claimed to maximize profit. Campaign design must '
        'separately handle repeated-player contacts, eligibility and governance.', '',
        'Freeze **0.50** as the diagnostic confusion-matrix threshold for the next '
        'notebook. It is not a deployment or financial optimum.', '',
        markdown_table(c['thresholds']), '',
        f'The maximum-F1 threshold on this **prespecified grid** is '
        f'{diagnostic.threshold:.2f} (F1 {diagnostic.f1:.4f}); this is a validation '
        'diagnostic reference only, not a replacement for the frozen 0.50 diagnostic '
        'or an exhaustive optimum over all possible scores.', '',
        '## Risk bands', '', markdown_table(c['risk_bands']), '',
        'Four bands partition the ranked population: bottom 50%, 50th-80th, 80th-90th '
        'and top 10%. Starting from descending score order, boundaries use ceil(n*.1), '
        'ceil(n*.2), ceil(n*.5), then n. Ties use stable original-panel order and can '
        'split identical scores across bands. Bands are relative to the scoring cohort, '
        'not fixed probability cutoffs; mean scores and observed rates need not be '
        'perfectly monotonic. They are presentation/prioritization tools, not clinical '
        'or regulatory categories. Recompute ranks within each future eligible campaign '
        'cohort without labels; deduplication requires a separate operational policy.', '',
        'Final selected variant/policy capacity results (the RF/XGB comparison above uses '
        'the original full-feature candidates):', '', markdown_table(c['final_targeting']), '',
        '## Exact pre-test refit protocol: defined, not executed', '',
        f'First test snapshot: {c["refit_summary"]["first_test_snapshot"]}; inclusive '
        f'30-day observation starts {c["refit_summary"]["first_test_observation_start"]}. '
        'Require S + 60 days < 2006-08-02, so the latest admissible calendar snapshot '
        f'is **{c["refit_summary"]["latest_safe_calendar_snapshot"]}**. For month-end '
        'snapshots the last safe one is **2006-05-31** (outcome ends 2006-07-30).', '',
        markdown_table(c['refit_boundary']), '',
        'January and February 2006 may join the future refit because their original '
        'embargo role separated development train/validation; their outcomes are '
        'available before the first test observation period. June and July remain '
        'excluded. No unsafe rows are used for the aggregate counts below.', '',
        markdown_table(c['refit_snapshots']), '',
        f'Planned final refit: **{c["refit_summary"]["refit_start"]} through '
        f'{c["refit_summary"]["refit_end"]}**, **{c["refit_summary"]["observations"]:,} '
        f'observations**, **{c["refit_summary"]["unique_players"]:,} players**, '
        f'**{c["refit_summary"]["churn_rate"]:.4%}** inactivity. This notebook reports '
        'eligibility only and does not perform this pre-test refit.', '',
        'Next notebook: verify the frozen specification and data hash, select only '
        'eligible snapshots by the strict date rule, fit the frozen preprocessing and '
        'estimator on that pre-test data, and retain the already frozen TRAIN-OOF '
        'calibration mapping if selected (otherwise no mapping). Missing indicators '
        'and category levels may be relearned by the same preprocessing algorithm '
        'on safe pre-test data; raw predictor definitions/hyperparameters remain fixed. '
        'Freeze the resulting fitted artifact before making test predictions.', '',
        '## One-time held-out evaluation plan', '',
        'Score only 2006-08-31 through 2006-11-30 once after the freeze/refit. Compute '
        'ROC-AUC; PR-AUC as average precision; Brier; log loss; precision, recall, F1 '
        'and specificity at diagnostic 0.50; lift/recall at top 10% and top 20%. '
        'Use 1,000 seeded (42) whole-UserID cluster bootstrap replicates for 95% '
        'percentile intervals. Ranking ties use stable original-panel order. Produce '
        'the calibration plot (same ten bins), ROC curve, PR curve, cumulative gains/lift '
        'and confusion matrix. Freeze risk bands/capacity definitions; do not choose '
        'a new threshold, calibrator, feature set or family after inspecting test outcomes. '
        'If results disappoint, report them; any later development requires genuinely '
        'new holdout data. No post-test tuning is allowed.', '',
        '## Limitations and status', '',
        '- Validation has been reused across model-family, feature-variant and calibration '
        'choices, creating selection optimism; bootstrap intervals do not correct it.\n'
        '- Repeated-player clustering respects dependence within a player but not shared '
        'calendar shocks; only three validation months are available.\n'
        '- Historical 2005-2006 data, cohort drift, correlated features and unverified '
        'demographic history limit generalization; no causal retention benefit or ROI '
        'is established.\n'
        '- Scores concern extended inactivity, not permanent departure, gambling harm, '
        'problem gambling, customer value or causal behavioral explanations.\n'
        '- Protected-group outcome analysis is unavailable; neither ablation nor SHAP '
        'is a fairness audit.\n'
        '- Calibration trained on earlier OOF periods may not transfer to a final '
        'pre-test refit or future populations.\n'
        '- TEST remains completely sealed. No pre-test final refit, test transformation, '
        'test predictions, probability inspection, or test evaluation occurred. '
        'No player-level predictions were exported; original artifacts were preserved. '
        'No commit or push was performed.', '',
        'All model-development decisions were frozen before evaluating the held-out test period.', '']
    return '\n'.join(lines)


def specification_report(c):
    spec = c['specification']
    predictors = '\n'.join('- `' + name + '`' for name in spec['predictors'])
    params = json.dumps(spec['estimator_parameters'], indent=2, allow_nan=False)
    return f'''# Frozen final model specification

All model-development decisions were frozen before evaluating the held-out test period.

## Selected family and exact estimator settings

**{spec['selected_family']}**; demographics **{'retained' if spec['demographics_retained'] else 'removed'}**;
calibration **{spec['calibration_choice']}**. Diagnostic threshold **0.50** only.
Selection evidence and uncertainty are documented in `final_validation_decision.md`.

Every estimator parameter (including defaults) is frozen below; `"NaN"` denotes
the numerical missing sentinel rather than a literal string passed to XGBoost.

```json
{params}
```

## Exact raw predictor set ({len(spec['predictors'])} columns)

{predictors}

Identifiers, snapshot date and outcome are excluded. The machine-readable JSON
also records the original-TRAIN transformed-column names for artifact validation.
Feature definitions remain those in `docs/feature_dictionary.md`.

## Preprocessing and missing values

Numerical: median imputation learned from the fitting partition plus missing-value
indicators for columns missing during that fit. No scaling. Categorical:
{spec['preprocessing']['categorical']}. Dense output preserves numerical zeros;
unlisted columns are dropped. No target encoding or class reweighting.
For the future safe pre-test refit, fit the same preprocessing algorithm afresh;
indicator columns/category levels may change with safe fitting data, but the raw
predictor set and treatment rules stay frozen.

## Probability mapping

Selected method: **{spec['calibration_choice']}**. Candidates were raw, TRAIN-only
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
**{c['refit_summary']['refit_start']} through {c['refit_summary']['refit_end']}**,
including January-February 2006; exclude June-July. This yields
**{c['refit_summary']['observations']:,} observations**, **{c['refit_summary']['unique_players']:,} players**,
and **{c['refit_summary']['churn_rate']:.4%}** churn. May's labels end 2006-07-30.

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
'''


def export_outputs(context):
    c = context
    reports = c['root'] / 'reports'
    freeze_specification(c)
    tables = dict(business_targeting_validation=c['targeting'], model_bootstrap_comparison=c['bootstrap'],
        demographic_ablation=c['ablation'], calibration_comparison=c['calibration_comparison'],
        final_candidate_validation_metrics=c['candidate_metrics'], calibration_temporal_folds=c['calibration_folds'],
        final_selected_generalization=c['selected_generalization'],
        final_calibration_validation_bins=c['calibration_curves'], final_validation_thresholds=c['thresholds'],
        final_validation_risk_bands=c['risk_bands'], final_policy_targeting_validation=c['final_targeting'],
        final_refit_snapshot_summary=c['refit_snapshots'], final_refit_boundary=c['refit_boundary'])
    for name, table in tables.items():
        table.to_csv(reports / (name + '.csv'), index=False)
    (reports / 'final_validation_decision.md').write_text(decision_report(c), encoding='utf-8')
    (reports / 'final_model_specification.md').write_text(specification_report(c), encoding='utf-8')
    (reports / 'final_model_specification.json').write_text(json.dumps(c['specification'], indent=2, allow_nan=False)+'\n', encoding='utf-8')
    create_figures(c)
    # Ignored local model outputs contain fitted artifacts, never prediction rows.
    joblib.dump(c['selected_model'], c['root'] / 'models/final_development_model.joblib')
    mapping = None if c['calibration_choice'] == 'raw' else c['calibrators'][c['calibration_choice']]
    joblib.dump(dict(method=c['calibration_choice'], mapping=mapping),
                c['root'] / 'models/final_development_calibration.joblib')
    model_hash = hashlib.sha256((c['root'] / 'models/final_development_model.joblib').read_bytes()).hexdigest()
    calibration_hash = hashlib.sha256((c['root'] / 'models/final_development_calibration.joblib').read_bytes()).hexdigest()
    metadata = dict(data_sha256=c['data_sha256'], original_model_hashes=c['artifact_hashes'],
        final_development_model_sha256=model_hash, final_development_calibration_sha256=calibration_hash,
        bootstrap_replicates=int(c['bootstrap'].valid_replicates.iloc[0]), random_state=42,
        original_train_observations=len(c['train']), validation_observations=len(c['validation']),
        oof_support=c['oof_summary'], ablation_loss_tolerances=ABLATION_TOLERANCES,
        final_family=c['selected_family'], demographics_retained=c['demographics_retained'],
        calibration=c['calibration_choice'], diagnostic_threshold=.5,
        validation_decisions_frozen=True, test_predictions_generated=False,
        test_probabilities_inspected=False, test_evaluated=False, pretest_refit_executed=False,
        player_level_predictions_exported=False)
    (reports / 'final_validation_run_metadata.json').write_text(json.dumps(metadata, indent=2)+'\n', encoding='utf-8')
    for name, before in c['artifact_hashes'].items():
        if hashlib.sha256((c['root'] / 'models' / name).read_bytes()).hexdigest() != before:
            raise RuntimeError('Original model artifact changed.')
    verify_specification(c['specification'], joblib.load(c['root'] / 'models/final_development_model.joblib'))


def run(root='.', replicates=1000):
    c = load_inputs(root)
    print('Saved candidate validation metrics reproduced.', flush=True)
    compare_candidates(c, replicates)
    print('Paired cluster bootstrap and single demographic ablation complete.', flush=True)
    print(c['ablation'].to_string(index=False), flush=True)
    assess_calibration(c)
    print(c['calibration_comparison'].to_string(index=False), flush=True)
    export_outputs(c)
    print(f"Frozen: {c['selected_family']}; demographics retained={c['demographics_retained']}; calibration={c['calibration_choice']}", flush=True)
    print(c['refit_summary'], flush=True)
    return c


if __name__ == '__main__':
    run()

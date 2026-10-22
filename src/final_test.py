"""Frozen one-time held-out evaluation, with ignored local scoring checkpoints.

Notebook reruns read the same saved predictions; they never refit or rescore.
Existing development guards are intentionally unchanged. This module is the
explicitly authorized final-test entry point, not a model-selection facility.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import (average_precision_score, brier_score_loss, confusion_matrix,
    f1_score, log_loss, precision_score, recall_score, roc_auc_score)

from src.business_policy import (_draw_clusters, calibration_curve_table, frozen_pipeline,
    policy_predictors, risk_band_table, verify_specification)
from src.evaluation import _validated_arrays, classification_metrics, cumulative_gains_table, ranking_metrics
from src.features import CATEGORICAL_COLUMNS, METADATA_COLUMNS, PREDICTOR_COLUMNS, TARGET_COLUMN
from src.preprocessing import validate_model_panel

TEST_DATES = pd.to_datetime(['2006-08-31', '2006-09-30', '2006-10-31', '2006-11-30'])
REFIT_DATES = pd.date_range('2005-03-31', '2006-05-31', freq='ME')
BOOTSTRAP_METRICS = ['roc_auc', 'average_precision', 'brier_score', 'precision', 'recall', 'f1',
    'lift_at_top10pct', 'recall_at_top10pct', 'lift_at_top20pct', 'recall_at_top20pct']
CAPACITIES = [.05, .10, .15, .20, .25, .30]


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def validate_frozen_policy(spec):
    expected = [c for c in PREDICTOR_COLUMNS if c not in CATEGORICAL_COLUMNS]
    checks = [spec['selected_family'] == 'XGBoost', spec['predictors'] == expected,
        spec['demographics_retained'] is False, spec['calibration_choice'] == 'raw',
        spec['diagnostic_threshold'] == .50, spec['observation_days'] == 30,
        spec['outcome_days'] == 60, spec['test'] == ['2006-08-31', '2006-11-30'],
        spec['refit_protocol']['refit_start'] == '2005-03-31',
        spec['refit_protocol']['refit_end'] == '2006-05-31',
        spec['capacity_fractions'] == CAPACITIES,
        spec['preprocessing']['categorical'] == 'excluded']
    exact = dict(n_estimators=400, learning_rate=.03, max_depth=3, min_child_weight=10,
        subsample=.8, colsample_bytree=.9, reg_alpha=.1, reg_lambda=10, gamma=.1,
        random_state=42, objective='binary:logistic', eval_metric='logloss', tree_method='hist')
    if not all(checks) or any(spec['estimator_parameters'][k] != v for k, v in exact.items()):
        raise ValueError('The frozen Day 10 policy does not match the authorized final evaluation.')


def require_final_dates(frame, split):
    """Exact month-end membership, never a loose date-range authorization."""
    expected = REFIT_DATES if split == 'refit' else TEST_DATES if split == 'test' else None
    if expected is None or frame.empty or frame.snapshot_date.isna().any():
        raise ValueError('Expected a nonempty final refit or test frame.')
    if not frame.snapshot_date.isin(expected).all() or set(frame.snapshot_date) != set(expected):
        raise ValueError('Final frame dates do not exactly match the frozen monthly windows.')
    if frame[METADATA_COLUMNS].isna().any().any() or frame.duplicated(METADATA_COLUMNS).any():
        raise ValueError('Invalid or duplicated player-snapshot keys.')
    if not frame[TARGET_COLUMN].isin([0, 1]).all():
        raise ValueError('Nonbinary target.')
    if split == 'refit' and not (frame.snapshot_date+pd.Timedelta(days=60) < TEST_DATES.min()-pd.Timedelta(days=29)).all():
        raise ValueError('Refit outcome overlaps the first test observation window.')


def final_partitions(panel, spec):
    validate_frozen_policy(spec)
    validate_model_panel(panel)
    refit = panel.loc[panel.snapshot_date.isin(REFIT_DATES)].copy()
    test = panel.loc[panel.snapshot_date.isin(TEST_DATES)].copy()
    require_final_dates(refit, 'refit')
    require_final_dates(test, 'test')
    if (len(refit), refit.UserID.nunique()) != (14242, 3651):
        raise ValueError('Refit population differs from the frozen specification.')
    np.testing.assert_allclose(refit[TARGET_COLUMN].mean(), spec['refit_protocol']['churn_rate'], atol=1e-12)
    if (len(test), test.UserID.nunique()) != (2838, 1455):
        raise ValueError('Unexpected final test population.')
    np.testing.assert_allclose(test[TARGET_COLUMN].mean(), .3407, atol=.0001)
    # Explicit exclusion is applied before every fit/prediction.
    for frame in (refit, test):
        if policy_predictors(frame, False).columns.tolist() != spec['predictors']:
            raise ValueError('Frozen predictors unavailable or in the wrong order.')
    return refit, test


def fit_final_model(template, refit, spec):
    require_final_dates(refit, 'refit')
    validate_frozen_policy(spec)
    verify_specification(spec, template)
    pipeline = frozen_pipeline(template, demographics=False)
    pipeline.fit(policy_predictors(refit, False), refit[TARGET_COLUMN])
    # New safe fitting data may learn new missing indicators under the SAME rule.
    fitted_spec = {**spec, 'transformed_features': pipeline['preprocessing'].get_feature_names_out().tolist()}
    verify_specification(fitted_spec, pipeline)
    transformed = pipeline['preprocessing'].transform(policy_predictors(refit, False))
    if not np.isfinite(transformed).all():
        raise ValueError('Nonfinite final-refit preprocessing output.')
    return pipeline


def predict_final_test_once(pipeline, test, spec):
    """One predict_proba invocation; binary labels are derived from raw scores."""
    require_final_dates(test, 'test')
    validate_frozen_policy(spec)
    fitted_spec = {**spec, 'transformed_features': pipeline['preprocessing'].get_feature_names_out().tolist()}
    verify_specification(fitted_spec, pipeline)
    x = policy_predictors(test, False)
    pre = pipeline['preprocessing']
    before = pre.named_transformers_['numeric']['imputer'].statistics_.copy()
    transformed = pre.transform(x)
    if not np.isfinite(transformed).all():
        raise ValueError('Nonfinite TEST preprocessing output.')
    probabilities = pipeline.predict_proba(x)[:, 1]
    _validated_arrays(test[TARGET_COLUMN], probabilities)
    np.testing.assert_array_equal(before, pre.named_transformers_['numeric']['imputer'].statistics_)
    return probabilities


def single_class_safe_metrics(y, p):
    """Bootstrap/monthly metrics with explicit undefined-value conventions."""
    y, p = _validated_arrays(y, p)
    predicted = p >= .50
    gains = ranking_metrics(y, p)
    return dict(roc_auc=roc_auc_score(y, p) if len(np.unique(y)) == 2 else np.nan,
        average_precision=average_precision_score(y, p) if y.sum() else np.nan,
        brier_score=brier_score_loss(y, p),
        precision=precision_score(y, predicted, zero_division=0),
        recall=recall_score(y, predicted, zero_division=0),
        f1=f1_score(y, predicted, zero_division=0), **gains)


def final_cluster_bootstrap(user_ids, y, p, replicates=1000, seed=42):
    y, p = _validated_arrays(y, p)
    ids = np.asarray(user_ids)
    if len(ids) != len(y) or pd.isna(ids).any() or replicates < 1:
        raise ValueError('Invalid player clusters or replicate count.')
    codes, unique = pd.factorize(ids, sort=False)
    clusters = [np.flatnonzero(codes == i) for i in range(len(unique))]
    rng = np.random.default_rng(seed)
    results = []
    for _ in range(replicates):
        indices = _draw_clusters(clusters, rng)
        results.append(single_class_safe_metrics(y[indices], p[indices]))
    draws = pd.DataFrame(results)
    observed = single_class_safe_metrics(y, p)
    rows = []
    for metric in BOOTSTRAP_METRICS:
        values = draws[metric].replace([np.inf, -np.inf], np.nan).dropna()
        low, high = np.quantile(values, [.025, .975]) if len(values) else [np.nan, np.nan]
        rows.append(dict(metric=metric, estimate=observed[metric], bootstrap_mean=values.mean(),
            ci95_low=low, ci95_high=high, valid_replicates=len(values),
            undefined_replicates=replicates-len(values), requested_replicates=replicates, random_state=seed))
    return pd.DataFrame(rows)


def build_aggregate_results(raw, root):
    """Calculations reuse the one saved probability vector; no estimator access."""
    root = Path(root)
    y, p = _validated_arrays(raw['y'], raw['probabilities'])
    metrics = {**classification_metrics(y, p, threshold=.50), **ranking_metrics(y, p),
        'threshold': .50, 'mean_predicted_probability': float(p.mean()),
        'observed_churn_prevalence': float(y.mean())}
    calibration = calibration_curve_table(y, p, bins=10)
    nonempty = calibration.loc[calibration.observations.gt(0)]
    metrics['ece'] = float(np.sum(nonempty.observations/len(y)*abs(nonempty.mean_probability-nonempty.observed_prevalence)))
    targeting = cumulative_gains_table(y, p, CAPACITIES)
    risk_bands = risk_band_table(y, p)
    intervals = final_cluster_bootstrap(raw['user_ids'], y, p)
    months = []
    dates = pd.to_datetime(raw['snapshot_dates'])
    for date in TEST_DATES:
        mask = dates == date
        month_y, month_p = y[mask], p[mask]
        detail = single_class_safe_metrics(month_y, month_p)
        months.append(dict(snapshot_date=str(date.date()), observations=int(mask.sum()),
            unique_players=len(np.unique(raw['user_ids'][mask])), churn_prevalence=month_y.mean(),
            mean_predicted_risk=month_p.mean(), roc_auc=detail['roc_auc'],
            average_precision=detail['average_precision'], brier_score=detail['brier_score'],
            lift_at_top10pct=detail['lift_at_top10pct'], targeted_top10pct=int(np.ceil(mask.sum()*.1))))
    validation = pd.read_csv(root / 'reports/final_selected_generalization.csv').set_index('metric').validation.to_dict()
    budgets = pd.read_csv(root / 'reports/final_policy_targeting_validation.csv')
    for percent in [10, 20]:
        row = budgets.loc[np.isclose(budgets.target_fraction, percent/100)].iloc[0]
        for metric in ['lift', 'recall']:
            validation[f'{metric}_at_top{percent}pct'] = row[metric]
    comparison = pd.DataFrame([dict(metric=metric, frozen_validation=validation[metric], final_test=metrics[metric],
        test_minus_validation=metrics[metric]-validation[metric]) for metric in
        ['roc_auc', 'average_precision', 'brier_score', 'lift_at_top10pct', 'recall_at_top10pct',
         'lift_at_top20pct', 'recall_at_top20pct']])
    return dict(root=root, raw=raw, metrics=pd.DataFrame([metrics]), intervals=intervals,
        targeting=targeting, risk_bands=risk_bands, calibration=calibration,
        monthly=pd.DataFrame(months), comparison=comparison)


def claim_evaluation(path, fingerprint):
    """Atomic exclusive claim: a partial run never silently refits or rescores."""
    with Path(path).open('x', encoding='utf-8') as stream:
        json.dump(dict(status='claimed', **fingerprint), stream, indent=2)


def verify_cached_run(raw, fingerprint, model_path):
    if raw['fingerprint'] != fingerprint or raw['fit_calls'] != 1 or raw['test_probability_calls'] != 1:
        raise ValueError('One-time evaluation checkpoint provenance mismatch.')
    if sha256(model_path) != raw['model_sha256']:
        raise ValueError('Final fitted model artifact has changed.')
    if raw['threshold'] != .50 or raw['calibration'] != 'raw':
        raise ValueError('Cached policy drift.')


def execute_or_load(root='.'):
    """The only real final-refit/test-scoring orchestration; subsequent runs replay."""
    root = Path(root).resolve()
    spec_path = root / 'reports/final_model_specification.json'
    spec = json.loads(spec_path.read_text())
    validate_frozen_policy(spec)
    prior = json.loads((root / 'reports/final_validation_run_metadata.json').read_text())
    fingerprint = {name: sha256(root / relative) for name, relative in {
        'data_sha256': 'data/processed/model_panel_60d.csv',
        'specification_sha256': 'reports/final_model_specification.json',
        'development_model_sha256': 'models/final_development_model.joblib',
        'validation_metrics_sha256': 'reports/final_selected_generalization.csv',
        'validation_policy_sha256': 'reports/final_policy_targeting_validation.csv'}.items()}
    if fingerprint['data_sha256'] != prior['data_sha256'] or fingerprint['development_model_sha256'] != prior['final_development_model_sha256']:
        raise ValueError('Frozen data or development artifact differs from Day 10.')
    cache_path = root / 'data/processed/final_test_evaluation.joblib'
    receipt = root / 'data/processed/final_test_evaluation_receipt.json'
    model_path = root / 'models/final_xgboost_churn_model.joblib'
    if cache_path.exists():
        raw = joblib.load(cache_path)
        verify_cached_run(raw, fingerprint, model_path)
        print('Reusing the one saved TEST probability vector; no model fit or prediction.', flush=True)
        return raw
    if receipt.exists() or model_path.exists():
        raise RuntimeError('A final evaluation was already started; refusing an automatic refit/rescore. Inspect its local checkpoint.')
    template = joblib.load(root / 'models/final_development_model.joblib')
    verify_specification(spec, template)
    panel = pd.read_csv(root / 'data/processed/model_panel_60d.csv', parse_dates=['snapshot_date'])
    refit, test = final_partitions(panel, spec)
    del panel
    claim_evaluation(receipt, fingerprint)
    print('Verified frozen specification, panel and exact safe refit/test populations.', flush=True)
    final_model = fit_final_model(template, refit, spec)
    joblib.dump(final_model, model_path)
    print('Frozen model fitted once to 14,242 safe rows and saved locally.', flush=True)
    probabilities = predict_final_test_once(final_model, test, spec)
    raw = dict(fingerprint=fingerprint, model_sha256=sha256(model_path), specification=spec,
        refit_summary=dict(start=str(refit.snapshot_date.min().date()), end=str(refit.snapshot_date.max().date()),
            observations=len(refit), unique_players=refit.UserID.nunique(), churn_rate=float(refit[TARGET_COLUMN].mean()),
            last_outcome_date=str((refit.snapshot_date.max()+pd.Timedelta(days=60)).date())),
        test_summary=dict(observations=len(test), unique_players=test.UserID.nunique(),
            churn_prevalence=float(test[TARGET_COLUMN].mean()), start='2006-08-31', end='2006-11-30'),
        transformed_features=final_model['preprocessing'].get_feature_names_out().tolist(),
        user_ids=test.UserID.to_numpy(), snapshot_dates=test.snapshot_date.to_numpy(),
        y=test[TARGET_COLUMN].to_numpy(), probabilities=probabilities,
        threshold=.50, calibration='raw', fit_calls=1, test_probability_calls=1)
    joblib.dump(raw, cache_path)
    receipt.write_text(json.dumps(dict(status='scored_once', **fingerprint, model_sha256=raw['model_sha256'],
        fit_calls=1, test_probability_calls=1), indent=2)+'\n', encoding='utf-8')
    print('One TEST predict_proba call completed; private results saved under ignored data/processed/.', flush=True)
    return raw


AGGREGATE_FILES = dict(metrics='final_test_metrics.csv', intervals='final_test_bootstrap_intervals.csv',
    targeting='final_test_targeting.csv', monthly='final_test_monthly_metrics.csv',
    risk_bands='final_test_risk_bands.csv', calibration='final_test_calibration_bins.csv',
    comparison='final_test_validation_comparison.csv')


def export_aggregate_tables(context):
    for key, filename in AGGREGATE_FILES.items():
        table = context[key]
        if set(table.columns) & {'UserID', 'churn_60d', 'probability', 'prediction', 'actual'} or len(table) >= 1000:
            raise ValueError('Only small aggregate tables may be exported under reports/.')
        table.to_csv(context['root'] / 'reports' / filename, index=False)

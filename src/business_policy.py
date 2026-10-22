"""Final validation decision tools. Test scoring is deliberately unavailable.

Only the original TRAIN partition may be fitted here. Pre-test refit eligibility
is a date/aggregate calculation, never permission to fit or predict that frame.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.special import expit, logit
from sklearn.base import clone
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import average_precision_score, brier_score_loss, log_loss, roc_auc_score

from src.evaluation import _validated_arrays, cumulative_gains_table, classification_metrics, ranking_metrics
from src.features import CATEGORICAL_COLUMNS, PREDICTOR_COLUMNS, TARGET_COLUMN
from src.preprocessing import predictor_frame, temporal_split_labels

CAPACITIES = [.05, .10, .15, .20, .25, .30]
COMPARISON_METRICS = ['roc_auc', 'average_precision', 'brier_score', 'lift_at_top10pct',
                      'recall_at_top10pct', 'lift_at_top20pct', 'recall_at_top20pct']
# Set before running the sensitivity analysis; practical tolerances, not inferential tests.
ABLATION_TOLERANCES = dict(roc_auc=.005, average_precision=.005, brier_score=.002,
    lift_at_top10pct=.05, recall_at_top10pct=.01, lift_at_top20pct=.05, recall_at_top20pct=.01)


def require_partition(frame, split):
    if split not in ('train', 'validation') or frame.empty:
        raise ValueError('Only nonempty original train/validation partitions are open.')
    if not temporal_split_labels(frame.snapshot_date).eq(split).all():
        raise ValueError('Mixed, embargoed or sealed dates are prohibited.')


def policy_predictors(frame, demographics=True):
    x = predictor_frame(frame)
    return x if demographics else x.drop(columns=CATEGORICAL_COLUMNS)


def policy_probabilities(pipeline, frame, split, demographics=True):
    require_partition(frame, split)
    p = pipeline.predict_proba(policy_predictors(frame, demographics))[:, 1]
    # Validate without accessing target labels, including for OOF scoring.
    if np.asarray(p).shape != (len(frame),) or not np.isfinite(p).all() or ((p < 0) | (p > 1)).any():
        raise ValueError('Invalid probabilities.')
    return p


def frozen_pipeline(template, demographics=True):
    """Clone exact estimator settings, removing only the categorical transformer."""
    pipeline = clone(template)
    if not demographics:
        pre = pipeline['preprocessing']
        pre.set_params(transformers=[item for item in pre.transformers if item[0] != 'categorical'])
    return pipeline


def fit_original_train(template, frame, demographics=True):
    require_partition(frame, 'train')
    pipeline = frozen_pipeline(template, demographics)
    return pipeline.fit(policy_predictors(frame, demographics), frame[TARGET_COLUMN])


def score_metrics(y, p):
    return {**{k: v for k, v in classification_metrics(y, p).items() if k in
               ['roc_auc', 'average_precision', 'brier_score', 'log_loss']}, **ranking_metrics(y, p)}


def capacity_table(y, predictions):
    """Ceiling capacity, stable input-order ties; units are player-snapshots."""
    return pd.concat([cumulative_gains_table(y, p, CAPACITIES).assign(model=name)
                      for name, p in predictions.items()], ignore_index=True)


def draw_cluster_indices(user_ids, rng):
    """Draw n unique-player clusters with replacement; retain every cluster row."""
    ids = np.asarray(user_ids)
    if ids.ndim != 1 or len(ids) == 0 or pd.isna(ids).any():
        raise ValueError('Expected nonmissing player clusters.')
    codes, unique = pd.factorize(ids, sort=False)
    clusters = [np.flatnonzero(codes == i) for i in range(len(unique))]
    return _draw_clusters(clusters, rng)


def _draw_clusters(clusters, rng):
    """Shared sampler for public checks and the cached production bootstrap."""
    draws = rng.integers(0, len(clusters), size=len(clusters))
    return np.concatenate([clusters[i] for i in draws])


def bootstrap_metric_vector(y, p):
    gains = cumulative_gains_table(y, p, [.1, .2])
    return np.array([roc_auc_score(y, p), average_precision_score(y, p), brier_score_loss(y, p),
                     gains.iloc[0].lift, gains.iloc[0].recall,
                     gains.iloc[1].lift, gains.iloc[1].recall])


def paired_cluster_bootstrap(validation, rf_probabilities, xgb_probabilities, replicates=1000, seed=42):
    """Paired percentile cluster-bootstrap intervals, conditional on fitted models."""
    require_partition(validation, 'validation')
    y, rf = _validated_arrays(validation[TARGET_COLUMN], rf_probabilities)
    _, xgb = _validated_arrays(y, xgb_probabilities)
    if replicates < 1:
        raise ValueError('At least one bootstrap replicate is required.')
    # Cache groups once, then resample clusters. Random row count is intentional.
    codes, ids = pd.factorize(validation.UserID, sort=False)
    clusters = [np.flatnonzero(codes == i) for i in range(len(ids))]
    rng, differences, skipped = np.random.default_rng(seed), [], 0
    for _ in range(replicates):
        indices = _draw_clusters(clusters, rng)
        labels = y[indices]
        if len(np.unique(labels)) < 2:
            skipped += 1
            continue
        differences.append(bootstrap_metric_vector(labels, xgb[indices]) -
                           bootstrap_metric_vector(labels, rf[indices]))
    if len(differences) < max(1, .9 * replicates):
        raise ValueError('Too many single-class bootstrap samples.')
    differences = np.asarray(differences)
    observed = bootstrap_metric_vector(y, xgb) - bootstrap_metric_vector(y, rf)
    rows = []
    for j, metric in enumerate(COMPARISON_METRICS):
        low, high = np.quantile(differences[:, j], [.025, .975])
        better = differences[:, j] < 0 if metric == 'brier_score' else differences[:, j] > 0
        rows.append(dict(metric=metric, observed_difference=observed[j],
                         bootstrap_mean_difference=differences[:, j].mean(), ci95_low=low, ci95_high=high,
                         proportion_xgboost_better=better.mean(),
                         proportion_tied=float((differences[:, j] == 0).mean()),
                         direction='lower is better' if metric == 'brier_score' else 'higher is better',
                         valid_replicates=len(differences), skipped_replicates=skipped, seed=seed))
    return pd.DataFrame(rows)


def ablation_comparison(y, full, demographic_free):
    original, removed = score_metrics(y, full), score_metrics(y, demographic_free)
    rows = []
    for metric in COMPARISON_METRICS:
        delta = removed[metric] - original[metric]
        loss = delta if metric == 'brier_score' else -delta
        rows.append(dict(metric=metric, full_xgboost=original[metric],
                         demographic_free_xgboost=removed[metric], difference_free_minus_full=delta,
                         absolute_difference=abs(delta), predictive_loss=loss,
                         practical_loss_tolerance=ABLATION_TOLERANCES[metric],
                         within_tolerance=bool(loss <= ABLATION_TOLERANCES[metric])))
    return pd.DataFrame(rows)


def calibration_curve_table(y, probabilities, bins=10):
    """Fixed equal-width [0,1] bins, last bin includes 1.0; empty bins retained.

    ECE = sum_b (n_b / n) * abs(mean(p_b) - mean(y_b)).
    """
    y, p = _validated_arrays(y, probabilities)
    index = np.minimum((p * bins).astype(int), bins - 1)
    rows = []
    for i in range(bins):
        mask = index == i
        rows.append(dict(bin=i, lower=i/bins, upper=(i+1)/bins, observations=int(mask.sum()),
                         mean_probability=float(p[mask].mean()) if mask.any() else np.nan,
                         observed_prevalence=float(y[mask].mean()) if mask.any() else np.nan))
    return pd.DataFrame(rows)


def logistic_mapping_parameters(y, probabilities, nonnegative_slope=False):
    """Two-parameter Bernoulli MLE on clipped logits, without validation reuse."""
    y, p = _validated_arrays(y, probabilities)
    if len(np.unique(y)) < 2:
        raise ValueError('Calibration requires both classes.')
    x = logit(np.clip(p, 1e-6, 1-1e-6))
    def objective(theta):
        z = theta[0] + theta[1] * x
        return float(np.mean(np.logaddexp(0, z) - y*z))
    def gradient(theta):
        residual = expit(theta[0] + theta[1]*x) - y
        return np.array([residual.mean(), np.mean(residual*x)])
    bounds = [(None, None), (1e-8, None)] if nonnegative_slope else None
    result = minimize(objective, [0., 1.], jac=gradient, method='L-BFGS-B', bounds=bounds,
                      options={'ftol': 1e-12, 'gtol': 1e-8, 'maxiter': 2000})
    if not result.success or not np.isfinite(result.x).all():
        raise ValueError('Calibration MLE did not converge.')
    return result.x


def calibration_metrics(y, p):
    table = calibration_curve_table(y, p)
    nonempty = table.loc[table.observations.gt(0)]
    intercept, slope = logistic_mapping_parameters(y, p)
    return dict(brier_score=brier_score_loss(y, p), log_loss=log_loss(y, p, labels=[0, 1]),
                ece=float(np.sum(nonempty.observations / len(y) *
                    abs(nonempty.mean_probability - nonempty.observed_prevalence))),
                mean_probability=float(np.mean(p)), observed_prevalence=float(np.mean(y)),
                calibration_intercept=float(intercept), calibration_slope=float(slope),
                roc_auc=roc_auc_score(y, p), average_precision=average_precision_score(y, p))


def temporal_calibration_folds(train):
    """Five expanding folds Aug-Dec 2005, purged through score-window start.

    Stronger than 60 days between snapshots: every fit outcome must finish
    strictly BEFORE the first day of the scoring row's 30-day observation.
    """
    require_partition(train, 'train')
    dates = sorted(train.snapshot_date.unique())
    folds = []
    for scoring_date in pd.to_datetime(dates):
        if scoring_date < pd.Timestamp('2005-08-31'):
            continue
        observation_start = scoring_date - pd.Timedelta(days=29)
        fit_mask = train.snapshot_date + pd.Timedelta(days=60) < observation_start
        scoring_mask = train.snapshot_date.eq(scoring_date)
        fit, scoring = train.loc[fit_mask], train.loc[scoring_mask]
        if fit.empty or scoring.empty:
            continue
        folds.append(dict(fold=len(folds)+1, fit_start=fit.snapshot_date.min(),
                          fit_end=fit.snapshot_date.max(), outcome_end=fit.snapshot_date.max()+pd.Timedelta(days=60),
                          score_date=scoring_date, score_observation_start=observation_start,
                          gap_snapshot_days=(scoring_date-fit.snapshot_date.max()).days,
                          fit_observations=len(fit), score_observations=len(scoring)))
    if not folds:
        raise ValueError('No safe training calibration folds.')
    return pd.DataFrame(folds)


def generate_train_oof(template, train, demographics=True):
    require_partition(train, 'train')
    folds = temporal_calibration_folds(train)
    parts = []
    for row in folds.itertuples():
        fitting = train.loc[train.snapshot_date.between(row.fit_start, row.fit_end)]
        scoring = train.loc[train.snapshot_date.eq(row.score_date)]
        pipeline = fit_original_train(template, fitting, demographics)
        p = policy_probabilities(pipeline, scoring, 'train', demographics)
        parts.append(pd.DataFrame(dict(UserID=scoring.UserID.to_numpy(),
            snapshot_date=scoring.snapshot_date.to_numpy(), label=scoring[TARGET_COLUMN].to_numpy(),
            probability=p, fold=row.fold, fit_end=row.fit_end,
            score_observation_start=row.score_observation_start)))
    oof = pd.concat(parts, ignore_index=True)
    validate_train_oof(oof)
    return oof, folds


def validate_train_oof(oof):
    require_partition(oof, 'train')
    required = {'UserID', 'snapshot_date', 'label', 'probability', 'fold', 'fit_end', 'score_observation_start'}
    if not required.issubset(oof.columns) or oof[list(required)].isna().any().any():
        raise ValueError('Incomplete OOF provenance.')
    if oof.duplicated(['UserID', 'snapshot_date']).any():
        raise ValueError('Duplicated OOF observations.')
    end = pd.to_datetime(oof.fit_end)
    start = pd.to_datetime(oof.score_observation_start)
    if not temporal_split_labels(end).eq('train').all():
        raise ValueError('OOF models must fit original training dates only.')
    if not (end + pd.Timedelta(days=60) < start).all():
        raise ValueError('OOF label/observation overlap.')
    if not (start == pd.to_datetime(oof.snapshot_date) - pd.Timedelta(days=29)).all():
        raise ValueError('Incorrect OOF scoring window.')
    _validated_arrays(oof.label, oof.probability)


def fit_train_calibrators(oof):
    """Sole calibration fitting entry point; validation/test provenance rejected."""
    validate_train_oof(oof)
    intercept, slope = logistic_mapping_parameters(oof.label, oof.probability, nonnegative_slope=True)
    mappings = {'sigmoid': dict(intercept=float(intercept), slope=float(slope),
                               source='purged temporal TRAIN OOF only')}
    # Prespecified support floor; enough rows alone does not guarantee stability.
    counts = oof.label.value_counts()
    if len(oof) >= 1000 and counts.min() >= 100 and oof.probability.nunique() >= 100:
        mappings['isotonic'] = IsotonicRegression(out_of_bounds='clip').fit(oof.probability, oof.label)
    return mappings


def apply_calibrator(mapping, probabilities):
    p = np.asarray(probabilities, dtype=float)
    if isinstance(mapping, dict):
        return expit(mapping['intercept'] + mapping['slope']*logit(np.clip(p, 1e-6, 1-1e-6)))
    return mapping.predict(p)


def select_calibration(comparison):
    """Prefer raw unless improvements are meaningful AND consistent.

    Require Brier decrease >= .001 and log loss decrease >= .002, ECE not
    worse by more than .002. Prefer sigmoid if eligible and essentially tied
    (Brier .0005, log loss .001) with the best eligible mapping.
    """
    rows = comparison.set_index('method')
    raw = rows.loc['raw']
    eligible = rows.drop(index='raw').loc[lambda d:
        (raw.brier_score-d.brier_score >= .001) &
        (raw.log_loss-d.log_loss >= .002) & (d.ece <= raw.ece+.002)]
    if eligible.empty:
        return 'raw'
    best = eligible.sort_values(['brier_score', 'log_loss']).iloc[0]
    if 'sigmoid' in eligible.index:
        sigmoid = eligible.loc['sigmoid']
        if sigmoid.brier_score <= best.brier_score+.0005 and sigmoid.log_loss <= best.log_loss+.001:
            return 'sigmoid'
    return best.name


def diagnostic_threshold_table(y, p):
    rows = []
    prevalence = np.mean(y)
    for threshold in np.arange(20, 81, 5)/100:
        metrics = classification_metrics(y, p, threshold)
        rows.append(dict(threshold=threshold, targeted_observations=metrics['predicted_positives'],
            targeting_percentage=100*metrics['predicted_positives']/len(y),
            precision=metrics['precision'], recall=metrics['recall'], f1=metrics['f1'],
            specificity=metrics['specificity'], churn_rate_targeted=metrics['precision'],
            lift=metrics['precision']/prevalence if metrics['predicted_positives'] else np.nan))
    return pd.DataFrame(rows)


def risk_band_table(y, p):
    """Disjoint deterministic rank bands; high-risk capacities use ceil(n*f)."""
    y, p = _validated_arrays(y, p)
    order = np.argsort(-p, kind='stable')
    n = len(y)
    cuts = [0, int(np.ceil(.1*n)), int(np.ceil(.2*n)), int(np.ceil(.5*n)), n]
    labels = ['Very high risk (top 10%)', 'High risk (80th-90th)',
              'Medium risk (50th-80th)', 'Low risk (bottom 50%)']
    rows = []
    for label, start, end in zip(labels, cuts[:-1], cuts[1:]):
        indices = order[start:end]
        if len(indices):
            rows.append(dict(band=label, observations=len(indices), actual_fraction=len(indices)/n,
                average_predicted_risk=float(p[indices].mean()), actual_churn_rate=float(y[indices].mean()),
                lift=float(y[indices].mean()/y.mean()) if y.mean() else np.nan))
    return pd.DataFrame(rows)


def refit_protocol(panel):
    """Compute eligible pre-test aggregates only; never return a scoring frame."""
    test_snapshot = pd.Timestamp('2006-08-31')
    observation_start = test_snapshot-pd.Timedelta(days=29)
    # Strict inequality at calendar-day resolution: S+60 <= observation_start-1.
    latest_safe_date = observation_start-pd.Timedelta(days=61)
    dates = pd.to_datetime(panel.snapshot_date)
    safe = dates.between('2005-03-31', latest_safe_date)
    eligible = panel.loc[safe, ['UserID', 'snapshot_date', TARGET_COLUMN]]
    if eligible.empty:
        raise ValueError('No safe pre-test snapshots.')
    table = eligible.groupby('snapshot_date').agg(observations=('UserID', 'size'),
        unique_players=('UserID', 'nunique'), churn_rate=(TARGET_COLUMN, 'mean')).reset_index()
    table['outcome_end'] = table.snapshot_date+pd.Timedelta(days=60)
    assert table.outcome_end.lt(observation_start).all()
    summary = dict(first_test_snapshot=str(test_snapshot.date()),
        first_test_observation_start=str(observation_start.date()),
        latest_safe_calendar_snapshot=str(latest_safe_date.date()),
        refit_start=str(eligible.snapshot_date.min().date()), refit_end=str(eligible.snapshot_date.max().date()),
        observations=len(eligible), unique_players=eligible.UserID.nunique(),
        churn_rate=float(eligible[TARGET_COLUMN].mean()))
    boundary = pd.DataFrame({'snapshot_date': pd.to_datetime(['2006-01-31', '2006-02-28',
                              '2006-05-31', '2006-06-30', '2006-07-31'])})
    boundary['outcome_end'] = boundary.snapshot_date+pd.Timedelta(days=60)
    boundary['safe_for_future_refit'] = boundary.outcome_end.lt(observation_start)
    return summary, table, boundary


def verify_specification(specification, pipeline):
    """Reject estimator, predictor or preprocessing drift after the decision."""
    actual = pipeline['model'].get_params()
    if set(specification['estimator_parameters']) != set(actual):
        raise ValueError('Frozen estimator parameter schema mismatch.')
    for key, expected in specification['estimator_parameters'].items():
        value = actual[key]
        if expected == 'NaN' and isinstance(value, float) and np.isnan(value):
            continue
        if value != expected:
            raise ValueError(f'Frozen estimator parameter mismatch: {key}')
    if specification['predictors'] != pipeline['preprocessing'].feature_names_in_.tolist():
        raise ValueError('Frozen raw predictor list mismatch.')
    names = pipeline['preprocessing'].get_feature_names_out().tolist()
    if names != specification['transformed_features']:
        raise ValueError('Frozen transformed feature mapping mismatch.')
    if specification['demographics_retained'] != any(c in specification['predictors'] for c in CATEGORICAL_COLUMNS):
        raise ValueError('Demographic policy mismatch.')
    pre = pipeline['preprocessing']
    numeric = pre.named_transformers_['numeric']
    if (numeric['imputer'].strategy != 'median' or not numeric['imputer'].add_indicator
            or 'scaler' in numeric.named_steps or pre.sparse_threshold != 0 or pre.remainder != 'drop'):
        raise ValueError('Frozen numeric preprocessing policy mismatch.')
    if specification['demographics_retained']:
        categorical = pre.named_transformers_['categorical']
        if categorical['imputer'].strategy != 'most_frequent' or categorical['encoder'].handle_unknown != 'ignore':
            raise ValueError('Frozen categorical preprocessing policy mismatch.')

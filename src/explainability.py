"""Validation-only explanations of the frozen Day 8 pipeline; never fits a model.

Tree SHAP is calculated in raw margin (log-odds) space, using training path
counts already stored in the fitted trees, without an external background.
"""
from __future__ import annotations

import re
import numpy as np
import pandas as pd
from scipy.special import expit

from src.features import CATEGORICAL_COLUMNS, METADATA_COLUMNS, PREDICTOR_COLUMNS, TARGET_COLUMN
from src.preprocessing import predictor_frame, temporal_split_labels

GROUPS = ['recency', 'activity', 'betting volume', 'monetary', 'trend',
          'lifecycle', 'demographics', 'missingness']


def require_validation(frame):
    if frame.empty or not temporal_split_labels(frame.snapshot_date).eq('validation').all():
        raise ValueError('Explanations require nonempty, exclusively VALIDATION observations.')


def validation_sample(frame, size=1000, random_state=42):
    """Reject mixed/sealed frames before sampling or accessing any predictors."""
    require_validation(frame)
    if size < 1:
        raise ValueError('Sample size must be positive.')
    return frame.sample(n=min(size, len(frame)), random_state=random_state).copy()


def feature_group(source, missing=False):
    if missing:
        return 'missingness'
    if source in CATEGORICAL_COLUMNS:
        return 'demographics'
    if source == 'days_since_last_active_bet':
        return 'recency'
    if source.startswith('days_since_'):
        return 'lifecycle'
    if any(token in source for token in ('change', 'share', 'recent15', 'early15')):
        return 'trend'
    if source.startswith(('active_days', 'max_gap', 'mean_gap')):
        return 'activity'
    if source.startswith(('bets_', 'avg_bets_')):
        return 'betting volume'
    if any(token in source for token in ('stake', 'winnings', 'net_loss')):
        return 'monetary'
    raise ValueError(f'Unmapped feature: {source}')


def transformed_feature_mapping(preprocessing, width):
    """Map the fitted transformer output, including learned missing indicators."""
    names = preprocessing.get_feature_names_out()
    if len(names) != width or len(set(names)) != width:
        raise ValueError('Transformed feature names do not match matrix columns.')
    rows = []
    for position, name in enumerate(names):
        bare = name.split('__', 1)[-1]
        missing = bare.startswith('missingindicator_')
        source = bare.removeprefix('missingindicator_')
        readable = f'{source} (missing)' if missing else bare
        for category in CATEGORICAL_COLUMNS:
            if bare.startswith(category + '_'):
                source = category
                readable = f'{category} = {bare[len(category) + 1:]}'
                break
        if source not in PREDICTOR_COLUMNS or source in METADATA_COLUMNS + [TARGET_COLUMN]:
            raise ValueError(f'Nonpredictor in SHAP features: {name}')
        rows.append(dict(position=position, transformed_feature=name, feature=readable,
                         source_feature=source, feature_group=feature_group(source, missing)))
    return pd.DataFrame(rows)


def validate_shap_matrix(values, matrix, mapping):
    if np.asarray(values).shape != np.asarray(matrix).shape or matrix.shape[1] != len(mapping):
        raise ValueError('SHAP dimensions do not match transformed observations/features.')
    if not np.isfinite(values).all():
        raise ValueError('SHAP values must be finite.')
    if not set(mapping.source_feature).issubset(PREDICTOR_COLUMNS):
        raise ValueError('Metadata/target found in SHAP features.')


def explain_validation(pipeline, frame):
    """Transform validation only and check log-odds/probability reconstruction."""
    import shap
    require_validation(frame)
    matrix = np.asarray(pipeline['preprocessing'].transform(predictor_frame(frame)))
    mapping = transformed_feature_mapping(pipeline['preprocessing'], matrix.shape[1])
    explainer = shap.TreeExplainer(pipeline['model'], feature_perturbation='tree_path_dependent',
                                   model_output='raw')
    explanation = explainer(matrix, check_additivity=True)
    validate_shap_matrix(explanation.values, matrix, mapping)
    probability = pipeline['model'].predict_proba(matrix)[:, 1]
    np.testing.assert_allclose(expit(explanation.base_values + explanation.values.sum(axis=1)),
                               probability, atol=2e-6, rtol=2e-6)
    explanation.feature_names = mapping.feature.tolist()
    return explanation, matrix, mapping


def importance_tables(explanation, mapping):
    table = mapping.copy()
    table['mean_abs_shap'] = np.abs(explanation.values).mean(axis=0)
    table = table.sort_values('mean_abs_shap', ascending=False).reset_index(drop=True)
    table['shap_rank'] = np.arange(1, len(table) + 1)
    groups = table.groupby('feature_group').mean_abs_shap.sum().reindex(GROUPS, fill_value=0)
    groups = groups.sort_values(ascending=False).rename('mean_abs_shap').reset_index()
    groups['share'] = groups.mean_abs_shap / groups.mean_abs_shap.sum()
    return table, groups


def compare_permutation(explanation, mapping, permutation):
    """Aggregate signed dummy/indicator SHAP first, then mean absolute by source.

    Day 8 permutes raw columns before the pipeline, jointly changing all their
    transformed descendants. Summing absolute dummy contributions would inflate
    attribution and give an inconsistent comparison unit.
    """
    rows = []
    for source, group in mapping.groupby('source_feature', sort=False):
        attribution = explanation.values[:, group.position.to_numpy()].sum(axis=1)
        rows.append(dict(feature=source, mean_abs_shap=np.abs(attribution).mean()))
    result = pd.DataFrame(rows).sort_values('mean_abs_shap', ascending=False)
    result['shap_rank'] = np.arange(1, len(result) + 1)
    perm = permutation.sort_values('mean_ap_drop', ascending=False).copy()
    perm['permutation_rank'] = np.arange(1, len(perm) + 1)
    result = result.merge(perm, on='feature', validate='one_to_one')
    result['notes'] = [
        'Both top 10' if a <= 10 and b <= 10 else
        'Ranks within 5 positions' if abs(a-b) <= 5 else
        'Different attribution/AP disruption; correlated features may redistribute importance'
        for a, b in zip(result.shap_rank, result.permutation_rank)]
    return result


def dependence_bins(frame, explanation, mapping, features):
    """Observed quantile-bin summaries, preserving missingness separately."""
    require_validation(frame)
    rows = []
    for feature in features:
        position = mapping.loc[mapping.transformed_feature.eq('numeric__' + feature), 'position'].item()
        data = pd.DataFrame({'value': frame[feature].to_numpy(),
                             'shap': explanation.values[:, position]})
        present = data.dropna(subset=['value']).copy()
        present['bin'] = pd.qcut(present.value, q=6, duplicates='drop')
        for _, part in present.groupby('bin', observed=True):
            rows.append(dict(feature=feature, low=part.value.min(), high=part.value.max(),
                             n=len(part), mean_shap=part.shap.mean(),
                             median_shap=part.shap.median(), missing=False))
        absent = data.loc[data.value.isna()]
        if len(absent):
            rows.append(dict(feature=feature, low=np.nan, high=np.nan, n=len(absent),
                             mean_shap=absent.shap.mean(), median_shap=absent.shap.median(), missing=True))
    return pd.DataFrame(rows)


def pairwise_margin_contrasts(pipeline, frame, pairs):
    """Small 2x2 model-response diagnostic, NOT SHAP interaction values.

    Hold all other transformed columns fixed. Replace each selected continuous
    predictor with its observed validation Q25/Q75. Nonzero mixed differences
    demonstrate a non-additive fitted response over these four synthetic points.
    Correlated/derived features are not made consistent: off-support extrapolation
    is possible, so these are neither causal effects nor observed-player effects.
    No full feature-by-feature interaction matrix is constructed.
    """
    require_validation(frame)
    matrix = np.asarray(pipeline['preprocessing'].transform(predictor_frame(frame)))
    mapping = transformed_feature_mapping(pipeline['preprocessing'], matrix.shape[1])
    rows = []
    for first, second in pairs:
        limits = [frame[f].quantile([.25, .75]).to_numpy() for f in (first, second)]
        if any(not np.isfinite(v).all() or v[0] == v[1] for v in limits):
            continue
        positions = [mapping.loc[mapping.transformed_feature.eq('numeric__' + f), 'position'].item()
                     for f in (first, second)]
        margins = {}
        for i, j in [(0, 0), (1, 0), (0, 1), (1, 1)]:
            changed = matrix.copy()
            changed[:, positions[0]] = limits[0][i]
            changed[:, positions[1]] = limits[1][j]
            margins[i, j] = pipeline['model'].predict(changed, output_margin=True)
        delta = margins[1, 1] - margins[1, 0] - margins[0, 1] + margins[0, 0]
        rows.append(dict(feature_1=first, feature_2=second,
                         first_low=limits[0][0], first_high=limits[0][1],
                         second_low=limits[1][0], second_high=limits[1][1],
                         mean_mixed_margin=float(delta.mean()), mean_abs_mixed_margin=float(np.abs(delta).mean()),
                         fraction_nonzero=float((np.abs(delta) > 1e-5).mean()), n=len(frame)))
    return pd.DataFrame(rows)


def validate_local_export(table):
    allowed = {'case', 'outcome', 'probability', 'actual', 'diagnostic_threshold',
               'increasing_risk', 'decreasing_risk', 'base_log_odds', 'sum_shap'}
    if set(table.columns) != allowed or not table.case.is_unique:
        raise ValueError('Local export schema must be anonymous and unique.')
    if not all(re.fullmatch(r'Validation Case [A-D]', label) for label in table.case):
        raise ValueError('Use anonymous Validation Case A-D labels only.')


def local_cases(frame, probabilities, explanation, mapping):
    """Nearest-to-cell-median probability example per outcome, sampled rows only."""
    require_validation(frame)
    actual = frame[TARGET_COLUMN].to_numpy()
    probabilities = np.asarray(probabilities)
    predicted = probabilities >= .50
    rows, positions = [], {}
    for letter, outcome, y, predicted_y in [('A', 'TP', 1, 1), ('B', 'FP', 0, 1),
                                          ('C', 'TN', 0, 0), ('D', 'FN', 1, 0)]:
        candidates = np.flatnonzero((actual == y) & (predicted == predicted_y))
        if not len(candidates):
            continue
        position = candidates[np.argmin(np.abs(probabilities[candidates] - np.median(probabilities[candidates])))]
        values = explanation.values[position]
        def drivers(positive):
            ordered = np.argsort(-values if positive else values)
            selected = [j for j in ordered if (values[j] > 0 if positive else values[j] < 0)][:3]
            return '; '.join(f'{mapping.iloc[j].feature} ({values[j]:+.3f} log-odds)' for j in selected)
        case = f'Validation Case {letter}'
        positions[case] = int(position)
        rows.append(dict(case=case, outcome=outcome, probability=float(probabilities[position]),
                         actual=int(actual[position]), diagnostic_threshold=.50,
                         increasing_risk=drivers(True), decreasing_risk=drivers(False),
                         base_log_odds=float(explanation.base_values[position]), sum_shap=float(values.sum())))
    table = pd.DataFrame(rows)
    validate_local_export(table)
    return table, positions


def load_candidate(root):
    """Load the saved pipeline and reproduce Day 8 validation metrics, no fit."""
    import hashlib
    import json
    from pathlib import Path
    import joblib
    from sklearn.metrics import average_precision_score, roc_auc_score
    from src.modelling import guarded_probabilities, modelling_partition

    root = Path(root)
    model_path = root / 'models/xgboost_model.joblib'
    data_path = root / 'data/processed/model_panel_60d.csv'
    metadata = json.loads((root / 'reports/xgboost_run_metadata.json').read_text())
    data_hash = hashlib.sha256(data_path.read_bytes()).hexdigest()
    if data_hash != metadata['data_sha256']:
        raise ValueError('Panel fingerprint differs from the Day 8 artifact.')
    model_hash = hashlib.sha256(model_path.read_bytes()).hexdigest()
    pipeline = joblib.load(model_path)
    expected = dict(n_estimators=400, learning_rate=.03, max_depth=3, min_child_weight=10,
                    subsample=.8, colsample_bytree=.9, reg_alpha=.1, reg_lambda=10,
                    gamma=.1, random_state=42)
    for key, value in expected.items():
        if pipeline['model'].get_params()[key] != value:
            raise ValueError(f'Saved model differs from frozen candidate: {key}')
    panel = pd.read_csv(data_path, parse_dates=['snapshot_date'])
    validation = modelling_partition(panel, 'validation')
    del panel  # No test frame is constructed, predicted, transformed or explained.
    probabilities = guarded_probabilities(pipeline, validation, 'validation')
    metrics = dict(roc_auc=roc_auc_score(validation[TARGET_COLUMN], probabilities),
                   average_precision=average_precision_score(validation[TARGET_COLUMN], probabilities))
    saved = pd.read_csv(root / 'reports/xgboost_validation_metrics.csv').iloc[0]
    for name, value in metrics.items():
        np.testing.assert_allclose(value, saved[name], atol=5e-5, rtol=0)
    np.testing.assert_allclose(list(metrics.values()), [.7464, .5630], atol=5e-5, rtol=0)
    return dict(root=root, pipeline=pipeline, validation=validation, metrics=metrics,
                model_sha256=model_hash, data_sha256=data_hash)


def compute_analysis(context, size=1000):
    """Explain a fixed validation sample without changing model or features."""
    from src.modelling import guarded_probabilities
    sample = validation_sample(context['validation'], size=size)
    explanation, matrix, mapping = explain_validation(context['pipeline'], sample)
    global_table, groups = importance_tables(explanation, mapping)
    comparison = compare_permutation(explanation, mapping, pd.read_csv(
        context['root'] / 'reports/xgboost_permutation_importance.csv'))
    preferred = ['days_since_last_active_bet', 'active_days_30d', 'winnings_7d',
                 'max_daily_stake_30d', 'days_since_first_casino_activity']
    # Require each continuous feature to be in the top 20 raw-source SHAP ranks.
    features = [f for f in preferred if f in comparison.head(20).feature.tolist()]
    bins = dependence_bins(sample, explanation, mapping, features)
    pairs = [('days_since_last_active_bet', 'active_days_30d'),
             ('days_since_last_active_bet', 'bets_30d'),
             ('active_days_30d', 'days_since_first_casino_activity'),
             ('winnings_7d', 'active_days_30d')]
    interactions = pairwise_margin_contrasts(context['pipeline'], sample, pairs)
    probabilities = guarded_probabilities(context['pipeline'], sample, 'validation')
    cases, case_positions = local_cases(sample, probabilities, explanation, mapping)
    context.update(sample=sample, explanation=explanation, matrix=matrix, mapping=mapping,
                   global_table=global_table, groups=groups, comparison=comparison,
                   bins=bins, features=features, interactions=interactions,
                   cases=cases, case_positions=case_positions)
    return context


def save_figures(context):
    """Export anonymous figures; local waterfalls omit raw feature values."""
    import matplotlib.pyplot as plt
    import shap
    figures = context['root'] / 'reports/figures'
    figures.mkdir(parents=True, exist_ok=True)
    explanation, mapping = context['explanation'], context['mapping']

    def save(name):
        plt.gcf().savefig(figures / name, dpi=160, bbox_inches='tight')
        plt.close('all')

    shap.plots.bar(explanation, max_display=20, show=False)
    plt.title('Validation mean absolute SHAP (log-odds)')
    save('shap_global_bar.png')
    shap.plots.beeswarm(explanation, max_display=20, show=False)
    plt.xlabel('SHAP value (log-odds; positive pushes toward churn)')
    save('shap_summary_beeswarm.png')
    groups = context['groups'].iloc[::-1]
    plt.figure(figsize=(9, 5))
    plt.barh(groups.feature_group, groups.mean_abs_shap)
    plt.xlabel('Sum of transformed-column mean absolute SHAP (log-odds)')
    plt.title('Feature groups: validation sample')
    save('shap_feature_group_bar.png')
    partners = {'days_since_last_active_bet': 'active_days_30d',
                'active_days_30d': 'days_since_first_casino_activity',
                'winnings_7d': 'active_days_30d',
                'max_daily_stake_30d': 'active_days_30d',
                'days_since_first_casino_activity': 'days_since_last_active_bet'}
    for feature in context['features']:
        position = mapping.loc[mapping.transformed_feature.eq('numeric__' + feature), 'position'].item()
        partner = partners[feature]
        # Display original continuous values, not imputed values; missing rows excluded.
        x = context['sample'][feature].to_numpy()
        color = context['sample'][partner].to_numpy()
        present = np.isfinite(x) & np.isfinite(color)
        plt.figure(figsize=(9, 5))
        dots = plt.scatter(x[present], explanation.values[present, position], c=color[present],
                           cmap='coolwarm', s=14, alpha=.6)
        plt.colorbar(dots, label=partner)
        part = context['bins'].loc[lambda d: d.feature.eq(feature) & ~d.missing]
        plt.plot((part.low + part.high) / 2, part.mean_shap, 'k.-', label='Quantile-bin mean')
        plt.axhline(0, color='gray', linewidth=.7)
        if feature in ['winnings_7d', 'max_daily_stake_30d']:
            plt.xscale('symlog', linthresh=1)
        plt.xlabel(feature + (' (symlog scale)' if feature in ['winnings_7d', 'max_daily_stake_30d'] else ''))
        plt.ylabel('SHAP value (log-odds)')
        plt.title('Validation association; color is exploratory, not proof of interaction')
        plt.legend()
        save('shap_dependence_' + feature + '.png')
    for case, position in context['case_positions'].items():
        # No feature values, row index, snapshot date or player identifier in exports.
        local = shap.Explanation(values=explanation.values[position],
                                 base_values=explanation.base_values[position],
                                 feature_names=mapping.feature.tolist())
        shap.plots.waterfall(local, max_display=12, show=False)
        record = context['cases'].loc[lambda d: d.case.eq(case)].iloc[0]
        plt.title(f'{case} ({record.outcome}), p={record.probability:.3f}, actual={record.actual}')
        save('shap_waterfall_case_' + case[-1].lower() + '.png')


def markdown_table(table, columns=None, digits=4):
    """Small Markdown renderer without an extra tabulate dependency."""
    table = table if columns is None else table[columns]
    def cell(value):
        return f'{value:.{digits}f}' if isinstance(value, (float, np.floating)) else str(value)
    lines = ['| ' + ' | '.join(table.columns) + ' |', '| ' + ' | '.join(['---'] * len(table.columns)) + ' |']
    lines += ['| ' + ' | '.join(cell(v).replace('|', '/') for v in row) + ' |'
              for row in table.itertuples(index=False, name=None)]
    return '\n'.join(lines)


def summary_markdown(context):
    """Evidence-backed interpretation generated from the actual fitted model."""
    c = context
    validation, sample, groups = c['validation'], c['sample'], c['groups']
    comparison = c['comparison']
    lines = ['# Day 9: SHAP explainability of the provisional XGBoost candidate', '',
        '## Model and population', '',
        'Frozen Day 8 `initial_conservative` XGBoost pipeline, 400 trees, learning rate 0.03, '
        'depth 3, min_child_weight 10, subsample 0.8, colsample_bytree 0.9, alpha 0.1, '
        'lambda 10, gamma 0.1, seed 42. No fitting, tuning, feature changes, calibration '
        'or final threshold selection occurred. Random Forest remains the targeting challenger.', '',
        f'Validation: {len(validation):,} player-snapshots from 2006-03-31 through 2006-05-31; '
        f'{validation.UserID.nunique():,} distinct players; observed inactivity prevalence '
        f'{validation[TARGET_COLUMN].mean():.2%}. SHAP sample: {len(sample):,} validation rows '
        '(uniform, without replacement, random_state=42). No test observations were '
        'transformed, predicted, explained or evaluated. Training remains 2005-03-31 '
        'through 2005-12-31; test remains sealed at 2006-08-31 through 2006-11-30; '
        'the original embargo windows are preserved.', '',
        f'Reproduced full-validation ROC-AUC **{c["metrics"]["roc_auc"]:.6f}**, '
        f'PR-AUC (average precision) **{c["metrics"]["average_precision"]:.6f}**.', '',
        'TreeExplainer uses tree_path_dependent perturbation, the stored training path counts '
        'as its implicit reference, and raw model output (log-odds). There is no separately '
        'sampled background. This reference is not the validation prevalence. '
        'For every explained row, sigmoid(base value + sum of SHAP) reproduces its '
        'model probability within numerical tolerance. Positive SHAP pushes toward churn; '
        'negative SHAP pushes away. Contributions are not probability percentage points.', '',
        '## Global explanation', '', markdown_table(c['global_table'].head(20),
            ['feature', 'mean_abs_shap', 'feature_group']), '',
        'The global table ranks transformed columns; category levels and missingness flags '
        'remain distinct. Red in the beeswarm denotes a higher transformed value, blue a '
        'lower value. For dummy columns these mean present/absent, and for missingness '
        'flags they mean missing/not missing. Numeric color values after preprocessing '
        'include median imputations.', '', '## Feature groups', '',
        markdown_table(groups), '',
        'Group scores sum per-column mean absolute SHAP. They describe attribution mass, '
        'not explained variance or percentages of predicted risk. Correlated and larger '
        'groups can accumulate attribution; signed contributions can cancel within a row. '
        'Recent/early half-window predictors and change/share predictors are grouped as '
        'trend; missingness always has its own group.', '', '## Direction and nonlinear patterns', '']
    for feature in c['features']:
        part = c['bins'].loc[lambda d: d.feature.eq(feature) & ~d.missing]
        first, last = part.iloc[0], part.iloc[-1]
        difference = last.mean_shap - first.mean_shap
        trend = 'higher' if difference > 0 else 'lower'
        steps = np.diff(part.mean_shap)
        monotonic = bool(np.all(steps >= 0) or np.all(steps <= 0))
        shape = ('Bin means are monotonic across this sample, but this does not establish a globally monotonic response.'
                 if monotonic else 'Intermediate bin means reverse direction: the observed pattern is nonlinear/nonmonotonic.')
        lines += [f'- **{feature}**: low observed bin [{first.low:.3g}, {first.high:.3g}] '
                  f'(n={first.n}) has mean SHAP {first.mean_shap:+.3f}; high bin '
                  f'[{last.low:.3g}, {last.high:.3g}] (n={last.n}) has {last.mean_shap:+.3f} '
                  f'log-odds. Higher values have {trend} average contributions in this '
                  f'low/high comparison. {shape}']
    lines += ['', 'These are marginal patterns in the observed validation sample, affected by '
        'other predictors and missingness. Quantile edges are descriptive bins, not learned '
        'operational thresholds. The dependence plots retain all finite values (monetary '
        'axes use a signed logarithmic scale); no cutoff or causal effect is inferred. '
        'Rows with a missing plotted value or color are excluded from that scatter; '
        'bin summaries separately record missing feature values.', '',
        '## Targeted interaction diagnostics', '', markdown_table(c['interactions']), '',
        'These are small 2x2 fitted log-odds contrasts at each feature\'s validation Q25/Q75, '
        'averaged over validation sample contexts. A mixed difference is '
        'f(high,high) - f(high,low) - f(low,high) + f(low,low). A nonzero contrast supports '
        'non-additivity of the fitted function at those synthetic points; a near-zero '
        'contrast does not rule out interactions elsewhere. Positive mean contrast means '
        'the joint high/high log-odds response exceeds the sum of the two separate changes. '
        'No full interaction matrix or SHAP interaction values were computed. '
        'Dependence-plot color alone does not prove an interaction. Holding correlated '
        'and derived predictors fixed can create inconsistent/off-support combinations, '
        'so these diagnostics do not establish real-player or causal effects.', '',
        '## Local explanations (diagnostic threshold 0.50)', '',
        'One example nearest to the within-cell median predicted probability is selected '
        'for each outcome within the SHAP sample. This is an illustrative selection rule, '
        'not a claim of population representativeness. Raw player records, identifiers, '
        'row indices, dates and local predictor values are omitted from exports. '
        'Waterfalls show anonymous feature contributions in log-odds.', '',
        markdown_table(c['cases'], ['case', 'outcome', 'probability', 'actual', 'increasing_risk', 'decreasing_risk']), '',
        'The same risk-increasing associations can occur in both TP/FP cases, and '
        'risk-decreasing associations in both TN/FN cases: an explanation describes '
        'the score, not why the observed outcome occurred. The 0.50 diagnostic does '
        'not select a deployment threshold.', '', '## Consistency with Day 8 permutation importance', '',
        markdown_table(comparison.head(20), ['feature', 'shap_rank', 'permutation_rank', 'notes']), '']
    earlier = ['days_since_last_active_bet', 'winnings_7d', 'active_days_30d',
               'days_since_first_casino_activity', 'max_daily_stake_30d']
    ranks = comparison.set_index('feature')
    overlap = len(set(comparison.head(10).feature) & set(comparison.nsmallest(10, 'permutation_rank').feature))
    rank_correlation = comparison.shap_rank.corr(comparison.permutation_rank, method='spearman')
    lines += [f'Top-10 overlap: {overlap}/10; source-level Spearman rank correlation '
              f'{rank_correlation:.3f}. The five earlier drivers have SHAP source ranks: '
              + ', '.join(f'{f} {int(ranks.loc[f, "shap_rank"])}' for f in earlier) + '.', '',
        ('SHAP broadly confirms the earlier five-driver interpretation; their exact ordering differs.'
         if all(ranks.loc[f, 'shap_rank'] <= 10 for f in earlier)
         else 'SHAP partly challenges the earlier ordering; inspect the source ranks above.'), '',
        'For comparability, signed SHAP contributions of every raw feature\'s descendants '
        '(including its missing indicator and category levels) are summed within each row, '
        'then absolute-valued and averaged. This differs from the transformed/group tables. '
        'Permutation importance measures full-validation average-precision loss after '
        'shuffling a raw predictor (five repeats); SHAP measures contribution magnitude '
        'in a 1,000-row sample. Correlated predictors can share SHAP credit; permutation '
        'can disrupt correlated structure, and nonlinear interactions redistribute '
        'contributions. Small negative AP drops reflect noise/redundancy, not protective '
        'effects. Neither ranking is absolute truth.', '', '## Demographics and missingness', '']
    for group in ['demographics', 'missingness']:
        subset = c['global_table'].loc[lambda d: d.feature_group.eq(group)]
        share = groups.set_index('feature_group').loc[group, 'share']
        top = subset.iloc[0]
        lines += [f'- **{group}**: {share:.2%} of summed mean absolute SHAP; highest column '
                  f'`{top.feature}` is rank {int(top.shap_rank)} ({top.mean_abs_shap:.5f} log-odds). '
                  + ('No column appears in the top 20.' if subset.shap_rank.min() > 20 else
                     'At least one column appears in the top 20; review its role.')]
    lines += ['', markdown_table(comparison.loc[comparison.feature.isin(CATEGORICAL_COLUMNS)],
                              ['feature', 'mean_abs_shap', 'shap_rank', 'permutation_rank']), '',
        'Missingness indicators with nonzero mean absolute SHAP:', '',
        markdown_table(c['global_table'].loc[lambda d: d.feature_group.eq('missingness') & d.mean_abs_shap.gt(0)],
                       ['feature', 'mean_abs_shap', 'shap_rank']), '',
        'Importance is reported neutrally: low demographic attribution does not establish '
        'fairness, and proxy effects can remain in behavioral predictors. Country/Language/Gender '
        'require governance review before operational use; this notebook performs no subgroup '
        'fairness audit or feature removal. Structural missingness may encode player behavior '
        '(for example, insufficient activity to define a gap) or source-data availability. '
        'A zero indicator attribution does not mean missingness is irrelevant: median-imputed '
        'values and correlated predictors may carry it indirectly.', '',
        '## Responsible interpretation and limitations', '',
        'The model predicts extended inactivity: no positive-bet day in the 60 days after '
        'the snapshot. It does **not** diagnose gambling harm, identify problem gambling, '
        'determine customer value, or prove why a player churns. Model explanations '
        'describe associations used by the model, not causal behavioral explanations.', '',
        '- Correlated/derived predictors make individual attributions reference-dependent; '
        'tree-path SHAP does not resolve confounding.\n'
        '- Repeated-player dependence means sampled snapshots are not independent; '
        'the analysis is not evidence of unseen-player generalization.\n'
        '- This historical 2005-2006 dataset may not represent current products, regulation '
        'or player populations; demographic history is not independently verified.\n'
        '- Model-family selection already used validation, so performance and interpretations '
        'can have validation-selection optimism. No uncertainty intervals or causal inference '
        'are provided.\n'
        '- Uniform sampling and quantile summaries can underrepresent rare categories/extremes. '
        'Local cases are illustrative, not a representative survey.\n'
        '- Synthetic interaction contrasts may be outside observed joint support and do not '
        'constitute exact SHAP interaction attribution.\n'
        '- Test remains sealed; probabilities remain uncalibrated; no final operational '
        'threshold was chosen.', '']
    return '\n'.join(lines)


def export_analysis(context):
    import hashlib
    import json
    import platform
    import shap
    import xgboost
    c = context
    reports = c['root'] / 'reports'
    outputs = {'shap_global_importance.csv': c['global_table'][['feature', 'mean_abs_shap', 'feature_group']],
               'shap_feature_mapping.csv': c['mapping'],
               'shap_feature_group_importance.csv': c['groups'],
               'shap_permutation_comparison.csv': c['comparison'],
               'shap_dependence_bins.csv': c['bins'],
               'shap_targeted_interactions.csv': c['interactions'],
               'shap_local_explanations.csv': c['cases']}
    validate_local_export(c['cases'])
    for filename, table in outputs.items():
        table.to_csv(reports / filename, index=False)
    save_figures(c)
    (reports / 'shap_summary.md').write_text(summary_markdown(c), encoding='utf-8')
    metadata = dict(model_sha256=c['model_sha256'], data_sha256=c['data_sha256'],
                    validation_rows=len(c['validation']), sample_rows=len(c['sample']),
                    transformed_columns=len(c['mapping']), random_state=42,
                    sampling='uniform validation rows without replacement',
                    validation_metrics=c['metrics'], python=platform.python_version(),
                    shap=shap.__version__, xgboost=xgboost.__version__,
                    feature_perturbation='tree_path_dependent', model_output='raw log-odds',
                    external_background=False, additivity_checked=True,
                    diagnostic_threshold=.5, final_threshold_selected=False,
                    test_predictions_generated=False, test_shap_computed=False,
                    model_refitted=False, calibrated=False)
    (reports / 'shap_run_metadata.json').write_text(json.dumps(metadata, indent=2) + '\n', encoding='utf-8')
    after = hashlib.sha256((c['root'] / 'models/xgboost_model.joblib').read_bytes()).hexdigest()
    if after != c['model_sha256']:
        raise RuntimeError('Frozen model artifact unexpectedly changed.')

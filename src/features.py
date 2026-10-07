"""Historical player-snapshot predictors and separately constructed churn targets.

Snapshot-day transactions are known at prediction time (end-of-day snapshots).
No model preprocessing is performed. See docs/feature_dictionary.md for conventions.
"""
from __future__ import annotations

from collections.abc import Iterable
import numpy as np
import pandas as pd

OBSERVATION_DAYS = 30
OUTCOME_DAYS = 60
METADATA_COLUMNS = ['UserID', 'snapshot_date']
TARGET_COLUMN = 'churn_60d'
CATEGORICAL_COLUMNS = ['Country', 'Language', 'Gender']


def feature_specs() -> list[dict[str, str]]:
    """Machine-readable definitions used to generate the Markdown dictionary."""
    specs = []

    def add(name, dtype, window, definition, notes='Historical transactions only.'):
        specs.append(dict(name=name, type=dtype, window=window, definition=definition,
                          role='predictor', leakage_notes=notes))

    for days in [30, 14, 7]:
        window = f'[S-{days-1}, S], {days} calendar dates'
        add(f'active_days_{days}d', 'integer', window, 'Distinct dates with Bets > 0.')
        add(f'bets_{days}d', 'integer', window, 'Sum of Bets over all cleaned records.')
    for days in [30, 7]:
        add(f'avg_bets_per_active_day_{days}d', 'float', f'{days} days through S',
            f'bets_{days}d / active_days_{days}d; missing if denominator is zero.')
    add('days_since_last_active_bet', 'integer', '30 days through S',
        'S minus latest date with Bets > 0; 0 means betting on snapshot day.')
    for kind in ['max', 'mean']:
        add(f'{kind}_gap_between_active_days_30d', 'float', '30 days through S',
            f'{kind.title()} full inactive dates between consecutive positive-bet dates '
            '(date difference minus 1); missing if fewer than two active dates. '
            'Excludes leading/trailing inactivity and gaps crossing the window boundary.')
    for metric in ['stake', 'winnings']:
        for days in [30, 14, 7]:
            add(f'{metric}_{days}d', 'float (EUR)', f'{days} days through S',
                f'Sum of cleaned {metric.title()} including zero-bet financial records '
                'and signed accounting corrections; no clipping.')
    for days in [30, 7]:
        add(f'net_loss_{days}d', 'float (EUR)', f'{days} days through S',
            f'stake_{days}d minus winnings_{days}d; negative values are net winnings.')
    add('avg_stake_per_bet_30d', 'float (EUR)', '30 days through S',
        'stake_30d / bets_30d; missing for a zero denominator (eligibility prevents this).')
    add('avg_stake_per_active_day_30d', 'float (EUR)', '30 days through S',
        'All-record stake_30d / positive-bet active_days_30d.')
    for kind in ['max', 'median', 'std']:
        add(f'{kind}_daily_stake_30d', 'float (EUR)', '30 calendar dates through S',
            f'{kind.title()} of 30 daily Stakes, including recorded zero-bet days and '
            'zero for dates with no record. Standard deviation uses ddof=0 (population).')
    for metric in ['active_days', 'bets', 'stake']:
        add(f'{metric}_7d_share', 'float', '7 days / 30 days, both ending S',
            f'{metric}_7d / {metric}_30d; missing if denominator is zero.')
    for metric in ['bets', 'stake', 'active_days']:
        for period, window in [('recent', '[S-14, S], 15 dates'), ('early', '[S-29, S-15], 15 dates')]:
            add(f'{metric}_{period}15', 'float (EUR)' if metric == 'stake' else 'integer', window,
                'Positive-bet active-day count.' if metric == 'active_days' else
                f'Sum of {metric.title()} over all cleaned records.')
        add(f'{metric}_change_15d', 'float (EUR)' if metric == 'stake' else 'integer',
            'Two disjoint 15-day halves of the 30-day window',
            f'{metric}_recent15 minus {metric}_early15; signed change.')
        add(f'{metric}_change_pct_15d', 'float (%)', 'Two disjoint 15-day halves ending S',
            f'100 * ({metric}_recent15 - {metric}_early15) / {metric}_early15. '
            'Missing whenever the early denominator is zero, including 0/0.')
    for name, source in [('days_since_registration', 'RegDate'),
                         ('days_since_first_pay', 'Fstpdate'),
                         ('days_since_first_casino_activity', 'Fstcadate')]:
        add(name, 'float (days)', 'Lifecycle event through S',
            f'S minus demographic {source}; missing if that source event is later than S.',
            'Only use event dates on/before S. Source event inconsistencies are not repaired '
            'using future activity; future-dated events are unknown as of S.')
    for col in CATEGORICAL_COLUMNS:
        add(col, 'raw coded category (integer)', 'Demographic record',
            'Original unencoded code; categorical interpretation, not a continuous magnitude.',
            'Assume these cohort attributes are available at S; no historical change log '
            'exists to verify past country/language/gender states.')
    return specs


PREDICTOR_COLUMNS = [s['name'] for s in feature_specs()]


def validate_inputs(demographics: pd.DataFrame, daily: pd.DataFrame) -> None:
    """Reject ambiguous grains, invalid dates, and invalid transaction values."""
    required_demo = ['UserID', 'Country', 'Language', 'Gender', 'RegDate', 'Fstpdate', 'Fstcadate']
    required_daily = ['UserID', 'Date', 'Stake', 'Winnings', 'Bets']
    for frame, required in [(demographics, required_demo), (daily, required_daily)]:
        if not set(required).issubset(frame.columns) or frame[required].isna().any().any():
            raise ValueError('Required input columns must exist and be nonmissing.')
    for frame, columns in [(demographics, ['RegDate', 'Fstpdate', 'Fstcadate']), (daily, ['Date'])]:
        for col in columns:
            if not pd.api.types.is_datetime64_any_dtype(frame[col]):
                raise TypeError(f'{col} must already be parsed as datetime.')
            if not frame[col].eq(frame[col].dt.normalize()).all():
                raise ValueError('Dates must be normalized calendar days.')
    if not demographics.UserID.is_unique or daily.duplicated(['UserID', 'Date']).any():
        raise ValueError('Expected unique demographic IDs and unique player-date activity.')
    if not set(daily.UserID).issubset(set(demographics.UserID)):
        raise ValueError('Activity contains players without demographics.')
    if not np.isfinite(daily[['Stake', 'Winnings', 'Bets']].to_numpy()).all():
        raise ValueError('Amounts and Bets must be finite.')
    if not daily[['Stake', 'Bets']].ge(0).all().all() or not daily.Bets.mod(1).eq(0).all():
        raise ValueError('Stake must be nonnegative and Bets nonnegative whole numbers.')


def safe_divide(numerator: pd.Series, denominator: pd.Series) -> pd.Series:
    """Return NaN for zero denominators; never fabricate infinite/zero ratios."""
    return numerator.div(denominator.where(denominator.ne(0)))


def _window_totals(history: pd.DataFrame, ids: pd.Index, start, end) -> pd.DataFrame:
    window = history.loc[history.Date.between(start, end)]
    totals = window.groupby('UserID')[['Bets', 'Stake', 'Winnings']].sum().reindex(ids, fill_value=0)
    totals['active_days'] = window.loc[window.Bets.gt(0)].groupby('UserID').Date.nunique().reindex(ids, fill_value=0)
    return totals


def build_snapshot_features(demographics: pd.DataFrame, daily: pd.DataFrame,
                            snapshot_date) -> pd.DataFrame:
    """Build predictors using a history-only view; never inspect future transactions.

    Returns eligible (recent positive-betting) players for a complete 30-day history.
    This function does not require or evaluate future outcome availability.
    """
    snapshot = pd.Timestamp(snapshot_date).normalize()
    observation_start = snapshot - pd.Timedelta(days=OBSERVATION_DAYS - 1)
    if daily.empty or observation_start < daily.Date.min():
        raise ValueError('A complete 30-day observation window is required.')
    # All downstream transaction aggregations receive only this bounded history.
    history = daily.loc[daily.Date.between(observation_start, snapshot)].copy()
    assert history.Date.le(snapshot).all() and history.Date.ge(observation_start).all()
    active = history.loc[history.Bets.gt(0)].sort_values(['UserID', 'Date']).copy()
    ids = pd.Index(sorted(active.UserID.unique()), name='UserID')
    f = pd.DataFrame(index=ids)
    for days in [30, 14, 7]:
        totals = _window_totals(history, ids, snapshot-pd.Timedelta(days=days-1), snapshot)
        for metric, source in [('active_days', 'active_days'), ('bets', 'Bets'),
                               ('stake', 'Stake'), ('winnings', 'Winnings')]:
            f[f'{metric}_{days}d'] = totals[source]
    for days in [30, 7]:
        f[f'avg_bets_per_active_day_{days}d'] = safe_divide(f[f'bets_{days}d'], f[f'active_days_{days}d'])
        f[f'net_loss_{days}d'] = f[f'stake_{days}d'] - f[f'winnings_{days}d']
    last_active = active.groupby('UserID').Date.max().reindex(ids)
    f['days_since_last_active_bet'] = (snapshot - last_active).dt.days
    active['inactive_gap'] = active.groupby('UserID').Date.diff().dt.days - 1
    gap_stats = active.groupby('UserID').inactive_gap.agg(['max', 'mean']).reindex(ids)
    f['max_gap_between_active_days_30d'] = gap_stats['max']
    f['mean_gap_between_active_days_30d'] = gap_stats['mean']
    f['avg_stake_per_bet_30d'] = safe_divide(f.stake_30d, f.bets_30d)
    f['avg_stake_per_active_day_30d'] = safe_divide(f.stake_30d, f.active_days_30d)
    calendar = pd.date_range(observation_start, snapshot, freq='D')
    daily_stake = history.pivot(index='UserID', columns='Date', values='Stake').reindex(
        index=ids, columns=calendar).fillna(0)
    f['max_daily_stake_30d'] = daily_stake.max(axis=1)
    f['median_daily_stake_30d'] = daily_stake.median(axis=1)
    f['std_daily_stake_30d'] = daily_stake.std(axis=1, ddof=0)
    for metric in ['active_days', 'bets', 'stake']:
        f[f'{metric}_7d_share'] = safe_divide(f[f'{metric}_7d'], f[f'{metric}_30d'])
    early = _window_totals(history, ids, observation_start, snapshot-pd.Timedelta(days=15))
    recent = _window_totals(history, ids, snapshot-pd.Timedelta(days=14), snapshot)
    for metric, source in [('bets', 'Bets'), ('stake', 'Stake'), ('active_days', 'active_days')]:
        f[f'{metric}_early15'] = early[source]
        f[f'{metric}_recent15'] = recent[source]
        f[f'{metric}_change_15d'] = f[f'{metric}_recent15'] - f[f'{metric}_early15']
        f[f'{metric}_change_pct_15d'] = 100*safe_divide(f[f'{metric}_change_15d'], f[f'{metric}_early15'])
    demo = demographics.set_index('UserID').reindex(ids)
    for name, source in [('days_since_registration', 'RegDate'),
                         ('days_since_first_pay', 'Fstpdate'),
                         ('days_since_first_casino_activity', 'Fstcadate')]:
        known_date = demo[source].where(demo[source].le(snapshot))
        f[name] = (snapshot - known_date).dt.days
    for col in CATEGORICAL_COLUMNS:
        f[col] = demo[col]
    f = f[PREDICTOR_COLUMNS]
    f.insert(0, 'snapshot_date', snapshot)
    return f.reset_index()


def build_snapshot_target(daily: pd.DataFrame, eligible_ids: Iterable[int],
                          snapshot_date, dataset_end) -> pd.Series:
    """Create 60-day labels separately; fail rather than label a censored outcome."""
    snapshot = pd.Timestamp(snapshot_date).normalize()
    outcome_end = snapshot + pd.Timedelta(days=OUTCOME_DAYS)
    if outcome_end > pd.Timestamp(dataset_end):
        raise ValueError('Cannot label a snapshot without the complete 60-day outcome.')
    ids = pd.Index(eligible_ids, name='UserID')
    future = daily.loc[daily.Date.gt(snapshot) & daily.Date.le(outcome_end) & daily.Bets.gt(0)]
    return pd.Series((~ids.isin(future.UserID)).astype('int64'), index=ids, name=TARGET_COLUMN)


def monthly_snapshot_dates(daily: pd.DataFrame) -> pd.DatetimeIndex:
    """Month ends with both full history and a full outcome window."""
    start, end = daily.Date.min(), daily.Date.max()
    dates = pd.date_range(start, end, freq='ME')
    return dates[(dates-pd.Timedelta(days=29) >= start) &
                 (dates+pd.Timedelta(days=OUTCOME_DAYS) <= end)]


def build_model_panel(demographics: pd.DataFrame, daily: pd.DataFrame) -> pd.DataFrame:
    """Return the in-memory panel; no files, training, encoding, or scaling."""
    validate_inputs(demographics, daily)
    blocks = []
    for snapshot in monthly_snapshot_dates(daily):
        features = build_snapshot_features(demographics, daily, snapshot)
        target = build_snapshot_target(daily, features.UserID, snapshot, daily.Date.max())
        block = features.join(target, on='UserID', validate='one_to_one')
        blocks.append(block)
    if not blocks:
        raise ValueError('No monthly snapshots with full history and outcome coverage.')
    return pd.concat(blocks, ignore_index=True).sort_values(METADATA_COLUMNS[::-1]).reset_index(drop=True)


def validate_panel(panel: pd.DataFrame, demographics: pd.DataFrame, daily: pd.DataFrame) -> pd.DataFrame:
    """Assert feature invariants and independently audit history/target boundaries.

    The audit uses per-player date arrays and searchsorted, independently of the
    groupby/filter operations in the builder. Returns snapshot-level coverage counts.
    """
    assert list(panel.columns) == METADATA_COLUMNS + PREDICTOR_COLUMNS + [TARGET_COLUMN]
    assert not panel.duplicated(METADATA_COLUMNS).any()
    assert panel[METADATA_COLUMNS+[TARGET_COLUMN]].notna().all().all()
    assert set(panel[TARGET_COLUMN].unique()) <= {0, 1}
    assert panel.active_days_30d.ge(1).all()
    assert panel.snapshot_date.sub(pd.Timedelta(days=29)).ge(daily.Date.min()).all()
    assert panel.snapshot_date.add(pd.Timedelta(days=60)).le(daily.Date.max()).all()
    assert not np.isinf(panel[PREDICTOR_COLUMNS].to_numpy(dtype=float)).any()
    nonnegative = [c for c in PREDICTOR_COLUMNS if c.startswith(('active_days_', 'bets_', 'stake_'))
                   and not any(x in c for x in ['change'])]
    nonnegative += [c for c in PREDICTOR_COLUMNS if c.startswith(('avg_', 'days_since_',
                    'max_', 'mean_gap_', 'median_', 'std_'))]
    assert panel[nonnegative].dropna(how='all').ge(0).where(panel[nonnegative].notna(), True).all().all()
    for days in [30, 14, 7]:
        assert panel[f'active_days_{days}d'].between(0, days).all()
        assert panel[f'active_days_{days}d'].le(panel[f'bets_{days}d']).all()
    assert panel.days_since_last_active_bet.between(0,29).all()
    for metric in ['active_days', 'bets', 'stake']:
        assert panel[f'{metric}_7d_share'].dropna().between(0,1).all()
        assert np.allclose(panel[f'{metric}_30d'], panel[f'{metric}_early15']+panel[f'{metric}_recent15'])
    for days in [30,7]:
        assert np.allclose(panel[f'net_loss_{days}d'], panel[f'stake_{days}d']-panel[f'winnings_{days}d'])
    allowed_missing = {'avg_bets_per_active_day_7d': panel.active_days_7d.eq(0),
        'max_gap_between_active_days_30d': panel.active_days_30d.lt(2),
        'mean_gap_between_active_days_30d': panel.active_days_30d.lt(2),
        'stake_7d_share': panel.stake_30d.eq(0)}
    for metric in ['bets','stake','active_days']:
        allowed_missing[f'{metric}_change_pct_15d'] = panel[f'{metric}_early15'].eq(0)
    d = demographics.set_index('UserID')
    for feature, source in [('days_since_registration','RegDate'),
                            ('days_since_first_pay','Fstpdate'),
                            ('days_since_first_casino_activity','Fstcadate')]:
        dates = panel.UserID.map(d[source])
        allowed_missing[feature] = dates.gt(panel.snapshot_date)
    for feature in PREDICTOR_COLUMNS:
        expected_missing = allowed_missing.get(feature, pd.Series(False, index=panel.index))
        assert panel[feature].isna().eq(expected_missing).all(), f'Unexpected missingness: {feature}'
    active_dates = {uid: np.sort(g.Date.to_numpy(dtype='datetime64[ns]')) for uid, g in
                    daily.loc[daily.Bets.gt(0)].groupby('UserID')}
    audit = []
    for snapshot, group in panel.groupby('snapshot_date'):
        start = snapshot-pd.Timedelta(days=29)
        end = snapshot+pd.Timedelta(days=60)
        for row in group.itertuples(index=False):
            dates = active_dates[row.UserID]
            s = snapshot.to_datetime64()
            n_history = np.searchsorted(dates, s, side='right') - np.searchsorted(dates, start.to_datetime64(), side='left')
            n_future = np.searchsorted(dates, end.to_datetime64(), side='right') - np.searchsorted(dates, s, side='right')
            assert row.active_days_30d == n_history > 0
            assert row.churn_60d == int(n_future == 0)
        audit.append({'snapshot_date': snapshot, 'feature_start': start, 'feature_end': snapshot,
            'outcome_start': snapshot+pd.Timedelta(days=1), 'outcome_end': end,
            'observations_audited': len(group)})
    return pd.DataFrame(audit)


def proposed_temporal_split(panel: pd.DataFrame) -> tuple[pd.Series, pd.DataFrame]:
    """A feasible 10/3/4-month design with two omitted month ends per gap.

    Later observation windows also begin after earlier outcome windows end.
    The split is a proposal for these study dates, not a universal automatic rule.
    """
    periods = {'train': ('2005-03-31','2005-12-31'),
               'validation': ('2006-03-31','2006-05-31'),
               'test': ('2006-08-31','2006-11-30')}
    assignment = pd.Series('embargo', index=panel.index, name='proposed_split')
    for name, (start, end) in periods.items():
        assignment.loc[panel.snapshot_date.between(start,end)] = name
    rows = []
    for name in ['train','validation','test','embargo']:
        group = panel.loc[assignment.eq(name)]
        if group.empty:
            raise ValueError(f'Proposed split {name} has no observations; reconsider dates.')
        rows.append({'split': name, 'first_snapshot': group.snapshot_date.min(),
            'last_snapshot': group.snapshot_date.max(), 'snapshots': group.snapshot_date.nunique(),
            'observations': len(group), 'unique_players': group.UserID.nunique(),
            'churn_count': int(group.churn_60d.sum()), 'churn_pct': group.churn_60d.mean()*100})
    summary = pd.DataFrame(rows).set_index('split')
    for earlier,later in [('train','validation'),('validation','test')]:
        last, first = summary.loc[earlier,'last_snapshot'], summary.loc[later,'first_snapshot']
        assert (first-last).days >= OUTCOME_DAYS
        assert last+pd.Timedelta(days=60) < first-pd.Timedelta(days=29)
    return assignment, summary


def write_feature_dictionary(path) -> None:
    """Write documentation only; never write the modelling panel automatically."""
    rows = [{'name':'UserID','type':'integer identifier','window':'N/A',
             'definition':'Player tracking and grouping key; never a predictor.','role':'metadata',
             'leakage_notes':'Repeated players may occur across temporal splits.'},
            {'name':'snapshot_date','type':'ISO calendar date','window':'Prediction at end of S',
             'definition':'Monthly snapshot for tracking, embargo, and temporal splitting.',
             'role':'metadata','leakage_notes':'Excluded from predictor list.'}]
    rows += feature_specs()
    rows += [{'name':TARGET_COLUMN,'type':'binary integer','window':'(S, S+60], 60 dates',
              'definition':'1 if no Bets > 0 day in the full future window; otherwise 0.',
              'role':'target','leakage_notes':'Future activity is used solely for this outcome; '
              'S+60 must be on/before the dataset end. Never a predictor.'}]
    preface = '''# Feature dictionary — 60-day operational churn panel

One row is an eligible player at an end-of-day month-end snapshot S. Eligibility requires
positive Bets in [S-29, S], full 30-day historical coverage, and full outcome coverage.
The target is extended inactivity, not guaranteed permanent customer departure.

There are 45 predictors, two metadata columns, and one target. Country, Language, and
Gender retain original integer codes but must be treated as categories during future
preprocessing. No categories are encoded and no numeric variables are scaled here.

All transaction predictors use only the 30 days through S. 7-/14-day windows contain
7/14 dates including S; early and recent trend halves contain 15 dates each. Monetary
totals include all cleaned records, including zero-bet transactions and signed Winnings
corrections. Negative Winnings or net loss are valid. Missing transaction dates are
zero-filled only for 30-calendar-day daily-stake statistics, assuming complete collection;
the source rows and all financial totals remain unchanged.

Division by zero yields missing values, never infinity, including 0/0 percentage changes.
No historical active days in the recent 7 days makes the 7-day bets-per-active-day ratio
missing. Fewer than two positive-bet days makes between-day gaps missing; no gap is
invented. Gaps count full inactive dates (date difference minus one), excluding window
edges. No missing predictors are imputed before temporal splitting.

Lifecycle dates are used only when on/before S. Demographic Fstcadate can be later than
observed gaming dates; future-dated source events are masked rather than replaced using
future data. Other inconsistencies are retained for transparency. Raw demographic codes
are assumed available historically; the source provides no attribute-change history.

The proposed split metadata is held separately and not exported in this panel. Training
uses 2005-03-31–2005-12-31; validation 2006-03-31–2006-05-31; test 2006-08-31–2006-11-30.
January/February and June/July 2006 snapshots are embargoed. Snapshot gaps are 90/92
days, and later observation windows start after earlier label windows end. Same-player
overlap is allowed for future behaviour of known/recently active customers; unseen-player
generalisation requires a separate design. Fit future preprocessing on training data only.

`data/processed/model_panel_60d.csv` is a local Git-ignored output and must not be uploaded.

| Feature name | Type | Time window | Definition | Modelling role | Leakage notes |
| --- | --- | --- | --- | --- | --- |
'''
    lines = ['| '+' | '.join(str(row[key]).replace('|','/') for key in
             ['name','type','window','definition','role','leakage_notes'])+' |' for row in rows]
    from pathlib import Path
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(preface+'\n'.join(lines)+'\n', encoding='utf-8')

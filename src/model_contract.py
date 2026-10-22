"""Public inference contract copied in order from the frozen Day 10 specification.

Tests compare this explicit tuple to reports/final_model_specification.json and
the feature dictionary. Changes require a new model version, not inference tuning.
"""
MODEL_VERSION = 'xgboost-60d-demographic-free-v1'
MODEL_FAMILY = 'XGBoost'
TARGET_NAME = 'churn_60d'
OBSERVATION_WINDOW_DAYS = 30
OUTCOME_WINDOW_DAYS = 60
DIAGNOSTIC_THRESHOLD = 0.50
TARGET_DEFINITION = (
    '60-day operational inactivity: no positive betting activity (Bets > 0) '
    'during the 60 days immediately after the snapshot; not permanent customer departure.'
)

FINAL_FEATURES = (
    'active_days_30d',
    'bets_30d',
    'active_days_14d',
    'bets_14d',
    'active_days_7d',
    'bets_7d',
    'avg_bets_per_active_day_30d',
    'avg_bets_per_active_day_7d',
    'days_since_last_active_bet',
    'max_gap_between_active_days_30d',
    'mean_gap_between_active_days_30d',
    'stake_30d',
    'stake_14d',
    'stake_7d',
    'winnings_30d',
    'winnings_14d',
    'winnings_7d',
    'net_loss_30d',
    'net_loss_7d',
    'avg_stake_per_bet_30d',
    'avg_stake_per_active_day_30d',
    'max_daily_stake_30d',
    'median_daily_stake_30d',
    'std_daily_stake_30d',
    'active_days_7d_share',
    'bets_7d_share',
    'stake_7d_share',
    'bets_recent15',
    'bets_early15',
    'bets_change_15d',
    'bets_change_pct_15d',
    'stake_recent15',
    'stake_early15',
    'stake_change_15d',
    'stake_change_pct_15d',
    'active_days_recent15',
    'active_days_early15',
    'active_days_change_15d',
    'active_days_change_pct_15d',
    'days_since_registration',
    'days_since_first_pay',
    'days_since_first_casino_activity',
)

# No fields are silently ignored: callers must explicitly strip metadata/labels.
EXCLUDED_FIELDS = ('Country', 'Language', 'Gender', 'UserID', 'snapshot_date', TARGET_NAME)
BATCH_RISK_BANDS = ('low', 'medium', 'high', 'very_high')


def get_model_info() -> dict:
    """Non-sensitive contract metadata, available even without the local model."""
    return dict(
        model_family=MODEL_FAMILY, model_version=MODEL_VERSION, target=TARGET_NAME,
        target_definition=TARGET_DEFINITION, observation_window_days=OBSERVATION_WINDOW_DAYS,
        outcome_window_days=OUTCOME_WINDOW_DAYS, predictor_count=len(FINAL_FEATURES),
        diagnostic_threshold=DIAGNOSTIC_THRESHOLD, calibration='none',
        demographics_excluded=True, metadata_excluded=True,
        intended_use='Prioritization and analysis of 60-day operational inactivity risk.',
        single_risk_band='unranked',
        batch_risk_bands=dict(low='bottom 50%', medium='50th-80th percentile',
                             high='80th-90th percentile', very_high='top 10%'),
        limitations=[
            'Historical 2005-2007 data require current-data validation and monitoring before deployment.',
            'Repeated-player dependence; results do not establish unseen-player generalization.',
            'Associations are not causal; inactivity can be followed by reactivation.',
            'Not gambling-harm diagnosis, responsible-gambling classification, customer-value scoring or automated adverse decisions.',
            'Demographic exclusion does not remove proxies or establish fairness.',
            'Raw probabilities and diagnostic 0.50 are not a financially optimal contact policy.',
            'Batch bands are relative prioritization tools, not regulatory categories.',
        ],
    )

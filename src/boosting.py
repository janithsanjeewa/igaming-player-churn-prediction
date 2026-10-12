"""Deliberate Day 8 XGBoost candidates and validation-only selection."""
import numpy as np
import pandas as pd

PARAMETER_COLUMNS = ['n_estimators', 'learning_rate', 'max_depth', 'min_child_weight',
                     'subsample', 'colsample_bytree', 'reg_lambda', 'reg_alpha', 'gamma']

# Fixed before evaluation; no adaptive follow-up search or early stopping.
_CONFIGURATIONS = [
    ('initial_conservative', 400, .03, 3, 10, .8, .9, 10, .1, .1),
    ('slow_long',            800, .03, 3, 10, .8, .9, 10, .1, .1),
    ('shallow_rate05',       400, .05, 3, 10, .8, .9, 10, .1, .1),
    ('shallow_rate08',       300, .08, 3, 5,  .8, .9, 5,  .1, 0),
    ('depth4_moderate',      500, .05, 4, 5,  .8, .9, 5,  .1, .1),
    ('depth4_regularized',   500, .05, 4, 10, .8, .9, 10, .5, .1),
    ('depth5_regularized',   400, .03, 5, 10, .8, .9, 10, .5, .1),
    ('depth5_flexible',      500, .05, 5, 1,  1., 1., 1,  0,  0),
    ('shallow_full_sample',  600, .03, 3, 5,  1., 1., 5,  0,  0),
    ('depth4_column_sample', 800, .03, 4, 10, .8, .7, 10, .5, .1),
]
XGB_CANDIDATES = [dict(name=row[0], **dict(zip(PARAMETER_COLUMNS, row[1:]))) for row in _CONFIGURATIONS]
CANDIDATE_REQUIRED_COLUMNS = ['name', *PARAMETER_COLUMNS, 'train_roc_auc', 'roc_auc',
    'train_average_precision', 'average_precision', 'brier_score', 'lift_at_top10pct',
    'recall_at_top10pct', 'roc_auc_gap', 'average_precision_gap']


def validate_candidate_results(results: pd.DataFrame) -> None:
    """Reject incomplete, duplicated or nonfinite candidate audit tables."""
    if not set(CANDIDATE_REQUIRED_COLUMNS).issubset(results.columns):
        raise ValueError('Incomplete XGBoost candidate results schema.')
    if results.empty or results.name.isna().any() or results.name.duplicated().any():
        raise ValueError('Candidate names must be nonmissing and unique.')
    if not np.isfinite(results[CANDIDATE_REQUIRED_COLUMNS[1:]].to_numpy(dtype=float)).all():
        raise ValueError('Candidate parameters and metrics must be finite.')


def select_xgboost_candidate(results: pd.DataFrame) -> str:
    """Rank first; regularization resolves performance ties, never accuracy.

    Use Day 7 tolerances: AP .002, ROC .002, top10 lift .02, recall .002,
    AP gap .02 and Brier .002. Within these ties prefer shallower depth,
    larger min_child_weight, larger lambda/alpha/gamma, then fewer trees.
    """
    validate_candidate_results(results)
    candidates = results.copy()
    for metric, tolerance in [('average_precision', .002), ('roc_auc', .002),
                              ('lift_at_top10pct', .02), ('recall_at_top10pct', .002)]:
        candidates = candidates.loc[candidates[metric] >= candidates[metric].max() - tolerance]
    for metric, tolerance in [('average_precision_gap', .02), ('brier_score', .002)]:
        candidates = candidates.loc[candidates[metric] <= candidates[metric].min() + tolerance]
    return candidates.sort_values(
        ['max_depth', 'min_child_weight', 'reg_lambda', 'reg_alpha', 'gamma', 'n_estimators', 'name'],
        ascending=[True, False, False, False, False, True, True]).iloc[0]['name']

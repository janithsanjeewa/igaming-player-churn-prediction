"""Fixed temporal split and train-fitted model preprocessing pipelines."""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.features import CATEGORICAL_COLUMNS, METADATA_COLUMNS, PREDICTOR_COLUMNS, TARGET_COLUMN

SPLIT_BOUNDARIES = {
    'train': ('2005-03-31', '2005-12-31'),
    'validation': ('2006-03-31', '2006-05-31'),
    'test': ('2006-08-31', '2006-11-30'),
}
NUMERIC_COLUMNS = [c for c in PREDICTOR_COLUMNS if c not in CATEGORICAL_COLUMNS]


def temporal_split_labels(snapshot_dates: pd.Series) -> pd.Series:
    """Label the approved windows; all other dates are excluded, never training."""
    dates = pd.to_datetime(snapshot_dates, errors='raise')
    if dates.isna().any():
        raise ValueError('Snapshot dates must be nonmissing.')
    labels = pd.Series('excluded', index=dates.index, dtype='str')
    for name, (start, end) in SPLIT_BOUNDARIES.items():
        labels.loc[dates.between(start, end)] = name
    embargo = dates.dt.to_period('M').isin(pd.PeriodIndex(
        ['2006-01', '2006-02', '2006-06', '2006-07'], freq='M'))
    labels.loc[embargo] = 'embargo'
    return labels


def validate_model_panel(panel: pd.DataFrame) -> None:
    """Verify the Day 5 artifact contract without modelling held-out rows."""
    expected = set(METADATA_COLUMNS + PREDICTOR_COLUMNS + [TARGET_COLUMN])
    if panel.shape != (18472, 48) or set(panel.columns) != expected:
        raise ValueError('Expected the Day 5 panel: 18,472 rows and the 48 documented columns.')
    if not pd.api.types.is_datetime64_any_dtype(panel.snapshot_date):
        raise TypeError('Parse snapshot_date before validation.')
    if panel[METADATA_COLUMNS].isna().any().any() or panel.duplicated(METADATA_COLUMNS).any():
        raise ValueError('Missing metadata or duplicate player-snapshot keys.')
    if not panel.snapshot_date.dt.is_month_end.all():
        raise ValueError('Expected month-end snapshots.')
    if panel.UserID.nunique() != 4025:
        raise ValueError('Unexpected unique player count.')
    if panel[TARGET_COLUMN].value_counts().to_dict() != {0: 11865, 1: 6607}:
        raise ValueError('Unexpected target values/counts.')


def predictor_frame(frame: pd.DataFrame) -> pd.DataFrame:
    """Select only dictionary predictors; identifiers, dates and target are excluded."""
    result = frame.loc[:, PREDICTOR_COLUMNS].copy()
    if not all(pd.api.types.is_numeric_dtype(result[c]) for c in NUMERIC_COLUMNS):
        raise TypeError('All documented numerical predictors must be numeric.')
    if np.isinf(result[NUMERIC_COLUMNS].to_numpy(dtype=float)).any():
        raise ValueError('Infinite predictors are invalid; undefined ratios must remain missing.')
    return result


def build_preprocessing(*, scale_numeric: bool, sparse_threshold: float = 0.3) -> ColumnTransformer:
    """Share feature types and imputation; fit only inside the training pipeline."""
    steps = [('imputer', SimpleImputer(strategy='median', add_indicator=True))]
    if scale_numeric:
        steps.append(('scaler', StandardScaler()))
    numeric = Pipeline(steps)
    categorical = Pipeline([
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('encoder', OneHotEncoder(handle_unknown='ignore')),
    ])
    return ColumnTransformer([
        ('numeric', numeric, NUMERIC_COLUMNS),
        ('categorical', categorical, CATEGORICAL_COLUMNS),
    ], remainder='drop', sparse_threshold=sparse_threshold)


def build_logistic_pipeline(*, class_weight: str | None = None) -> Pipeline:
    """Return a fresh, unfitted pipeline; caller must fit on TRAIN only."""
    return Pipeline([
        ('preprocessing', build_preprocessing(scale_numeric=True)),
        ('model', LogisticRegression(max_iter=5000, random_state=42, class_weight=class_weight)),
    ])


def build_random_forest_pipeline(*, n_estimators=500, max_depth=None,
                                min_samples_leaf=1, max_features='sqrt',
                                class_weight=None) -> Pipeline:
    """Unscaled median/indicator and categorical pipeline, with fixed RF seed."""
    return Pipeline([
        ('preprocessing', build_preprocessing(scale_numeric=False)),
        ('model', RandomForestClassifier(
            n_estimators=n_estimators, max_depth=max_depth,
            min_samples_leaf=min_samples_leaf, max_features=max_features,
            class_weight=class_weight, random_state=42, n_jobs=-1)),
    ])


def build_xgboost_pipeline(**parameters) -> Pipeline:
    """Train-only unscaled preprocessing and reproducible histogram boosting.

    Dense output preserves legitimate numeric zeros: XGBoost treats implicit
    sparse entries as missing. No class reweighting or early stopping is used.
    """
    from xgboost import XGBClassifier

    settings = dict(objective='binary:logistic', eval_metric='logloss',
                    tree_method='hist', random_state=42, n_jobs=-1,
                    importance_type='gain')
    settings.update(parameters)
    return Pipeline([
        ('preprocessing', build_preprocessing(scale_numeric=False, sparse_threshold=0)),
        ('model', XGBClassifier(**settings)),
    ])

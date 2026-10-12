"""Small Day 7 experiment helpers with explicit temporal prediction guards."""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.evaluation import classification_metrics, ranking_metrics
from src.features import TARGET_COLUMN
from src.preprocessing import predictor_frame, temporal_split_labels

# Deliberate alternatives, not a Cartesian grid. All primary models unweighted.
RF_CANDIDATES = [
    dict(name='initial', max_depth=None, min_samples_leaf=1, max_features='sqrt'),
    dict(name='unlimited_leaf20', max_depth=None, min_samples_leaf=20, max_features='sqrt'),
    dict(name='depth8_leaf5', max_depth=8, min_samples_leaf=5, max_features='sqrt'),
    dict(name='depth12_leaf10', max_depth=12, min_samples_leaf=10, max_features='sqrt'),
    dict(name='depth16_leaf20', max_depth=16, min_samples_leaf=20, max_features='sqrt'),
    dict(name='depth12_leaf10_half', max_depth=12, min_samples_leaf=10, max_features=0.5),
    dict(name='depth8_leaf20_half', max_depth=8, min_samples_leaf=20, max_features=0.5),
]


def modelling_partition(panel: pd.DataFrame, split: str) -> pd.DataFrame:
    """Only training/validation may become modelling frames; test stays aggregate."""
    if split not in ('train', 'validation'):
        raise ValueError('Only train and validation are open for modelling.')
    return panel.loc[temporal_split_labels(panel.snapshot_date).eq(split)].copy()


def guarded_probabilities(pipeline, frame: pd.DataFrame, split: str) -> np.ndarray:
    """Validate every date before prediction; reject test, embargo and mixed frames."""
    if split not in ('train', 'validation') or frame.empty:
        raise ValueError('Prediction requires a nonempty train or validation frame.')
    if not temporal_split_labels(frame.snapshot_date).eq(split).all():
        raise ValueError('Prediction frame contains dates outside the authorized split.')
    probabilities = pipeline.predict_proba(predictor_frame(frame))[:, 1]
    if not np.isfinite(probabilities).all() or ((probabilities < 0) | (probabilities > 1)).any():
        raise ValueError('Invalid model probabilities.')
    return probabilities


def evaluate_candidate(pipeline, train, validation) -> dict:
    """Fit TRAIN only; report train/validation performance without any test input."""
    if train.empty or not temporal_split_labels(train.snapshot_date).eq('train').all():
        raise ValueError('Fit requires the approved training dates.')
    if validation.empty or not temporal_split_labels(validation.snapshot_date).eq('validation').all():
        raise ValueError('Evaluation requires the approved validation dates.')
    pipeline.fit(predictor_frame(train), train[TARGET_COLUMN])
    training = classification_metrics(train[TARGET_COLUMN], guarded_probabilities(pipeline, train, 'train'))
    validation_probabilities = guarded_probabilities(pipeline, validation, 'validation')
    heldout = {**classification_metrics(validation[TARGET_COLUMN], validation_probabilities),
               **ranking_metrics(validation[TARGET_COLUMN], validation_probabilities)}
    return {**heldout, 'train_roc_auc': training['roc_auc'],
            'train_average_precision': training['average_precision'],
            'roc_auc_gap': training['roc_auc'] - heldout['roc_auc'],
            'average_precision_gap': training['average_precision'] - heldout['average_precision']}


def select_candidate(results: pd.DataFrame) -> str:
    """Predeclared near-tie tolerances; ranking first, then simplicity/gap/Brier.

    Sequentially retain configurations within 0.002 AP, 0.002 ROC-AUC,
    0.02 lift and 0.002 top-decile recall of the best remaining candidate.
    Within that ranking tie, prefer lower AP gap (tolerance 0.02), then
    Brier (tolerance 0.002), then shallower depth and larger leaves.
    Sensitivity rows cannot compete with the unweighted primary family.
    """
    candidates = results.loc[results.class_weight.eq('unweighted')].copy()
    if candidates.empty:
        raise ValueError('No unweighted primary candidates.')
    for metric, tolerance in [('average_precision', .002), ('roc_auc', .002),
                              ('lift_at_top10pct', .02), ('recall_at_top10pct', .002)]:
        candidates = candidates.loc[candidates[metric] >= candidates[metric].max() - tolerance]
    for metric, tolerance in [('average_precision_gap', .02), ('brier_score', .002)]:
        candidates = candidates.loc[candidates[metric] <= candidates[metric].min() + tolerance]
    candidates['depth_order'] = candidates.max_depth.fillna(np.inf)
    return candidates.sort_values(['depth_order', 'min_samples_leaf', 'average_precision', 'name'],
                                  ascending=[True, False, False, True]).iloc[0]['name']

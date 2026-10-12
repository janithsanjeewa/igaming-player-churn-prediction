"""Shared training/validation evaluation with sealed temporal prediction guards."""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.evaluation import classification_metrics, ranking_metrics
from src.features import TARGET_COLUMN
from src.preprocessing import predictor_frame, temporal_split_labels


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

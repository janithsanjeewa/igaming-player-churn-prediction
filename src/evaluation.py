"""Validation metrics and deterministic observation-level retention ranking."""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import (accuracy_score, average_precision_score,
    balanced_accuracy_score, brier_score_loss, confusion_matrix, f1_score,
    log_loss, precision_score, recall_score, roc_auc_score)


def _validated_arrays(y_true, probabilities):
    y = np.asarray(y_true)
    p = np.asarray(probabilities, dtype=float)
    if y.ndim != 1 or p.ndim != 1 or len(y) != len(p) or len(y) == 0:
        raise ValueError('Expected nonempty aligned one-dimensional arrays.')
    if not np.isin(y, [0, 1]).all() or not np.isfinite(p).all() or ((p < 0) | (p > 1)).any():
        raise ValueError('Expected binary labels and finite probabilities in [0, 1].')
    return y.astype(int), p


def classification_metrics(y_true, probabilities, threshold=0.5) -> dict:
    """Ranking, probability and threshold metrics; AP is the PR summary statistic."""
    y, p = _validated_arrays(y_true, probabilities)
    if not 0 <= threshold <= 1:
        raise ValueError('Threshold must be in [0, 1].')
    predicted = p >= threshold
    tn, fp, fn, tp = confusion_matrix(y, predicted, labels=[0, 1]).ravel()
    return {
        'roc_auc': roc_auc_score(y, p),
        'average_precision': average_precision_score(y, p),
        'log_loss': log_loss(y, p, labels=[0, 1]),
        'brier_score': brier_score_loss(y, p),
        'accuracy': accuracy_score(y, predicted),
        'balanced_accuracy': balanced_accuracy_score(y, predicted),
        'precision': precision_score(y, predicted, zero_division=0),
        'recall': recall_score(y, predicted, zero_division=0),
        'f1': f1_score(y, predicted, zero_division=0),
        'specificity': tn / (tn + fp) if tn + fp else np.nan,
        'true_positives': int(tp), 'false_positives': int(fp),
        'true_negatives': int(tn), 'false_negatives': int(fn),
        'predicted_positives': int(predicted.sum()),
    }


def cumulative_gains_table(y_true, probabilities, fractions=None) -> pd.DataFrame:
    """Ceil(n*fraction) rows; ties preserve input order, with no label tie-breaking.

    These are player-snapshot observations, not deduplicated campaign recipients.
    Lift is top-group prevalence divided by full validation prevalence.
    """
    y, p = _validated_arrays(y_true, probabilities)
    fractions = np.arange(1, 11) / 10 if fractions is None else np.asarray(fractions, dtype=float)
    if fractions.ndim != 1 or not np.isfinite(fractions).all() or ((fractions <= 0) | (fractions > 1)).any():
        raise ValueError('Fractions must be finite and in (0, 1].')
    cumulative = np.cumsum(y[np.argsort(-p, kind='stable')])
    total = y.sum()
    records = []
    for fraction in fractions:
        count = int(np.ceil(len(y) * fraction))
        captured = int(cumulative[count - 1])
        precision = captured / count
        records.append({'target_fraction': fraction, 'selected_observations': count,
            'actual_fraction': count / len(y), 'churn_observations_captured': captured,
            'precision': precision, 'recall': captured / total if total else np.nan,
            'lift': precision / y.mean() if total else np.nan})
    return pd.DataFrame(records)


def ranking_metrics(y_true, probabilities) -> dict:
    table = cumulative_gains_table(y_true, probabilities, [0.1, 0.2])
    return {f'{metric}_at_top{int(row.target_fraction * 100)}pct': float(getattr(row, metric))
            for row in table.itertuples() for metric in ['recall', 'precision', 'lift']}


def threshold_analysis(y_true, probabilities, thresholds=(.2, .3, .4, .5, .6, .7, .8)) -> pd.DataFrame:
    columns = ['predicted_positives', 'precision', 'recall', 'f1', 'specificity']
    return pd.DataFrame([{'threshold': t, **{k: v for k, v in
        classification_metrics(y_true, probabilities, t).items() if k in columns}}
        for t in thresholds])


def coefficient_table(pipeline) -> pd.DataFrame:
    """Coefficients use scaled numeric/indicator units and unscaled one-hot units."""
    names = pipeline.named_steps['preprocessing'].get_feature_names_out()
    coefficients = pipeline.named_steps['model'].coef_[0]
    return pd.DataFrame({'feature': names, 'coefficient': coefficients,
        'odds_ratio': np.exp(coefficients), 'absolute_coefficient': np.abs(coefficients)
        }).sort_values('absolute_coefficient', ascending=False).reset_index(drop=True)

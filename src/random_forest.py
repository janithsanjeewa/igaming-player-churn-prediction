"""Small Day 7 experiment helpers with explicit temporal prediction guards."""
from __future__ import annotations

import numpy as np
import pandas as pd

# Re-export shared helpers to preserve Day 7 notebook/import compatibility.
from src.modelling import modelling_partition, guarded_probabilities, evaluate_candidate

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

"""Validated inference around the frozen final pipeline; no fit/calibration APIs.

No request logging, persistence, real-player examples or inference-time feature
engineering. Inputs must already follow docs/feature_dictionary.md.
"""
from __future__ import annotations

import argparse
from collections.abc import Mapping
from functools import lru_cache
import hashlib
from io import BytesIO
import json
import os
from pathlib import Path
from threading import RLock

import joblib
import numpy as np
import pandas as pd
from pandas.api.types import is_bool_dtype, is_complex_dtype, is_numeric_dtype

from src.model_contract import (DIAGNOSTIC_THRESHOLD, FINAL_FEATURES, MODEL_VERSION,
                               TARGET_DEFINITION, get_model_info)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH_ENV = 'IGAMING_CHURN_MODEL_PATH'
DEFAULT_MODEL_PATH = Path('models/final_xgboost_churn_model.joblib')
_LOAD_LOCK = RLock()


class FeatureValidationError(ValueError):
    """Invalid feature schema/type; messages never include record values or IDs."""


class ModelLoadError(RuntimeError):
    """Missing, unreadable or incompatible frozen artifact/manifest."""


def validate_features(data: pd.DataFrame) -> pd.DataFrame:
    """Return a detached ordered float64 frame, preserving index and NaN.

    Strict: no metadata, labels, demographics or unknown fields are ignored.
    Numeric strings, bools, complex numbers and infinities are rejected. Genuine
    NaN/pd.NA/None missing values are accepted and left missing for the imputer.
    Empty batches are rejected; single observations are accepted.
    """
    if not isinstance(data, pd.DataFrame):
        raise FeatureValidationError('Batch input must be a pandas DataFrame.')
    if data.columns.has_duplicates:
        raise FeatureValidationError('Duplicate feature columns are not allowed.')
    if not all(isinstance(column, str) for column in data.columns):
        raise FeatureValidationError('Feature column names must be strings.')
    missing = [column for column in FINAL_FEATURES if column not in data.columns]
    if missing:
        raise FeatureValidationError('Missing required features: ' + ', '.join(missing))
    if set(data.columns) != set(FINAL_FEATURES):
        # Unknown field names can contain user data: do not echo them.
        raise FeatureValidationError('Unexpected fields are not allowed; remove metadata, labels and unknown features.')
    if data.empty:
        raise FeatureValidationError('Prediction input must contain at least one observation.')
    ordered = data.loc[:, list(FINAL_FEATURES)].copy()
    for column in FINAL_FEATURES:
        values = ordered[column]
        if is_bool_dtype(values.dtype) or is_complex_dtype(values.dtype):
            raise FeatureValidationError(f'{column} must contain real numerical values or missing values.')
        if not is_numeric_dtype(values.dtype):
            # Object columns may contain real numbers and explicit missing values;
            # reject strings/containers without coercing numeric-looking strings.
            present = values.loc[values.notna()]
            if not all(isinstance(value, (int, float, np.integer, np.floating))
                       and not isinstance(value, (bool, np.bool_)) for value in present):
                raise FeatureValidationError(f'{column} must contain real numerical values or missing values.')
        try:
            converted = values.to_numpy(dtype=np.float64, na_value=np.nan)
        except (TypeError, ValueError, OverflowError):
            raise FeatureValidationError(f'{column} cannot be represented as real numerical values.') from None
        if np.isinf(converted).any():
            raise FeatureValidationError(f'{column} contains infinity; use missing values only when genuinely undefined.')
        ordered[column] = converted
    return ordered


def _model_path(model_path=None) -> Path:
    configured = model_path if model_path is not None else os.environ.get(MODEL_PATH_ENV, DEFAULT_MODEL_PATH)
    path = Path(configured).expanduser()
    return (path if path.is_absolute() else PROJECT_ROOT / path).resolve()


@lru_cache(maxsize=4)
def _load_verified(path, modified_ns, size, specification_bytes, metadata_bytes):
    """Load once per artifact/manifest revision; invoked under a thread lock."""
    del modified_ns, size  # Cache keys, not model settings.
    from sklearn.pipeline import Pipeline
    from xgboost import XGBClassifier
    from src.business_policy import verify_specification

    try:
        spec, metadata = json.loads(specification_bytes), json.loads(metadata_bytes)
        if (spec['predictors'] != list(FINAL_FEATURES) or spec['demographics_retained'] is not False
                or spec['calibration_choice'] != 'raw' or spec['diagnostic_threshold'] != DIAGNOSTIC_THRESHOLD
                or spec['selected_family'] != 'XGBoost' or spec['observation_days'] != 30
                or spec['outcome_days'] != 60 or metadata['calibration'] != 'raw'
                or metadata['demographics_retained'] is not False
                or metadata['diagnostic_threshold'] != DIAGNOSTIC_THRESHOLD):
            raise ValueError('Frozen policy mismatch.')
        if hashlib.sha256(specification_bytes).hexdigest() != metadata['specification_sha256']:
            raise ValueError('Frozen specification fingerprint mismatch.')
        artifact = Path(path).read_bytes()
        if hashlib.sha256(artifact).hexdigest() != metadata['final_model_sha256']:
            raise ValueError('Final model fingerprint mismatch.')
        # Deserialize only the bytes whose hash was verified, avoiding a second read.
        pipeline = joblib.load(BytesIO(artifact))
        if not isinstance(pipeline, Pipeline) or set(pipeline.named_steps) != {'preprocessing', 'model'}:
            raise ValueError('Expected the frozen two-step pipeline.')
        if not isinstance(pipeline['model'], XGBClassifier) or not callable(getattr(pipeline, 'predict_proba', None)):
            raise ValueError('Expected a fitted XGBoost probability classifier.')
        if not np.array_equal(pipeline.classes_, [0, 1]):
            raise ValueError('Unexpected probability class ordering.')
        fitted_spec = {**spec, 'transformed_features': metadata['transformed_features']}
        verify_specification(fitted_spec, pipeline)
        pre = pipeline['preprocessing']
        if (pre.n_features_in_ != len(FINAL_FEATURES)
                or pipeline['model'].n_features_in_ != len(metadata['transformed_features'])
                or any(name != 'numeric' for name, _, _ in pre.transformers)):
            raise ValueError('Frozen feature schema mismatch.')
        return pipeline
    except Exception:
        # Do not expose deserializer payloads, filesystem details or feature values.
        raise ModelLoadError('Cannot load the verified frozen model. Check artifact integrity, manifests and compatible dependencies; inference never retrains.') from None


def load_model(model_path=None):
    """Resolve the default/configured local artifact and cache a verified pipeline."""
    path = _model_path(model_path)
    with _LOAD_LOCK:
        if not path.is_file():
            raise ModelLoadError('Final model artifact is missing. Produce models/final_xgboost_churn_model.joblib through the final evaluation/training workflow first, or configure IGAMING_CHURN_MODEL_PATH. Inference never retrains; retain the one-time evaluation checkpoint.')
        try:
            stat = path.stat()
            spec = (PROJECT_ROOT / 'reports/final_model_specification.json').read_bytes()
            metadata = (PROJECT_ROOT / 'reports/final_test_run_metadata.json').read_bytes()
        except OSError:
            raise ModelLoadError('Cannot read the model artifact or its frozen specification/audit manifests.') from None
        return _load_verified(str(path), stat.st_mtime_ns, stat.st_size, spec, metadata)


def clear_model_cache():
    """Explicit cache invalidation for controlled reloads and tests; never fits."""
    with _LOAD_LOCK:
        _load_verified.cache_clear()


def _probabilities(model, frame):
    result = np.asarray(model.predict_proba(frame), dtype=float)
    if (result.shape != (len(frame), 2) or not np.isfinite(result).all()
            or ((result < 0) | (result > 1)).any()
            or not np.allclose(result.sum(axis=1), 1, atol=1e-6, rtol=0)):
        raise ModelLoadError('Model returned invalid class probabilities.')
    return result[:, 1]


def batch_risk_bands(probabilities) -> np.ndarray:
    """Frozen descending-rank slices, stable input-position ties, no labels."""
    p = np.asarray(probabilities, dtype=float)
    if p.ndim != 1 or not len(p) or not np.isfinite(p).all() or ((p < 0) | (p > 1)).any():
        raise ValueError('Risk bands require nonempty finite probabilities in [0, 1].')
    order = np.argsort(-p, kind='stable')
    n = len(p)
    top10, top20, top50 = (int(np.ceil(n*fraction)) for fraction in [.1, .2, .5])
    bands = np.full(n, 'low', dtype=object)
    bands[order[:top10]] = 'very_high'
    bands[order[top10:top20]] = 'high'
    bands[order[top20:top50]] = 'medium'
    return bands


def predict_churn(features: dict, *, model_path=None, include_diagnostic=True) -> dict:
    if not isinstance(features, Mapping):
        raise FeatureValidationError('Single input must be a mapping of feature names to scalar numerical values.')
    # Object dtype preserves scalar types so mixed booleans are never coerced to 1.
    frame = validate_features(pd.DataFrame([dict(features)], dtype=object))
    probability = float(_probabilities(load_model(model_path), frame)[0])
    output = dict(churn_probability=probability, risk_band='unranked',
                  target_definition=TARGET_DEFINITION, model_version=MODEL_VERSION)
    if include_diagnostic:
        output['diagnostic_prediction'] = int(probability >= DIAGNOSTIC_THRESHOLD)
    return output


def predict_churn_batch(data: pd.DataFrame, *, model_path=None, include_diagnostic=True) -> pd.DataFrame:
    frame = validate_features(data)
    p = _probabilities(load_model(model_path), frame)
    output = pd.DataFrame(dict(churn_probability=p, risk_band=batch_risk_bands(p),
                              target_definition=TARGET_DEFINITION, model_version=MODEL_VERSION), index=data.index.copy())
    if include_diagnostic:
        output['diagnostic_prediction'] = (p >= DIAGNOSTIC_THRESHOLD).astype(int)
    return output


def model_health_check(*, model_path=None) -> dict:
    """Load/schema checks only; no synthetic risk score is generated."""
    try:
        load_model(model_path)
        synthetic = pd.DataFrame([{feature: 0.0 for feature in FINAL_FEATURES}])
        validate_features(synthetic)
    except (ModelLoadError, FeatureValidationError) as error:
        return dict(healthy=False, model_version=MODEL_VERSION, error=str(error))
    return dict(healthy=True, model_version=MODEL_VERSION, predictor_count=len(FINAL_FEATURES),
                artifact_verified=True, synthetic_schema_checked=True, prediction_performed=False)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description='Frozen model health and public contract metadata only.')
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--health', action='store_true')
    group.add_argument('--model-info', action='store_true')
    args = parser.parse_args(argv)
    result = model_health_check() if args.health else get_model_info()
    print(json.dumps(result, indent=2))
    return 1 if args.health and not result['healthy'] else 0


if __name__ == '__main__':
    raise SystemExit(main())

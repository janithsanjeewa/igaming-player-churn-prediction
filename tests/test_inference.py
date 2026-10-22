"""Synthetic inference fixtures only; never rescore historical player records."""
from contextlib import redirect_stderr, redirect_stdout
from io import StringIO
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock, patch

import joblib
import numpy as np
import pandas as pd

from src.business_policy import risk_band_table
from src.inference import (FeatureValidationError, ModelLoadError, batch_risk_bands,
    clear_model_cache, load_model, main, model_health_check, predict_churn,
    predict_churn_batch, validate_features)
from src.model_contract import FINAL_FEATURES, MODEL_VERSION

ROOT = Path(__file__).resolve().parents[1]
ARTIFACT = ROOT / 'models/final_xgboost_churn_model.joblib'


def synthetic_record():
    """Artificial schema fixture, not a real or copied player observation."""
    values = {feature: 0.0 for feature in FINAL_FEATURES}
    values.update(active_days_30d=5, bets_30d=20, days_since_last_active_bet=3,
                  days_since_registration=200, days_since_first_pay=180,
                  days_since_first_casino_activity=175)
    return values


def probability_model(probabilities):
    model = Mock()
    p = np.asarray(probabilities, dtype=float)
    model.predict_proba.return_value = np.column_stack([1-p, p])
    return model


class FeatureValidationTests(unittest.TestCase):
    def test_order_copy_and_input_index_are_preserved_without_mutation(self):
        frame = pd.DataFrame([synthetic_record()], index=['synthetic_row'])
        reversed_frame = frame.loc[:, list(reversed(FINAL_FEATURES))]
        validated = validate_features(reversed_frame)
        self.assertEqual(validated.columns.tolist(), list(FINAL_FEATURES))
        self.assertEqual(validated.index.tolist(), ['synthetic_row'])
        validated.iloc[0, 0] = -999
        self.assertEqual(frame.iloc[0, 0], 5)

    def test_missing_duplicate_unknown_and_nonstring_columns_rejected(self):
        frame = pd.DataFrame([synthetic_record()])
        with self.assertRaises(FeatureValidationError): validate_features(frame.drop(columns=FINAL_FEATURES[0]))
        with self.assertRaises(FeatureValidationError): validate_features(pd.concat([frame, frame.iloc[:, :1]], axis=1))
        for field in ['UserID', 'snapshot_date', 'churn_60d', 'Country', 'Language', 'Gender', 'unknown']:
            with self.assertRaises(FeatureValidationError): validate_features(frame.assign(**{field: 'synthetic'}))
        bad_names = frame.copy()
        bad_names.columns = [None, *FINAL_FEATURES[1:]]
        with self.assertRaises(FeatureValidationError): validate_features(bad_names)
        with self.assertRaises(FeatureValidationError): validate_features(frame.iloc[:0])
        with self.assertRaises(FeatureValidationError): validate_features(synthetic_record())

    def test_non_numeric_strings_bools_complex_and_dates_rejected(self):
        for value in ['3.0', 'not_numeric', True, np.bool_(False), 1+2j, pd.Timestamp('2006-01-01'), [1, 2], {'value': 3}]:
            record = synthetic_record()
            record[FINAL_FEATURES[0]] = value
            with self.assertRaises(FeatureValidationError): validate_features(pd.DataFrame([record], dtype=object))

    def test_nan_none_nullable_numbers_and_signed_amounts_accepted(self):
        frame = pd.DataFrame([synthetic_record(), synthetic_record()])
        frame[FINAL_FEATURES[0]] = pd.Series([pd.NA, 3], dtype='Float64')
        frame[FINAL_FEATURES[1]] = [np.nan, None]
        frame['winnings_7d'] = [-10., 0.]
        before = frame.copy(deep=True)
        validated = validate_features(frame)
        self.assertTrue(np.isnan(validated.iloc[0, 0]))
        self.assertTrue(validated[FINAL_FEATURES[1]].isna().all())
        self.assertEqual(validated.winnings_7d.tolist(), [-10., 0.])
        pd.testing.assert_frame_equal(frame, before)

    def test_infinity_rejected_without_echoing_payload(self):
        for value in [np.inf, -np.inf]:
            frame = pd.DataFrame([synthetic_record()])
            frame[FINAL_FEATURES[0]] = value
            with self.assertRaises(FeatureValidationError): validate_features(frame)
        with self.assertRaises(FeatureValidationError) as context:
            validate_features(pd.DataFrame([synthetic_record()]).assign(private_identifier='SENSITIVE_SYNTHETIC_VALUE'))
        self.assertNotIn('SENSITIVE_SYNTHETIC_VALUE', str(context.exception))
        self.assertNotIn('private_identifier', str(context.exception))


class InferenceTests(unittest.TestCase):
    def test_single_is_unranked_and_uses_frozen_boundary(self):
        for p, diagnostic in [(.49, 0), (.50, 1), (1., 1), (0., 0)]:
            model = probability_model([p])
            with patch('src.inference.load_model', return_value=model):
                result = predict_churn(dict(reversed(list(synthetic_record().items()))))
            self.assertEqual(result['churn_probability'], p)
            self.assertEqual(result['diagnostic_prediction'], diagnostic)
            self.assertEqual(result['risk_band'], 'unranked')
            self.assertEqual(result['model_version'], MODEL_VERSION)
            self.assertEqual(model.predict_proba.call_args.args[0].columns.tolist(), list(FINAL_FEATURES))
            model.fit.assert_not_called()
            model.predict.assert_not_called()

    def test_optional_diagnostic_and_single_bad_input(self):
        with patch('src.inference.load_model', return_value=probability_model([.6])):
            self.assertNotIn('diagnostic_prediction', predict_churn(synthetic_record(), include_diagnostic=False))
        with self.assertRaises(FeatureValidationError): predict_churn([synthetic_record()])
        with patch('src.inference.load_model') as loading:
            with self.assertRaises(FeatureValidationError): predict_churn({'invalid': 1})
            loading.assert_not_called()

    def test_batch_preserves_duplicate_index_position_and_input(self):
        frame = pd.DataFrame([synthetic_record()]*4, index=['row_c', 'row_a', 'row_a', 'row_b'])
        original = frame.copy(deep=True)
        model = probability_model([.1, .9, .6, .3])
        with patch('src.inference.load_model', return_value=model):
            result = predict_churn_batch(frame)
        self.assertEqual(result.index.tolist(), frame.index.tolist())
        self.assertEqual(result.churn_probability.tolist(), [.1, .9, .6, .3])
        self.assertEqual(result.risk_band.tolist(), ['low', 'very_high', 'medium', 'low'])
        self.assertEqual(result.diagnostic_prediction.tolist(), [0, 1, 1, 0])
        self.assertEqual(set(result.columns), {'churn_probability', 'diagnostic_prediction', 'risk_band', 'target_definition', 'model_version'})
        self.assertEqual(model.predict_proba.call_args.args[0].shape, (4, 42))
        pd.testing.assert_frame_equal(frame, original)

    def test_rank_bands_match_day10_policy_and_stable_ties(self):
        probabilities = np.linspace(.95, .05, 20)
        bands = batch_risk_bands(probabilities)
        self.assertEqual(bands.tolist(), ['very_high']*2+['high']*2+['medium']*6+['low']*10)
        np.testing.assert_array_equal(batch_risk_bands(np.repeat(.4, 20)), bands)
        for n in [1, 2, 3, 7, 21, 103]:
            p = np.random.default_rng(42).uniform(size=n)
            names = {'Very high risk (top 10%)': 'very_high', 'High risk (80th-90th)': 'high',
                     'Medium risk (50th-80th)': 'medium', 'Low risk (bottom 50%)': 'low'}
            reference = risk_band_table(np.ones(n), p)
            actual = pd.Series(batch_risk_bands(p)).value_counts().to_dict()
            self.assertEqual(actual, {names[row.band]: row.observations for row in reference.itertuples()})

    def test_invalid_probability_outputs_rejected(self):
        for result in [np.array([[.3, np.inf]]), np.array([[.3, -.1]]), np.array([[.3, .8]]), np.array([.6]), np.array([[np.nan, .6]])]:
            model = Mock()
            model.predict_proba.return_value = result
            with patch('src.inference.load_model', return_value=model):
                with self.assertRaises(ModelLoadError): predict_churn(synthetic_record())
        for p in [[], [np.inf], [-.1], [[.1]]]:
            with self.assertRaises(ValueError): batch_risk_bands(p)

    def test_no_default_logging_or_prediction_request_persistence(self):
        out, err = StringIO(), StringIO()
        with patch('src.inference.load_model', return_value=probability_model([.6])), \
                patch('src.inference.joblib.dump') as saving, redirect_stdout(out), redirect_stderr(err):
            predict_churn(synthetic_record())
        self.assertEqual(out.getvalue()+err.getvalue(), '')
        saving.assert_not_called()


class ModelLoadingTests(unittest.TestCase):
    def tearDown(self):
        clear_model_cache()

    def test_missing_model_has_actionable_error_no_retraining(self):
        with tempfile.TemporaryDirectory() as directory:
            missing = Path(directory) / 'missing.joblib'
            with self.assertRaisesRegex(ModelLoadError, 'final evaluation/training workflow'):
                load_model(missing)
            health = model_health_check(model_path=missing)
            self.assertFalse(health['healthy'])

    def test_model_info_cli_does_not_require_artifact(self):
        stream = StringIO()
        with patch.dict(os.environ, {'IGAMING_CHURN_MODEL_PATH': 'missing.joblib'}), redirect_stdout(stream):
            self.assertEqual(main(['--model-info']), 0)
        self.assertEqual(json.loads(stream.getvalue())['predictor_count'], 42)
        with patch('src.inference.model_health_check', return_value={'healthy': False, 'error': 'missing'}), redirect_stdout(StringIO()):
            self.assertEqual(main(['--health']), 1)

    def test_wrong_artifact_rejected_before_unpickling(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'wrong.joblib'
            path.write_bytes(b'not the final frozen model')
            with patch('src.inference.joblib.load') as loading:
                with self.assertRaises(ModelLoadError): load_model(path)
                loading.assert_not_called()

    @unittest.skipUnless(ARTIFACT.exists(), 'Final model is a local ignored artifact.')
    def test_actual_artifact_cached_healthy_and_deterministic_on_synthetic_rows(self):
        clear_model_cache()
        actual_load = joblib.load
        with patch('src.inference.joblib.load', wraps=actual_load) as loading:
            first = load_model()
            self.assertIs(first, load_model())
            with patch.object(first, 'predict_proba', wraps=first.predict_proba) as predicting:
                health = model_health_check()
                self.assertTrue(health['healthy'])
                self.assertFalse(health['prediction_performed'])
                predicting.assert_not_called()
            record = synthetic_record()
            record['avg_bets_per_active_day_7d'] = np.nan
            first_result = predict_churn(record)
            self.assertEqual(first_result, predict_churn(record))
            batch = predict_churn_batch(pd.DataFrame([record]*20))
            self.assertTrue(batch.churn_probability.between(0, 1).all())
            self.assertAlmostEqual(first_result['churn_probability'], batch.churn_probability.iloc[0])
            self.assertEqual(loading.call_count, 1)
        with patch.dict(os.environ, {'IGAMING_CHURN_MODEL_PATH': str(ARTIFACT)}):
            self.assertIs(load_model(), first)


if __name__ == '__main__':
    unittest.main()

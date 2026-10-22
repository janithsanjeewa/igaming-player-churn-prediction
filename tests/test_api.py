"""HTTP contract tests with synthetic fixtures; no historical player requests."""
from contextlib import ExitStack, redirect_stderr, redirect_stdout
from io import StringIO
import json
import os
from pathlib import Path
import unittest
from unittest.mock import Mock, patch

from fastapi.testclient import TestClient
import numpy as np

from api.config import Settings
from api.main import create_app
from api.schemas import PredictionRequest
from src.inference import FeatureValidationError, ModelLoadError
from src.model_contract import FINAL_FEATURES, MODEL_VERSION

FIXTURE = Path(__file__).resolve().parent / 'fixtures/synthetic_prediction.json'
ARTIFACT = Path(__file__).resolve().parents[1] / 'models/final_xgboost_churn_model.joblib'


def synthetic_record(score=.50):
    """Artificial record, never a copied bwin row; fake model maps one value to p."""
    record = json.loads(FIXTURE.read_text())
    record['active_days_30d'] = score * 100
    return record


class APITests(unittest.TestCase):
    def setUp(self):
        self.stack = ExitStack()
        self.addCleanup(self.stack.close)
        self.model = Mock()
        def probabilities(frame):
            p = frame.active_days_30d.to_numpy(dtype=float)/100
            return np.column_stack([1-p, p])
        self.model.predict_proba.side_effect = probabilities
        self.loading = self.stack.enter_context(patch('src.inference.load_model', return_value=self.model))
        self.app = create_app(Settings(max_batch_size=20))
        self.client = self.stack.enter_context(TestClient(self.app, raise_server_exceptions=False))

    def test_root_docs_redoc_and_public_info(self):
        root = self.client.get('/')
        self.assertEqual(root.status_code, 200)
        self.assertEqual(root.json()['version'], MODEL_VERSION)
        self.assertEqual(root.json()['docs'], '/docs')
        self.assertEqual(self.client.get('/docs').status_code, 200)
        self.assertEqual(self.client.get('/redoc').status_code, 200)
        info = self.client.get('/model-info')
        self.assertEqual(info.status_code, 200)
        self.assertEqual(info.json()['predictor_count'], 42)
        self.assertEqual(info.json()['diagnostic_threshold'], .50)
        self.assertEqual(info.json()['calibration'], 'none')
        self.assertTrue(info.json()['demographics_excluded'])
        self.assertNotIn('model_path', info.json())
        self.assertNotIn('UserID', info.text)

    def test_health_validates_startup_without_scoring(self):
        self.assertTrue(self.app.state.startup_model_available)
        response = self.client.get('/health')
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()['status'], 'healthy')
        self.assertTrue(response.json()['model_loaded'])
        self.assertEqual(response.json()['predictor_count'], 42)
        self.model.predict_proba.assert_not_called()

    def test_single_raw_threshold_and_unranked(self):
        for p, label in [(.49, 0), (.50, 1), (.51, 1), (0., 0), (1., 1)]:
            response = self.client.post('/predict', json=synthetic_record(p))
            self.assertEqual(response.status_code, 200, response.text)
            result = response.json()
            self.assertAlmostEqual(result['churn_probability'], p)
            self.assertEqual(result['diagnostic_prediction'], label)
            self.assertEqual(result['risk_band'], 'unranked')
            self.assertEqual(set(result), {'churn_probability', 'diagnostic_prediction', 'risk_band', 'model_version', 'target_definition'})
        self.model.fit.assert_not_called()
        self.model.predict.assert_not_called()

    def test_missing_feature_invalid_scalar_and_numeric_string(self):
        record = synthetic_record()
        del record[FINAL_FEATURES[0]]
        self.assertEqual(self.client.post('/predict', json=record).status_code, 422)
        for value in ['0.5', True, [], {}, 'SENSITIVE_SYNTHETIC_PAYLOAD']:
            record = synthetic_record()
            record['winnings_7d'] = value
            response = self.client.post('/predict', json=record)
            self.assertEqual(response.status_code, 422)
            self.assertNotIn('SENSITIVE_SYNTHETIC_PAYLOAD', response.text)

    def test_demographics_metadata_target_and_unknown_fields_rejected(self):
        for field in ['Country', 'Language', 'Gender', 'UserID', 'snapshot_date', 'churn_60d', 'unknown_sensitive_key']:
            record = synthetic_record()
            record[field] = 'SENSITIVE_SYNTHETIC_PAYLOAD'
            response = self.client.post('/predict', json=record)
            self.assertEqual(response.status_code, 422)
            self.assertNotIn('SENSITIVE_SYNTHETIC_PAYLOAD', response.text)
            self.assertNotIn('unknown_sensitive_key', response.text)

    def test_json_null_is_preserved_for_fitted_imputer(self):
        record = synthetic_record()
        record['winnings_7d'] = None
        response = self.client.post('/predict', json=record)
        self.assertEqual(response.status_code, 200)
        self.assertTrue(np.isnan(self.model.predict_proba.call_args.args[0].winnings_7d.iloc[0]))

    def test_nonstandard_json_nan_infinity_and_overflow_rejected_safely(self):
        for value in [np.inf, -np.inf, np.nan]:
            record = synthetic_record()
            record['winnings_7d'] = value
            response = self.client.post('/predict', content=json.dumps(record), headers={'Content-Type': 'application/json'})
            self.assertEqual(response.status_code, 422)
            self.assertEqual(set(response.json()), {'detail'})
        payload = json.dumps(synthetic_record()).replace('"winnings_7d": 0.0', '"winnings_7d": 1e999')
        self.assertEqual(self.client.post('/predict', content=payload, headers={'Content-Type': 'application/json'}).status_code, 422)
        self.assertEqual(self.client.post('/predict', content='{invalid', headers={'Content-Type': 'application/json'}).status_code, 422)

    def test_batch_order_count_probability_and_rank_bands(self):
        scores = np.random.default_rng(42).permutation(np.linspace(.05, .95, 20))
        response = self.client.post('/predict-batch', json={'records': [synthetic_record(p) for p in scores]})
        self.assertEqual(response.status_code, 200, response.text)
        payload = response.json()
        self.assertEqual(payload['count'], 20)
        results = payload['predictions']
        np.testing.assert_allclose([row['churn_probability'] for row in results], scores)
        expected = np.empty(20, dtype=object)
        rank = np.argsort(-scores, kind='stable')
        expected[rank[:2]], expected[rank[2:4]], expected[rank[4:10]], expected[rank[10:]] = 'very_high', 'high', 'medium', 'low'
        self.assertEqual([row['risk_band'] for row in results], expected.tolist())
        self.assertTrue(all(0 <= row['churn_probability'] <= 1 for row in results))
        self.assertTrue(all('index' not in row and 'UserID' not in row for row in results))

    def test_batch_empty_oversized_and_bad_record(self):
        for rows in [[], [synthetic_record()]*21]:
            response = self.client.post('/predict-batch', json={'records': rows})
            self.assertEqual(response.status_code, 422)
            self.assertIn('between 1 and 20', response.json()['detail'])
        bad = synthetic_record()
        del bad['bets_30d']
        self.assertEqual(self.client.post('/predict-batch', json={'records': [synthetic_record(), bad]}).status_code, 422)
        self.assertEqual(self.client.post('/predict-batch', json={'records': [synthetic_record()], 'UserID': 0}).status_code, 422)

    def test_validation_errors_do_not_call_model(self):
        self.model.predict_proba.reset_mock()
        response = self.client.post('/predict', json={'not_a_feature': 'SENSITIVE_SYNTHETIC_PAYLOAD'})
        self.assertEqual(response.status_code, 422)
        self.model.predict_proba.assert_not_called()

    def test_inference_validation_and_unavailable_errors_are_sanitized(self):
        private_message = 'SENSITIVE_SYNTHETIC_PAYLOAD at C:\\private\\model.joblib'
        for error, code in [(FeatureValidationError(private_message), 422), (ModelLoadError(private_message), 503), (RuntimeError(private_message), 500)]:
            with patch('api.main.inference.predict_churn', side_effect=error):
                response = self.client.post('/predict', json=synthetic_record())
            self.assertEqual(response.status_code, code)
            self.assertNotIn('SENSITIVE_SYNTHETIC_PAYLOAD', response.text)
            self.assertNotIn('private', response.text)
            self.assertNotIn('Traceback', response.text)
        with patch('api.main.inference.predict_churn_batch', side_effect=ModelLoadError(private_message)):
            self.assertEqual(self.client.post('/predict-batch', json={'records': [synthetic_record()]}).status_code, 503)

    def test_health_unavailable_hides_internal_errors(self):
        with patch('api.main.inference.model_health_check', return_value={'healthy': False, 'error': 'SENSITIVE_INTERNAL_PATH'}):
            response = self.client.get('/health')
        self.assertEqual(response.status_code, 503)
        self.assertFalse(response.json()['model_loaded'])
        self.assertEqual(response.json()['status'], 'unavailable')
        self.assertNotIn('SENSITIVE_INTERNAL_PATH', response.text)

    def test_cors_explicit_local_origins_and_no_wildcard(self):
        allowed = self.client.options('/predict', headers={'Origin': 'http://localhost:5173', 'Access-Control-Request-Method': 'POST', 'Access-Control-Request-Headers': 'Content-Type'})
        self.assertEqual(allowed.status_code, 200)
        self.assertEqual(allowed.headers['access-control-allow-origin'], 'http://localhost:5173')
        self.assertNotIn('access-control-allow-credentials', allowed.headers)
        denied = self.client.options('/predict', headers={'Origin': 'https://unlisted.invalid', 'Access-Control-Request-Method': 'POST'})
        self.assertEqual(denied.status_code, 400)
        self.assertNotIn('access-control-allow-origin', denied.headers)

    def test_openapi_exact_feature_and_response_contract(self):
        spec = self.client.get('/openapi.json').json()
        self.assertEqual(spec['info']['version'], MODEL_VERSION)
        schemas = spec['components']['schemas']
        request = schemas['PredictionRequest']
        self.assertEqual(list(request['properties']), list(FINAL_FEATURES))
        self.assertEqual(set(request['required']), set(FINAL_FEATURES))
        self.assertFalse(request['additionalProperties'])
        for name in ['Country', 'Language', 'Gender', 'UserID', 'snapshot_date', 'churn_60d']:
            self.assertNotIn(name, request['properties'])
        array = schemas['BatchPredictionRequest']['properties']['records']
        self.assertEqual(array['maxItems'], 20)
        self.assertEqual(array['minItems'], 1)
        self.assertEqual(array['items']['$ref'], '#/components/schemas/PredictionRequest')
        self.assertEqual(schemas['SinglePredictionResponse']['properties']['risk_band']['const'], 'unranked')
        for path in ['/predict', '/predict-batch']:
            for status in ['200', '422', '503', '500']:
                self.assertIn(status, spec['paths'][path]['post']['responses'])
        self.assertEqual(set(PredictionRequest.model_fields), set(FINAL_FEATURES))

    def test_requests_and_predictions_not_logged_or_persisted(self):
        out, err = StringIO(), StringIO()
        with redirect_stdout(out), redirect_stderr(err), patch('src.inference.joblib.dump') as saving:
            response = self.client.post('/predict', json=synthetic_record())
        self.assertEqual(response.status_code, 200)
        self.assertEqual(out.getvalue()+err.getvalue(), '')
        saving.assert_not_called()


class APIConfigurationTests(unittest.TestCase):
    def test_environment_config_and_local_defaults(self):
        with patch.dict(os.environ, {}, clear=True):
            self.assertEqual(Settings.from_environment(), Settings())
        with patch.dict(os.environ, {'MODEL_PATH': 'models/relocated.joblib', 'ALLOWED_ORIGINS': ' https://frontend.example, http://localhost:3000,https://frontend.example ', 'MAX_BATCH_SIZE': '7'}):
            settings = Settings.from_environment()
        self.assertEqual(settings.model_path, 'models/relocated.joblib')
        self.assertEqual(settings.allowed_origins, ('https://frontend.example', 'http://localhost:3000'))
        self.assertEqual(settings.max_batch_size, 7)
        self.assertEqual(create_app(settings).openapi()['components']['schemas']['BatchPredictionRequest']['properties']['records']['maxItems'], 7)

    def test_invalid_config_and_empty_origin_list(self):
        for value in ['0', '-1', 'not-an-integer']:
            with patch.dict(os.environ, {'MAX_BATCH_SIZE': value}):
                with self.assertRaises(ValueError): Settings.from_environment()
        for origin in ['*', 'https://frontend.example/path', 'https://user:secret@frontend.example']:
            with self.assertRaises(ValueError): Settings(allowed_origins=(origin,))
        with patch.dict(os.environ, {'ALLOWED_ORIGINS': ''}):
            self.assertEqual(Settings.from_environment().allowed_origins, ())

    def test_startup_missing_artifact_stays_available_for_docs_and_info(self):
        with patch('api.main.inference.model_health_check', return_value={'healthy': False, 'error': 'private'}), \
                patch('api.main.inference.predict_churn', side_effect=ModelLoadError('private')):
            with TestClient(create_app(Settings(model_path='missing.joblib'))) as client:
                self.assertEqual(client.get('/docs').status_code, 200)
                self.assertEqual(client.get('/model-info').status_code, 200)
                self.assertEqual(client.get('/health').status_code, 503)
                self.assertEqual(client.post('/predict', json=synthetic_record()).status_code, 503)

    @unittest.skipUnless(ARTIFACT.exists(), 'Final model is a local ignored artifact.')
    def test_actual_local_model_health_and_synthetic_prediction(self):
        with TestClient(create_app(Settings())) as client:
            self.assertEqual(client.get('/health').status_code, 200)
            record = json.loads(FIXTURE.read_text())
            response = client.post('/predict', json=record)
            self.assertEqual(response.status_code, 200, response.text)
            self.assertTrue(0 <= response.json()['churn_probability'] <= 1)
            self.assertEqual(response.json()['risk_band'], 'unranked')


if __name__ == '__main__':
    unittest.main()

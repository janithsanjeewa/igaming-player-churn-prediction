"""Synthetic preflight checks and replay-only verification of final aggregates."""
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock, patch

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import brier_score_loss, roc_auc_score

from src.business_policy import _draw_clusters, frozen_pipeline
from src.business_policy import verify_specification
from src.evaluation import classification_metrics, cumulative_gains_table, ranking_metrics
from src.features import CATEGORICAL_COLUMNS, METADATA_COLUMNS, PREDICTOR_COLUMNS, TARGET_COLUMN
from src.final_test import (REFIT_DATES, TEST_DATES, claim_evaluation, final_cluster_bootstrap,
    fit_final_model, predict_final_test_once, require_final_dates, sha256,
    single_class_safe_metrics, validate_frozen_policy, verify_cached_run)
from src.preprocessing import build_xgboost_pipeline

ROOT = Path(__file__).resolve().parents[1]


class FinalTestTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.spec = json.loads((ROOT / 'reports/final_model_specification.json').read_text())
        rng = np.random.default_rng(42)
        def frame(dates):
            result = pd.DataFrame(rng.normal(size=(len(dates), len(PREDICTOR_COLUMNS))), columns=PREDICTOR_COLUMNS)
            for category in CATEGORICAL_COLUMNS:
                result[category] = 999
            result['snapshot_date'] = dates
            result['UserID'] = np.arange(len(dates))
            result[TARGET_COLUMN] = np.arange(len(dates)) % 2
            result.loc[0, 'days_since_first_casino_activity'] = np.nan
            return result
        cls.refit, cls.test = frame(REFIT_DATES), frame(TEST_DATES)
        # Tiny synthetic model only: no real panel, saved final model or test predictions.
        cls.small = frozen_pipeline(build_xgboost_pipeline(n_estimators=8, max_depth=2), False)
        cls.small.fit(cls.refit[cls.spec['predictors']], cls.refit[TARGET_COLUMN])

    def test_exact_refit_and_test_months_and_outcome_boundary(self):
        require_final_dates(self.refit, 'refit')
        require_final_dates(self.test, 'test')
        self.assertEqual(set(self.refit.snapshot_date.dt.month[self.refit.snapshot_date.dt.year.eq(2006)]), {1, 2, 3, 4, 5})
        self.assertEqual(self.refit.snapshot_date.max()+pd.Timedelta(days=60), pd.Timestamp('2006-07-30'))
        self.assertLess(self.refit.snapshot_date.max()+pd.Timedelta(days=60), TEST_DATES.min()-pd.Timedelta(days=29))
        for date in ['2006-06-30', '2006-07-31', '2006-08-31']:
            changed = self.refit.copy()
            changed.loc[0, 'snapshot_date'] = pd.Timestamp(date)
            with self.assertRaises(ValueError): require_final_dates(changed, 'refit')
        for invalid in [self.test.iloc[:-1], self.test.assign(snapshot_date=pd.Timestamp('2006-05-31'))]:
            with self.assertRaises(ValueError): require_final_dates(invalid, 'test')

    def test_raw_threshold_feature_and_parameter_contracts_reject_drift(self):
        validate_frozen_policy(self.spec)
        for key, value in [('diagnostic_threshold', .4), ('calibration_choice', 'sigmoid'),
                           ('demographics_retained', True), ('outcome_days', 30)]:
            with self.assertRaises(ValueError): validate_frozen_policy({**self.spec, key: value})
        for column in CATEGORICAL_COLUMNS + METADATA_COLUMNS + [TARGET_COLUMN]:
            self.assertNotIn(column, self.spec['predictors'])
        changed = {**self.spec, 'estimator_parameters': {**self.spec['estimator_parameters'], 'max_depth': 99}}
        with self.assertRaises(ValueError): validate_frozen_policy(changed)

    def test_bad_frames_rejected_before_fit_or_prediction(self):
        spy = Mock()
        with self.assertRaises(ValueError): fit_final_model(spy, self.test, self.spec)
        with self.assertRaises(ValueError): predict_final_test_once(spy, self.refit, self.spec)
        self.assertEqual(spy.mock_calls, [])

    def test_one_raw_probability_call_and_no_preprocessing_fit_on_scoring(self):
        # Patch specification verification solely for this small synthetic estimator;
        # production always compares every parameter to the frozen 400-tree settings.
        spy = Mock(wraps=self.small)
        spy.__getitem__ = Mock(side_effect=self.small.__getitem__)
        before = self.small['preprocessing'].named_transformers_['numeric']['imputer'].statistics_.copy()
        with patch('src.final_test.verify_specification'):
            p = predict_final_test_once(spy, self.test, self.spec)
        self.assertEqual(spy.predict_proba.call_count, 1)
        spy.predict.assert_not_called()
        spy.fit.assert_not_called()
        passed = spy.predict_proba.call_args.args[0]
        self.assertEqual(passed.columns.tolist(), self.spec['predictors'])
        self.assertTrue(np.isfinite(self.small['preprocessing'].transform(passed)).all())
        self.assertTrue(np.isfinite(p).all() and ((p >= 0) & (p <= 1)).all())
        np.testing.assert_array_equal(before, self.small['preprocessing'].named_transformers_['numeric']['imputer'].statistics_)

    def test_atomic_claim_and_cached_run_cannot_silently_rescore(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'receipt.json'
            claim_evaluation(path, {'data_sha256': 'frozen'})
            with self.assertRaises(FileExistsError): claim_evaluation(path, {'data_sha256': 'frozen'})
            model_path = Path(directory) / 'model.bin'
            model_path.write_bytes(b'fixed model')
            raw = dict(fingerprint={'a': 'b'}, fit_calls=1, test_probability_calls=1,
                       model_sha256=sha256(model_path), threshold=.5, calibration='raw')
            verify_cached_run(raw, {'a': 'b'}, model_path)
            for change in [dict(test_probability_calls=2), dict(threshold=.4), dict(calibration='isotonic')]:
                with self.assertRaises(ValueError): verify_cached_run({**raw, **change}, {'a': 'b'}, model_path)

    def test_cluster_bootstrap_uses_whole_players_and_reports_valid_counts(self):
        ids = [1, 1, 2, 2, 2, 2]
        y = np.array([0, 1, 0, 1, 0, 1])
        p = np.array([.1, .8, .2, .7, .3, .6])
        with patch('src.final_test._draw_clusters', wraps=_draw_clusters) as sampler:
            result = final_cluster_bootstrap(ids, y, p, replicates=15)
        self.assertEqual(sampler.call_count, 15)
        self.assertEqual([c.tolist() for c in sampler.call_args.args[0]], [[0, 1], [2, 3, 4, 5]])
        self.assertTrue(result.valid_replicates.eq(15).all())
        pd.testing.assert_frame_equal(result, final_cluster_bootstrap(ids, y, p, replicates=15))
        single = final_cluster_bootstrap([1, 1], [0, 0], [.1, .2], replicates=4).set_index('metric')
        self.assertEqual(single.loc['roc_auc', 'valid_replicates'], 0)
        self.assertEqual(single.loc['brier_score', 'valid_replicates'], 4)
        self.assertEqual(single.loc['lift_at_top10pct', 'valid_replicates'], 0)

    def test_safe_metrics_match_existing_definitions(self):
        y, p = [0, 1, 0, 1], [.1, .8, .7, .3]
        result = single_class_safe_metrics(y, p)
        expected = {**classification_metrics(y, p), **ranking_metrics(y, p)}
        for metric, value in result.items():
            self.assertAlmostEqual(value, expected[metric])
        self.assertTrue(np.isnan(single_class_safe_metrics([0, 0], [.1, .2])['roc_auc']))

    def test_actual_exports_are_aggregate_and_reproduce_saved_probability_vector(self):
        cache = ROOT / 'data/processed/final_test_evaluation.joblib'
        if not cache.exists():
            self.skipTest('One-time evaluation has not run yet; no model is scored by tests.')
        raw = joblib.load(cache)
        metrics = pd.read_csv(ROOT / 'reports/final_test_metrics.csv').iloc[0]
        expected = {**classification_metrics(raw['y'], raw['probabilities']), **ranking_metrics(raw['y'], raw['probabilities'])}
        for metric, value in expected.items():
            self.assertAlmostEqual(metrics[metric], value)
        self.assertAlmostEqual(metrics.roc_auc, roc_auc_score(raw['y'], raw['probabilities']))
        self.assertAlmostEqual(metrics.brier_score, brier_score_loss(raw['y'], raw['probabilities']))
        expected_targeting = cumulative_gains_table(raw['y'], raw['probabilities'], [.05, .1, .15, .2, .25, .3])
        pd.testing.assert_frame_equal(pd.read_csv(ROOT / 'reports/final_test_targeting.csv'), expected_targeting,
                                      check_dtype=False, atol=1e-12)
        for path in (ROOT / 'reports').glob('final_test_*.csv'):
            table = pd.read_csv(path)
            self.assertNotIn('UserID', table.columns)
            self.assertFalse({'probability', 'prediction', 'actual', 'churn_60d'} & set(table.columns))
            self.assertLess(len(table), 1000)

    def test_saved_final_model_matches_spec_and_safe_refit_medians_without_rescoring(self):
        cache = ROOT / 'data/processed/final_test_evaluation.joblib'
        if not cache.exists():
            self.skipTest('One-time evaluation has not run yet.')
        raw = joblib.load(cache)
        pipeline = joblib.load(ROOT / 'models/final_xgboost_churn_model.joblib')
        fitted_spec = {**self.spec, 'transformed_features': pipeline['preprocessing'].get_feature_names_out().tolist()}
        verify_specification(fitted_spec, pipeline)
        self.assertEqual(pipeline['preprocessing'].feature_names_in_.tolist(), self.spec['predictors'])
        self.assertEqual(raw['threshold'], .50)
        self.assertEqual(raw['calibration'], 'raw')
        self.assertEqual(raw['fit_calls'], 1)
        self.assertEqual(raw['test_probability_calls'], 1)
        panel = pd.read_csv(ROOT / 'data/processed/model_panel_60d.csv', parse_dates=['snapshot_date'])
        refit = panel.loc[panel.snapshot_date.isin(REFIT_DATES)]
        np.testing.assert_allclose(pipeline['preprocessing'].named_transformers_['numeric']['imputer'].statistics_,
                                   refit[self.spec['predictors']].median().to_numpy())
        self.assertEqual(sha256(ROOT / 'models/final_xgboost_churn_model.joblib'), raw['model_sha256'])


if __name__ == '__main__':
    unittest.main()

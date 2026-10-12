"""XGBoost leakage guards, dense-zero semantics, metrics and candidate audit."""
import unittest
from unittest.mock import Mock

import numpy as np
import pandas as pd

from src.boosting import XGB_CANDIDATES, select_xgboost_candidate, validate_candidate_results
from src.evaluation import classification_metrics, ranking_metrics
from src.features import CATEGORICAL_COLUMNS, PREDICTOR_COLUMNS
from src.modelling import evaluate_candidate, guarded_probabilities, modelling_partition
from src.preprocessing import NUMERIC_COLUMNS, build_xgboost_pipeline, temporal_split_labels


class XGBoostTests(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(42)
        self.frame = pd.DataFrame(rng.normal(size=(60, len(PREDICTOR_COLUMNS))), columns=PREDICTOR_COLUMNS)
        for column in CATEGORICAL_COLUMNS:
            self.frame[column] = np.tile([1., 2., 3.], 20)
        self.frame.loc[:4, NUMERIC_COLUMNS[0]] = np.nan
        self.frame.loc[5, NUMERIC_COLUMNS[0]] = 0
        self.frame.loc[0, 'Country'] = np.nan
        self.frame = self.frame.assign(UserID=np.arange(60), snapshot_date=pd.Timestamp('2005-03-31'),
                                       churn_60d=np.tile([0, 1], 30))
        self.validation = self.frame.iloc[:12].assign(snapshot_date=pd.Timestamp('2006-03-31'))

    def small_pipeline(self):
        return build_xgboost_pipeline(n_estimators=15, learning_rate=.05, max_depth=3, min_child_weight=1)

    def test_metadata_excluded_train_only_preprocessing_dense_zero_and_shape(self):
        pipeline = self.small_pipeline().fit(self.frame, self.frame.churn_60d)
        pre = pipeline['preprocessing']
        numeric = pre.named_transformers_['numeric']
        self.assertNotIn('scaler', numeric.named_steps)
        np.testing.assert_allclose(numeric['imputer'].statistics_, self.frame[NUMERIC_COLUMNS].median())
        self.assertFalse(any(c in name for name in pre.get_feature_names_out()
                             for c in ['UserID', 'snapshot_date', 'churn_60d']))
        changed = self.frame.assign(UserID=-1, snapshot_date=pd.Timestamp('2099-01-01'), churn_60d=1)
        np.testing.assert_array_equal(pipeline.predict_proba(self.frame), pipeline.predict_proba(changed))
        heldout = self.validation.copy()
        heldout[NUMERIC_COLUMNS] = 1e8
        heldout.loc[0, NUMERIC_COLUMNS[0]] = np.nan
        heldout.loc[1, NUMERIC_COLUMNS[0]] = 0
        heldout['Country'] = 999
        before = numeric['imputer'].statistics_.copy()
        matrix = pre.transform(heldout)
        self.assertIsInstance(matrix, np.ndarray)
        self.assertEqual(matrix.shape[0], len(heldout))
        self.assertTrue(np.isfinite(matrix).all())
        self.assertEqual(matrix[0, 0], before[0])
        self.assertEqual(matrix[1, 0], 0)
        np.testing.assert_array_equal(before, numeric['imputer'].statistics_)
        self.assertNotIn(999, pre.named_transformers_['categorical']['encoder'].categories_[0])

    def test_seeded_predictions_range_and_default_settings(self):
        first = self.small_pipeline().fit(self.frame, self.frame.churn_60d)
        second = self.small_pipeline().fit(self.frame, self.frame.churn_60d)
        p = guarded_probabilities(first, self.validation, 'validation')
        np.testing.assert_allclose(p, guarded_probabilities(second, self.validation, 'validation'), rtol=0, atol=1e-7)
        self.assertTrue(np.isfinite(p).all() and ((p >= 0) & (p <= 1)).all())
        params = first['model'].get_params()
        for key, value in dict(objective='binary:logistic', eval_metric='logloss', tree_method='hist',
                               n_jobs=-1, random_state=42, importance_type='gain').items():
            self.assertEqual(params[key], value)
        self.assertIsNone(params['scale_pos_weight'])
        self.assertIsNone(params['early_stopping_rounds'])

    def test_metric_consistency_and_fit_prediction_rows(self):
        test = self.frame.iloc[:6].assign(snapshot_date=pd.Timestamp('2006-08-31'))
        panel = pd.concat([self.frame, self.validation, test], ignore_index=True)
        train, validation = modelling_partition(panel, 'train'), modelling_partition(panel, 'validation')
        spy = Mock(wraps=self.small_pipeline())
        metrics = evaluate_candidate(spy, train, validation)
        self.assertEqual(spy.fit.call_args.args[0].index.tolist(), train.index.tolist())
        self.assertEqual(spy.predict_proba.call_count, 2)
        for call in spy.predict_proba.call_args_list:
            self.assertTrue(temporal_split_labels(panel.loc[call.args[0].index, 'snapshot_date']).isin(['train', 'validation']).all())
            self.assertFalse({'UserID', 'snapshot_date', 'churn_60d'}.intersection(call.args[0].columns))
        p = guarded_probabilities(spy, validation, 'validation')
        expected = {**classification_metrics(validation.churn_60d, p), **ranking_metrics(validation.churn_60d, p)}
        for key, value in expected.items():
            self.assertAlmostEqual(metrics[key], value)

    def test_sealed_period_rejected_before_predict_or_fit(self):
        spy = Mock()
        for date, split in [('2006-08-31', 'test'), ('2006-08-31', 'validation'),
                            ('2006-06-30', 'validation'), ('2006-02-28', 'train')]:
            frame = self.frame.assign(snapshot_date=pd.Timestamp(date))
            with self.assertRaises(ValueError): guarded_probabilities(spy, frame, split)
        mixed = self.validation.copy()
        mixed.loc[0, 'snapshot_date'] = pd.Timestamp('2006-08-31')
        with self.assertRaises(ValueError): evaluate_candidate(spy, self.frame, mixed)
        with self.assertRaises(ValueError): evaluate_candidate(spy, mixed, self.validation)
        with self.assertRaises(ValueError): modelling_partition(self.frame, 'test')
        spy.fit.assert_not_called()
        spy.predict.assert_not_called()
        spy.predict_proba.assert_not_called()
        dates = pd.Series(['2005-03-30', '2005-03-31', '2005-12-31', '2006-01-31', '2006-02-28',
                           '2006-03-30', '2006-03-31', '2006-05-31', '2006-06-30', '2006-07-31',
                           '2006-08-30', '2006-08-31', '2006-11-30', '2006-12-01'])
        self.assertEqual(temporal_split_labels(dates).tolist(), ['excluded', 'train', 'train',
            'embargo', 'embargo', 'excluded', 'validation', 'validation', 'embargo', 'embargo',
            'excluded', 'test', 'test', 'excluded'])

    def test_candidates_schema_and_regularized_tie_selection(self):
        metrics = dict(train_roc_auc=.8, roc_auc=.75, train_average_precision=.7, average_precision=.56,
                       brier_score=.19, lift_at_top10pct=1.7, recall_at_top10pct=.17,
                       roc_auc_gap=.05, average_precision_gap=.14)
        rows = pd.DataFrame([{**config, **metrics} for config in XGB_CANDIDATES])
        validate_candidate_results(rows)
        self.assertEqual(len(rows), 10)
        for invalid in [rows.drop(columns='roc_auc'), pd.concat([rows, rows.iloc[:1]]),
                        rows.assign(brier_score=np.inf), rows.iloc[:0]]:
            with self.assertRaises(ValueError): validate_candidate_results(invalid)
        two = rows.iloc[[0, 7]].copy()
        self.assertEqual(select_xgboost_candidate(two), 'initial_conservative')
        two.loc[two.name.eq('depth5_flexible'), 'average_precision'] = .57
        self.assertEqual(select_xgboost_candidate(two), 'depth5_flexible')
        two.loc[two.name.eq('depth5_flexible'), 'average_precision'] = .561
        two.loc[two.name.eq('depth5_flexible'), 'average_precision_gap'] = .3
        self.assertEqual(select_xgboost_candidate(two), 'initial_conservative')


if __name__ == '__main__':
    unittest.main()

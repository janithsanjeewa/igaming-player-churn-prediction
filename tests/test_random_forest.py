"""RF train-only preprocessing, reproducibility and sealed-period safeguards."""
import unittest
from unittest.mock import Mock

import numpy as np
import pandas as pd

from src.features import CATEGORICAL_COLUMNS, PREDICTOR_COLUMNS
from src.preprocessing import NUMERIC_COLUMNS, build_random_forest_pipeline, temporal_split_labels
from src.random_forest import (RF_CANDIDATES, evaluate_candidate, guarded_probabilities,
                               modelling_partition, select_candidate)


class RandomForestTests(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(42)
        self.frame = pd.DataFrame(rng.normal(size=(60, len(PREDICTOR_COLUMNS))), columns=PREDICTOR_COLUMNS)
        for column in CATEGORICAL_COLUMNS:
            self.frame[column] = np.tile([1., 2., 3.], 20)
        self.frame.loc[:5, NUMERIC_COLUMNS[0]] = np.nan
        self.frame.loc[0, 'Country'] = np.nan
        self.frame = self.frame.assign(UserID=np.arange(60), snapshot_date=pd.Timestamp('2005-03-31'),
                                       churn_60d=np.tile([0, 1], 30))

    def small_pipeline(self):
        return build_random_forest_pipeline(n_estimators=20, max_depth=4, min_samples_leaf=2)

    def test_metadata_excluded_train_only_imputation_no_scaling(self):
        pipeline = self.small_pipeline().fit(self.frame, self.frame.churn_60d)
        pre = pipeline['preprocessing']
        numeric = pre.named_transformers_['numeric']
        self.assertNotIn('scaler', numeric.named_steps)
        np.testing.assert_allclose(numeric['imputer'].statistics_, self.frame[NUMERIC_COLUMNS].median())
        self.assertFalse(any(c in name for name in pre.get_feature_names_out()
                             for c in ['UserID', 'snapshot_date', 'churn_60d']))
        changed = self.frame.assign(UserID=-1, snapshot_date=pd.Timestamp('2099-01-01'), churn_60d=1)
        np.testing.assert_allclose(pipeline.predict_proba(self.frame), pipeline.predict_proba(changed), rtol=0, atol=1e-14)
        heldout = self.frame.iloc[:7].copy()
        heldout[NUMERIC_COLUMNS] = 1e8
        heldout.loc[heldout.index[0], NUMERIC_COLUMNS[0]] = np.nan
        heldout['Country'] = 999
        before = numeric['imputer'].statistics_.copy()
        transformed = pre.transform(heldout)
        values = transformed.toarray() if hasattr(transformed, 'toarray') else transformed
        self.assertEqual(values.shape[0], 7)
        self.assertTrue(np.isfinite(values).all())
        self.assertEqual(values[0, 0], before[0])
        np.testing.assert_array_equal(before, numeric['imputer'].statistics_)
        self.assertNotIn(999, pre.named_transformers_['categorical']['encoder'].categories_[0])

    def test_reproducible_valid_probabilities(self):
        first = self.small_pipeline().fit(self.frame, self.frame.churn_60d)
        second = self.small_pipeline().fit(self.frame, self.frame.churn_60d)
        p = guarded_probabilities(first, self.frame, 'train')
        # Parallel summation order can differ at machine precision; seeded trees match.
        np.testing.assert_allclose(p, guarded_probabilities(second, self.frame, 'train'), rtol=0, atol=1e-14)
        for tree_a, tree_b in zip(first['model'].estimators_, second['model'].estimators_):
            np.testing.assert_array_equal(tree_a.tree_.feature, tree_b.tree_.feature)
            np.testing.assert_array_equal(tree_a.tree_.threshold, tree_b.tree_.threshold)
        self.assertTrue(((p >= 0) & (p <= 1)).all())
        self.assertEqual(first['model'].random_state, 42)
        default = build_random_forest_pipeline()['model']
        self.assertEqual(default.n_estimators, 500)
        self.assertEqual(default.n_jobs, -1)
        self.assertIsNone(default.class_weight)

    def test_test_embargo_mixed_dates_rejected_before_prediction(self):
        spy = Mock()
        for date, split in [('2006-08-31', 'test'), ('2006-08-31', 'validation'),
                            ('2006-06-30', 'validation'), ('2005-03-31', 'validation')]:
            heldout = self.frame.assign(snapshot_date=pd.Timestamp(date))
            with self.assertRaises(ValueError):
                guarded_probabilities(spy, heldout, split)
        mixed = self.frame.copy()
        mixed.loc[0, 'snapshot_date'] = pd.Timestamp('2006-08-31')
        with self.assertRaises(ValueError):
            guarded_probabilities(spy, mixed, 'train')
        with self.assertRaises(ValueError):
            evaluate_candidate(spy, self.frame, mixed)
        with self.assertRaises(ValueError):
            evaluate_candidate(spy, mixed, self.frame)
        spy.fit.assert_not_called()
        spy.predict.assert_not_called()
        spy.predict_proba.assert_not_called()

    def test_orchestration_fits_and_predicts_only_authorized_rows(self):
        validation = self.frame.iloc[:10].assign(snapshot_date=pd.Timestamp('2006-03-31'))
        test = self.frame.iloc[:5].assign(snapshot_date=pd.Timestamp('2006-08-31'))
        panel = pd.concat([self.frame, validation, test], ignore_index=True)
        spy = Mock(wraps=self.small_pipeline())
        training = modelling_partition(panel, 'train')
        heldout = modelling_partition(panel, 'validation')
        evaluate_candidate(spy, training, heldout)
        self.assertEqual(spy.fit.call_args.args[0].index.tolist(), training.index.tolist())
        for call in spy.predict_proba.call_args_list:
            indices = call.args[0].index
            self.assertTrue(temporal_split_labels(panel.loc[indices, 'snapshot_date']).isin(['train', 'validation']).all())
        self.assertEqual(spy.predict_proba.call_count, 2)
        with self.assertRaises(ValueError):
            modelling_partition(panel, 'test')

    def test_selection_prioritizes_ranking_and_simplifies_ties(self):
        base = dict(class_weight='unweighted', average_precision=.55, roc_auc=.74,
                    lift_at_top10pct=1.7, recall_at_top10pct=.17,
                    average_precision_gap=.10, brier_score=.19, min_samples_leaf=10)
        results = pd.DataFrame([
            dict(base, name='deep', max_depth=16),
            dict(base, name='simple', max_depth=8, average_precision=.549),
            dict(base, name='poor_ranking', max_depth=4, average_precision=.50),
            dict(base, name='weighted', max_depth=4, average_precision=.99, class_weight='balanced'),
        ])
        self.assertEqual(select_candidate(results), 'simple')
        results.loc[0, 'average_precision'] = .56
        self.assertEqual(select_candidate(results), 'deep')
        self.assertEqual(len(RF_CANDIDATES), 7)


if __name__ == '__main__':
    unittest.main()

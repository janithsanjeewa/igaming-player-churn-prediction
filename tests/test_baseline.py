"""Temporal isolation, preprocessing, probabilities and hand-computed lift checks."""
import unittest

import numpy as np
import pandas as pd

from src.evaluation import classification_metrics, cumulative_gains_table, threshold_analysis
from src.features import CATEGORICAL_COLUMNS, PREDICTOR_COLUMNS
from src.preprocessing import (NUMERIC_COLUMNS, build_logistic_pipeline,
                               predictor_frame, temporal_split_labels)


class BaselineTests(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(42)
        self.train = pd.DataFrame(rng.normal(size=(40, len(PREDICTOR_COLUMNS))),
                                  columns=PREDICTOR_COLUMNS)
        for c in CATEGORICAL_COLUMNS:
            self.train[c] = np.tile([1., 2.], 20)
        self.train.loc[:4, NUMERIC_COLUMNS[0]] = np.nan
        self.train.loc[0, 'Country'] = np.nan
        self.y = np.tile([0, 1], 20)

    def test_split_boundaries_and_embargo(self):
        dates = pd.Series(['2005-03-30', '2005-03-31', '2005-12-31', '2006-01-01',
            '2006-02-28', '2006-03-30', '2006-03-31', '2006-05-31', '2006-06-01',
            '2006-07-31', '2006-08-30', '2006-08-31', '2006-11-30', '2006-12-01'])
        self.assertEqual(temporal_split_labels(dates).tolist(), [
            'excluded', 'train', 'train', 'embargo', 'embargo', 'excluded',
            'validation', 'validation', 'embargo', 'embargo', 'excluded',
            'test', 'test', 'excluded'])
        # Later historical windows start after earlier 60-day label windows end.
        self.assertGreater(pd.Timestamp('2006-03-31') - pd.Timedelta(days=29),
                           pd.Timestamp('2005-12-31') + pd.Timedelta(days=60))
        self.assertGreater(pd.Timestamp('2006-08-31') - pd.Timedelta(days=29),
                           pd.Timestamp('2006-05-31') + pd.Timedelta(days=60))

    def test_metadata_and_target_cannot_enter_transform(self):
        frame = self.train.assign(UserID=np.arange(40), snapshot_date='2005-03-31', churn_60d=self.y)
        x = predictor_frame(frame)
        self.assertEqual(list(x.columns), PREDICTOR_COLUMNS)
        pipe = build_logistic_pipeline().fit(frame, self.y)
        names = pipe['preprocessing'].get_feature_names_out()
        self.assertFalse(any(any(c in n for c in ['UserID', 'snapshot_date', 'churn_60d']) for n in names))
        changed = frame.assign(UserID=-1, snapshot_date='2099-01-01', churn_60d=1-self.y)
        np.testing.assert_allclose(pipe.predict_proba(frame), pipe.predict_proba(changed))

    def test_train_only_medians_scaling_categories_and_validation_transform(self):
        pipe = build_logistic_pipeline().fit(self.train, self.y)
        pre = pipe['preprocessing']
        numeric = pre.named_transformers_['numeric']
        imputer, scaler = numeric['imputer'], numeric['scaler']
        np.testing.assert_allclose(imputer.statistics_, self.train[NUMERIC_COLUMNS].median())
        imputed_train = imputer.transform(self.train[NUMERIC_COLUMNS])
        np.testing.assert_allclose(scaler.mean_, imputed_train.mean(axis=0))
        self.assertEqual(scaler.n_samples_seen_, len(self.train))
        self.assertEqual(pre.named_transformers_['categorical']['imputer'].statistics_[0], 2.)
        self.assertIn('numeric__missingindicator_' + NUMERIC_COLUMNS[0], pre.get_feature_names_out())
        heldout = self.train.iloc[:6].copy()
        heldout[NUMERIC_COLUMNS] = 1e8
        heldout.loc[0, NUMERIC_COLUMNS[0]] = np.nan
        heldout['Country'] = 999  # unseen validation category must not refit encoder
        means_before = scaler.mean_.copy()
        transformed = pre.transform(heldout)
        values = transformed.toarray() if hasattr(transformed, 'toarray') else transformed
        self.assertTrue(np.isfinite(values).all())
        np.testing.assert_array_equal(scaler.mean_, means_before)
        self.assertNotIn(999, pre.named_transformers_['categorical']['encoder'].categories_[0])
        # Missing value maps to the training median before scaling.
        self.assertAlmostEqual(values[0, 0], (imputer.statistics_[0]-scaler.mean_[0])/scaler.scale_[0])

    def test_no_infinity_and_valid_probabilities(self):
        pipe = build_logistic_pipeline().fit(self.train, self.y)
        transformed = pipe['preprocessing'].transform(self.train)
        values = transformed.toarray() if hasattr(transformed, 'toarray') else transformed
        self.assertTrue(np.isfinite(values).all())
        probabilities = pipe.predict_proba(self.train)
        self.assertTrue(np.isfinite(probabilities).all())
        self.assertTrue(((probabilities >= 0) & (probabilities <= 1)).all())
        np.testing.assert_allclose(probabilities.sum(axis=1), 1)
        invalid = self.train.copy()
        invalid.iloc[0, 0] = np.inf
        with self.assertRaises(ValueError):
            predictor_frame(invalid)

    def test_lift_recall_precision_ceiling_and_stable_ties(self):
        y = [1, 0, 1, 0, 0, 1, 0, 0, 0, 0]
        p = np.arange(10, 0, -1)/10
        table = cumulative_gains_table(y, p, [.1, .2, 1])
        np.testing.assert_allclose(table.precision, [1, .5, .3])
        np.testing.assert_allclose(table.recall, [1/3, 1/3, 1])
        np.testing.assert_allclose(table.lift, [10/3, 5/3, 1])
        ties = cumulative_gains_table([0, 1, 1], [.5, .5, .5], [.1, .5])
        self.assertEqual(ties.selected_observations.tolist(), [1, 2])
        self.assertEqual(ties.churn_observations_captured.tolist(), [0, 1])

    def test_confusion_counts_thresholds_and_invalid_probabilities(self):
        y, p = [0, 0, 1, 1], [.1, .6, .4, .8]
        metrics = classification_metrics(y, p)
        for key in ['true_positives', 'false_positives', 'true_negatives', 'false_negatives']:
            self.assertEqual(metrics[key], 1)
        self.assertEqual(metrics['specificity'], .5)
        table = threshold_analysis(y, p)
        self.assertEqual(table.threshold.tolist(), [.2, .3, .4, .5, .6, .7, .8])
        self.assertEqual(table.predicted_positives.tolist(), [3, 3, 3, 2, 2, 1, 1])
        for invalid in [[.1, np.inf, .4, .8], [.1, -.1, .4, .8], [.1, 1.1, .4, .8]]:
            with self.assertRaises(ValueError):
                classification_metrics(y, invalid)


if __name__ == '__main__':
    unittest.main()

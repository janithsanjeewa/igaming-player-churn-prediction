"""SHAP contracts, reconstruction, anonymity and sealed-split regression tests."""
import unittest
from unittest.mock import Mock

import numpy as np
import pandas as pd
import shap
from scipy.special import expit

from src.explainability import (compare_permutation, dependence_bins, explain_validation,
    importance_tables, local_cases, pairwise_margin_contrasts, transformed_feature_mapping,
    validate_local_export, validate_shap_matrix, validation_sample)
from src.features import CATEGORICAL_COLUMNS, PREDICTOR_COLUMNS, TARGET_COLUMN
from src.preprocessing import build_xgboost_pipeline


class ExplainabilityTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        rng = np.random.default_rng(42)
        train = pd.DataFrame(rng.normal(size=(100, len(PREDICTOR_COLUMNS))), columns=PREDICTOR_COLUMNS)
        for column in CATEGORICAL_COLUMNS:
            train[column] = np.tile([1, 2], 50)
        train.loc[:10, 'days_since_first_casino_activity'] = np.nan
        train['UserID'] = np.arange(100)
        train['snapshot_date'] = pd.Timestamp('2005-03-31')
        train[TARGET_COLUMN] = (train.active_days_30d < 0).astype(int)
        cls.pipeline = build_xgboost_pipeline(n_estimators=12, max_depth=2, min_child_weight=1).fit(train, train[TARGET_COLUMN])
        cls.validation = train.iloc[:20].copy().assign(snapshot_date=pd.Timestamp('2006-03-31'))
        cls.explanation, cls.matrix, cls.mapping = explain_validation(cls.pipeline, cls.validation)

    def test_exact_names_numeric_missingness_and_categories(self):
        self.assertEqual(len(self.mapping), self.matrix.shape[1])
        self.assertEqual(self.explanation.values.shape, self.matrix.shape)
        self.assertTrue(np.isfinite(self.explanation.values).all())
        self.assertEqual(self.mapping.transformed_feature.tolist(),
                         self.pipeline['preprocessing'].get_feature_names_out().tolist())
        self.assertTrue(set(self.mapping.source_feature).issubset(PREDICTOR_COLUMNS))
        for category in CATEGORICAL_COLUMNS:
            rows = self.mapping.loc[self.mapping.source_feature.eq(category)]
            self.assertEqual(len(rows), 2)
            self.assertTrue(rows.feature_group.eq('demographics').all())
        indicator = self.mapping.loc[self.mapping.feature_group.eq('missingness')]
        self.assertEqual(indicator.source_feature.tolist(), ['days_since_first_casino_activity'])
        with self.assertRaises(ValueError):
            transformed_feature_mapping(self.pipeline['preprocessing'], self.matrix.shape[1] + 1)

    def test_invalid_shap_dimensions_nonfinite_and_metadata_rejected(self):
        values = self.explanation.values
        for bad in [values[:, :-1], np.full_like(values, np.inf), np.full_like(values, np.nan)]:
            with self.assertRaises(ValueError): validate_shap_matrix(bad, self.matrix, self.mapping)
        for source in ['UserID', 'snapshot_date', TARGET_COLUMN]:
            bad_mapping = self.mapping.copy()
            bad_mapping.loc[0, 'source_feature'] = source
            with self.assertRaises(ValueError):
                validate_shap_matrix(values, self.matrix, bad_mapping)
            fake = Mock()
            fake.get_feature_names_out.return_value = np.array(['numeric__' + source])
            with self.assertRaises(ValueError): transformed_feature_mapping(fake, 1)

    def test_additivity_reconstructs_predictions(self):
        probability = self.pipeline['model'].predict_proba(self.matrix)[:, 1]
        np.testing.assert_allclose(expit(self.explanation.base_values + self.explanation.values.sum(axis=1)),
                                   probability, atol=2e-6)

    def test_sample_reproducible_and_small_population_uses_all(self):
        first = validation_sample(self.validation, size=7)
        self.assertEqual(len(first), 7)
        pd.testing.assert_frame_equal(first, validation_sample(self.validation, size=7))
        all_rows = validation_sample(self.validation)
        self.assertEqual(set(all_rows.index), set(self.validation.index))
        self.assertFalse(all_rows.index.duplicated().any())
        with self.assertRaises(ValueError): validation_sample(self.validation, size=0)

    def test_test_train_embargo_mixed_and_empty_rejected_before_model_access(self):
        spy = Mock()
        invalid = [self.validation.assign(snapshot_date=pd.Timestamp(date))
                   for date in ['2006-08-31', '2005-03-31', '2006-06-30']]
        mixed = self.validation.copy()
        mixed.iloc[0, mixed.columns.get_loc('snapshot_date')] = pd.Timestamp('2006-08-31')
        invalid += [mixed, self.validation.iloc[:0]]
        for frame in invalid:
            with self.assertRaises(ValueError): explain_validation(spy, frame)
            with self.assertRaises(ValueError): validation_sample(frame)
            with self.assertRaises(ValueError): pairwise_margin_contrasts(spy, frame, [])
            with self.assertRaises(ValueError): dependence_bins(frame, None, None, [])
            with self.assertRaises(ValueError): local_cases(frame, None, None, None)
        self.assertEqual(spy.mock_calls, [])

    def test_anonymous_local_export_and_all_four_diagnostic_cells(self):
        frame = self.validation.iloc[:4].copy()
        frame[TARGET_COLUMN] = [1, 0, 0, 1]
        explanation = self.explanation[:4]
        table, positions = local_cases(frame, [.8, .7, .2, .3], explanation, self.mapping)
        self.assertEqual(table.case.tolist(), ['Validation Case A', 'Validation Case B', 'Validation Case C', 'Validation Case D'])
        self.assertEqual(table.outcome.tolist(), ['TP', 'FP', 'TN', 'FN'])
        self.assertEqual(len(positions), 4)
        self.assertNotIn('UserID', table.to_csv(index=False))
        self.assertNotIn('snapshot_date', table.columns)
        validate_local_export(table)
        with self.assertRaises(ValueError): validate_local_export(table.assign(UserID=1))
        with self.assertRaises(ValueError): validate_local_export(table.assign(case='Player 123'))

    def test_source_level_aggregation_signed_before_absolute_and_groups_conserve_mass(self):
        importance, groups = importance_tables(self.explanation, self.mapping)
        self.assertAlmostEqual(importance.mean_abs_shap.sum(), groups.mean_abs_shap.sum())
        self.assertAlmostEqual(groups.share.sum(), 1.)
        # Opposite category-level contributions cancel within each player.
        mapping = pd.DataFrame({'source_feature': ['Country', 'Country', 'active_days_30d'],
                                'position': [0, 1, 2]})
        explanation = shap.Explanation(values=np.array([[2., -2., 1.], [3., -3., -1.]]))
        permutation = pd.DataFrame({'feature': ['active_days_30d', 'Country'],
                                    'mean_ap_drop': [.1, .01], 'std_ap_drop': [.01, .001]})
        result = compare_permutation(explanation, mapping, permutation).set_index('feature')
        self.assertEqual(result.loc['Country', 'mean_abs_shap'], 0)
        self.assertEqual(result.loc['active_days_30d', 'shap_rank'], 1)

    def test_pair_contrast_additive_model_is_zero_and_interaction_is_detected(self):
        # Replace fitted predictor only; transformer/matrix remain realistic.
        class MarginModel:
            def __init__(self, interact): self.interact = interact
            def predict(self, matrix, output_margin):
                self_check = matrix[:, 0] + matrix[:, 1]
                return self_check + (matrix[:, 0] * matrix[:, 1] if self.interact else 0)
        pair = [('active_days_30d', 'bets_30d')]
        for interact in [False, True]:
            pipeline = {'preprocessing': self.pipeline['preprocessing'], 'model': MarginModel(interact)}
            result = pairwise_margin_contrasts(pipeline, self.validation, pair)
            if interact:
                self.assertGreater(result.mean_abs_mixed_margin.iloc[0], .01)
            else:
                self.assertAlmostEqual(result.mean_abs_mixed_margin.iloc[0], 0)


if __name__ == '__main__':
    unittest.main()

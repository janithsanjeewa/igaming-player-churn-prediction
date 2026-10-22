"""Final policy leakage guards, clustered uncertainty and frozen-contract tests."""
import unittest
from unittest.mock import Mock, patch

import numpy as np
import pandas as pd

from src.business_policy import (ablation_comparison, apply_calibrator, calibration_curve_table,
    capacity_table, diagnostic_threshold_table, draw_cluster_indices, fit_original_train,
    fit_train_calibrators, frozen_pipeline, generate_train_oof, paired_cluster_bootstrap,
    policy_predictors, policy_probabilities, refit_protocol, risk_band_table,
    select_calibration, temporal_calibration_folds, validate_train_oof, verify_specification)
from src.features import CATEGORICAL_COLUMNS, PREDICTOR_COLUMNS, TARGET_COLUMN
from src.final_validation import json_safe
from src.preprocessing import build_xgboost_pipeline


class BusinessPolicyTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        rng = np.random.default_rng(42)
        dates = pd.date_range('2005-03-31', '2005-12-31', freq='ME')
        train = pd.DataFrame(rng.normal(size=(len(dates)*20, len(PREDICTOR_COLUMNS))), columns=PREDICTOR_COLUMNS)
        for column in CATEGORICAL_COLUMNS:
            train[column] = np.tile([1, 2], len(train)//2)
        train['snapshot_date'] = np.repeat(dates, 20)
        train['UserID'] = np.tile(np.arange(20), len(dates))
        train[TARGET_COLUMN] = np.tile([0, 1], len(train)//2)
        cls.train = train
        cls.validation = train.iloc[:20].assign(snapshot_date=pd.Timestamp('2006-03-31'))
        cls.template = build_xgboost_pipeline(n_estimators=8, max_depth=2, min_child_weight=1)
        cls.free = fit_original_train(cls.template, train, demographics=False)

    def test_test_embargo_mixed_dates_never_reach_predict_or_fit(self):
        spy = Mock()
        invalid = [self.validation.assign(snapshot_date=pd.Timestamp(date)) for date in
                   ['2006-08-31', '2006-06-30', '2006-02-28']]
        mixed = self.validation.copy()
        mixed.iloc[0, mixed.columns.get_loc('snapshot_date')] = pd.Timestamp('2006-08-31')
        invalid += [mixed, self.validation.iloc[:0]]
        for frame in invalid:
            for split in ['validation', 'train', 'test']:
                with self.assertRaises(ValueError): policy_probabilities(spy, frame, split)
            with self.assertRaises(ValueError): fit_original_train(spy, frame)
            with self.assertRaises(ValueError): generate_train_oof(spy, frame)
        self.assertEqual(spy.mock_calls, [])

    def test_whole_clusters_retained_with_multiplicity(self):
        rng = Mock()
        rng.integers.return_value = np.array([1, 1, 0])
        result = draw_cluster_indices([10, 10, 20, 20, 20, 30], rng)
        np.testing.assert_array_equal(result, [2, 3, 4, 2, 3, 4, 0, 1])
        # Both RF and XGB use the same resampled arrays; strong Brier improvement
        # must have the lower-is-better direction in every paired draw.
        frame = pd.DataFrame({'UserID': [1, 1, 2, 2, 2, 2, 3, 3],
            'snapshot_date': pd.Timestamp('2006-03-31'), TARGET_COLUMN: [0, 1]*4})
        rf = np.repeat(.5, len(frame))
        xgb = .2+.6*frame[TARGET_COLUMN].to_numpy()
        from src.business_policy import _draw_clusters
        with patch('src.business_policy._draw_clusters', wraps=_draw_clusters) as sampler:
            first = paired_cluster_bootstrap(frame, rf, xgb, replicates=40)
        self.assertEqual(sampler.call_count, 40)
        clusters = sampler.call_args.args[0]
        self.assertEqual([part.tolist() for part in clusters], [[0, 1], [2, 3, 4, 5], [6, 7]])
        second = paired_cluster_bootstrap(frame, rf, xgb, replicates=40)
        pd.testing.assert_frame_equal(first, second)
        row = first.set_index('metric').loc['brier_score']
        self.assertAlmostEqual(row.observed_difference, -.21)
        self.assertEqual(row.proportion_xgboost_better, 1)
        self.assertLess(row.ci95_high, 0)
        self.assertEqual(first.valid_replicates.unique().tolist(), [40])

    def test_capacity_rounding_ties_and_hand_computed_capture(self):
        table = capacity_table([1, 0, 1, 0, 1], {'example': [.9, .9, .8, .2, .1]})
        top20 = table.loc[table.target_fraction.eq(.2)].iloc[0]
        self.assertEqual(top20.selected_observations, 1)
        self.assertEqual(top20.churn_observations_captured, 1)
        self.assertEqual(top20.precision, 1)
        self.assertAlmostEqual(top20.recall, 1/3)
        self.assertAlmostEqual(top20.lift, 1/.6)
        top25 = table.loc[table.target_fraction.eq(.25)].iloc[0]
        self.assertEqual(top25.selected_observations, 2)
        self.assertEqual(top25.precision, .5)

    def test_demographic_free_really_removes_inputs_and_encoded_columns(self):
        self.assertFalse(set(CATEGORICAL_COLUMNS) & set(policy_predictors(self.train, False).columns))
        self.assertFalse(set(CATEGORICAL_COLUMNS) & set(self.free['preprocessing'].feature_names_in_))
        names = self.free['preprocessing'].get_feature_names_out()
        self.assertFalse(any(category in name for category in CATEGORICAL_COLUMNS for name in names))
        original = policy_probabilities(self.free, self.validation, 'validation', demographics=False)
        changed = self.validation.assign(Country=9999, Language=-99, Gender=88)
        np.testing.assert_array_equal(original, policy_probabilities(self.free, changed, 'validation', False))
        self.assertEqual(self.free['model'].get_params(), frozen_pipeline(self.template, False)['model'].get_params())

    def test_train_oof_purge_and_calibrator_provenance(self):
        from src.business_policy import fit_original_train as actual_fit
        with patch('src.business_policy.fit_original_train', wraps=actual_fit) as fitting:
            oof, folds = generate_train_oof(self.template, self.train, demographics=False)
        self.assertEqual(len(folds), 5)
        self.assertTrue((folds.outcome_end < folds.score_observation_start).all())
        self.assertGreaterEqual(folds.gap_snapshot_days.min(), 90)
        self.assertEqual(fitting.call_count, 5)
        for call, row in zip(fitting.call_args_list, folds.itertuples()):
            dates = call.args[1].snapshot_date
            self.assertLess(dates.max()+pd.Timedelta(days=60), row.score_observation_start)
            self.assertLess(dates.max(), row.score_date)
        validate_train_oof(oof)
        mappings = fit_train_calibrators(oof)
        self.assertIn('sigmoid', mappings)
        self.assertNotIn('isotonic', mappings)  # Too few rows under prespecified rule.
        self.assertGreater(mappings['sigmoid']['slope'], 0)
        p = apply_calibrator(mappings['sigmoid'], np.array([.1, .5, .9]))
        self.assertTrue(np.isfinite(p).all())
        self.assertTrue((np.diff(p) >= 0).all())
        for date in ['2006-03-31', '2006-08-31']:
            invalid = oof.assign(snapshot_date=pd.Timestamp(date))
            with self.assertRaises(ValueError): fit_train_calibrators(invalid)
        with self.assertRaises(ValueError): fit_train_calibrators(oof.assign(fit_end=oof.snapshot_date))
        with self.assertRaises(ValueError): fit_train_calibrators(oof.drop(columns='fit_end'))

    def test_calibration_selection_prefers_no_mapping_for_inconsistent_gains(self):
        rows = pd.DataFrame([dict(method='raw', brier_score=.19, log_loss=.55, ece=.03),
            dict(method='sigmoid', brier_score=.1895, log_loss=.549, ece=.02),
            dict(method='isotonic', brier_score=.188, log_loss=.548, ece=.04)])
        self.assertEqual(select_calibration(rows), 'raw')
        rows.loc[rows.method.eq('sigmoid'), ['brier_score', 'log_loss']] = [.1883, .547]
        rows.loc[rows.method.eq('isotonic'), ['ece', 'log_loss']] = [.02, .5465]
        self.assertEqual(select_calibration(rows), 'sigmoid')

    def test_ece_bins_include_endpoints_and_threshold_empty_targeting(self):
        table = calibration_curve_table([0, 1, 0, 1], [0, 1, .1, .9])
        self.assertEqual(table.observations.sum(), 4)
        self.assertEqual(table.iloc[0].observations, 1)
        self.assertEqual(table.iloc[1].observations, 1)
        self.assertEqual(table.iloc[-1].observations, 2)
        thresholds = diagnostic_threshold_table([0, 1], [.1, .15])
        self.assertEqual(thresholds.targeted_observations.sum(), 0)
        self.assertTrue(thresholds.lift.isna().all())
        bands = risk_band_table([0, 1]*10, np.repeat(.4, 20))
        self.assertEqual(bands.observations.tolist(), [2, 2, 6, 10])
        self.assertEqual(bands.observations.sum(), 20)

    def test_specification_must_match_frozen_estimator_and_raw_features(self):
        spec = dict(estimator_parameters=json_safe(self.free['model'].get_params()),
            predictors=self.free['preprocessing'].feature_names_in_.tolist(),
            transformed_features=self.free['preprocessing'].get_feature_names_out().tolist(),
            demographics_retained=False)
        verify_specification(spec, self.free)
        wrong = {**spec, 'estimator_parameters': {**spec['estimator_parameters'], 'max_depth': 99}}
        with self.assertRaises(ValueError): verify_specification(wrong, self.free)
        with self.assertRaises(ValueError): verify_specification({**spec, 'predictors': ['UserID']}, self.free)
        with self.assertRaises(ValueError): verify_specification({**spec, 'demographics_retained': True}, self.free)

    def test_exact_pretest_boundary_and_safe_months_only(self):
        frame = pd.DataFrame({'UserID': range(8), TARGET_COLUMN: [0, 1]*4,
            'snapshot_date': pd.to_datetime(['2005-03-31', '2006-01-31', '2006-02-28',
                '2006-05-31', '2006-06-02', '2006-06-03', '2006-06-30', '2006-08-31'])})
        summary, snapshots, boundary = refit_protocol(frame)
        self.assertEqual(summary['first_test_observation_start'], '2006-08-02')
        self.assertEqual(summary['latest_safe_calendar_snapshot'], '2006-06-02')
        self.assertEqual(summary['observations'], 5)
        self.assertTrue((snapshots.outcome_end < pd.Timestamp('2006-08-02')).all())
        self.assertEqual(boundary.safe_for_future_refit.tolist(), [True, True, True, False, False])
        self.assertEqual(pd.Timestamp('2006-06-02')+pd.Timedelta(days=60), pd.Timestamp('2006-08-01'))

    def test_ablation_decision_uses_explicit_metric_tolerances(self):
        y = np.array([0, 1]*50)
        p = np.where(y, .7, .3)
        comparison = ablation_comparison(y, p, p)
        self.assertTrue(comparison.within_tolerance.all())
        self.assertEqual(comparison.absolute_difference.sum(), 0)


if __name__ == '__main__':
    unittest.main()

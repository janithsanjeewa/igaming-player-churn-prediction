"""Frozen contract comparisons; no player data or model fitting."""
import json
from pathlib import Path
import unittest

from src.features import CATEGORICAL_COLUMNS, METADATA_COLUMNS, PREDICTOR_COLUMNS
from src.model_contract import (DIAGNOSTIC_THRESHOLD, FINAL_FEATURES, MODEL_VERSION,
    OBSERVATION_WINDOW_DAYS, OUTCOME_WINDOW_DAYS, TARGET_NAME, get_model_info)

ROOT = Path(__file__).resolve().parents[1]


class ModelContractTests(unittest.TestCase):
    def test_exact_order_matches_frozen_specification_and_dictionary(self):
        spec = json.loads((ROOT / 'reports/final_model_specification.json').read_text())
        self.assertEqual(len(FINAL_FEATURES), 42)
        self.assertEqual(len(set(FINAL_FEATURES)), 42)
        self.assertEqual(list(FINAL_FEATURES), spec['predictors'])
        self.assertEqual(list(FINAL_FEATURES), [c for c in PREDICTOR_COLUMNS if c not in CATEGORICAL_COLUMNS])
        self.assertIsInstance(FINAL_FEATURES, tuple)

    def test_excluded_fields_and_frozen_windows_threshold(self):
        self.assertFalse(set(FINAL_FEATURES) & set(CATEGORICAL_COLUMNS + METADATA_COLUMNS + [TARGET_NAME]))
        self.assertEqual(TARGET_NAME, 'churn_60d')
        self.assertEqual((OBSERVATION_WINDOW_DAYS, OUTCOME_WINDOW_DAYS, DIAGNOSTIC_THRESHOLD), (30, 60, .50))
        self.assertTrue(MODEL_VERSION)

    def test_info_is_accurate_independent_and_non_sensitive(self):
        info = get_model_info()
        self.assertEqual(info['predictor_count'], 42)
        self.assertEqual(info['calibration'], 'none')
        self.assertEqual(info['single_risk_band'], 'unranked')
        self.assertTrue(info['demographics_excluded'])
        self.assertTrue(info['metadata_excluded'])
        self.assertNotIn('UserID', json.dumps(info))
        self.assertNotIn('model_path', info)
        info['limitations'].clear()
        self.assertTrue(get_model_info()['limitations'])


if __name__ == '__main__':
    unittest.main()

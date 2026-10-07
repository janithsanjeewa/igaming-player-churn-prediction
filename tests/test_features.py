"""Boundary and future-invariance tests for the player-snapshot feature contract."""
import unittest
import numpy as np
import pandas as pd
from src.features import (PREDICTOR_COLUMNS, build_snapshot_features,
                          build_snapshot_target, validate_inputs)


class FeatureContractTests(unittest.TestCase):
    def setUp(self):
        self.snapshot = pd.Timestamp('2020-01-30')
        self.daily = pd.DataFrame([
            (1,'2020-01-01',10,5,1), (1,'2020-01-15',20,10,2),
            (1,'2020-01-16',30,15,3), (1,'2020-01-24',40,20,4),
            (1,'2020-01-30',50,25,5), (1,'2020-01-20',0,-5,0),
            (1,'2020-03-30',100,70,10), (2,'2020-01-01',0,2,1),
        ], columns=['UserID','Date','Stake','Winnings','Bets'])
        self.daily.Date = pd.to_datetime(self.daily.Date)
        self.demo = pd.DataFrame({'UserID':[1,2], 'Country':[276,40],
            'Language':[2,2], 'Gender':[1,0],
            'RegDate':pd.to_datetime(['2020-01-01']*2),
            'Fstpdate':pd.to_datetime(['2020-01-01']*2),
            'Fstcadate':pd.to_datetime(['2020-02-05','2020-01-01'])})

    def test_hand_calculated_history_and_financial_corrections(self):
        validate_inputs(self.demo,self.daily)
        rows = build_snapshot_features(self.demo,self.daily,self.snapshot).set_index('UserID')
        r = rows.loc[1]
        expected = {'active_days_30d':5,'active_days_14d':2,'active_days_7d':2,
            'bets_30d':15,'bets_7d':9,'stake_30d':150,'winnings_30d':70,
            'net_loss_30d':80,'bets_early15':3,'bets_recent15':12,
            'bets_change_15d':9,'bets_change_pct_15d':300,
            'active_days_early15':2,'active_days_recent15':3,
            'active_days_change_pct_15d':50,'days_since_last_active_bet':0,
            'max_gap_between_active_days_30d':13,'mean_gap_between_active_days_30d':6.25,
            'max_daily_stake_30d':50,'median_daily_stake_30d':0,
            'days_since_registration':29}
        for name,value in expected.items():
            self.assertAlmostEqual(r[name],value,msg=name)
        self.assertAlmostEqual(r.std_daily_stake_30d,np.std([10,20,30,40,50]+[0]*25))
        self.assertTrue(pd.isna(r.days_since_first_casino_activity))
        self.assertEqual(len(PREDICTOR_COLUMNS),45)
        self.assertEqual(len(set(PREDICTOR_COLUMNS)),45)
        self.assertNotIn('UserID',PREDICTOR_COLUMNS)
        self.assertNotIn('churn_60d',PREDICTOR_COLUMNS)

    def test_legitimate_missing_values_and_negative_net_loss(self):
        r=build_snapshot_features(self.demo,self.daily,self.snapshot).set_index('UserID').loc[2]
        self.assertEqual(r.active_days_30d,1)
        self.assertEqual(r.days_since_last_active_bet,29)
        self.assertEqual(r.net_loss_30d,-2)
        for feature in ['max_gap_between_active_days_30d','mean_gap_between_active_days_30d',
                        'avg_bets_per_active_day_7d','stake_7d_share','stake_change_pct_15d']:
            self.assertTrue(pd.isna(r[feature]),msg=feature)
        self.assertEqual(r.bets_change_pct_15d,-100)

    def test_features_unchanged_when_future_is_removed_or_changed(self):
        full=build_snapshot_features(self.demo,self.daily,self.snapshot)
        past=self.daily.loc[self.daily.Date.le(self.snapshot)].copy()
        pd.testing.assert_frame_equal(full,build_snapshot_features(self.demo,past,self.snapshot))
        changed=self.daily.copy()
        changed.loc[changed.Date.gt(self.snapshot),['Stake','Winnings','Bets']]=[1e9,-1e9,999999]
        pd.testing.assert_frame_equal(full,build_snapshot_features(self.demo,changed,self.snapshot))

    def test_target_exact_boundaries_zero_bets_and_censoring(self):
        end=self.snapshot+pd.Timedelta(days=60)
        for day, bets, expected in [(0,1,1),(1,1,0),(60,1,0),(61,1,1),(1,0,1)]:
            frame=pd.DataFrame({'UserID':[1], 'Date':[self.snapshot+pd.Timedelta(days=day)], 'Bets':[bets]})
            self.assertEqual(build_snapshot_target(frame,[1],self.snapshot,end).iloc[0],expected)
        with self.assertRaises(ValueError):
            build_snapshot_target(self.daily,[1],self.snapshot,end-pd.Timedelta(days=1))

    def test_reject_incomplete_history_and_duplicate_days(self):
        with self.assertRaises(ValueError):
            build_snapshot_features(self.demo,self.daily,self.snapshot-pd.Timedelta(days=1))
        with self.assertRaises(ValueError):
            validate_inputs(self.demo,pd.concat([self.daily,self.daily.iloc[:1]]))


if __name__ == '__main__':
    unittest.main()

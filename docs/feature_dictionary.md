# Feature dictionary — 60-day operational churn panel

One row is an eligible player at an end-of-day month-end snapshot S. Eligibility requires
positive Bets in [S-29, S], full 30-day historical coverage, and full outcome coverage.
The target is extended inactivity, not guaranteed permanent customer departure.

There are 45 predictors, two metadata columns, and one target. Country, Language, and
Gender retain original integer codes but must be treated as categories during future
preprocessing. No categories are encoded and no numeric variables are scaled here.

All transaction predictors use only the 30 days through S. 7-/14-day windows contain
7/14 dates including S; early and recent trend halves contain 15 dates each. Monetary
totals include all cleaned records, including zero-bet transactions and signed Winnings
corrections. Negative Winnings or net loss are valid. Missing transaction dates are
zero-filled only for 30-calendar-day daily-stake statistics, assuming complete collection;
the source rows and all financial totals remain unchanged.

Division by zero yields missing values, never infinity, including 0/0 percentage changes.
No historical active days in the recent 7 days makes the 7-day bets-per-active-day ratio
missing. Fewer than two positive-bet days makes between-day gaps missing; no gap is
invented. Gaps count full inactive dates (date difference minus one), excluding window
edges. No missing predictors are imputed before temporal splitting.

Lifecycle dates are used only when on/before S. Demographic Fstcadate can be later than
observed gaming dates; future-dated source events are masked rather than replaced using
future data. Other inconsistencies are retained for transparency. Raw demographic codes
are assumed available historically; the source provides no attribute-change history.

The proposed split metadata is held separately and not exported in this panel. Training
uses 2005-03-31–2005-12-31; validation 2006-03-31–2006-05-31; test 2006-08-31–2006-11-30.
January/February and June/July 2006 snapshots are embargoed. Snapshot gaps are 90/92
days, and later observation windows start after earlier label windows end. Same-player
overlap is allowed for future behaviour of known/recently active customers; unseen-player
generalisation requires a separate design. Fit future preprocessing on training data only.

`data/processed/model_panel_60d.csv` is a local Git-ignored output and must not be uploaded.

| Feature name | Type | Time window | Definition | Modelling role | Leakage notes |
| --- | --- | --- | --- | --- | --- |
| UserID | integer identifier | N/A | Player tracking and grouping key; never a predictor. | metadata | Repeated players may occur across temporal splits. |
| snapshot_date | ISO calendar date | Prediction at end of S | Monthly snapshot for tracking, embargo, and temporal splitting. | metadata | Excluded from predictor list. |
| active_days_30d | integer | [S-29, S], 30 calendar dates | Distinct dates with Bets > 0. | predictor | Historical transactions only. |
| bets_30d | integer | [S-29, S], 30 calendar dates | Sum of Bets over all cleaned records. | predictor | Historical transactions only. |
| active_days_14d | integer | [S-13, S], 14 calendar dates | Distinct dates with Bets > 0. | predictor | Historical transactions only. |
| bets_14d | integer | [S-13, S], 14 calendar dates | Sum of Bets over all cleaned records. | predictor | Historical transactions only. |
| active_days_7d | integer | [S-6, S], 7 calendar dates | Distinct dates with Bets > 0. | predictor | Historical transactions only. |
| bets_7d | integer | [S-6, S], 7 calendar dates | Sum of Bets over all cleaned records. | predictor | Historical transactions only. |
| avg_bets_per_active_day_30d | float | 30 days through S | bets_30d / active_days_30d; missing if denominator is zero. | predictor | Historical transactions only. |
| avg_bets_per_active_day_7d | float | 7 days through S | bets_7d / active_days_7d; missing if denominator is zero. | predictor | Historical transactions only. |
| days_since_last_active_bet | integer | 30 days through S | S minus latest date with Bets > 0; 0 means betting on snapshot day. | predictor | Historical transactions only. |
| max_gap_between_active_days_30d | float | 30 days through S | Max full inactive dates between consecutive positive-bet dates (date difference minus 1); missing if fewer than two active dates. Excludes leading/trailing inactivity and gaps crossing the window boundary. | predictor | Historical transactions only. |
| mean_gap_between_active_days_30d | float | 30 days through S | Mean full inactive dates between consecutive positive-bet dates (date difference minus 1); missing if fewer than two active dates. Excludes leading/trailing inactivity and gaps crossing the window boundary. | predictor | Historical transactions only. |
| stake_30d | float (EUR) | 30 days through S | Sum of cleaned Stake including zero-bet financial records and signed accounting corrections; no clipping. | predictor | Historical transactions only. |
| stake_14d | float (EUR) | 14 days through S | Sum of cleaned Stake including zero-bet financial records and signed accounting corrections; no clipping. | predictor | Historical transactions only. |
| stake_7d | float (EUR) | 7 days through S | Sum of cleaned Stake including zero-bet financial records and signed accounting corrections; no clipping. | predictor | Historical transactions only. |
| winnings_30d | float (EUR) | 30 days through S | Sum of cleaned Winnings including zero-bet financial records and signed accounting corrections; no clipping. | predictor | Historical transactions only. |
| winnings_14d | float (EUR) | 14 days through S | Sum of cleaned Winnings including zero-bet financial records and signed accounting corrections; no clipping. | predictor | Historical transactions only. |
| winnings_7d | float (EUR) | 7 days through S | Sum of cleaned Winnings including zero-bet financial records and signed accounting corrections; no clipping. | predictor | Historical transactions only. |
| net_loss_30d | float (EUR) | 30 days through S | stake_30d minus winnings_30d; negative values are net winnings. | predictor | Historical transactions only. |
| net_loss_7d | float (EUR) | 7 days through S | stake_7d minus winnings_7d; negative values are net winnings. | predictor | Historical transactions only. |
| avg_stake_per_bet_30d | float (EUR) | 30 days through S | stake_30d / bets_30d; missing for a zero denominator (eligibility prevents this). | predictor | Historical transactions only. |
| avg_stake_per_active_day_30d | float (EUR) | 30 days through S | All-record stake_30d / positive-bet active_days_30d. | predictor | Historical transactions only. |
| max_daily_stake_30d | float (EUR) | 30 calendar dates through S | Max of 30 daily Stakes, including recorded zero-bet days and zero for dates with no record. Standard deviation uses ddof=0 (population). | predictor | Historical transactions only. |
| median_daily_stake_30d | float (EUR) | 30 calendar dates through S | Median of 30 daily Stakes, including recorded zero-bet days and zero for dates with no record. Standard deviation uses ddof=0 (population). | predictor | Historical transactions only. |
| std_daily_stake_30d | float (EUR) | 30 calendar dates through S | Std of 30 daily Stakes, including recorded zero-bet days and zero for dates with no record. Standard deviation uses ddof=0 (population). | predictor | Historical transactions only. |
| active_days_7d_share | float | 7 days / 30 days, both ending S | active_days_7d / active_days_30d; missing if denominator is zero. | predictor | Historical transactions only. |
| bets_7d_share | float | 7 days / 30 days, both ending S | bets_7d / bets_30d; missing if denominator is zero. | predictor | Historical transactions only. |
| stake_7d_share | float | 7 days / 30 days, both ending S | stake_7d / stake_30d; missing if denominator is zero. | predictor | Historical transactions only. |
| bets_recent15 | integer | [S-14, S], 15 dates | Sum of Bets over all cleaned records. | predictor | Historical transactions only. |
| bets_early15 | integer | [S-29, S-15], 15 dates | Sum of Bets over all cleaned records. | predictor | Historical transactions only. |
| bets_change_15d | integer | Two disjoint 15-day halves of the 30-day window | bets_recent15 minus bets_early15; signed change. | predictor | Historical transactions only. |
| bets_change_pct_15d | float (%) | Two disjoint 15-day halves ending S | 100 * (bets_recent15 - bets_early15) / bets_early15. Missing whenever the early denominator is zero, including 0/0. | predictor | Historical transactions only. |
| stake_recent15 | float (EUR) | [S-14, S], 15 dates | Sum of Stake over all cleaned records. | predictor | Historical transactions only. |
| stake_early15 | float (EUR) | [S-29, S-15], 15 dates | Sum of Stake over all cleaned records. | predictor | Historical transactions only. |
| stake_change_15d | float (EUR) | Two disjoint 15-day halves of the 30-day window | stake_recent15 minus stake_early15; signed change. | predictor | Historical transactions only. |
| stake_change_pct_15d | float (%) | Two disjoint 15-day halves ending S | 100 * (stake_recent15 - stake_early15) / stake_early15. Missing whenever the early denominator is zero, including 0/0. | predictor | Historical transactions only. |
| active_days_recent15 | integer | [S-14, S], 15 dates | Positive-bet active-day count. | predictor | Historical transactions only. |
| active_days_early15 | integer | [S-29, S-15], 15 dates | Positive-bet active-day count. | predictor | Historical transactions only. |
| active_days_change_15d | integer | Two disjoint 15-day halves of the 30-day window | active_days_recent15 minus active_days_early15; signed change. | predictor | Historical transactions only. |
| active_days_change_pct_15d | float (%) | Two disjoint 15-day halves ending S | 100 * (active_days_recent15 - active_days_early15) / active_days_early15. Missing whenever the early denominator is zero, including 0/0. | predictor | Historical transactions only. |
| days_since_registration | float (days) | Lifecycle event through S | S minus demographic RegDate; missing if that source event is later than S. | predictor | Only use event dates on/before S. Source event inconsistencies are not repaired using future activity; future-dated events are unknown as of S. |
| days_since_first_pay | float (days) | Lifecycle event through S | S minus demographic Fstpdate; missing if that source event is later than S. | predictor | Only use event dates on/before S. Source event inconsistencies are not repaired using future activity; future-dated events are unknown as of S. |
| days_since_first_casino_activity | float (days) | Lifecycle event through S | S minus demographic Fstcadate; missing if that source event is later than S. | predictor | Only use event dates on/before S. Source event inconsistencies are not repaired using future activity; future-dated events are unknown as of S. |
| Country | raw coded category (integer) | Demographic record | Original unencoded code; categorical interpretation, not a continuous magnitude. | predictor | Assume these cohort attributes are available at S; no historical change log exists to verify past country/language/gender states. |
| Language | raw coded category (integer) | Demographic record | Original unencoded code; categorical interpretation, not a continuous magnitude. | predictor | Assume these cohort attributes are available at S; no historical change log exists to verify past country/language/gender states. |
| Gender | raw coded category (integer) | Demographic record | Original unencoded code; categorical interpretation, not a continuous magnitude. | predictor | Assume these cohort attributes are available at S; no historical change log exists to verify past country/language/gender states. |
| churn_60d | binary integer | (S, S+60], 60 dates | 1 if no Bets > 0 day in the full future window; otherwise 0. | target | Future activity is used solely for this outcome; S+60 must be on/before the dataset end. Never a predictor. |

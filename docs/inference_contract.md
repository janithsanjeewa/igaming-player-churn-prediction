# Frozen inactivity-risk inference contract

The prediction score estimates the probability of 60-day operational inactivity based on the preceding 30-day behavioural observation window.

It does not represent probability of permanent customer departure.

## Purpose and version

`src/inference.py` wraps the already fitted demographic-free XGBoost pipeline.
It validates prepared features and returns raw probabilities without fitting,
calibration, threshold search or feature engineering. It adds no API server or UI.
Model version: `xgboost-60d-demographic-free-v1`.

The target is `churn_60d`: 1 means no positive-bet day (`Bets > 0`) in (S,S+60];
0 means at least one such day. Observation features use [S-29,S], inclusive.
All features must be available at the prediction snapshot. Callers construct
them using the existing [feature dictionary](feature_dictionary.md); this layer
cannot verify historical availability or recompute features from transactions.

## Ordered feature schema

`src/model_contract.py` defines the immutable ordered `FINAL_FEATURES` tuple,
copied from `reports/final_model_specification.json`. Automated tests compare it
to that frozen specification and the dictionary's numerical predictor list.
Exactly these 42 fields are required:

```text
active_days_30d
bets_30d
active_days_14d
bets_14d
active_days_7d
bets_7d
avg_bets_per_active_day_30d
avg_bets_per_active_day_7d
days_since_last_active_bet
max_gap_between_active_days_30d
mean_gap_between_active_days_30d
stake_30d
stake_14d
stake_7d
winnings_30d
winnings_14d
winnings_7d
net_loss_30d
net_loss_7d
avg_stake_per_bet_30d
avg_stake_per_active_day_30d
max_daily_stake_30d
median_daily_stake_30d
std_daily_stake_30d
active_days_7d_share
bets_7d_share
stake_7d_share
bets_recent15
bets_early15
bets_change_15d
bets_change_pct_15d
stake_recent15
stake_early15
stake_change_15d
stake_change_pct_15d
active_days_recent15
active_days_early15
active_days_change_15d
active_days_change_pct_15d
days_since_registration
days_since_first_pay
days_since_first_casino_activity
```

Fields may arrive in any column/key order; validation copies and reorders them.
The model never receives the DataFrame index as a feature.

## Input validation and missingness

- Single input: a mapping of names to scalar real numerical or missing values.
- Batch input: a nonempty pandas DataFrame. Duplicate column names are rejected.
- Every required field must be present, even if its value is genuinely missing.
- Unknown fields are rejected. **No extra fields are silently ignored**, including
  UserID, snapshot_date, churn_60d, Country, Language and Gender. Callers must
  explicitly select the contract fields before calling; labels are never echoed.
- Numeric strings, booleans, complex numbers, dates, containers and infinity are
  rejected with `FeatureValidationError`. Values are not included in error text.
- `NaN`, `pd.NA` and `None` are accepted missing values. Validation normalizes
  their representation to numerical NaN, without replacing them with an estimate.
  Median imputation and learned missing indicators remain inside the fitted
  pipeline. An all-missing field remains NaN until that pipeline processes it.
- Negative monetary corrections, signed net loss and signed changes are allowed.
  This is a numeric/schema validator, not a replacement for source-data quality
  checks or the feature definitions. It does not impose new feature ranges.

Validation returns an independent, ordered float64 frame without changing the
caller input. It completes before loading/scoring the model.

## Single prediction

```python
from src.inference import predict_churn

# prepared_features is a dictionary with all 42 fields, constructed by the caller.
result = predict_churn(prepared_features)
```

The returned dictionary contains:

| Field | Type | Meaning |
| --- | --- | --- |
| churn_probability | float in [0,1] | Raw probability of 60-day inactivity |
| diagnostic_prediction | int, 0 or 1 | 1 when probability >= 0.50 |
| risk_band | string | Always `unranked` for independent single scoring |
| target_definition | string | Explicit inactivity target, not permanent departure |
| model_version | string | Frozen inference contract version |

`include_diagnostic=False` omits the diagnostic field. The threshold is fixed
at **0.50** and cannot be supplied as a prediction argument. It is a diagnostic
reference, not a contact-cost or financially optimal decision threshold.

No frozen validation-derived probability cutoffs were saved. Therefore a single
independent observation cannot receive a population-percentile band. No new
cutoffs or model explanations are invented here.

## Batch prediction and relative risk bands

```python
from src.inference import predict_churn_batch

# prepared_frame contains only the 42 required numerical columns.
results = predict_churn_batch(prepared_frame)
```

The output DataFrame has the same fields as the single output, with batch rank
bands. It preserves input row count, row order and index, including duplicate
index labels. It does not echo input features, IDs or labels. Retaining an
input index does not log or save it; callers remain responsible for their own
index contents and any downstream export.

Descending raw score order assigns bands using the exact Day 10 rounding:

| Output band | Rank allocation |
| --- | --- |
| very_high | First ceil(n * 0.10) observations |
| high | Next observations through ceil(n * 0.20) |
| medium | Next observations through ceil(n * 0.50) |
| low | Remaining observations (approximately bottom 50%) |

Ties retain input position, never index value, player identifier or outcome.
Identical scores can span bands. The four bands are disjoint and are relative
to this batch only; changing batch composition can change an observation's
band without changing its probability. Scores/diagnostics are independently
determined by features and the frozen model.

Small batches can have empty bands because of ceiling rounding. An explicitly
submitted one-row **batch** receives `very_high` under this allocation rule;
that means first within that one-row cohort and provides no external percentile
evidence. Use `predict_churn` for an independent observation: it returns
`unranked`. Choose a coherent campaign/scoring cohort for useful batch bands.
Bands are prioritization tools, not clinical or regulatory categories.

## Model loading and artifact integrity

Default local artifact: `models/final_xgboost_churn_model.joblib`, resolved from
the repository root regardless of the working directory. Set
`IGAMING_CHURN_MODEL_PATH` or pass the keyword-only `model_path` to use a different
location for the **same frozen artifact**. Relative paths resolve from the
project root. This setting does not authorize an alternative model.

The loader requires the frozen specification and final evaluation audit manifest
under `reports/`. It checks SHA-256 fingerprints before deserialization, model
family, exact estimator parameters, raw/transformed feature schema and count,
class ordering, fitted preprocessing and the raw/0.50 policy. These manifests
must travel with the project. Only trusted local joblib files should be used;
a checksum is not a signature or permission to load arbitrary pickle files.

A missing artifact raises `ModelLoadError` explaining that the final
evaluation/training workflow must produce it first. Inference never silently
trains or reruns final evaluation. Preserve the one-time evaluation checkpoint
when restoring missing artifacts. Unreadable, corrupted or incompatible artifacts
also raise `ModelLoadError` without exposing payload values or internal paths.

Loading is lazy and cached by resolved path, file modification time/size and
manifest contents. A lock prevents duplicate concurrent first loads. The cache
holds up to four verified artifact revisions. `clear_model_cache()` explicitly
invalidates it. Returned model objects are shared and must be treated as read-only.
Prediction functions validate finite, normalized two-class outputs in [0,1].

## Metadata and health

`get_model_info()` returns public contract metadata, target/windows, predictor
count, calibration (`none`), exclusions, intended use and limitations. It does
not require the artifact and exposes no dataset rows or local model path.

`model_health_check()` checks artifact loading/integrity, expected fitted methods
and schema, and a deterministic all-zero **synthetic schema fixture**. It returns
`healthy`, model version and check status; it does **not** call predict_proba or
present a synthetic risk score as meaningful. Failure returns `healthy=False`
with an actionable non-payload error. Health indicates loading/schema readiness,
not current-world model validity, calibration, latency or fairness.

```powershell
python -m src.inference --health
python -m src.inference --model-info
```

The CLI accepts no customer records or feature payloads. Health failures return
exit status 1; successful health/info commands return 0.

## Privacy, limitations and verification

There is no default request logging, feature printing or automatic prediction
persistence. The model and datasets remain Git-ignored. Examples/tests use
synthetic fixtures rather than real bwin player records. Existing final reports,
historical notebooks and model decisions are unchanged. No new dependency is
required.

Historical 2005-2007 data, repeated-player structure, correlated predictors and
temporal drift limit transferability. Scores do not diagnose gambling harm,
identify responsible-gambling status, determine customer value or establish
causality; they should not drive automated adverse decisions. Demographic
exclusion is not a fairness guarantee. Obtain current-data validation, governance
review and monitoring before operational use. See the existing model card.

Run all checks from the repository root:

```powershell
.\venv\Scripts\python.exe -m unittest discover -s tests -v
```

Inference tests verify schema/type/missingness behavior, raw threshold application,
single/batch bands, ordering, cache behavior, privacy and repeatable synthetic
predictions. They do not refit the final model or rescore historical test players.

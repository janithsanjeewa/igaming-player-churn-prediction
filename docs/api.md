# Frozen inactivity-risk prediction API

Probability estimates 60-day operational inactivity.

It does not represent permanent customer departure or gambling-harm risk.

This FastAPI application wraps `src.inference` without changing model settings,
features, calibration or diagnostic threshold. It accepts prepared numerical
features, not transactions. See [inference_contract.md](inference_contract.md)
and [feature_dictionary.md](feature_dictionary.md) for the full feature definitions.
No frontend, Docker image, model fitting or test-set reevaluation is included.

## Local startup

From the repository root, install dependencies into the existing virtual environment:

```powershell
.\venv\Scripts\python.exe -m pip install -r requirements.txt
.\venv\Scripts\python.exe -m uvicorn api.main:app --reload --host 127.0.0.1 --port 8000
```

With the environment activated, the equivalent command is:

```powershell
uvicorn api.main:app --reload
```

Keep this demonstration bound to loopback. Open `http://127.0.0.1:8000/docs`
for Swagger UI, `/redoc` for ReDoc, and `/openapi.json` for the generated schema.
The service version matches `MODEL_VERSION` in `src/model_contract.py`.

## Endpoints

| Method | Path | Behavior |
| --- | --- | --- |
| GET | `/` | API name/version and navigation paths only |
| GET | `/health` | Verify cached artifact/schema readiness without predicting |
| GET | `/model-info` | Public frozen contract metadata and limitations |
| POST | `/predict` | One prepared feature record; independent score is `unranked` |
| POST | `/predict-batch` | Ordered records; bands are relative to the submitted cohort |

`/model-info` returns model family/version, target and definition, observation/
outcome windows, predictor count, diagnostic threshold, demographic exclusion,
calibration (`none`), intended use, band definitions and limitations. It exposes
no file paths, training data, player records or secrets.

## Requests

`POST /predict` accepts a **flat JSON object with exactly 42 required fields**.
`PredictionRequest` is generated directly from `FINAL_FEATURES`; there is no
second handwritten feature list in the API. Generated OpenAPI documents every
field, its required status and nullable numerical type.

All keys must be present. Values may be JSON numbers or **JSON `null`** for
genuine missingness. Null remains missing until the existing fitted median
imputer processes it. The API rejects numeric strings, booleans, arrays/objects
as feature values, missing fields, unknown fields and non-finite values.
JSON `NaN`, `Infinity` and `-Infinity` are not supported: use `null`, not these
nonstandard JSON tokens. Overflow such as `1e999` is rejected as non-finite.

Country, Language, Gender, UserID, snapshot_date and churn_60d are forbidden.
No labels or identifiers are needed, and extra fields are never silently ignored.
The API does not construct or verify the historical transaction windows; callers
must supply correctly prepared predictors available at the scoring snapshot.

### Synthetic example

`tests/fixtures/synthetic_prediction.json` is a **fully artificial schema fixture**,
not a copied bwin observation. Its score is a software demonstration, not a
meaningful assessment of a real person. From the project root:

```powershell
curl.exe -X POST "http://127.0.0.1:8000/predict" -H "Content-Type: application/json" --data-binary "@tests/fixtures/synthetic_prediction.json"
```

For a batch, send an object with a `records` array containing complete 42-field
feature objects. A valid synthetic batch file can be constructed locally without
placing feature records in shell arguments:

```python
import json
from pathlib import Path

record = json.loads(Path("tests/fixtures/synthetic_prediction.json").read_text())
batch = {"records": [record, {**record, "days_since_last_active_bet": 10}]}
# Optional manual synthetic fixture only; the API never saves requests itself.
Path("data/processed/synthetic_api_batch.json").write_text(json.dumps(batch))
```

Then post the ignored synthetic file with `--data-binary
"@data/processed/synthetic_api_batch.json"` to `/predict-batch`.

## Prediction responses

Single responses contain:

| Field | Type | Meaning |
| --- | --- | --- |
| churn_probability | number in [0,1] | Raw 60-day operational inactivity probability |
| diagnostic_prediction | 0 or 1 | 1 when probability >= **0.50** |
| risk_band | `unranked` | No independent single-person population percentile |
| model_version | string | Frozen model contract version |
| target_definition | string | Inactivity definition, not permanent departure |

Both endpoints retain the frozen raw-probability policy. No calibrator is used.
There is no request field to change the diagnostic threshold or select another
model. No fabricated explanation is returned.

Batch responses have this structure:

```text
{
  "predictions": [prediction row, prediction row, ...],
  "count": number of rows,
  "model_version": frozen version
}
```

Each prediction row contains the five fields above, with batch risk bands.
Result order exactly matches `records` order. There are no DataFrame indexes,
input features, player IDs or labels in the response.

## Single versus batch bands

Single predictions always use **`unranked`**. No frozen probability cutoffs exist
for assigning an independent observation to the population-percentile bands.

Batch allocation uses descending raw score, breaking ties by input position:

| Band | Frozen allocation |
| --- | --- |
| `very_high` | First ceil(n * 0.10) rows |
| `high` | Next rows through ceil(n * 0.20) |
| `medium` | Next rows through ceil(n * 0.50) |
| `low` | Remaining rows, approximately bottom 50% |

Bands are disjoint and depend on batch composition; equal scores may span bands.
Small batches can have empty bands. An explicitly submitted one-row **batch**
receives `very_high` solely because it is first within that cohort; that is not
evidence of an external top-10% rank. Use `/predict` for independent single scoring.
Use a coherent campaign cohort for meaningful prioritization. These bands are
not regulatory or clinical categories, and 0.50 is not a financial optimum.

## Configuration

Environment variables are read when the app is created; restart/reload after
changing settings. `.env.example` contains safe examples. No `.env` file is
loaded automatically or required for tests.

| Variable | Default | Notes |
| --- | --- | --- |
| `MODEL_PATH` | `models/final_xgboost_churn_model.joblib` | Relative paths resolve from the project root, not server working directory |
| `ALLOWED_ORIGINS` | `http://localhost:3000,http://localhost:5173` | Comma-separated explicit HTTP(S) origins; no wildcard |
| `MAX_BATCH_SIZE` | `1000` | Positive integer; controls validation and generated OpenAPI `maxItems` |

If `MODEL_PATH` is unset, the existing inference variable
`IGAMING_CHURN_MODEL_PATH` remains supported before the default path. An explicit
`MODEL_PATH` takes precedence. A relocated path must contain the **same verified
frozen artifact**, not an alternative model. The frozen specification and final
audit manifest must remain under `reports/` for integrity/schema verification.
There are no hard-coded machine-specific paths or secrets.

`ALLOWED_ORIGINS` ignores surrounding spaces and duplicate entries. Set it to
an empty string to allow no cross-origin browser clients. Origins must not
contain paths, query strings, fragments, credentials or `*`. Localhost defaults
are for development; set explicit deployment origins later. No Lovable URL is
assumed. Invalid configuration fails clearly at app creation.

CORS permits GET/POST with Content-Type/Accept and handles browser preflights.
Credentialed cross-origin requests are disabled. CORS is a browser policy,
not authentication or network access control; it does not expose the local API
publicly or prevent non-browser clients from making requests.

## Startup, health and errors

Lifespan startup checks the model through the cached inference loader. If it
cannot load, the API **still starts**: navigation, documentation and model info
remain available. `/health` returns 503 with `status="unavailable"` and
`model_loaded=false`; prediction calls return a sanitized 503. No training is
attempted. Health and prediction calls check again, allowing a restored artifact
to recover without retraining. A healthy response is:

```json
{
  "status": "healthy",
  "model_loaded": true,
  "model_version": "xgboost-60d-demographic-free-v1",
  "predictor_count": 42
}
```

Health checks load/verify the model and validate a synthetic feature schema;
they generate no prediction. Health indicates software/artifact readiness,
not current-world accuracy or governance approval.

| Status | Meaning |
| --- | --- |
| 200 | Successful health, information or prediction response |
| 422 | Invalid/missing/extra feature, malformed JSON, empty batch or batch above configured maximum |
| 503 | Model artifact/service unavailable or incompatible |
| 500 | Unexpected server error |

Error responses use a fixed `detail` message. They do not echo validation inputs,
unknown field names, exceptions, local file paths or stack traces. Batch-length
errors state the configured minimum/maximum. The health failure response uses
the documented health fields with a generic availability message.

## Privacy and limitations

The API adds no request-body logging, analytics, request persistence or automatic
prediction export. Uvicorn's ordinary access logs contain method/path/status,
not feature bodies; do not put sensitive values in URLs. Tests/examples use only
synthetic fixtures. No UserID is accepted as a predictor.

The score concerns extended inactivity, allows later reactivation, and does not
measure harm, responsible-gambling status, customer value or causality. Historical
2005-2007 data, repeated-player dependence, correlated predictors and temporal
drift limit transferability. Demographic removal is not a fairness guarantee.
Current-data validation, governance review and monitoring remain necessary
before operational use. See the frozen model card for full limitations.

## Verification

Run the complete existing and new test suite from the root:

```powershell
.\venv\Scripts\python.exe -m unittest discover -s tests -v
```

API tests cover endpoints, startup without a model, safe errors, strict/null
input validation, ordering, bands, configured batch bounds, CORS and generated
OpenAPI. The actual-model integration check submits a synthetic fixture only;
no historical test player is rescored and no model/policy is changed.

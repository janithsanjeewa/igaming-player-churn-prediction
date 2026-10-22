"""Pydantic HTTP schemas derived from the single frozen feature-name contract."""
from functools import lru_cache
from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, create_model

from src.model_contract import FINAL_FEATURES


class StrictRequest(BaseModel):
    model_config = ConfigDict(extra='forbid')


RealNumber = Annotated[float, Field(strict=True, allow_inf_nan=False)]

PredictionRequest = create_model(
    'PredictionRequest',
    __base__=StrictRequest,
    __doc__=(
        'All 42 prepared numerical features are required. Use JSON null for genuine '
        'missingness. Probability estimates 60-day operational inactivity. '
        'It does not represent permanent customer departure or gambling-harm risk.'
    ),
    **{name: (RealNumber | None, Field(..., description=(
        f'Frozen predictor {name}; real number or null for a genuinely missing value. '
        'Definition and units: docs/feature_dictionary.md.')))
       for name in FINAL_FEATURES},
)


@lru_cache(maxsize=8)
def batch_request_model(maximum: int):
    return create_model(
        'BatchPredictionRequest', __base__=StrictRequest,
        __doc__='One coherent scoring cohort; input order is retained and bands are relative to this batch.',
        records=(list[PredictionRequest], Field(..., min_length=1, max_length=maximum,
            description=f'Between 1 and {maximum} feature records, in desired result order.')),
    )


BatchPredictionRequest = batch_request_model(1000)


class PredictionResponse(BaseModel):
    model_config = ConfigDict(extra='forbid')
    churn_probability: float = Field(ge=0, le=1, description='Raw probability of 60-day operational inactivity; not permanent departure or gambling harm.')
    diagnostic_prediction: Literal[0, 1] = Field(description='1 when raw probability >= frozen diagnostic threshold 0.50.')
    risk_band: Literal['unranked', 'low', 'medium', 'high', 'very_high']
    model_version: str
    target_definition: str


class SinglePredictionResponse(PredictionResponse):
    risk_band: Literal['unranked'] = Field(description='No population percentile can be assigned to an independent single prediction.')


class BatchPredictionRow(PredictionResponse):
    risk_band: Literal['low', 'medium', 'high', 'very_high'] = Field(description='Relative band within the submitted batch; not a regulatory category.')


class BatchPredictionResponse(BaseModel):
    predictions: list[BatchPredictionRow]
    count: int = Field(ge=1)
    model_version: str


class HealthResponse(BaseModel):
    status: Literal['healthy', 'unavailable']
    model_loaded: bool
    model_version: str
    predictor_count: int
    detail: str | None = None


class ModelInfoResponse(BaseModel):
    model_family: str
    model_version: str
    target: str
    target_definition: str
    observation_window_days: int
    outcome_window_days: int
    predictor_count: int
    diagnostic_threshold: float
    calibration: Literal['none']
    demographics_excluded: bool
    metadata_excluded: bool
    intended_use: str
    single_risk_band: Literal['unranked']
    batch_risk_bands: dict[str, str]
    limitations: list[str]


class ErrorResponse(BaseModel):
    detail: str


class RootResponse(BaseModel):
    name: str
    version: str
    docs: str
    health: str

"""Local FastAPI wrapper for frozen inference; no training, storage or payload logs."""
from contextlib import asynccontextmanager

import pandas as pd
from fastapi import FastAPI, Request, Response
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from starlette.concurrency import run_in_threadpool

from api.config import Settings
from api.schemas import (BatchPredictionResponse, ErrorResponse, HealthResponse,
    ModelInfoResponse, PredictionRequest, RootResponse, SinglePredictionResponse,
    batch_request_model)
from src import inference
from src.model_contract import FINAL_FEATURES, MODEL_VERSION

TITLE = 'iGaming Player Churn Prediction API'
DESCRIPTION = (
    'API for estimating 60-day operational inactivity risk from a preceding '
    '30-day behavioural observation window. Probability estimates 60-day operational '
    'inactivity. It does not represent permanent customer departure or gambling-harm risk. '
    'Single scores are unranked; batch bands are relative to the submitted cohort.'
)
UNAVAILABLE = 'Model service is unavailable. Check the configured artifact and frozen manifests.'
INVALID = 'Invalid feature input. Supply all 42 frozen predictors as numbers or null, with no extra fields.'
SERVER_ERROR = 'An unexpected server error occurred.'
ERRORS = {422: {'model': ErrorResponse, 'description': 'Invalid request; payload values are not echoed.'},
          503: {'model': ErrorResponse, 'description': 'Verified frozen model is unavailable.'},
          500: {'model': ErrorResponse, 'description': 'Unexpected server error; internal details are withheld.'}}


def create_app(settings: Settings | None = None) -> FastAPI:
    settings = settings if settings is not None else Settings.from_environment()
    BatchRequest = batch_request_model(settings.max_batch_size)

    @asynccontextmanager
    async def lifespan(application):
        # Failure does not prevent documentation/metadata availability. Predictions
        # retry the cached loader so a restored artifact can recover without a fit.
        try:
            health = await run_in_threadpool(inference.model_health_check, model_path=settings.model_path)
            application.state.startup_model_available = bool(health.get('healthy'))
        except Exception:
            application.state.startup_model_available = False
        yield

    application = FastAPI(title=TITLE, description=DESCRIPTION, version=MODEL_VERSION,
                          lifespan=lifespan, debug=False)
    application.state.settings = settings
    application.add_middleware(CORSMiddleware, allow_origins=list(settings.allowed_origins),
                               allow_credentials=False, allow_methods=['GET', 'POST'],
                               allow_headers=['Content-Type', 'Accept'])

    @application.exception_handler(RequestValidationError)
    async def request_validation_error(request: Request, error: RequestValidationError):
        # Pydantic's default detail includes input values and unknown field names.
        # Return only fixed messages and the configured public batch capacity.
        batch_size_error = any(item['type'] in ('too_long', 'too_short') and
                               item['loc'] == ('body', 'records') for item in error.errors())
        message = (f'Batch must contain between 1 and {settings.max_batch_size} records.'
                   if batch_size_error else INVALID)
        return JSONResponse(status_code=422, content={'detail': message})

    @application.exception_handler(inference.FeatureValidationError)
    async def feature_validation_error(request: Request, error: inference.FeatureValidationError):
        return JSONResponse(status_code=422, content={'detail': INVALID})

    @application.exception_handler(inference.ModelLoadError)
    async def model_unavailable(request: Request, error: inference.ModelLoadError):
        return JSONResponse(status_code=503, content={'detail': UNAVAILABLE})

    @application.exception_handler(Exception)
    async def server_error(request: Request, error: Exception):
        return JSONResponse(status_code=500, content={'detail': SERVER_ERROR})

    @application.get('/', response_model=RootResponse, summary='Service navigation')
    def root():
        return dict(name=TITLE, version=MODEL_VERSION, docs='/docs', health='/health')

    @application.get('/health', response_model=HealthResponse, response_model_exclude_none=True,
                     responses={503: {'model': HealthResponse}, 500: ERRORS[500]},
                     summary='Artifact and schema readiness; no prediction generated')
    def health(response: Response):
        result = inference.model_health_check(model_path=settings.model_path)
        ready = bool(result.get('healthy'))
        response.status_code = 200 if ready else 503
        return dict(status='healthy' if ready else 'unavailable', model_loaded=ready,
                    model_version=MODEL_VERSION, predictor_count=len(FINAL_FEATURES),
                    detail=None if ready else UNAVAILABLE)

    @application.get('/model-info', response_model=ModelInfoResponse,
                     summary='Public frozen contract metadata; no player data')
    def model_info():
        return inference.get_model_info()

    @application.post('/predict', response_model=SinglePredictionResponse, responses=ERRORS,
                      summary='Score one independent observation; risk band is unranked')
    def predict(payload: PredictionRequest):
        return inference.predict_churn(payload.model_dump(), model_path=settings.model_path)

    @application.post('/predict-batch', response_model=BatchPredictionResponse, responses=ERRORS,
                      summary='Score a batch in input order and assign relative rank bands')
    def predict_batch(payload: BatchRequest):
        data = pd.DataFrame([record.model_dump() for record in payload.records],
                            columns=list(FINAL_FEATURES), dtype=object)
        result = inference.predict_churn_batch(data, model_path=settings.model_path)
        return dict(predictions=result.to_dict(orient='records'), count=len(result), model_version=MODEL_VERSION)

    return application


app = create_app()

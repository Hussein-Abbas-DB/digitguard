"""FastAPI application for the trained DigitGuard model."""

import os
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Annotated

import joblib
import sklearn
from fastapi import FastAPI, Query, Request
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
from sklearn.datasets import load_digits

from .core import ARTIFACT_VERSION, infer
from .train import split_indices

WEB = Path(__file__).with_name("web")
Pixel = Annotated[float, Field(strict=True, ge=0, le=16, allow_inf_nan=False)]
Digit = Annotated[int, Field(ge=0, le=9)]


class PredictionRequest(BaseModel):
    pixels: Annotated[list[Pixel], Field(min_length=64, max_length=64)]


class PredictionResponse(BaseModel):
    prediction: Digit | None
    confidence: float | None
    candidates: list[Digit]
    probabilities: list[float] | None
    reason: str


def create_app(model_path=None):
    @asynccontextmanager
    async def lifespan(app):
        path = Path(model_path or os.getenv("DIGITGUARD_MODEL", "artifacts/model.joblib"))
        if not path.is_file():
            raise RuntimeError("Model missing. Run python -m digitguard.train first.")
        bundle = joblib.load(path)
        if bundle["version"] != ARTIFACT_VERSION or bundle["sklearn_version"] != sklearn.__version__:
            raise RuntimeError("Model version mismatch. Retrain in this environment.")
        app.state.bundle = bundle
        pixels, labels = load_digits(return_X_y=True)
        test = split_indices(labels)["test"]
        app.state.examples = {
            digit: pixels[test[labels[test] == digit][0]].tolist() for digit in range(10)
        }
        yield

    app = FastAPI(title="DigitGuard", version="1.0.0", lifespan=lifespan)
    app.mount("/static", StaticFiles(directory=WEB), name="static")

    @app.get("/", include_in_schema=False)
    def index():
        return FileResponse(WEB / "index.html")

    @app.get("/healthz")
    def health(request: Request):
        return {"status": "ready", "model": request.app.state.bundle["name"]}

    @app.get("/v1/example")
    def example(request: Request, digit: Annotated[int, Query(ge=0, le=9)] = 3):
        return {"digit": digit, "pixels": request.app.state.examples[digit]}

    @app.post("/v1/predict", response_model=PredictionResponse)
    def predict(body: PredictionRequest, request: Request):
        return infer(request.app.state.bundle, body.pixels)

    return app


app = create_app()

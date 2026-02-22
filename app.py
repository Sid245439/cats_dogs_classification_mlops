"""
FastAPI Inference Service for Cats vs Dogs Classification.
Endpoints: /health, /predict, /metrics
"""

import logging
import os
import time
from pathlib import Path

from fastapi import FastAPI, File, HTTPException, Request, UploadFile
from fastapi.responses import Response
from prometheus_client import CONTENT_TYPE_LATEST, Counter, Histogram, generate_latest

# Setup logging
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()
LOG_DIR = Path("logs")
LOG_DIR.mkdir(exist_ok=True)

logging.basicConfig(
    level=LOG_LEVEL,
    format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
)
logger = logging.getLogger("cats_dogs_api")
file_handler = logging.FileHandler(LOG_DIR / "api.log")
file_handler.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(name)s - %(message)s"))
logger.addHandler(file_handler)

# Prometheus metrics
REQUEST_COUNT = Counter(
    "cats_dogs_api_requests_total",
    "Total API requests",
    ["method", "endpoint", "http_status"],
)
REQUEST_LATENCY = Histogram(
    "cats_dogs_api_request_latency_seconds",
    "Request latency (s)",
    ["endpoint"],
)

MODEL = None
CLASSES = ["cat", "dog"]


def load_model():
    global MODEL
    try:
        import tensorflow as tf
        model_path = Path("models/model.h5")
        if not model_path.exists():
            model_path = Path("models/model.keras")
        if model_path.exists():
            MODEL = tf.keras.models.load_model(str(model_path))
            logger.info("✓ Model loaded")
        else:
            logger.warning("No model file found at models/model.h5 or models/model.keras")
    except Exception as e:
        logger.error(f"Failed to load model: {e}")


app = FastAPI(title="Cats vs Dogs Prediction API")


@app.middleware("http")
async def log_and_measure(request: Request, call_next):
    start_time = time.perf_counter()
    status_code = 500
    try:
        response = await call_next(request)
        status_code = response.status_code
        return response
    finally:
        duration = time.perf_counter() - start_time
        endpoint = request.url.path
        REQUEST_COUNT.labels(
            method=request.method, endpoint=endpoint, http_status=status_code
        ).inc()
        REQUEST_LATENCY.labels(endpoint=endpoint).observe(duration)
        logger.info(f"{request.method} {endpoint} status={status_code} duration_ms={duration*1000:.2f}")


@app.on_event("startup")
async def startup_event():
    load_model()


@app.get("/health")
async def health():
    return {"status": "healthy", "model_loaded": MODEL is not None}


@app.get("/metrics")
async def metrics():
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    if MODEL is None:
        raise HTTPException(500, "Model not loaded")
    try:
        contents = await file.read()
        from src.preprocessing import preprocess_for_inference
        img_array = preprocess_for_inference(contents)
        probs = MODEL.predict(img_array, verbose=0)[0]
        pred_idx = int(probs.argmax())
        label = CLASSES[pred_idx]
        prob = float(probs[pred_idx])
        logger.info(f"prediction={label} prob={prob:.3f}")
        return {
            "label": label,
            "class_id": pred_idx,
            "probabilities": {CLASSES[i]: float(probs[i]) for i in range(len(CLASSES))},
            "confidence": prob,
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.exception("Prediction failed")
        raise HTTPException(400, str(e))


@app.get("/")
async def root():
    return {
        "service": "Cats vs Dogs Prediction API",
        "endpoints": {"/health": "GET", "/predict": "POST (multipart image)", "/metrics": "GET"},
    }

# uvicorn src.api:app --reload
"""
🩺 Clinical Heart Disease AI - FastAPI REST API
World-class API for serving heart-disease predictions and SHAP explainability.

Author: Ridwan Oladipo, MD | AI Specialist
"""

from datetime import datetime
import logging
import time
import traceback
from typing import Any, Dict, List

import numpy as np
import uvicorn
from fastapi import FastAPI, HTTPException, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field, field_validator

from src.model import initialize_model, predict_heart_disease, predictor

# =========================
# Logging
# =========================
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%d-%m-%Y | %I:%M%p",
    handlers=[logging.FileHandler("outputs/api.log"), logging.StreamHandler()],
)
logger = logging.getLogger(__name__)

# =========================
# FastAPI app
# =========================
app = FastAPI(
    title="Clinical Heart Disease AI API",
    description="Production API for heart disease prediction & clinical-grade explainability",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.middleware("http")
async def add_process_time_header(request: Request, call_next):
    start = time.time()
    response = await call_next(request)
    logger.info(f"{request.method} {request.url.path} in {(time.time()-start)*1000:.2f} ms")
    return response

# =========================
# Schemas
# =========================
class PatientInput(BaseModel):
    age: int = Field(..., ge=18, le=101)
    sex: int = Field(..., ge=0, le=1)
    cp: int = Field(..., ge=0, le=3)
    trestbps: int = Field(..., ge=80, le=301)
    chol: int = Field(..., ge=100, le=601)
    fbs: int = Field(..., ge=0, le=1)
    restecg: int = Field(..., ge=0, le=2)
    thalach: int = Field(..., ge=60, le=221)
    exang: int = Field(..., ge=0, le=1)
    oldpeak: float = Field(..., ge=0.0, le=10.1)
    slope: int = Field(..., ge=0, le=2)
    ca: int = Field(..., ge=0, le=3)
    thal: int = Field(..., ge=1, le=3)

    @field_validator("thalach")
    @classmethod
    def validate_hr_vs_age(cls, v, info):
        age = info.data.get("age")
        if age and v > (220 - age) + 20:
            raise ValueError(f"Heart rate {v} unusually high for age {age}")
        return v

# =========================
# State & helpers
# =========================
model_loaded = False
startup_time = None

def current_time_iso():
    return datetime.now().isoformat()

# =========================
# Startup & shutdown
# =========================
@app.on_event("startup")
async def startup_event():
    global model_loaded, startup_time
    startup_time = current_time_iso()
    logger.info("Starting API...")
    try:
        model_loaded = initialize_model()
        logger.info("Model ready" if model_loaded else "Model failed to load")
    except Exception as e:
        logger.error(f"Startup error: {e}")
        logger.error(traceback.format_exc())

@app.on_event("shutdown")
async def shutdown_event():
    logger.info("Shutting down API...")

# =========================
# Info & health routes
# =========================
@app.get("/", summary="API overview", tags=["App Info"])
async def root():
    return {
        "app": "Clinical Heart Disease AI",
        "purpose": "Machine learning API for coronary artery disease risk prediction with explainability.",
        "model": {
            "type": "XGBoost (optimized)",
            "explainability": "SHAP",
            "performance": {"roc_auc": 0.91, "sensitivity": 0.97, "specificity": 0.71},
        },
        "author": "Ridwan Oladipo, MD | AI Specialist",
        "version": "1.0.0",
        "documentation": "/docs",
    }

@app.get("/health", tags=["Health"])
async def health_check():
    return {
        "status": "ok" if predictor.model else "error",
        "model_loaded": bool(predictor.model),
        "version": "1.0.0",
        "startup_time": startup_time,
        "timestamp": current_time_iso(),
    }

@app.get("/metrics", tags=["Model Metrics"])
async def get_model_metrics():
    if not model_loaded:
        raise HTTPException(status.HTTP_503_SERVICE_UNAVAILABLE, detail="Model not loaded")
    try:
        logger.info("Metrics requested")
        m = predictor.get_metrics()
        return {
            "model": m["model"],
            "roc_auc": m["roc_auc"],
            "accuracy": m["accuracy"],
            "sensitivity": m["sensitivity"],
            "specificity": m["specificity"],
            "ppv": m["ppv"],
            "npv": m["npv"],
            "confusion_matrix": m["confusion_matrix"],
        }
    except Exception as e:
        logger.error(f"Metrics retrieval error: {e}")
        logger.error(traceback.format_exc())
        raise HTTPException(status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Metrics retrieval failed")

# =========================
# Prediction endpoints
# =========================
@app.post("/predict", tags=["Predictions"])
async def predict(patient: PatientInput):
    if not model_loaded:
        raise HTTPException(status.HTTP_503_SERVICE_UNAVAILABLE, detail="Model not loaded")
    results = predict_heart_disease(patient.dict())
    p = results["prediction"]
    return {
        "probability": p["probability"],
        "risk_class": p["risk_class"],
        "clinical_summary": p["clinical_summary"],
        "timestamp": current_time_iso(),
    }

@app.post("/shap", tags=["Explainability"])
async def get_shap_explanation(patient: PatientInput):
    results = predict_heart_disease(patient.dict())
    e = results["explanations"]
    shap_vals = [float(v) for v in e["shap_values"][0]]
    top_feats = [
        {k: float(v) if isinstance(v, (np.float32, np.float64)) else v for k, v in feat.items()}
        for feat in e["top_features"]
    ]
    return {
        "shap_values": shap_vals,
        "top_features": top_feats,
        "feature_names": results["model_info"]["feature_names"],
        "timestamp": current_time_iso(),
    }

@app.post("/positions", tags=["Patient Comparisons"])
async def get_feature_positions_route(patient: PatientInput):
    results = predict_heart_disease(patient.dict())
    c = results["comparisons"]
    return {
        "feature_positions": c["feature_positions"],
        "guideline_categories": c["guideline_categories"],
        "timestamp": current_time_iso(),
    }

# =========================
# Uvicorn entry
# =========================
if __name__ == "__main__":
    uvicorn.run("api:app", host="0.0.0.0", port=8000, reload=True, log_level="info")

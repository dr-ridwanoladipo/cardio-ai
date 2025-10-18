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

def current_time_iso():
    return datetime.now().isoformat()

# =========================
# Startup
# =========================
@app.on_event("startup")
async def startup_event():
    global model_loaded
    try:
        model_loaded = initialize_model()
        logger.info("Model ready" if model_loaded else "Model failed to load")
    except Exception as e:
        logger.error(f"Startup error: {e}")
        logger.error(traceback.format_exc())

# =========================
# Prediction endpoints
# =========================
@app.post("/predict", tags=["Predictions"])
async def predict(patient: PatientInput):
    if not model_loaded:
        raise HTTPException(status.HTTP_503_SERVICE_UNAVAILABLE, detail="Model not loaded")
    try:
        logger.info(f"Predict: age={patient.age}, sex={patient.sex}")
        results = predict_heart_disease(patient.dict())
        p = results["prediction"]
        return {
            "probability": p["probability"],
            "risk_class": p["risk_class"],
            "clinical_summary": p["clinical_summary"],
            "timestamp": current_time_iso(),
        }
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        logger.error(traceback.format_exc())
        raise HTTPException(status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Prediction failed")

@app.post("/shap", tags=["Explainability"])
async def get_shap_explanation(patient: PatientInput):
    if not predictor.model:
        raise HTTPException(status.HTTP_503_SERVICE_UNAVAILABLE, detail="Model not loaded")
    try:
        logger.info(f"SHAP: age={patient.age}")
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
    except Exception as e:
        logger.error(f"SHAP error: {e}")
        logger.error(traceback.format_exc())
        raise HTTPException(status.HTTP_500_INTERNAL_SERVER_ERROR, detail="SHAP explanation failed")

@app.post("/positions", tags=["Patient Comparisons"])
async def get_feature_positions_route(patient: PatientInput):
    if not model_loaded:
        raise HTTPException(status.HTTP_503_SERVICE_UNAVAILABLE, detail="Model not loaded")
    try:
        logger.info(f"Positions: age={patient.age}")
        results = predict_heart_disease(patient.dict())
        c = results["comparisons"]
        return {
            "feature_positions": c["feature_positions"],
            "guideline_categories": c["guideline_categories"],
            "timestamp": current_time_iso(),
        }
    except Exception as e:
        logger.error(f"Position analysis error: {e}")
        logger.error(traceback.format_exc())
        raise HTTPException(status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Position analysis failed")

# =========================
# Uvicorn entry
# =========================
if __name__ == "__main__":
    uvicorn.run("api:app", host="0.0.0.0", port=8000, reload=True, log_level="info")

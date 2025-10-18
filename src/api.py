# uvicorn src.api:app --reload
"""
🩺 Clinical Heart Disease AI - FastAPI REST API
World-class API for serving heart-disease predictions and SHAP explainability.

Author: Ridwan Oladipo, MD | AI Specialist
"""

from datetime import datetime
import logging
import time
from typing import Any, Dict, List

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, field_validator

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

# =========================
# Middleware
# =========================
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
    duration = (time.time() - start) * 1000
    logger.info(f"{request.method} {request.url.path} in {duration:.2f} ms")
    return response

# =========================
# Schemas
# =========================
class PatientInput(BaseModel):
    age:      int   = Field(..., ge=18,  le=101, description="Patient age in years (18-100)")
    sex:      int   = Field(..., ge=0,   le=1,   description="Sex (0=Female, 1=Male)")
    cp:       int   = Field(..., ge=0,   le=3,   description="Chest pain type")
    trestbps: int   = Field(..., ge=80,  le=301, description="Resting blood pressure (80-300 mmHg)")
    chol:     int   = Field(..., ge=100, le=601, description="Cholesterol (100-600 mg/dl)")
    fbs:      int   = Field(..., ge=0,   le=1,   description="Fasting blood sugar >120 mg/dl")
    restecg:  int   = Field(..., ge=0,   le=2,   description="Resting ECG result")
    thalach:  int   = Field(..., ge=60,  le=221, description="Max heart rate achieved")
    exang:    int   = Field(..., ge=0,   le=1,   description="Exercise induced angina")
    oldpeak:  float = Field(..., ge=0.0, le=10.1,description="ST depression")
    slope:    int   = Field(..., ge=0,   le=2,   description="Slope of ST segment")
    ca:       int   = Field(..., ge=0,   le=3,   description="Number of major vessels")
    thal:     int   = Field(..., ge=1,   le=3,   description="Thalassemia status")

    @field_validator("thalach")
    @classmethod
    def validate_hr_vs_age(cls, v, info):
        age = info.data.get("age")
        if age and v > (220 - age) + 20:
            raise ValueError(f"Heart rate {v} unusually high for age {age}")
        return v

    class Config:
        json_schema_extra = {
            "example": {
                "age": 63, "sex": 1, "cp": 0, "trestbps": 145, "chol": 233,
                "fbs": 1, "restecg": 0, "thalach": 150, "exang": 0,
                "oldpeak": 2.3, "slope": 0, "ca": 0, "thal": 1
            }
        }

# =========================
# Base route
# =========================
@app.get("/", tags=["App Info"])
async def root():
    return {
        "app": "Clinical Heart Disease AI",
        "version": "1.0.0",
        "message": "API scaffold initialized successfully.",
    }

# =========================
# Uvicorn entry
# =========================
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("api:app", host="0.0.0.0", port=8000, reload=True, log_level="info")

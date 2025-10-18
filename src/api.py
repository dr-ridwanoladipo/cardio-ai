# uvicorn src.api:app --reload
"""
🩺 Clinical Heart Disease AI - FastAPI REST API
World-class API for serving heart-disease predictions and SHAP explainability.

Author: Ridwan Oladipo, MD | AI Specialist
"""

from datetime import datetime
import logging
import time
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware

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

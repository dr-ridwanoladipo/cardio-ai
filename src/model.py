"""
🩺 Clinical Heart Disease AI - Model Module
Reusable prediction and explanation functions for both FastAPI and Streamlit.

Author: Ridwan Oladipo, MD | AI Specialist
"""

import json
import logging
import warnings
from typing import Dict, Any

import joblib
import pandas as pd

warnings.filterwarnings("ignore")

# ── Logging ────────────────────────────────────────────────────────────────────
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════════════
#                            Core Predictor Class
# ═══════════════════════════════════════════════════════════════════════════════
class HeartDiseasePredictor:
    """Clinical Heart Disease Prediction and Explanation System."""

    def __init__(self):
        """Create empty placeholders; populate later via load_artifacts()."""
        self.model = None
        self.scaler = None
        self.explainer = None
        self.feature_names = None
        self.numerical_features = None
        self.metrics = None
        self.cohort_data = None

    def load_artifacts(self) -> bool:
        """Load model, scaler, explainer, feature lists, metrics, cohort data."""
        try:
            logger.info("Loading model artifacts...")

            # Core model components
            self.model = joblib.load("artifacts/xgb_model.pkl")
            self.scaler = joblib.load("artifacts/scaler.pkl")
            self.explainer = joblib.load("artifacts/shap_explainer.pkl")

            # Feature metadata
            with open("artifacts/feature_names.json") as f:
                self.feature_names = json.load(f)
            with open("artifacts/numerical_features.json") as f:
                self.numerical_features = json.load(f)

            # Metrics
            with open("artifacts/metrics.json") as f:
                self.metrics = json.load(f)

            # Cohort data for population comparisons
            self.cohort_data = pd.read_csv("data/heartdisease_processed.csv")

            logger.info("All artifacts loaded successfully")
            return True

        except Exception as e:
            logger.error(f"Failed to load artifacts: {str(e)}")
            return False


# ═══════════════════════════════════════════════════════════════════════════════
#                        Global instance + Convenience wrapper
# ═══════════════════════════════════════════════════════════════════════════════
predictor = HeartDiseasePredictor()


def initialize_model() -> bool:
    """Initialize the global predictor instance."""
    return predictor.load_artifacts()

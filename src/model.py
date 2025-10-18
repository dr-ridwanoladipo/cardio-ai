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

    # ───────────────────────────────────────────────────────────────────────────
    # Artifact loading
    # ───────────────────────────────────────────────────────────────────────────
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

    # ───────────────────────────────────────────────────────────────────────────
    # Derived feature engineering
    # ───────────────────────────────────────────────────────────────────────────
    def compute_auto_fields(self, patient_dict: Dict[str, Any]) -> Dict[str, Any]:
        """Compute derived clinical fields from basic patient inputs."""
        enhanced_patient = patient_dict.copy()

        # Age group
        age = patient_dict['age']
        if age < 40:
            enhanced_patient['age_group'] = 0  # Young
        elif age < 55:
            enhanced_patient['age_group'] = 1  # Middle-aged
        elif age < 65:
            enhanced_patient['age_group'] = 2  # Older
        else:
            enhanced_patient['age_group'] = 3  # Elderly

        # Chest pain severity
        cp_severity_map = {0: 4, 1: 3, 2: 2, 3: 1}
        enhanced_patient['cp_severity'] = cp_severity_map[patient_dict['cp']]

        # Blood pressure category
        bp = patient_dict['trestbps']
        if bp < 120:
            enhanced_patient['bp_category'] = 0
        elif bp < 130:
            enhanced_patient['bp_category'] = 1
        elif bp < 140:
            enhanced_patient['bp_category'] = 2
        else:
            enhanced_patient['bp_category'] = 3

        # Cholesterol risk category
        chol = patient_dict['chol']
        if chol < 200:
            enhanced_patient['chol_risk'] = 0
        elif chol < 240:
            enhanced_patient['chol_risk'] = 1
        else:
            enhanced_patient['chol_risk'] = 2

        # Heart rate achievement
        enhanced_patient['hr_achievement'] = patient_dict['thalach'] / (220 - age)

        # Interaction terms
        enhanced_patient['age_chol_interaction'] = age * chol / 1000
        enhanced_patient['cp_exang_interaction'] = patient_dict['cp'] * patient_dict['exang']

        return enhanced_patient


# ═══════════════════════════════════════════════════════════════════════════════
#                        Global instance + Convenience wrapper
# ═══════════════════════════════════════════════════════════════════════════════
predictor = HeartDiseasePredictor()


def initialize_model() -> bool:
    """Initialize the global predictor instance."""
    return predictor.load_artifacts()

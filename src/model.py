"""
🩺 Clinical Heart Disease AI - Model Module
Reusable prediction and explanation functions for both FastAPI and Streamlit.

Author: Ridwan Oladipo, MD | AI Specialist
"""

import json
import logging
import warnings
from typing import Dict, Any, Tuple, List

import joblib
import pandas as pd
import numpy as np

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

    # ───────────────────────────────────────────────────────────────────────────
    # Prediction & clinical summary
    # ───────────────────────────────────────────────────────────────────────────
    def predict_proba(self, input_df: pd.DataFrame) -> Tuple[float, str, str]:
        """Predict probability and return risk classification + summary."""
        if self.model is None:
            raise ValueError("Model not loaded. Call load_artifacts() first.")

        input_scaled = input_df.copy()
        input_scaled[self.numerical_features] = self.scaler.transform(input_df[self.numerical_features])
        probability = self.model.predict_proba(input_scaled)[0, 1]

        if probability < 0.3:
            risk_class = "Low Risk"
        elif probability < 0.7:
            risk_class = "Moderate Risk"
        else:
            risk_class = "High Risk"

        clinical_summary = self._generate_clinical_summary(probability, input_df.iloc[0])
        return probability, risk_class, clinical_summary

    def _generate_clinical_summary(self, probability: float, patient_data: pd.Series) -> str:
        """Generate a clinical summary based on prediction and key features."""
        risk_pct = probability * 100
        high_risk_factors = []

        if patient_data['ca'] >= 2:
            high_risk_factors.append("multi-vessel coronary disease")
        if patient_data['thal'] == 3:
            high_risk_factors.append("reversible perfusion defect")
        if patient_data['exang'] == 1:
            high_risk_factors.append("exercise-induced angina")
        if patient_data['slope'] == 2:
            high_risk_factors.append("downsloping ST segment")
        if patient_data['cp'] == 0:
            high_risk_factors.append("typical angina")

        if probability >= 0.7:
            summary = (
                f"🟥 **High Risk** ({risk_pct:.1f}%): likely driven by "
                f"{', '.join(high_risk_factors[:2]) if high_risk_factors else 'multiple adverse factors'}."
            )
        elif probability >= 0.3:
            summary = f"🟧 **Moderate Risk** ({risk_pct:.1f}%): further evaluation recommended."
        else:
            summary = f"🟩 **Low Risk** ({risk_pct:.1f}%): maintain heart-healthy lifestyle."
        return summary

    # ───────────────────────────────────────────────────────────────────────────
    # SHAP explainability
    # ───────────────────────────────────────────────────────────────────────────
    def get_shap_values(self, input_df: pd.DataFrame) -> np.ndarray:
        """Calculate SHAP values for explanation."""
        if self.explainer is None:
            raise ValueError("SHAP explainer not loaded. Call load_artifacts() first.")

        input_scaled = input_df.copy()
        input_scaled[self.numerical_features] = self.scaler.transform(input_df[self.numerical_features])
        shap_values = self.explainer.shap_values(input_scaled)
        return shap_values

    def get_top_features(self, shap_values: np.ndarray, top_n: int = 5) -> List[Dict[str, Any]]:
        """Return top contributing features with explanations."""
        feature_contributions = []
        for i, feature_name in enumerate(self.feature_names):
            contribution = shap_values[0, i]
            feature_contributions.append({
                'feature': feature_name,
                'shap_value': float(contribution),
                'abs_contribution': abs(contribution),
                'impact': 'Increases Risk' if contribution > 0 else 'Decreases Risk'
            })
        feature_contributions.sort(key=lambda x: x['abs_contribution'], reverse=True)
        return feature_contributions[:top_n]


# ═══════════════════════════════════════════════════════════════════════════════
#                        Global instance + Convenience wrapper
# ═══════════════════════════════════════════════════════════════════════════════
predictor = HeartDiseasePredictor()


def initialize_model() -> bool:
    """Initialize the global predictor instance."""
    return predictor.load_artifacts()

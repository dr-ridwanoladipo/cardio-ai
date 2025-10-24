# $env:PYTHONPATH="."
# streamlit run src/app.py
"""
🩺 Clinical Heart Disease AI - Streamlit Application
Cardiovascular risk assessment with AI-powered explainability

Author: Ridwan Oladipo, MD | AI Specialist
"""

# ── Imports ────────────────────────────────────────────────────────────────────
import streamlit as st
from src.app_helpers import load_custom_css, check_api_health, get_sample_patients

# ================ 🛠 SIDEBAR TOGGLE ================
if 'sidebar_state' not in st.session_state:
    st.session_state.sidebar_state = 'expanded'

st.set_page_config(
    page_title="Clinical Heart Disease AI",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state=st.session_state.sidebar_state
)

if st.button("🩺", help="Toggle sidebar"):
    st.session_state.sidebar_state = (
        'collapsed' if st.session_state.sidebar_state == 'expanded' else 'expanded'
    )
    st.rerun()

st.markdown(
    '<div style="font-size:0.75rem; color:#6b7280; margin-top:-10px;">Menu</div>',
    unsafe_allow_html=True
)

# ================ 💅 LOAD CUSTOM STYLING ================
load_custom_css()

# ================ 🏥 MAIN APPLICATION ================
def main():
    """Main Streamlit application"""
    st.markdown("""
    <div class="medical-header">
        <h1>🩺 Clinical Heart Disease AI</h1>
        <p>Cardiovascular risk assessment with AI-powered explainability</p>
        <p><strong>By Ridwan Oladipo, MD | AI Specialist</strong></p>
    </div>
    """, unsafe_allow_html=True)

    # ---------- 🔧 API HEALTH CHECK ----------
    health_status = check_api_health()

    if not health_status:
        st.error("🚨 **API Connection Failed** - Please ensure the FastAPI service is running on localhost:8000")
        st.code("uvicorn src.api:app --reload", language="bash")
        st.stop()

    if not health_status.get('model_loaded', False):
        st.error("🚨 **Model Not Loaded** - Please check API logs")
        st.stop()

    st.markdown("""
    <div style="background-color: #d4edda; border: 1px solid #c3e6cb; color: #155724; 
               padding: 0.5rem 1rem; margin-bottom: 1rem; border-radius: 0.375rem; font-size: 0.85rem;">
    ✅ <strong>System Online</strong> - Model loaded and ready for predictions
    </div>
    """, unsafe_allow_html=True)

    # ---------- 🩺 PATIENT INPUT PANEL ----------
    st.markdown("## 🩺 Patient Input Panel")

    with st.container():
        # Sidebar for sample data
        with st.sidebar:
            st.markdown("### 🎯 Quick Demo")
            sample_patients = get_sample_patients()
            selected_sample = st.selectbox(
                "Load Sample Patient:",
                ["Custom Input"] + list(sample_patients.keys())
            )

            if st.button("🔄 Load Sample Data"):
                if selected_sample != "Custom Input":
                    for key, value in sample_patients[selected_sample].items():
                        st.session_state[key] = value
                    st.rerun()

            st.markdown("---")
            st.markdown("### ℹ️ About This Tool")
            st.markdown("""
            This AI system predicts coronary artery disease risk using:
            - **XGBoost** machine learning model  
            - **SHAP** explainable AI  
            - **Clinical guidelines** integration  
            - **Population comparisons**
            """)

        # Main input form
        col1, col2 = st.columns(2)

        with col1:
            st.markdown("### 👤 Demographics & Vitals")

            age = st.slider("Age (years)", 18, 100,
                            st.session_state.get('age', 54),
                            help="Patient's age in years")

            sex = st.selectbox("Sex",
                               options=[0, 1],
                               format_func=lambda x: "Female" if x == 0 else "Male",
                               index=st.session_state.get('sex', 1))

            trestbps = st.slider("Resting Blood Pressure (mmHg)", 80, 300,
                                 st.session_state.get('trestbps', 132),
                                 help="Resting blood pressure measurement")

            chol = st.slider("Cholesterol (mg/dl)", 100, 600,
                             st.session_state.get('chol', 246),
                             help="Serum cholesterol level")

            fbs = st.selectbox("Fasting Blood Sugar > 120 mg/dl",
                               options=[0, 1],
                               format_func=lambda x: "No" if x == 0 else "Yes",
                               index=st.session_state.get('fbs', 0))

            thalach = st.slider("Maximum Heart Rate", 60, 220,
                                st.session_state.get('thalach', 150),
                                help="Maximum heart rate achieved during exercise")

        with col2:
            st.markdown("### ❤️ Cardiac Assessment")

            cp = st.selectbox("Chest Pain Type",
                              options=[0, 1, 2, 3],
                              format_func=lambda x: ["Typical Angina", "Atypical Angina",
                                                     "Non-anginal Pain", "Asymptomatic"][x],
                              index=st.session_state.get('cp', 0))

            restecg = st.selectbox("Resting ECG",
                                   options=[0, 1, 2],
                                   format_func=lambda x: ["Normal", "ST-T Abnormality",
                                                          "LV Hypertrophy"][x],
                                   index=st.session_state.get('restecg', 0))

            exang = st.selectbox("Exercise Induced Angina",
                                 options=[0, 1],
                                 format_func=lambda x: "No" if x == 0 else "Yes",
                                 index=st.session_state.get('exang', 0))

            oldpeak = st.slider("ST Depression", 0.0, 10.0,
                                st.session_state.get('oldpeak', 1.0),
                                step=0.1,
                                help="ST depression induced by exercise")

            slope = st.selectbox("ST Slope",
                                 options=[0, 1, 2],
                                 format_func=lambda x: ["Upsloping", "Flat", "Downsloping"][x],
                                 index=st.session_state.get('slope', 1))

            ca = st.selectbox("Major Vessels (Angiography)",
                              options=[0, 1, 2, 3],
                              format_func=lambda x: f"{x} vessels",
                              index=st.session_state.get('ca', 0),
                              help="Number of major coronary vessels with significant stenosis")

            thal = st.selectbox("Thalassemia",
                                options=[1, 2, 3],
                                format_func=lambda x: ["Normal", "Fixed Defect", "Reversible Defect"][x - 1],
                                index=st.session_state.get('thal', 2) - 1)

    st.success("✅ Patient input section active and ready for predictions.")

# ================ 🚀 ENTRY POINT ================
if __name__ == "__main__":
    main()

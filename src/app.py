# $env:PYTHONPATH="."
# streamlit run src/app.py
"""
🩺 Clinical Heart Disease AI - Streamlit Application
Cardiovascular risk assessment with AI-powered explainability

Author: Ridwan Oladipo, MD | AI Specialist
"""

# ── Imports ────────────────────────────────────────────────────────────────────
import streamlit as st
from markdown import markdown
from src.app_helpers import (
    load_custom_css, check_api_health, get_sample_patients,
    call_api_endpoint, create_risk_gauge, create_shap_waterfall
)

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
            - **XGBoost** model  
            - **SHAP** explainable AI  
            - **Clinical guidelines** integration
            """)

        col1, col2 = st.columns(2)

        with col1:
            st.markdown("### 👤 Demographics & Vitals")
            age = st.slider("Age (years)", 18, 100, st.session_state.get('age', 54))
            sex = st.selectbox("Sex", [0, 1],
                               format_func=lambda x: "Female" if x == 0 else "Male",
                               index=st.session_state.get('sex', 1))
            trestbps = st.slider("Resting Blood Pressure (mmHg)", 80, 300, st.session_state.get('trestbps', 132))
            chol = st.slider("Cholesterol (mg/dl)", 100, 600, st.session_state.get('chol', 246))
            fbs = st.selectbox("Fasting Blood Sugar > 120 mg/dl", [0, 1],
                               format_func=lambda x: "No" if x == 0 else "Yes",
                               index=st.session_state.get('fbs', 0))
            thalach = st.slider("Maximum Heart Rate", 60, 220, st.session_state.get('thalach', 150))

        with col2:
            st.markdown("### ❤️ Cardiac Assessment")
            cp = st.selectbox("Chest Pain Type",
                              [0, 1, 2, 3],
                              format_func=lambda x: ["Typical Angina", "Atypical Angina",
                                                     "Non-anginal Pain", "Asymptomatic"][x],
                              index=st.session_state.get('cp', 0))
            restecg = st.selectbox("Resting ECG",
                                   [0, 1, 2],
                                   format_func=lambda x: ["Normal", "ST-T Abnormality", "LV Hypertrophy"][x],
                                   index=st.session_state.get('restecg', 0))
            exang = st.selectbox("Exercise Induced Angina",
                                 [0, 1],
                                 format_func=lambda x: "No" if x == 0 else "Yes",
                                 index=st.session_state.get('exang', 0))
            oldpeak = st.slider("ST Depression", 0.0, 10.0, st.session_state.get('oldpeak', 1.0), step=0.1)
            slope = st.selectbox("ST Slope", [0, 1, 2],
                                 format_func=lambda x: ["Upsloping", "Flat", "Downsloping"][x],
                                 index=st.session_state.get('slope', 1))
            ca = st.selectbox("Major Vessels (Angiography)", [0, 1, 2, 3],
                              format_func=lambda x: f"{x} vessels",
                              index=st.session_state.get('ca', 0))
            thal = st.selectbox("Thalassemia", [1, 2, 3],
                                format_func=lambda x: ["Normal", "Fixed Defect", "Reversible Defect"][x - 1],
                                index=st.session_state.get('thal', 2) - 1)

    patient_data = {
        'age': age, 'sex': sex, 'cp': cp, 'trestbps': trestbps, 'chol': chol,
        'fbs': fbs, 'restecg': restecg, 'thalach': thalach, 'exang': exang,
        'oldpeak': oldpeak, 'slope': slope, 'ca': ca, 'thal': thal
    }
    for k, v in patient_data.items():
        st.session_state[k] = v

    st.markdown("---")

    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        predict_button = st.button("🔮 **Analyze Cardiovascular Risk**",
                                   use_container_width=True, type="primary")

    if predict_button:
        with st.spinner("🧠 Analyzing patient data..."):
            prediction_data, pred_error = call_api_endpoint("predict", patient_data)
            shap_data, shap_error = call_api_endpoint("shap", patient_data)

        if pred_error:
            st.error(f"❌ {pred_error}")
            return

        st.markdown("## 📊 Risk Assessment")
        col1a, col1b = st.columns([1, 1])

        with col1a:
            st.markdown('<div class="risk-gauge-container">', unsafe_allow_html=True)
            risk_fig = create_risk_gauge(prediction_data['probability'], prediction_data['risk_class'])
            st.plotly_chart(risk_fig, use_container_width=True)
            st.markdown('</div>', unsafe_allow_html=True)

        with col1b:
            risk_class = prediction_data['risk_class']
            probability = prediction_data['probability']
            box = "risk-low" if "Low" in risk_class else "risk-moderate" if "Moderate" in risk_class else "risk-high"
            st.markdown(f'<div class="{box}">{risk_class}<br>{probability:.1%} Risk</div>', unsafe_allow_html=True)

        # ---------- 🧠 SHAP EXPLAINABILITY ----------
        if shap_data and not shap_error:
            st.markdown("## 🧠 Explainable AI Dashboard")
            s_col1, s_col2 = st.columns([2, 1])

            with s_col1:
                shap_fig = create_shap_waterfall(shap_data)
                st.plotly_chart(shap_fig, use_container_width=True)

            with s_col2:
                st.markdown("### 🎯 Top Risk Drivers")
                for feature in shap_data['top_features'][:5]:
                    impact_class = "feature-increase" if feature['shap_value'] > 0 else "feature-decrease"
                    st.markdown(f"""
                    <div class="feature-card {impact_class}">
                        <strong>{feature['feature'].replace('_', ' ').title()}</strong><br>
                        <small>{feature['clinical_explanation']}</small><br>
                        <span style="font-size: 0.8em;">Impact: {abs(feature['shap_value']):.3f}</span>
                    </div>
                    """, unsafe_allow_html=True)

        # ---------- ⚕️ CLINICAL SUMMARY ----------
        summary_html = markdown(prediction_data['clinical_summary'])
        st.markdown(f"""
        <div class="clinical-summary">
            <h4>🩺 Clinical Interpretation & Recommendations</h4>
            {summary_html}
        </div>
        """, unsafe_allow_html=True)

# ================ 🚀 ENTRY POINT ================
if __name__ == "__main__":
    main()

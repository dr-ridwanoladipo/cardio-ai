# $env:PYTHONPATH="."
# streamlit run src/app.py
"""
🩺 Clinical Heart Disease AI - Streamlit Application
Cardiovascular risk assessment with AI-powered explainability

Author: Ridwan Oladipo, MD | AI Specialist
"""

# ── Imports ────────────────────────────────────────────────────────────────────
import streamlit as st
from src.app_helpers import load_custom_css

# ================ 🔧 PAGE CONFIGURATION ================
st.set_page_config(
    page_title="Clinical Heart Disease AI",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="expanded"
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

    st.success("✅ App initialized successfully! Ready for next development phase.")

# ================ 🚀 ENTRY POINT ================
if __name__ == "__main__":
    main()

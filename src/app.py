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

    st.success("✅ App initialized successfully with sidebar toggle and layout ready.")

# ================ 🚀 ENTRY POINT ================
if __name__ == "__main__":
    main()

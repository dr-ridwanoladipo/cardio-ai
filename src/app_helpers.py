"""
🩺 Clinical Heart Disease AI - Streamlit Helper Functions
Utility functions, API calls, visualizations, and styling for the Streamlit app

Author: Ridwan Oladipo, MD | AI Specialist
"""

# ── Imports ────────────────────────────────────────────────────────────────────
import base64
import json
import time
import warnings
from datetime import datetime
from io import BytesIO
from typing import Any, Dict, List
import os
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import requests
import streamlit as st
from plotly.subplots import make_subplots

warnings.filterwarnings("ignore")

# ===============================
# 🔧 CONFIGURATION
# ===============================
API_BASE_URL = os.environ.get("API_BASE_URL", "http://localhost:8000")
TIMEOUT = 30


# ===============================
# CUSTOM CSS & STYLING
# ===============================
def load_custom_css():
    """Load custom CSS for professional medical interface"""
    st.markdown("""
    <style>
    /* Import fonts */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

    /* Global styling */
    .main .block-container {
        padding-top: 0rem !important;
        padding-bottom: 2rem;
        max-width: 1200px;
    }

    /* Main content area */
    div[data-testid="stMainBlockContainer"], section[data-testid="stMain"] {
        padding-top: 0.5rem !important;
    }

    /* Header styling */
    .medical-header {
        background: linear-gradient(135deg, #1e3c72 0%, #2a5298 100%);
        padding: 2rem;
        border-radius: 15px;
        color: white;
        text-align: center;
        margin-bottom: 2rem;
        box-shadow: 0 8px 32px rgba(0, 0, 0, 0.1);
    }

    .medical-header h1 {
        font-family: 'Inter', sans-serif;
        font-weight: 700;
        font-size: 2.5rem;
        margin-bottom: 0.5rem;
        text-shadow: 2px 2px 4px rgba(0, 0, 0, 0.3);
    }

    .medical-header p {
        font-size: 1.1rem;
        opacity: 0.9;
        margin: 0;
    }

    /* Risk gauge container */
    .risk-gauge-container {
        background: white;
        border-radius: 20px;
        padding: 2rem;
        box-shadow: 0 8px 32px rgba(0, 0, 0, 0.1);
        border: 1px solid #e0e7ff;
        margin: 1rem 0;
    }

    /* Risk level styling */
    .risk-low {
        background: linear-gradient(135deg, #10b981 0%, #059669 100%);
        color: white;
        padding: 1rem;
        border-radius: 12px;
        text-align: center;
        font-weight: 600;
        margin: 1rem 0;
    }

    .risk-moderate {
        background: linear-gradient(135deg, #f59e0b 0%, #d97706 100%);
        color: white;
        padding: 1rem;
        border-radius: 12px;
        text-align: center;
        font-weight: 600;
        margin: 1rem 0;
    }

    .risk-high {
        background: linear-gradient(135deg, #ef4444 0%, #dc2626 100%);
        color: white;
        padding: 1rem;
        border-radius: 12px;
        text-align: center;
        font-weight: 600;
        margin: 1rem 0;
    }

    /* Medical panel styling */
    .medical-panel {
        background: #f8fafc;
        border: 2px solid #e2e8f0;
        border-radius: 15px;
        padding: 1.5rem;
        margin: 1rem 0;
    }

    .medical-panel-header {
        color: #1e40af;
        font-weight: 600;
        font-size: 1.2rem;
        margin-bottom: 1rem;
        padding-bottom: 0.5rem;
        border-bottom: 2px solid #ddd6fe;
    }

    /* Feature importance styling */
    .feature-card {
        background: white;
        border-radius: 10px;
        padding: 1rem;
        margin: 0.5rem 0;
        border-left: 4px solid;
        box-shadow: 0 2px 8px rgba(0, 0, 0, 0.1);
    }

    .feature-increase {
        border-left-color: #ef4444;
        background: linear-gradient(90deg, #fef2f2 0%, #ffffff 100%);
    }

    .feature-decrease {
        border-left-color: #10b981;
        background: linear-gradient(90deg, #f0fdf4 0%, #ffffff 100%);
    }

    /* Metrics display */
    .metric-card {
        background: white;
        border-radius: 12px;
        padding: 1.5rem;
        text-align: center;
        box-shadow: 0 4px 16px rgba(0, 0, 0, 0.1);
        border: 1px solid #e5e7eb;
    }

    .metric-value {
        font-size: 2rem;
        font-weight: 700;
        color: #1f2937;
    }

    .metric-label {
        color: #6b7280;
        font-size: 0.9rem;
        margin-top: 0.5rem;
    }

    /* Clinical summary styling */
    .clinical-summary {
        background: linear-gradient(135deg, #f0f9ff 0%, #e0f2fe 100%);
        border: 2px solid #0ea5e9;
        border-radius: 15px;
        padding: 1.5rem;
        margin: 1rem 0;
    }

    .clinical-summary h4 {
        color: #0c4a6e;
        margin-bottom: 1rem;
    }

    /* Warning box */
    .insight-box {
        background: linear-gradient(135deg, #fef3c7 0%, #fde68a 100%);
        border: 2px solid #f59e0b;
        border-radius: 12px;
        padding: 1rem;
        margin: 1rem 0;
    }

    .insight-box h5 {
        color: #92400e;
        margin-bottom: 0.5rem;
    }

    /* Sidebar styling */
    .sidebar .sidebar-content {
        background: linear-gradient(180deg, #f8fafc 0%, #f1f5f9 100%);
    }

    /* Button styling */
    .stButton > button {
        background: linear-gradient(135deg, #3b82f6 0%, #1d4ed8 100%);
        color: white;
        border: none;
        border-radius: 10px;
        padding: 0.75rem 1.5rem;
        font-weight: 600;
        transition: all 0.3s ease;
    }

    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 16px rgba(59, 130, 246, 0.3);
    }

    /* Footer styling */
    .medical-footer {
        background: #1f2937;
        color: white;
        padding: 2rem;
        border-radius: 15px;
        text-align: center;
        margin-top: 3rem;
    }

    /* Hide Streamlit default elements */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}

    /* Custom scrollbar */
    ::-webkit-scrollbar {
        width: 8px;
    }

    ::-webkit-scrollbar-track {
        background: #f1f1f1;
        border-radius: 10px;
    }

    ::-webkit-scrollbar-thumb {
        background: #3b82f6;
        border-radius: 10px;
    }

    ::-webkit-scrollbar-thumb:hover {
        background: #1d4ed8;
    }

    /* Expander header styling */
    .streamlit-expander .streamlit-expanderHeader {
        background: linear-gradient(135deg, #3b82f6 0%, #1d4ed8 100%) !important;
        color: white !important;
        border-radius: 12px !important;
        padding: 1rem !important;
        border: none !important;
        font-weight: 600 !important;
        box-shadow: 0 4px 12px rgba(59, 130, 246, 0.3) !important;
    }

    /* Alternative expander targeting */
    details[open] > summary,
    details > summary {
        background: linear-gradient(135deg, #3b82f6 0%, #1d4ed8 100%) !important;
        color: white !important;
        padding: 1rem !important;
        border-radius: 12px !important;
        border: none !important;
        font-weight: 600 !important;
        list-style: none !important;
    }

    /* Hide default arrow */
    details > summary::-webkit-details-marker {
        display: none;
    }

    /* Content styling */
    details[open] {
        border: 2px solid #3b82f6 !important;
        border-radius: 12px !important;
        overflow: hidden !important;
    }

    /* Mobile responsive */
    @media (max-width: 768px) {
        .stColumns > div {
            text-align: center !important;
        }

        .stColumns h3,
        div[data-testid="column"] h3,
        .element-container h3 {
            text-align: center !important;
            margin-left: auto !important;
            margin-right: auto !important;
        }

        .stColumns p,
        .stColumns div,
        div[data-testid="column"] p,
        div[data-testid="column"] div {
            text-align: center !important;
        }

        .element-container p {
            text-align: center !important;
        }

        .stColumns .stMarkdown,
        div[data-testid="column"] .stMarkdown {
            text-align: center !important;
        }

        .stColumns * {
            text-align: center !important;
        }
    }

    </style>
    """, unsafe_allow_html=True)
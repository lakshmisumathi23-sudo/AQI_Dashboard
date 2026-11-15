# ============================================================
# INDIAN AIR QUALITY INDEX (AQI) PREDICTION DASHBOARD
# Simple blue-and-white Streamlit application
# ============================================================

import streamlit as st
import pandas as pd
import joblib
from datetime import date
from io import BytesIO

from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle


# -------------------- PAGE CONFIGURATION --------------------

st.set_page_config(
    page_title="Indian AQI Dashboard",
    page_icon="🌍",
    layout="wide"
)

MODEL_PATH = "indian_aqi_prediction_package.pkl"
DATA_PATH = "aqi.csv"


# -------------------- SIMPLE BLUE-WHITE DESIGN --------------------

st.markdown("""
<style>
.stApp {
    background-color: #DCEEFF;
}
.block-container {
    padding-top: 1.7rem;
    padding-bottom: 2rem;
}
h1, h2, h3, h4, p, label, .stMarkdown {
    color: #000000 !important;
}
h1, h2, h3 {
    font-weight: 800 !important;
}
[data-testid="stSidebar"] {
    background-color: #B8DCFF;
}
div[data-testid="stForm"] {
    background-color: #FFFFFF;
    padding: 24px;
    border-radius: 12px;
    border: 1px solid #90CAF9;
}
[data-testid="stMetric"] {
    background-color: #FFFFFF;
    padding: 18px;
    border-radius: 10px;
    border-left: 5px solid #1976D2;
}
[data-testid="stMetricLabel"] {
    color: #000000 !important;
    font-weight: 700 !important;
}
[data-testid="stMetricValue"] {
    color: #000000 !important;
    font-weight: 800 !important;
}
.stButton > button, .stDownloadButton > button {
    background-color: #1976D2;
    color: #FFFFFF !important;
    border-radius: 8px;
    border: none;
    font-weight: 800;
    min-height: 44px;
}
.stButton > button:hover, .stDownloadButton > button:hover {
    background-color: #0D47A1;
    color: #FFFFFF !important;
}
</style>
""", unsafe_allow_html=True)


# -------------------- LOAD TRAINED MODEL --------------------

@st.cache_resource
def load_model_package():
    return joblib.load(MODEL_PATH)


try:
    model_package = load_model_package()
    best_model = model_package["model"]
except FileNotFoundError:

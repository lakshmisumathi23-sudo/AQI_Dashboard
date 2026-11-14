
# ============================================================
# INDIAN AIR QUALITY INDEX DASHBOARD
# Simple Blue Theme + Downloadable AQI PDF Report
# ============================================================

# 1. IMPORT LIBRARIES

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import xgboost as xgb
import matplotlib.pyplot as plt

from datetime import date
from io import BytesIO

from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.platypus import (
    SimpleDocTemplate,
    Paragraph,
    Spacer,
    Table,
    TableStyle
)


# 2. PAGE CONFIGURATION

st.set_page_config(
    page_title="Indian AQI Dashboard",
    page_icon="🌍",
    layout="wide"
)


# 3. DASHBOARD DESIGN

st.markdown("""
<style>

.stApp {
    background-color: #DCEEFF;
}

.block-container {
    padding-top: 1.5rem;
}

h1, h2, h3 {
    color: #000000 !important;
    font-weight: 800;
}

[data-testid="stSidebar"] {
    background-color: #B8DCFF;
}

[data-testid="stMetric"] {
    background-color: #FFFFFF;
    padding: 20px;
    border-radius: 10px;
    border-left: 5px solid #1976D2;
    box-shadow: 0 3px 8px rgba(0,0,0,0.08);
}

[data-testid="stMetricLabel"] {
    color: #000000 !important;
    font-weight: bold;
}

[data-testid="stMetricValue"] {
    color: #1565C0 !important;
    font-weight: 800;
}

.stButton > button {
    background-color: #1976D2;
    color: white;
    border-radius: 8px;
    border: none;
    font-weight: bold;
}

.stButton > button:hover {
    background-color: #0D47A1;
    color: white;
}

[data-testid="stDataFrame"] {
    border: 1px solid #90CAF9;
    border-radius: 8px;

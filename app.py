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
DATA_PATH = "locations.csv"


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
    best_model = None

with col1:
    selected_state = st.selectbox("Select State", state_options)

    area_options = get_areas_for_state(selected_state)
    selected_area = st.selectbox("Select City / Area", area_options)

    stations = st.number_input(
        "Number of Monitoring Stations",
        min_value=1,
        max_value=100,
        value=5,
        step=1
    )
        
with col2:
        selected_pollutant = st.selectbox(
            "Prominent Pollutant",
            pollutant_options
        )

        selected_date = st.date_input(
            "Select Date",
            value=date.today()
        )

    submitted = st.form_submit_button(
        "CALCULATE AQI",
        use_container_width=True
    )


# -------------------- PREDICTION --------------------

if submitted:
    input_data = pd.DataFrame({
        "state": [selected_state],
        "area": [selected_area],
        "number_of_monitoring_stations": [int(stations)],
        "prominent_pollutants": [selected_pollutant],
        "year": [selected_date.year],
        "month": [selected_date.month],
        "day": [selected_date.day],
        "day_of_week": [selected_date.weekday()],
        "day_of_year": [selected_date.timetuple().tm_yday],
        "quarter": [((selected_date.month - 1) // 3) + 1]
    })

    try:
        prediction = float(best_model.predict(input_data)[0])
        prediction = max(0.0, min(500.0, prediction))
        category, health_message, category_color = classify_aqi(prediction)

        st.divider()
        st.subheader("AQI Prediction Result")

        result_col1, result_col2 = st.columns(2)

        with result_col1:
            st.metric("Predicted AQI", f"{prediction:.2f}")

        with result_col2:
            st.markdown(
                f"""
                <div style="
                    background-color: white;
                    border-left: 8px solid {category_color};
                    border-radius: 10px;
                    padding: 18px;
                    min-height: 95px;">
                    <div style="font-size:16px; color:#000000; font-weight:bold;">
                        AIR QUALITY CATEGORY
                    </div>
                    <div style="font-size:30px; color:{category_color}; font-weight:800;">
                        {category}
                    </div>
                </div>
                """,
                unsafe_allow_html=True
            )

        st.markdown("### Health Advisory")
        st.info(health_message)

        st.caption(
            f"Location: {selected_area}, {selected_state} | "
            f"Date: {selected_date} | Pollutant: {selected_pollutant}"
        )

        pdf_file = create_pdf_report(
            selected_state,
            selected_area,
            int(stations),
            selected_pollutant,
            selected_date,
            prediction,
            category
        )

        st.download_button(
            label="DOWNLOAD AQI REPORT (PDF)",
            data=pdf_file,
            file_name=f"AQI_Report_{selected_area}_{selected_date}.pdf",
            mime="application/pdf"
        )

        st.warning(
            "This is an ML-generated prediction based on the inputs provided. "
            "It is not a live AQI measurement or an official CPCB reading."
        )

    except Exception as error:
        st.error("Prediction could not be completed. Please check the inputs and model.")
        st.code(str(error))


# -------------------- FOOTER --------------------

st.divider()
st.caption("Indian AQI Prediction Dashboard | Machine Learning Deployment")


# ============================================================
# INDIAN AIR QUALITY INDEX (AQI) PREDICTION DASHBOARD
# Blue-and-white Streamlit application
# ============================================================

from datetime import date
from io import BytesIO
from pathlib import Path

import joblib
import pandas as pd
import streamlit as st

from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.platypus import (
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)


# ============================================================
# 1. PAGE CONFIGURATION AND FILE PATHS
# ============================================================

st.set_page_config(
    page_title="Indian AQI Dashboard",
    page_icon="🌍",
    layout="wide",
)

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "indian_aqi_prediction_package.pkl"
DATA_PATH = BASE_DIR / "locations.csv"


# ============================================================
# 2. DASHBOARD DESIGN
# ============================================================

st.markdown(
    """
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

    .stButton > button,
    .stDownloadButton > button,
    [data-testid="stFormSubmitButton"] > button {
        background-color: #1976D2;
        color: #FFFFFF !important;
        border-radius: 8px;
        border: none;
        font-weight: 800;
        min-height: 44px;
    }

    .stButton > button:hover,
    .stDownloadButton > button:hover,
    [data-testid="stFormSubmitButton"] > button:hover {
        background-color: #0D47A1;
        color: #FFFFFF !important;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# 3. HELPER FUNCTIONS
# ============================================================

def find_column(dataframe, possible_names):
    """Find a dataframe column without depending on capitalization."""
    normalized_columns = {
        str(column).strip().lower().replace(" ", "_"): column
        for column in dataframe.columns
    }

    for name in possible_names:
        key = name.strip().lower().replace(" ", "_")
        if key in normalized_columns:
            return normalized_columns[key]

    return None


@st.cache_data
def load_locations():
    """Load state, area and optional pollutant values from locations.csv."""
    if not DATA_PATH.exists():
        raise FileNotFoundError(
            "locations.csv was not found. Upload locations.csv "
            "to the same GitHub repository folder as app.py."
        )

    dataframe = pd.read_csv(DATA_PATH)

    if dataframe.empty:
        raise ValueError("locations.csv is empty.")

    state_column = find_column(
        dataframe,
        ["state", "state_name"],
    )
    area_column = find_column(
        dataframe,
        ["area", "city", "city_area", "location", "district"],
    )
    pollutant_column = find_column(
        dataframe,
        [
            "prominent_pollutants",
            "prominent_pollutant",
            "pollutant",
        ],
    )

    if state_column is None or area_column is None:
        raise ValueError(
            "locations.csv must contain a state column and an area/city "
            "column. Accepted names include State and Area or State and City."
        )

    dataframe = dataframe.copy()
    dataframe[state_column] = dataframe[state_column].astype(str).str.strip()
    dataframe[area_column] = dataframe[area_column].astype(str).str.strip()

    dataframe = dataframe[
        (dataframe[state_column] != "")
        & (dataframe[area_column] != "")
        & (dataframe[state_column].str.lower() != "nan")
        & (dataframe[area_column].str.lower() != "nan")
    ]

    state_values = sorted(dataframe[state_column].dropna().unique().tolist())

    pollutant_values = []
    if pollutant_column is not None:
        pollutant_values = sorted(
            dataframe[pollutant_column]
            .dropna()
            .astype(str)
            .str.strip()
            .loc[lambda values: values != ""]
            .unique()
            .tolist()
        )

    if not pollutant_values:
        pollutant_values = [
            "PM2.5",
            "PM10",
            "NO2",
            "SO2",
            "CO",
            "O3",
            "Multiple Pollutants",
        ]

    return dataframe, state_column, area_column, state_values, pollutant_values


def get_areas_for_state(dataframe, state_column, area_column, state_name):
    """Return the areas available for the selected state."""
    areas = (
        dataframe.loc[
            dataframe[state_column] == state_name,
            area_column,
        ]
        .dropna()
        .astype(str)
        .str.strip()
        .unique()
        .tolist()
    )

    return sorted([area for area in areas if area and area.lower() != "nan"])


def classify_aqi(aqi_value):
    """Classify the predicted AQI using the dashboard's AQI bands."""
    if aqi_value <= 50:
        return (
            "Good",
            "Air quality is considered satisfactory. Enjoy normal outdoor activities.",
            "#008000",
        )

    if aqi_value <= 100:
        return (
            "Satisfactory",
            "Air quality is acceptable. Unusually sensitive people should consider limiting prolonged outdoor exertion.",
            "#7CB342",
        )

    if aqi_value <= 200:
        return (
            "Moderate",
            "People with breathing or heart-related conditions should reduce prolonged outdoor exertion.",
            "#F9A825",
        )

    if aqi_value <= 300:
        return (
            "Poor",
            "People may experience breathing discomfort. Reduce prolonged outdoor activities.",
            "#EF6C00",
        )

    if aqi_value <= 400:
        return (
            "Very Poor",
            "Prolonged exposure may cause respiratory effects. Avoid strenuous outdoor activities.",
            "#D32F2F",
        )

    return (
        "Severe",
        "Health effects may occur even in healthy people. Avoid outdoor exertion and follow local health advisories.",
        "#7B1FA2",
    )


def create_pdf_report(
    state_name,
    area_name,
    station_count,
    pollutant_name,
    selected_day,
    predicted_aqi,
    category,
):
    """Create a downloadable PDF report for the prediction."""
    pdf_buffer = BytesIO()

    document = SimpleDocTemplate(
        pdf_buffer,
        pagesize=A4,
        rightMargin=40,
        leftMargin=40,
        topMargin=45,
        bottomMargin=40,
    )

    styles = getSampleStyleSheet()
    story = []

    story.append(
        Paragraph(
            "<b>INDIAN AIR QUALITY INDEX (AQI) REPORT</b>",
            styles["Title"],
        )
    )
    story.append(Spacer(1, 18))

    report_rows = [
        ["Parameter", "Details"],
        ["State", str(state_name)],
        ["City / Area", str(area_name)],
        ["Monitoring Stations", str(station_count)],
        ["Prominent Pollutant", str(pollutant_name)],
        ["Selected Date", str(selected_day)],
        ["Predicted AQI", f"{predicted_aqi:.2f}"],
        ["Air Quality Category", str(category)],
    ]

    report_table = Table(
        report_rows,
        colWidths=[190, 300],
    )

    report_table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#1976D2")),
                ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                ("FONTNAME", (0, 1), (0, -1), "Helvetica-Bold"),
                ("BACKGROUND", (0, 1), (-1, -1), colors.HexColor("#EAF4FF")),
                ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
                ("PADDING", (0, 0), (-1, -1), 9),
                ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
            ]
        )
    )

    story.append(report_table)
    story.append(Spacer(1, 18))
    story.append(
        Paragraph(
            "Note: This is a machine-learning-generated prediction based "
            "on the inputs provided. It is not a live AQI measurement or "
            "an official CPCB reading.",
            styles["BodyText"],
        )
    )

    document.build(story)
    pdf_buffer.seek(0)

    return pdf_buffer.getvalue()


# ============================================================
# 4. LOAD LOCATION DATA
# ============================================================

st.title("🌍 Indian AQI Prediction Dashboard")
st.write(
    "Select a state, city/area, monitoring station count, pollutant "
    "and date to estimate the Air Quality Index."
)

try:
    (
        locations_df,
        state_column,
        area_column,
        state_options,
        pollutant_options,
    ) = load_locations()

except Exception as error:
    st.error("Location data could not be loaded.")
    st.write(str(error))
    st.stop()

if not state_options:
    st.error("No valid state names were found in locations.csv.")
    st.stop()


# ============================================================
# 5. LOAD THE TRAINED MACHINE-LEARNING MODEL
# ============================================================

@st.cache_resource
def load_model_package():
    """Load the trained model package from the app folder."""
    return joblib.load(MODEL_PATH)


best_model = None
model_error = None

try:
    model_package = load_model_package()

    if isinstance(model_package, dict):
        best_model = model_package.get("model")
    else:
        best_model = model_package

    if best_model is None:
        model_error = (
            "The model package does not contain a usable model under "
            "the key 'model'. Check the saved model package."
        )

except FileNotFoundError:
    model_error = (
        "The trained model file was not found. Upload "
        "indian_aqi_prediction_package.pkl to the same repository folder "
        "as app.py."
    )

except Exception as error:
    model_error = f"The trained model could not be loaded: {error}"

if model_error:
    st.warning(model_error)


# ============================================================
# 6. USER INPUT FORM
# ============================================================

st.subheader("Enter AQI Prediction Details")

with st.form("aqi_prediction_form"):

    col1, col2 = st.columns(2)

    with col1:
        selected_state = st.selectbox(
            "Select State",
            options=state_options,
        )

        area_options = get_areas_for_state(
            locations_df,
            state_column,
            area_column,
            selected_state,
        )

        if not area_options:
            area_options = ["No area available"]

        selected_area = st.selectbox(
            "Select City / Area",
            options=area_options,
        )

        stations = st.number_input(
            "Number of Monitoring Stations",
            min_value=1,
            max_value=100,
            value=5,
            step=1,
        )

    with col2:
        selected_pollutant = st.selectbox(
            "Prominent Pollutant",
            options=pollutant_options,
        )

        selected_date = st.date_input(
            "Select Date",
            value=date.today(),
        )

    submitted = st.form_submit_button(
        "CALCULATE AQI",
        use_container_width=True,
    )


# ============================================================
# 7. AQI PREDICTION
# ============================================================

if submitted:

    if best_model is None:
        st.error(
            "Prediction cannot be performed because the trained model "
            "is unavailable. Check the model file in your GitHub repository."
        )

    elif selected_area == "No area available":
        st.error("Please select a valid city or area.")

    else:
        input_data = pd.DataFrame(
            {
                "state": [selected_state],
                "area": [selected_area],
                "number_of_monitoring_stations": [int(stations)],
                "prominent_pollutants": [selected_pollutant],
                "year": [selected_date.year],
                "month": [selected_date.month],
                "day": [selected_date.day],
                "day_of_week": [selected_date.weekday()],
                "day_of_year": [selected_date.timetuple().tm_yday],
                "quarter": [((selected_date.month - 1) // 3) + 1],
            }
        )

        try:
            prediction = float(best_model.predict(input_data)[0])
            prediction = max(0.0, min(500.0, prediction))

            category, health_message, category_color = classify_aqi(
                prediction
            )

            st.divider()
            st.subheader("AQI Prediction Result")

            result_col1, result_col2 = st.columns(2)

            with result_col1:
                st.metric(
                    label="Predicted AQI",
                    value=f"{prediction:.2f}",
                )

            with result_col2:
                st.markdown(
                    f"""
                    <div style="
                        background-color: white;
                        border-left: 8px solid {category_color};
                        border-radius: 10px;
                        padding: 18px;
                        min-height: 95px;
                    ">
                        <div style="
                            font-size: 16px;
                            color: #000000;
                            font-weight: bold;
                        ">
                            AIR QUALITY CATEGORY
                        </div>
                        <div style="
                            font-size: 30px;
                            color: {category_color};
                            font-weight: 800;
                        ">
                            {category}
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

            st.markdown("### Health Advisory")
            st.info(health_message)

            st.caption(
                f"Location: {selected_area}, {selected_state} | "
                f"Date: {selected_date} | "
                f"Pollutant: {selected_pollutant}"
            )

            pdf_file = create_pdf_report(
                selected_state,
                selected_area,
                int(stations),
                selected_pollutant,
                selected_date,
                prediction,
                category,
            )

            st.download_button(
                label="DOWNLOAD AQI REPORT (PDF)",
                data=pdf_file,
                file_name=f"AQI_Report_{selected_area}_{selected_date}.pdf",
                mime="application/pdf",
            )

            st.warning(
                "This is an ML-generated prediction based on the inputs "
                "provided. It is not a live AQI measurement or an official "
                "CPCB reading."
            )

        except Exception as error:
            st.error(
                "Prediction could not be completed. Please check the "
                "model and input feature names."
            )
            st.code(str(error))


# ============================================================
# 8. FOOTER
# ============================================================

st.divider()
st.caption(
    "Indian AQI Prediction Dashboard | Machine Learning Deployment"
)


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
}

</style>
""", unsafe_allow_html=True)


# 4. AQI CATEGORY FUNCTION

def get_aqi_category(aqi):

    if aqi <= 50:
        return "Good"
    elif aqi <= 100:
        return "Satisfactory"
    elif aqi <= 200:
        return "Moderate"
    elif aqi <= 300:
        return "Poor"
    elif aqi <= 400:
        return "Very Poor"
    else:
        return "Severe"


# 5. AQI COLOR FUNCTION

def get_aqi_color(aqi):

    if aqi <= 50:
        return "#16A34A"
    elif aqi <= 100:
        return "#65A30D"
    elif aqi <= 200:
        return "#EAB308"
    elif aqi <= 300:
        return "#EA580C"
    elif aqi <= 400:
        return "#DC2626"
    else:
        return "#7E22CE"


# 6. LOAD DATASET

@st.cache_data
def load_data():

    df = pd.read_csv("aqi.csv")

    df = df.dropna(axis=1, how="all")

    df["date"] = pd.to_datetime(
        df["date"],
        dayfirst=True,
        errors="coerce"
    )

    df["aqi_value"] = pd.to_numeric(
        df["aqi_value"],
        errors="coerce"
    )

    df = df.dropna(
        subset=["date", "aqi_value"]
    )

    df["state"] = df["state"].astype(str).str.strip()
    df["area"] = df["area"].astype(str).str.strip()

    df = df[
        (df["aqi_value"] >= 0) &
        (df["aqi_value"] <= 500)
    ]

    return df


# 7. LOAD XGBOOST MODEL

@st.cache_resource
def load_model():

    preprocessor = joblib.load(
        "aqi_preprocessor.pkl"
    )

    model = xgb.XGBRegressor()

    model.load_model(
        "aqi_xgboost_model.json"
    )

    return preprocessor, model


# 8. LOAD FILES

try:

    df = load_data()
    preprocessor, model = load_model()

except Exception as e:

    st.error("Unable to load dataset or model.")
    st.write("Please check your project files.")
    st.code(str(e))
    st.stop()


# 9. MAIN DASHBOARD HEADING

st.markdown("""
<div style="
    background-color: #B8DCFF;
    padding: 25px;
    border-radius: 12px;
    text-align: center;
    border: 2px solid #1976D2;
">

<h1 style="
    color: #000000 !important;
    font-size: 38px;
    font-weight: 900;
    margin: 0;
">
INDIAN AIR QUALITY INDEX DASHBOARD
</h1>

<p style="
    color: #000000;
    font-size: 17px;
    margin-top: 10px;
">
Air Quality Monitoring and Machine Learning Prediction
</p>

</div>
""", unsafe_allow_html=True)

st.divider()


# 10. SIDEBAR NAVIGATION

st.sidebar.title("Dashboard Navigation")

page = st.sidebar.radio(
    "Select Dashboard Page",
    [
        "Dashboard Overview",
        "AQI Prediction",
        "Historical Analysis",
        "AQI Report"
    ]
)

st.sidebar.markdown("---")
st.sidebar.info("Machine Learning Model: XGBoost")


# ============================================================
# PAGE 1: DASHBOARD OVERVIEW
# ============================================================

if page == "Dashboard Overview":

    st.header("Dashboard Overview")

    st.write("Overview of air quality across India.")

    average_aqi = df["aqi_value"].mean()
    maximum_aqi = df["aqi_value"].max()
    minimum_aqi = df["aqi_value"].min()
    total_states = df["state"].nunique()

    # AQI STATISTICS

    st.subheader("Air Quality Statistics")

    col1, col2, col3, col4 = st.columns(4)

    col1.metric("Average AQI", f"{average_aqi:.2f}")
    col2.metric("Maximum AQI", f"{maximum_aqi:.0f}")
    col3.metric("Minimum AQI", f"{minimum_aqi:.0f}")
    col4.metric("States Covered", total_states)

    st.divider()

    # AQI TREND

    st.subheader("AQI Trend Over Time")

    trend_data = (
        df.groupby("date")["aqi_value"]
        .mean()
        .reset_index()
    )

    fig, ax = plt.subplots(figsize=(12, 5))

    ax.plot(
        trend_data["date"],
        trend_data["aqi_value"],
        color="#1976D2",
        linewidth=2
    )

    ax.set_xlabel("Date")
    ax.set_ylabel("Average AQI")
    ax.set_title("Overall Air Quality Trend")
    ax.grid(alpha=0.25)

    st.pyplot(fig)
    plt.close(fig)

    st.divider()

    # STATE-WISE AQI

    st.subheader("Top 15 States by Average AQI")

    state_data = (
        df.groupby("state")["aqi_value"]
        .mean()
        .sort_values(ascending=False)
        .head(15)
    )

    fig, ax = plt.subplots(figsize=(12, 5))

    state_data.plot(
        kind="bar",
        ax=ax,
        color="#42A5F5"
    )

    ax.set_xlabel("State")
    ax.set_ylabel("Average AQI")
    ax.set_title("State-wise AQI Comparison")
    plt.xticks(rotation=45, ha="right")

    st.pyplot(fig)
    plt.close(fig)

    st.divider()

    # AQI CATEGORY DISTRIBUTION

    st.subheader("AQI Category Distribution")

    category_data = df.copy()

    category_data["AQI Category"] = (
        category_data["aqi_value"]
        .apply(get_aqi_category)
    )

    category_order = [
        "Good", "Satisfactory", "Moderate",
        "Poor", "Very Poor", "Severe"
    ]

    category_counts = (
        category_data["AQI Category"]
        .value_counts()
        .reindex(category_order, fill_value=0)
    )

    fig, ax = plt.subplots(figsize=(9, 5))

    category_counts.plot(
        kind="bar",
        ax=ax,
        color="#64B5F6"
    )

    ax.set_xlabel("AQI Category")
    ax.set_ylabel("Number of Records")
    ax.set_title("Distribution of AQI Categories")
    plt.xticks(rotation=30)

    st.pyplot(fig)
    plt.close(fig)


# ============================================================
# PAGE 2: AQI PREDICTION
# ============================================================

elif page == "AQI Prediction":

    st.header("AQI Prediction")

    st.write(
        "Enter the required information to predict air quality."
    )

    st.divider()

    # INPUT 1: STATE

    st.subheader("1. Select State")

    state = st.selectbox(
        "Choose State",
        sorted(df["state"].dropna().unique())
    )

    # INPUT 2: AREA

    st.subheader("2. Select Area")

    state_areas = sorted(
        df[df["state"] == state]["area"]
        .dropna()
        .unique()
    )

    area = st.selectbox(
        "Choose Area",
        state_areas
    )

    # INPUT 3: MONITORING STATIONS

    st.subheader("3. Number of Monitoring Stations")

    station_count = st.number_input(
        "Enter Monitoring Stations",
        min_value=1,
        max_value=100,
        value=5
    )

    # INPUT 4: POLLUTANT

    st.subheader("4. Select Prominent Pollutant")

    pollutants = sorted(
        df["prominent_pollutants"]
        .dropna()
        .astype(str)
        .unique()
    )

    pollutant = st.selectbox(
        "Choose Pollutant",
        pollutants
    )

    # INPUT 5: DATE

    st.subheader("5. Select Prediction Date")

    prediction_date = st.date_input(
        "Choose Date",
        value=date(2025, 4, 30)
    )

    st.divider()

    # INPUT 6: PREDICTION

    st.subheader("6. Generate AQI Prediction")

    if st.button("Predict AQI"):

        input_data = pd.DataFrame({

            "state": [state],
            "area": [area],

            "number_of_monitoring_stations": [
                station_count
            ],

            "prominent_pollutants": [pollutant],

            "year": [prediction_date.year],
            "month": [prediction_date.month],
            "day": [prediction_date.day],

            "day_of_week": [
                prediction_date.weekday()
            ],

            "day_of_year": [
                prediction_date.timetuple().tm_yday
            ],

            "quarter": [
                (prediction_date.month - 1) // 3 + 1
            ]

        })

        try:

            processed_input = preprocessor.transform(
                input_data
            )

            prediction = model.predict(
                processed_input
            )[0]

            prediction = np.clip(prediction, 0, 500)

            category = get_aqi_category(prediction)
            color = get_aqi_color(prediction)

            st.success("Prediction Completed")

            col1, col2 = st.columns(2)

            col1.metric(
                "Predicted AQI",
                f"{prediction:.2f}"
            )

            col2.metric(
                "Air Quality Category",
                category
            )

            st.markdown(
                f"""
                <div style="
                    background-color: {color};
                    color: white;
                    padding: 20px;
                    text-align: center;
                    border-radius: 10px;
                    font-size: 24px;
                    font-weight: bold;">
                    Predicted Air Quality: {category}
                </div>
                """,
                unsafe_allow_html=True
            )

            st.caption(
                "This is a machine learning estimate, "
                "not an official real-time AQI measurement."
            )

        except Exception as e:

            st.error("Prediction failed.")
            st.code(str(e))


# ============================================================
# PAGE 3: HISTORICAL ANALYSIS
# ============================================================

elif page == "Historical Analysis":

    st.header("Historical AQI Analysis")

    st.write("Explore historical air quality records.")

    st.divider()

    # FILTER 1: STATE

    st.subheader("1. Select State")

    selected_state = st.selectbox(
        "Choose State",
        ["All States"] + sorted(
            df["state"].dropna().unique()
        )
    )

    filtered_df = df.copy()

    if selected_state != "All States":

        filtered_df = filtered_df[
            filtered_df["state"] == selected_state
        ]

    # FILTER 2: AREA

    st.subheader("2. Select Area")

    area_options = sorted(
        filtered_df["area"].dropna().unique()
    )

    selected_area = st.selectbox(
        "Choose Area",
        ["All Areas"] + area_options
    )

    if selected_area != "All Areas":

        filtered_df = filtered_df[
            filtered_df["area"] == selected_area
        ]

    st.divider()

    # HISTORICAL STATISTICS

    st.subheader("Historical AQI Statistics")

    col1, col2, col3 = st.columns(3)

    col1.metric(
        "Average AQI",
        f"{filtered_df['aqi_value'].mean():.2f}"
    )

    col2.metric(
        "Maximum AQI",
        f"{filtered_df['aqi_value'].max():.0f}"
    )

    col3.metric(
        "Minimum AQI",
        f"{filtered_df['aqi_value'].min():.0f}"
    )

    st.divider()

    # HISTORICAL TREND

    st.subheader("Historical AQI Trend")

    historical_trend = (
        filtered_df.groupby("date")["aqi_value"]
        .mean()
        .reset_index()
    )

    fig, ax = plt.subplots(figsize=(12, 5))

    ax.plot(
        historical_trend["date"],
        historical_trend["aqi_value"],
        color="#1976D2",
        linewidth=2
    )

    ax.set_xlabel("Date")
    ax.set_ylabel("AQI")
    ax.set_title("Historical Air Quality Trend")
    ax.grid(alpha=0.25)

    st.pyplot(fig)
    plt.close(fig)

    st.divider()

    # HISTORICAL RECORDS

    st.subheader("Historical AQI Records")

    st.dataframe(
        filtered_df.sort_values(
            "date",
            ascending=False
        ),
        use_container_width=True
    )


# ============================================================
# PAGE 4: AQI REPORT
# ============================================================

elif page == "AQI Report":

    st.header("AQI Report")

    st.write(
        "View the air quality summary and download "
        "a PDF report based on the uploaded dataset."
    )

    st.divider()

    # REPORT DATE RANGE

    report_start = df["date"].min().strftime("%d-%m-%Y")
    report_end = df["date"].max().strftime("%d-%m-%Y")

    st.subheader("1. Report Information")

    st.write("Report coverage:", report_start, "to", report_end)

    st.write("Total records:", f"{len(df):,}")

    st.write("States covered:", df["state"].nunique())

    st.write("Areas covered:", df["area"].nunique())

    st.divider()

    # AQI SUMMARY

    st.subheader("2. Overall AQI Statistics")

    average_aqi = df["aqi_value"].mean()
    maximum_aqi = df["aqi_value"].max()
    minimum_aqi = df["aqi_value"].min()

    col1, col2, col3 = st.columns(3)

    col1.metric("Average AQI", f"{average_aqi:.2f}")
    col2.metric("Maximum AQI", f"{maximum_aqi:.0f}")
    col3.metric("Minimum AQI", f"{minimum_aqi:.0f}")

    st.divider()

    # CATEGORY REPORT

    st.subheader("3. AQI Category Report")

    report_df = df.copy()

    report_df["AQI Category"] = (
        report_df["aqi_value"].apply(get_aqi_category)
    )

    category_order = [
        "Good", "Satisfactory", "Moderate",
        "Poor", "Very Poor", "Severe"
    ]

    category_report = (
        report_df["AQI Category"]
        .value_counts()
        .reindex(category_order, fill_value=0)
        .rename_axis("AQI Category")
        .reset_index(name="Number of Records")
    )

    category_report["Percentage"] = (
        category_report["Number of Records"]
        / len(report_df) * 100
    ).round(2)

    st.dataframe(
        category_report,
        use_container_width=True,
        hide_index=True
    )

    st.divider()

    # STATE-WISE REPORT

    st.subheader("4. State-wise Average AQI Report")

    state_report = (
        df.groupby("state")["aqi_value"]
        .agg(["mean", "max", "min", "count"])
        .reset_index()
    )

    state_report.columns = [
        "State",
        "Average AQI",
        "Maximum AQI",
        "Minimum AQI",
        "Number of Records"
    ]

    state_report["Average AQI"] = (
        state_report["Average AQI"].round(2)
    )

    state_report = state_report.sort_values(
        "Average AQI",
        ascending=False
    )

    st.dataframe(
        state_report,
        use_container_width=True,
        hide_index=True
    )

    st.divider()

    # PDF REPORT GENERATION

    st.subheader("5. Download AQI PDF Report")

    def create_pdf_report():

        buffer = BytesIO()

        document = SimpleDocTemplate(
            buffer,
            pagesize=A4,
            rightMargin=18 * mm,
            leftMargin=18 * mm,
            topMargin=18 * mm,
            bottomMargin=18 * mm
        )

        styles = getSampleStyleSheet()

        story = []

        # PDF title

        story.append(
            Paragraph(
                "INDIAN AIR QUALITY INDEX REPORT",
                styles["Title"]
            )
        )

        story.append(Spacer(1, 8 * mm))

        story.append(
            Paragraph(
                f"Dataset coverage: {report_start} to {report_end}",
                styles["Normal"]
            )
        )

        story.append(
            Paragraph(
                f"Total records: {len(df):,}",
                styles["Normal"]
            )
        )

        story.append(
            Paragraph(
                f"States covered: {df['state'].nunique()}",
                styles["Normal"]
            )
        )

        story.append(
            Paragraph(
                f"Areas covered: {df['area'].nunique()}",
                styles["Normal"]
            )
        )

        story.append(Spacer(1, 8 * mm))

        # Overall AQI table

        story.append(
            Paragraph(
                "Overall AQI Statistics",
                styles["Heading2"]
            )
        )

        summary_table = [
            ["Metric", "Value"],
            ["Average AQI", f"{average_aqi:.2f}"],
            ["Maximum AQI", f"{maximum_aqi:.0f}"],
            ["Minimum AQI", f"{minimum_aqi:.0f}"]
        ]

        table = Table(
            summary_table,
            colWidths=[75 * mm, 75 * mm]
        )

        table.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#1976D2")),
            ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
            ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
            ("PADDING", (0, 0), (-1, -1), 7),
            ("BACKGROUND", (0, 1), (-1, -1), colors.HexColor("#EAF4FF"))
        ]))

        story.append(table)

        story.append(Spacer(1, 8 * mm))

        # AQI category table

        story.append(
            Paragraph(
                "AQI Category Distribution",
                styles["Heading2"]
            )
        )

        category_pdf = [
            ["AQI Category", "Records", "Percentage"]
        ]

        for _, row in category_report.iterrows():

            category_pdf.append([
                str(row["AQI Category"]),
                str(int(row["Number of Records"])),
                f"{row['Percentage']:.2f}%"
            ])

        category_table = Table(
            category_pdf,
            colWidths=[60 * mm, 45 * mm, 45 * mm],
            repeatRows=1
        )

        category_table.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#1976D2")),
            ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
            ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
            ("PADDING", (0, 0), (-1, -1), 6)
        ]))

        story.append(category_table)

        story.append(Spacer(1, 8 * mm))

        # State-wise AQI table

        story.append(
            Paragraph(
                "State-wise Average AQI",
                styles["Heading2"]
            )
        )

        state_pdf = [[
            "State",
            "Average",
            "Maximum",
            "Minimum"
        ]]

        for _, row in state_report.iterrows():

            state_pdf.append([
                str(row["State"]),
                f"{row['Average AQI']:.2f}",
                f"{row['Maximum AQI']:.0f}",
                f"{row['Minimum AQI']:.0f}"
            ])

        state_table = Table(
            state_pdf,
            colWidths=[65 * mm, 30 * mm, 30 * mm, 30 * mm],
            repeatRows=1
        )

        state_table.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#1976D2")),
            ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
            ("GRID", (0, 0), (-1, -1), 0.4, colors.grey),
            ("PADDING", (0, 0), (-1, -1), 5),
            ("FONTSIZE", (0, 0), (-1, -1), 8)
        ]))

        story.append(state_table)

        story.append(Spacer(1, 8 * mm))

        # Report note

        story.append(
            Paragraph(
                "Note: This report summarizes the historical "
                "records available in the uploaded dataset. "
                "It is not an official real-time AQI report.",
                styles["Italic"]
            )
        )

        document.build(story)

        buffer.seek(0)

        return buffer.getvalue()

    pdf_file = create_pdf_report()

    st.download_button(
        label="Download AQI Report (PDF)",
        data=pdf_file,
        file_name="Indian_AQI_Report.pdf",
        mime="application/pdf"
    )


# ============================================================
# FOOTER
# ============================================================

st.divider()

st.markdown("""
<div style="
    text-align: center;
    color: #000000;
    padding: 10px;
    font-weight: bold;">
    Indian AQI Dashboard | XGBoost Machine Learning
</div>
""", unsafe_allow_html=True)

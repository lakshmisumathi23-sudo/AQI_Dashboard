
import streamlit as st

# Page configuration
st.set_page_config(
    page_title="AQI Dashboard",
    layout="centered"
)

# Blue background and dark input boxes
st.markdown("""
<style>
.stApp {
    background-color: #DCEBFA;
}

h1, h2, h3, p, label {
    color: #163A5F !important;
}

div[data-baseweb="select"] > div,
div[data-baseweb="input"] {
    background-color: #343A46 !important;
    border-radius: 8px;
}

div[data-baseweb="select"] span {
    color: white !important;
}

div[data-baseweb="input"] input {
    color: white !important;
    background-color: #343A46 !important;
}
</style>
""", unsafe_allow_html=True)

# Dashboard title
st.title("Indian AQI Dashboard")

st.write("Air Quality Index Monitoring")

# State selection
st.subheader("1. Select State")

states = [
    "Tamil Nadu",
    "Karnataka",
    "Kerala",
    "Delhi",
    "Maharashtra",
    "Andaman and Nicobar Islands"
]

state = st.selectbox("Select State", states)

# Area selection
st.subheader("2. Select Area")

areas = [
    "Chennai",
    "Coimbatore",
    "Madurai",
    "Bengaluru",
    "Kochi",
    "Sri Vijaya Puram"
]

area = st.selectbox("Select Area", areas)

# Monitoring stations
st.subheader("3. Number of Monitoring Stations")

stations = st.number_input(
    "Select Stations",
    min_value=1,
    max_value=100,
    value=5
)

# AQI score
st.subheader("4. AQI Score")

aqi = st.number_input(
    "Enter AQI Value",
    min_value=0,
    max_value=500,
    value=50
)

# Air quality category and colors

if aqi <= 50:
    quality = "Good"
    color = "#008000"

elif aqi <= 100:
    quality = "Satisfactory"
    color = "#66A000"

elif aqi <= 200:
    quality = "Moderate"
    color = "#FFA500"

elif aqi <= 300:
    quality = "Poor"
    color = "#FF0000"

elif aqi <= 400:
    quality = "Very Poor"
    color = "#800000"

else:
    quality = "Severe"
    color = "#800080"
# Display results
st.markdown("---")

st.subheader("AQI Results")

st.write("Selected State:", state)
st.write("Selected Area:", area)
st.write("Monitoring Stations:", stations)

st.metric("AQI Score", aqi)

st.markdown(
    f"<h3 style='color:{color}; font-weight:bold;'>"
    f"Air Quality: {quality}</h3>",
    unsafe_allow_html=True
)
# AQI quality reference
st.subheader("AQI Quality Reference")

st.write("""
0–50: Good

51–100: Satisfactory

101–200: Moderate

201–300: Poor

301–400: Very Poor

401–500: Severe
""")

import streamlit as st # type: ignore
import pandas as pd
from src.pipeline.prediction_pipeline import PredictionPipeline

st.set_page_config(page_title="House Price Predictor", layout="centered")

# Custom Colorful CSS
st.markdown("""
    <style>
    /* Main background gradient */
    .stApp {
        background: linear-gradient(135deg, #fdfbfb 0%, #ebedee 100%);
    }
    
    /* Header styling */
    h1 {
        color: #ff4b4b;
        background: -webkit-linear-gradient(45deg, #f093fb 0%, #f5576c 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-size: 3rem !important;
        font-weight: 800 !important;
        text-align: center;
        margin-bottom: 30px;
    }
    
    /* Input label styling */
    .stNumberInput label, .stSelectbox label {
        color: #4a4a4a !important;
        font-weight: 600;
        font-size: 1.1rem;
    }
    
    /* Make the predict button awesome */
    .stButton > button {
        background: linear-gradient(45deg, #0ba360 0%, #3cba92 100%) !important;
        color: white !important;
        border-radius: 30px !important;
        font-size: 1.2rem !important;
        font-weight: bold !important;
        padding: 10px 24px !important;
        border: none !important;
        box-shadow: 0 4px 15px rgba(11, 163, 96, 0.4) !important;
        transition: all 0.3s ease !important;
        width: 100%;
        margin-top: 20px;
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 20px rgba(11, 163, 96, 0.6) !important;
    }

    /* Success message pop */
    .stSuccess {
        background-color: #d4edda !important;
        color: #155724 !important;
        border-radius: 10px !important;
        padding: 20px !important;
        font-size: 1.5rem !important;
        font-weight: 700 !important;
        text-align: center;
        box-shadow: 0 4px 15px rgba(0,0,0,0.1);
        border-left: 5px solid #28a745 !important;
    }
    </style>
""", unsafe_allow_html=True)

st.title("House Price Prediction")

st.write("Enter the housing details below to predict the house price:")

# Input fields
longitude = st.number_input("Longitude", value=-122.25, step=0.01)
latitude = st.number_input("Latitude", value=37.85, step=0.01)
housing_median_age = st.number_input("Housing Median Age", value=25, min_value=0)
total_rooms = st.number_input("Total Rooms", value=1500, min_value=0)
total_bedrooms = st.number_input("Total Bedrooms", value=4, min_value=0)
population = st.number_input("Population", value=1200, min_value=0)
households = st.number_input("Households", value=400, min_value=0)
median_income = st.number_input("Median Income", value=5.5, step=0.1)
ocean_proximity = st.selectbox("Ocean Proximity", options=["<1H OCEAN", "INLAND", "NEAR OCEAN", "NEAR BAY", "ISLAND"])

if st.button("Predict House Price"):
    data = pd.DataFrame({
        "longitude": [longitude],
        "latitude": [latitude],
        "housing_median_age": [housing_median_age],
        "total_rooms": [total_rooms],
        "total_bedrooms": [total_bedrooms],
        "population": [population],
        "households": [households],
        "median_income": [median_income],
        "ocean_proximity": [ocean_proximity]
    })

    pipeline = PredictionPipeline()
    result = pipeline.predict(data)

    st.success(f"Predicted House Price: ${result[0]:,.2f}")

st.markdown("---")
st.subheader("📊 Model Performance (Random Forest)")
st.write("These metrics represent the accuracy boundaries of this prediction model on unseen test data.")

col1, col2, col3, col4 = st.columns(4)
col1.metric(label="Accuracy (R²)", value="81.73%")
col2.metric(label="RMSE", value="$48,926")
col3.metric(label="MAE", value="$31,660")
col4.metric(label="Error Rate (MAPE)", value="17.78%")
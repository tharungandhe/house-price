import streamlit as st # type: ignore
import pandas as pd
from src.pipeline.prediction_pipeline import PredictionPipeline

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

if st.button("Predict House Price"):
    data = pd.DataFrame({
        "longitude": [longitude],
        "latitude": [latitude],
        "housing_median_age": [housing_median_age],
        "total_rooms": [total_rooms],
        "total_bedrooms": [total_bedrooms],
        "population": [population],
        "households": [households],
        "median_income": [median_income]
    })

    pipeline = PredictionPipeline()
    result = pipeline.predict(data)

    st.success(f"Predicted House Price: ${result[0]:,.2f}")
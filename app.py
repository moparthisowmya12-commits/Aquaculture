import streamlit as st
import joblib
import pandas as pd

st.set_page_config(page_title="Disease Risk Prediction", page_icon="🐟")

st.title("🐟 Aquaculture & Livestock Disease Risk Prediction")
st.write("Enter the required parameters to predict disease-risk level.")

# Load trained model
model = joblib.load("aquaculture_livestock_best_model.pkl")

# Sector selection
sector = st.selectbox(
    "Select Sector",
    ["Aquaculture", "Livestock"]
)

st.subheader("Enter Parameters")

if sector == "Aquaculture":

    water_temperature = st.number_input(
        "Water Temperature (°C)",
        min_value=0.0,
        max_value=50.0,
        value=25.0
    )

    ph_level = st.number_input(
        "pH Level",
        min_value=0.0,
        max_value=14.0,
        value=7.0
    )

    dissolved_oxygen = st.number_input(
        "Dissolved Oxygen (mg/L)",
        min_value=0.0,
        max_value=20.0,
        value=6.0
    )

    fish_count = st.number_input(
        "Fish Count",
        min_value=1,
        value=100
    )

    feed_quantity = st.number_input(
        "Feed Quantity (kg/day)",
        min_value=0.0,
        value=5.0
    )

    rainfall = st.number_input(
        "Rainfall (mm)",
        min_value=0.0,
        value=10.0
    )

    input_data = pd.DataFrame([{
        "WaterTemperature": water_temperature,
        "PH": ph_level,
        "DissolvedOxygen": dissolved_oxygen,
        "FishCount": fish_count,
        "FeedQuantity": feed_quantity,
        "Rainfall": rainfall
    }])

else:

    animal_count = st.number_input(
        "Animal Count",
        min_value=1,
        value=10
    )

    animal_age = st.number_input(
        "Animal Age (months)",
        min_value=0,
        value=12
    )

    body_temperature = st.number_input(
        "Body Temperature (°C)",
        min_value=30.0,
        max_value=45.0,
        value=38.5
    )

    feed_quantity = st.number_input(
        "Feed Quantity (kg/day)",
        min_value=0.0,
        value=5.0
    )

    rainfall = st.number_input(
        "Rainfall (mm)",
        min_value=0.0,
        value=10.0
    )

    input_data = pd.DataFrame([{
        "AnimalCount": animal_count,
        "AnimalAge": animal_age,
        "BodyTemperature": body_temperature,
        "FeedQuantity": feed_quantity,
        "Rainfall": rainfall
    }])


if st.button("🔍 Predict Disease Risk"):

    prediction = model.predict(input_data)

    st.success(f"Predicted Disease Risk: {prediction[0]}")

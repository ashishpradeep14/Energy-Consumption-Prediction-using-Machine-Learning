import streamlit as st
import pandas as pd
import joblib

# Load the saved pipeline (feature engineering + Random Forest model)
pipeline = joblib.load("random_forest_pipeline.pkl")

# Define the app title
st.title("Energy Consumption Prediction App")

# Provide a description
st.markdown("""
This app predicts *energy consumption* based on various input parameters.  
Please provide the following values for prediction:
""")

# Create input fields for user input
lights = st.number_input("Lights (in kWh):", min_value=0, value=20)
T1 = st.number_input("Temperature in Kitchen (T1, in °C):", min_value=-10.0, max_value=50.0, value=23.0, step=0.1)
RH_1 = st.number_input("Humidity in Kitchen (RH_1, in %):", min_value=0.0, max_value=100.0, value=30.0, step=0.1)
T2 = st.number_input("Temperature in Living Room (T2, in °C):", min_value=-10.0, max_value=50.0, value=24.0, step=0.1)
RH_2 = st.number_input("Humidity in Living Room (RH_2, in %):", min_value=0.0, max_value=100.0, value=35.0, step=0.1)
T_out = st.number_input("Outside Temperature (T_out, in °C):", min_value=-20.0, max_value=50.0, value=15.0, step=0.1)
Appliances_lag_1 = st.number_input("Previous Appliances Consumption (Lag 1, in kWh):", min_value=0, value=50)
Appliances_lag_2 = st.number_input("Previous Appliances Consumption (Lag 2, in kWh):", min_value=0, value=45)

# Collect input data into a dictionary
input_data = {
    "lights": lights,
    "T1": T1,
    "RH_1": RH_1,
    "T2": T2,
    "RH_2": RH_2,
    "T_out": T_out,
    "Appliances_lag_1": Appliances_lag_1,
    "Appliances_lag_2": Appliances_lag_2
}

# Convert input data to a DataFrame
input_df = pd.DataFrame([input_data])

# Add a button for prediction
if st.button("Predict Energy Consumption"):
    try:
        # Make prediction using the loaded pipeline
        prediction = pipeline.predict(input_df)
        
        # Display the prediction result
        st.success(f"Predicted Energy Consumption: {prediction[0]:.2f} kWh")
    except Exception as e:
        st.error(f"An error occurred during prediction: {e}")
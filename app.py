import streamlit as st
import joblib
import numpy as np

# Load the saved model and scaler (Make sure you uploaded them to Colab first!)
model = joblib.load('crop_svm_model.joblib')
scaler = joblib.load('app_scaler.joblib')

st.title("🌱 AgroSmart: Crop Recommendation")

# Input fields
n = st.number_input("Nitrogen", value=50)
p = st.number_input("Phosphorus", value=50)
k = st.number_input("Potassium", value=50)
temp = st.number_input("Temperature", value=25.0)
hum = st.number_input("Humidity", value=60.0)
ph = st.number_input("pH", value=6.5)
rain = st.number_input("Rainfall", value=100.0)

if st.button("Predict"):
    input_data = np.array([[n, p, k, temp, hum, ph, rain]])
    scaled_data = scaler.transform(input_data)
    prediction = model.predict(scaled_data)
    st.success(f"Best Crop: {prediction[0]}")
